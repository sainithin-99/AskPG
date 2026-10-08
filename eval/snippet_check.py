r"""
Deterministic check of rule 4 of the AskPG system prompt: "Include a short SQL or config snippet only if a
source contains it." No LLM involved.

Usage (from the repo root, cmd):
    python eval\snippet_check.py g1
    python eval\snippet_check.py g1 --chunks data\chunks\chunks.jsonl --quiet

Reads eval\results\gen-<name>.json (status=ok answers only) and data\chunks\chunks.jsonl, and writes
eval\results\gen-<name>_snippets.json.

What counts as a snippet
  1. every fenced block (three backticks);
  2. every inline `code` span that has at least 3 whitespace-separated words (a statement or setting, for
     example `GRANT SELECT ON phone_number TO assistant;`). Single identifiers such as `work_mem` are NOT
     checked: they are names, not snippets.

How a snippet is checked
  Each line is normalised (lowercase, all whitespace removed, curly quotes and
  non-breaking hyphens unified, a trailing "-- comment" or "# comment" dropped, a line split on "...", an
  ellipsis character or ';' into segments). A segment shorter than 8 characters is ignored (')', 'END;' ...). A segment is FOUND when it is a
  substring of the normalised text (breadcrumb + body, exactly what the LLM saw) of
     (a) any chunk shown to the LLM, and separately (b) any chunk the answer cited.
  If some shown chunk ids are missing from chunks.jsonl, a non-match is reported as UNKNOWN, not UNSUPPORTED.
  Verdict per snippet: supported (all lines found in shown chunks), PARTIAL (some), UNSUPPORTED (none),
  trivial (no checkable line). "only uncited" means found in a shown chunk the answer did not cite.

What this does NOT prove (read before trusting)
  - A snippet that IS found can still be used wrongly (the claim around it may be false).
  - A snippet that is NOT found is not automatically wrong: the model may have joined lines, renamed a value,
    or abbreviated. UNSUPPORTED means "read it", and it breaks rule 4 either way (invented text).
  - A short line can match by coincidence (the 8-character floor is a heuristic, not a calibrated threshold).
  - It only sees snippets in backticks. A snippet written as plain prose is invisible to it.
  - Chunk text comes from chunks.jsonl as it is NOW. If you re-chunked after the run, the result is invalid:
    the script compares chunks_md5 with the run meta when the meta has it (g1 does not).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (RESULTS_DIR, CHUNKS_PATH)

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
INLINE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
COMMENT_RE = re.compile(r"(?:^|\s)(?:--|#)\s.*$")   # "x -- note", "# note"; NOT "--flag"
SPLIT_RE = re.compile(r"…|\.\.\.|;")   # ellipsis or statement separator
MIN_CHARS = 8
MIN_WORDS_INLINE = 3
TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‑": "-", "–": "-", "—": "-"})


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s.translate(TRANS).lower())


def segments(line: str) -> list[str]:
    """Normalised, checkable segments of one snippet line."""
    line = COMMENT_RE.sub("", line.strip())
    out = []
    for part in SPLIT_RE.split(line):
        n = norm(part)
        if len(n) >= MIN_CHARS:
            out.append(n)
    return out


def extract_snippets(answer: str) -> list[tuple[str, str]]:
    """[(kind, text)] with kind 'fence' or 'inline'."""
    snippets = [("fence", m.strip("\n")) for m in FENCE_RE.findall(answer)]
    rest = FENCE_RE.sub(" ", answer)
    for m in INLINE_RE.findall(rest):
        if len(m.split()) >= MIN_WORDS_INLINE:
            snippets.append(("inline", m))
    return snippets


def check_snippet(kind: str, text: str, shown: dict, cited: dict, n_missing: int = 0) -> dict:
    segs = [s for line in text.split("\n") for s in segments(line)]
    hay_shown = "\n".join(shown.values())
    hay_cited = "\n".join(cited.values())
    in_shown = [s for s in segs if s in hay_shown]
    in_cited = [s for s in segs if s in hay_cited]
    unmatched = [s for s in segs if s not in hay_shown]
    if not segs:
        verdict = "trivial"
    elif len(in_shown) == len(segs):
        verdict = "supported" if len(in_cited) == len(segs) else "supported (only uncited)"
    elif in_shown:
        verdict = "PARTIAL"
    else:
        verdict = "UNSUPPORTED"
    if n_missing and verdict in ("PARTIAL", "UNSUPPORTED"):
        verdict = f"UNKNOWN ({n_missing} chunk text(s) not loaded)"
    return {"kind": kind, "text": text, "n_segments": len(segs), "found_shown": len(in_shown),
            "found_cited": len(in_cited), "verdict": verdict, "unmatched": unmatched}


def load_text_by_id(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["chunk_id"]] = norm(f"{r['context']}\n{r['text']}")   # what build_messages shows the LLM
    return out


def main(name: str, chunks_path: Path, quiet: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_path = ev.RESULTS_DIR / f"gen-{name}.json"
    if not run_path.exists():
        raise SystemExit(f"No {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    text_by_id = load_text_by_id(chunks_path)

    now_md5 = hashlib.md5(chunks_path.read_bytes()).hexdigest()
    was = run["meta"].get("chunks_md5")
    if was is None:
        print(f"NOTE: run meta has no chunks_md5, so I cannot prove chunks.jsonl is unchanged since the run "
              f"(current md5 {now_md5}). Only valid if you have not re-chunked since {run['meta']['time']}.")
    elif was != now_md5:
        raise SystemExit(f"chunks.jsonl changed since the run (meta {was}, now {now_md5}). Result would be invalid.")
    else:
        print(f"chunks_md5 matches the run meta ({now_md5}).")

    records, missing_total = [], 0
    ok = [q for q in run["per_question"] if q["status"] == "ok"]
    for q in ok:
        shown_ids, cited_ids = q["sources_shown"], q["cited_chunks"]
        missing = [c for c in shown_ids if c not in text_by_id]
        missing_total += len(missing)
        shown = {c: text_by_id[c] for c in shown_ids if c in text_by_id}
        cited = {c: text_by_id[c] for c in cited_ids if c in text_by_id}
        snips = [check_snippet(k, t, shown, cited, len(missing)) for k, t in extract_snippets(q["answer"])]
        records.append({"id": q["id"], "type": q["type"], "snippets": snips, "missing_chunks": missing})

    print(f"\nRun gen-{name}: {len(run['per_question'])} questions, {len(ok)} with status=ok checked "
          f"(refusals and errors have no snippets to check).")
    if missing_total:
        print(f"WARNING: {missing_total} shown chunk id(s) not found in {chunks_path.name}; "
              f"verdicts for those answers may be too harsh.")
    for r in records:
        if not r["snippets"]:
            continue
        print(f"\n{r['id']} ({r['type']})" + (f"   [missing chunks: {r['missing_chunks']}]" if r["missing_chunks"] else ""))
        for s in r["snippets"]:
            one_line = s["text"].replace("\n", " ⏎ ")
            print(f"  {s['verdict']:<24} {s['kind']:<6} {s['found_shown']}/{s['n_segments']} lines  {one_line[:90]}")
            if not quiet:
                for u in s["unmatched"]:
                    print(f"      not found: {u[:100]}")

    verdicts = Counter(s["verdict"] for r in records for s in r["snippets"])
    with_snip = [r for r in records if r["snippets"]]
    flagged = [r for r in with_snip if any(s["verdict"] in ("PARTIAL", "UNSUPPORTED") for s in r["snippets"])]
    print(f"\nAnswers with status=ok: {len(ok)}   with at least one checked snippet: {len(with_snip)}   "
          f"with a PARTIAL/UNSUPPORTED snippet: {len(flagged)} {[r['id'] for r in flagged]}")
    print("Snippets by verdict: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))

    out = ev.RESULTS_DIR / f"gen-{name}_snippets.json"
    out.write_text(json.dumps({"run": name, "chunks_md5": now_md5, "answers_ok": len(ok),
                               "answers_with_snippet": len(with_snip), "flagged": [r["id"] for r in flagged],
                               "by_verdict": dict(verdicts), "per_answer": records},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="run label, e.g. g1 (reads eval\\results\\gen-<name>.json)")
    ap.add_argument("--chunks", type=Path, default=ev.CHUNKS_PATH)
    ap.add_argument("--quiet", action="store_true", help="do not list the unmatched lines")
    a = ap.parse_args()
    main(a.name, a.chunks, a.quiet)
