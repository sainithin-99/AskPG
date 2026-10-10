r"""
Audit of the LLM judge: writes a text file with, for a chosen set of answers, the answer, the numbered sources the
generator saw, and every judged claim with the judge's verdict and evidence, so a human can read them and find
judge FALSE POSITIVES (flagged but actually supported) and FALSE NEGATIVES (marked supported but not in the source).

Usage (from the repo root, cmd):
    python eval\audit_judge.py g1                 # writes eval\audit_g1.txt (UTF-8) and prints a short summary
    python eval\audit_judge.py g1 --extra 4 --seed 0

Selection (never by hand, so the sample is not cherry-picked):
  - every answer that has at least one not_supported / contradicted / invalid claim (tests false positives), plus
  - --extra randomly drawn answers (seeded) from the FULLY-SUPPORTED ones (tests false negatives).
Also prints the attribution split: claims "supported only by an uncited source" among claims WITH citations vs
claims WITHOUT any citation (a block citation at the end of a paragraph leaves earlier claims with cited=[], where
"uncited" is not meaningful).
Untested on the real files when written.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (RESULTS_DIR, load_chunks)


def main(name: str, extra: int, seed: int) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    judge = json.loads((ev.RESULTS_DIR / f"gen-{name}_judge.json").read_text(encoding="utf-8"))
    run = {q["id"]: q for q in json.loads((ev.RESULTS_DIR / f"gen-{name}.json").read_text(encoding="utf-8"))["per_question"]}
    by_id = {r["chunk_id"]: r for r in ev.load_chunks()}
    recs = [r for r in judge["per_question"] if not r["error"]]

    bad = lambda r: any(c["verdict"] != "supported" for c in r["claims"])  # noqa: E731
    flagged = [r for r in recs if bad(r)]
    clean = [r for r in recs if not bad(r) and r["claims"]]
    picked_clean = random.Random(seed).sample(clean, min(extra, len(clean)))
    chosen = sorted(flagged + picked_clean, key=lambda r: [x["id"] for x in recs].index(r["id"]))

    claims = [c for r in recs for c in r["claims"]]
    with_cit = [c for c in claims if c["cited"]]
    no_cit = [c for c in claims if not c["cited"]]
    print(f"claims {len(claims)}: with citations {len(with_cit)}, without {len(no_cit)}")
    print(f"found_in_other_source=True: with citations {sum(c['found_in_other_source'] for c in with_cit)} of {len(with_cit)}, "
          f"without citations {sum(c['found_in_other_source'] for c in no_cit)} of {len(no_cit)}")
    print(f"answers in the audit: {len(chosen)} ({len(flagged)} with flagged claims + {len(picked_clean)} random fully-supported: "
          f"{[r['id'] for r in picked_clean]})")

    lines = [f"AUDIT of gen-{name}_judge (judge {judge['meta']['judge_model']}, prompt {judge['meta']['prompt_md5']})",
             "Read each claim against the sources. Mark in your head: judge right / judge wrong.\n"]
    for r in chosen:
        q = run[r["id"]]
        lines += ["=" * 100, f"{r['id']} ({r['type']})  correctness={r['correctness']} completeness={r['completeness']}",
                  "-" * 100, "ANSWER:", q["answer"], "-" * 100, "SOURCES SHOWN TO THE GENERATOR:"]
        for i, cid in enumerate(q["sources_shown"], 1):
            c = by_id[cid]
            lines += [f"[{i}] {cid}  {c['context']}", c["text"], ""]
        lines.append("JUDGED CLAIMS:")
        for c in r["claims"]:
            lines.append(f"  {c['verdict'].upper():<13} cited={c['cited']} other_src={c['found_in_other_source']}  {c['claim']}")
            if c["evidence"]:
                lines.append(f"{'':<16}evidence: {c['evidence']}")
        if r["conflicts"]:
            lines.append(f"  CONFLICTS vs reference: {r['conflicts']}")
        if r["missing_points"]:
            lines.append(f"  MISSING vs reference: {r['missing_points']}")
        lines.append("")
    out = ev.RESULTS_DIR.parent / f"audit_{name}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--extra", type=int, default=4, help="random fully-supported answers to add")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.name, a.extra, a.seed)
