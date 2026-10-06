r"""
One-off helper: check every gold label in the eval dataset against data/chunks/chunks.jsonl.

Usage (from the repo root):
    python eval\check_gold.py
    python eval\check_gold.py --dataset eval\dataset.jsonl --chunks data\chunks\chunks.jsonl

A gold entry is {"page": <html file stem, e.g. sql-comment>, "heading": <section heading>} or
{"page": ..., "anchor": ...}. It is OK when some chunk of that page lists the section in its `covers`
(the same rule the retrieval eval uses). The heading is compared with the LAST element of each covers
section_path, case-insensitively and ignoring extra spaces, so numbered headings must include the number
exactly as the docs print it (for example "19.4.1. Memory").

For every label that does not match, the script prints the headings that really exist on that page
(or the closest page names), so you can fix the label instead of guessing again.

Also checks the dataset shape: unique ids, valid types, and gold counts per type
(factual >= 1, multi_hop >= 2, ambiguous >= 2, unanswerable = 0).

Exit code 1 if anything is wrong. Tested on synthetic data only, not yet on the real chunks.jsonl.
"""
import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("factual", "multi_hop", "ambiguous", "unanswerable")


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def load_index(chunks_path: Path):
    headings: dict[str, dict[str, str]] = {}  # page -> {normalised heading: original heading}
    anchors: dict[str, set] = {}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            page = r["chunk_id"].split(":")[0]
            for c in r.get("covers", []):
                path = c.get("section_path") or []
                if path:
                    headings.setdefault(page, {})[norm(path[-1])] = path[-1]
                if c.get("anchor"):
                    anchors.setdefault(page, set()).add(c["anchor"])
    return headings, anchors


def check_gold(g: dict, headings: dict, anchors: dict) -> str | None:
    """Return None if the label is fine, otherwise a human-readable problem."""
    page = g.get("page")
    if page not in headings:
        near = difflib.get_close_matches(page or "", list(headings), n=5, cutoff=0.5)
        return f"page {page!r} not found. Closest pages: {near or 'none'}"
    if "anchor" in g and g["anchor"] in anchors.get(page, set()):
        return None
    if "heading" in g:
        if norm(g["heading"]) in headings[page]:
            return None
        avail = list(headings[page].values())
        near = difflib.get_close_matches(norm(g["heading"]), list(headings[page]), n=3, cutoff=0.4)
        near = [headings[page][n] for n in near]
        return (f"heading {g['heading']!r} not on page {page!r}. Closest: {near or 'none'}. "
                f"All {len(avail)} headings on that page: {avail[:15]}{' ...' if len(avail) > 15 else ''}")
    return f"gold entry needs a 'heading' or an 'anchor' that exists on page {page!r}: {g}"


def main(dataset_path: Path, chunks_path: Path) -> int:
    if not chunks_path.exists():
        raise SystemExit(f"No {chunks_path}. Run: python src\\ingest.py chunk")
    headings, anchors = load_index(chunks_path)
    rows = [json.loads(line) for line in dataset_path.open(encoding="utf-8") if line.strip()]

    problems = 0
    ids = Counter(r["id"] for r in rows)
    for i, n in ids.items():
        if n > 1:
            print(f"BAD   duplicate id {i}")
            problems += 1

    for r in rows:
        t, gold = r.get("type"), r.get("gold", [])
        if t not in TYPES:
            print(f"BAD   {r['id']}: unknown type {t!r}")
            problems += 1
        need = {"factual": 1, "multi_hop": 2, "ambiguous": 2}.get(t)
        if t == "unanswerable" and gold:
            print(f"BAD   {r['id']}: unanswerable question has gold labels")
            problems += 1
        if need and len(gold) < need:
            print(f"BAD   {r['id']}: {t} needs at least {need} gold section(s), has {len(gold)}")
            problems += 1
        for g in gold:
            msg = check_gold(g, headings, anchors)
            if msg:
                print(f"BAD   {r['id']}: {msg}")
                problems += 1
            else:
                print(f"ok    {r['id']}: {g.get('page')} / {g.get('heading') or g.get('anchor')}")

    print("\nQuestions per type:", dict(Counter(r.get("type") for r in rows)))
    print(f"{len(rows)} questions, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    ap.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks" / "chunks.jsonl")
    a = ap.parse_args()
    raise SystemExit(main(a.dataset, a.chunks))
