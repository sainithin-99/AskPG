r"""
Apply the pending d2 label edits to eval\dataset.jsonl, JSON-aware (no line numbers, no regex on text).

Usage (from the repo root, cmd):
    python eval\relabel_d2.py                      # DRY RUN: shows what would change, writes nothing
    python eval\relabel_d2.py --apply              # writes eval\dataset.jsonl
    python eval\relabel_d2.py --apply --add amb-03 <page> <anchor> --add amb-03 <page> <anchor>

Edits (replace = swap one gold entry, matched by id + page + heading; aborts if it is not found):
    multi-03: ddl-partitioning '5.12.2. Declarative Partitioning' -> '5.12.2.1. Example'
    multi-05: ssl-tcp '18.9. Secure TCP/IP Connections with SSL' -> '18.9.1. Basic Setup'
--add appends an extra acceptable gold {"page", "anchor"} to a question (used for amb-03; get the exact
page and anchor from `python eval\run_eval.py find auto_explain` and `... find pg_stat_statements`).

Line endings and every untouched line are preserved byte for byte, so `fc` shows only the changed lines.
"""
import argparse
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "dataset.jsonl"

REPLACE = [
    ("multi-03", {"page": "ddl-partitioning", "heading": "5.12.2. Declarative Partitioning"},
     {"page": "ddl-partitioning", "heading": "5.12.2.1. Example"}),
    ("multi-05", {"page": "ssl-tcp", "heading": "18.9. Secure TCP/IP Connections with SSL"},
     {"page": "ssl-tcp", "heading": "18.9.1. Basic Setup"}),
]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def same(a: dict, b: dict) -> bool:
    return a.get("page") == b["page"] and norm(a.get("heading", "")) == norm(b["heading"])


def main(apply: bool, adds: list) -> None:
    with PATH.open(encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)

    rows = {}  # id -> line index
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            rows[json.loads(line)["id"]] = i

    changed = set()

    def edit(qid: str, fn) -> None:
        if qid not in rows:
            raise SystemExit(f"ABORT: question id {qid!r} not in dataset")
        i = rows[qid]
        body = lines[i].rstrip("\r\n")
        eol = lines[i][len(body):]
        rec = json.loads(body)
        fn(rec)
        lines[i] = json.dumps(rec, ensure_ascii=False) + eol
        changed.add(i)

    for qid, old, new in REPLACE:
        def swap(rec, old=old, new=new, qid=qid):
            for k, g in enumerate(rec["gold"]):
                if same(g, old):
                    rec["gold"][k] = new
                    return
            raise SystemExit(f"ABORT: {qid} has no gold {old}. Current gold: {rec['gold']}")
        edit(qid, swap)

    for qid, page, anchor in adds:
        def add(rec, page=page, anchor=anchor):
            new = {"page": page, "anchor": anchor}
            if new in rec["gold"]:
                print(f"  {rec['id']}: {new} already present, skipped")
                return
            rec["gold"].append(new)
        edit(qid, add)

    for i in sorted(changed):
        print(f"line {i + 1}: {lines[i].strip()}\n")
    print(f"{len(changed)} line(s) changed.")
    if apply:
        with PATH.open("w", encoding="utf-8", newline="") as f:
            f.write("".join(lines))
        print(f"Written {PATH}")
    else:
        print("Dry run: nothing written. Re-run with --apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--add", nargs=3, action="append", default=[], metavar=("ID", "PAGE", "ANCHOR"))
    a = ap.parse_args()
    main(a.apply, a.add)