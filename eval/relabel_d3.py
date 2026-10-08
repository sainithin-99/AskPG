r"""
One-off: apply the d3 label edits to eval\dataset.jsonl (d2 -> d3), in ONE round.

Usage (from the repo root, cmd):
    python eval\relabel_d3.py

What it does
  1. Refuses to run if eval\dataset_pre_d3.jsonl already exists (so a re-run cannot overwrite the d2 backup).
  2. Copies eval\dataset.jsonl to eval\dataset_pre_d3.jsonl.
  3. Adds the gold sections below to the listed questions (nothing is removed or reordered).
  4. Runs eval\check_gold.py. A pass proves each section EXISTS, not that it answers the question.

Edits (each added section was read in the docs text, rule 5, 7 Oct 2026):
  fact-05  + sql-vacuum / Description, routine-vacuuming / 24.1.1. Vacuuming Basics,
             routine-vacuuming / 24.1.2. Recovering Disk Space   (all state the VACUUM FULL difference)
  amb-02   + backup / anchor BACKUP                               (names the three approaches; thin but valid)
  fact-09  + sql-explain / Description, sql-explain / Parameters  (state what ANALYZE does)
  amb-01   + ddl-partitioning / 5.12.1. Overview                  (states the query-performance benefit)
Deliberately NOT added (read and rejected, or borderline): fact-03, fact-06, fact-07, fact-11, fact-01.

The heading "24.1.1. Vacuuming Basics" is not confirmed by your `find` output (only 24.1.2 and 24.1.3 were
printed); check_gold.py at the end will say if it is misspelled.

Untested when written.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
BACKUP = HERE / "dataset_pre_d3.jsonl"

ADDS = {
    "fact-05": [
        {"page": "sql-vacuum", "heading": "Description"},
        {"page": "routine-vacuuming", "heading": "24.1.1. Vacuuming Basics"},
        {"page": "routine-vacuuming", "heading": "24.1.2. Recovering Disk Space"},
    ],
    "amb-02": [
        {"page": "backup", "anchor": "BACKUP"},
    ],
    "fact-09": [
        {"page": "sql-explain", "heading": "Description"},
        {"page": "sql-explain", "heading": "Parameters"},
    ],
    "amb-01": [
        {"page": "ddl-partitioning", "heading": "5.12.1. Overview"},
    ],
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if BACKUP.exists():
        raise SystemExit(f"{BACKUP.name} already exists: d3 was probably applied already. "
                         f"Delete it by hand only if you really want to redo the relabelling.")

    lines = DATASET.read_text(encoding="utf-8").splitlines()
    out, seen, n_added = [], set(), 0
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        row = json.loads(line)
        if row["id"] in ADDS:
            seen.add(row["id"])
            for g in ADDS[row["id"]]:
                if g in row["gold"]:
                    raise SystemExit(f"{row['id']}: {g} is already in gold; refusing to add twice.")
                row["gold"].append(g)
                n_added += 1
        out.append(json.dumps(row, ensure_ascii=False))

    missing = set(ADDS) - seen
    if missing:
        raise SystemExit(f"Question id(s) not found in the dataset: {sorted(missing)}. Nothing written.")

    shutil.copyfile(DATASET, BACKUP)
    DATASET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Backed up d2 labels to {BACKUP.name}; added {n_added} gold section(s) to {len(ADDS)} questions.")

    total = sum(len(json.loads(l)["gold"]) for l in out if l.strip() and not l.lstrip().startswith("#"))
    print(f"Total gold sections now: {total} (expected 42)\n")
    print("Running check_gold.py ...\n")
    return subprocess.run([sys.executable, str(HERE / "check_gold.py")]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
