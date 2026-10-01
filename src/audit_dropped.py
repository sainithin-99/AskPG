import collections
import json
import random
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "chunks" / "dropped.jsonl"
rows = [json.loads(line) for line in path.open(encoding="utf-8")]

whole = [r for r in rows if r["whole_section_lost"]]
print(f"Total dropped: {len(rows)}")
print(f"  whole section lost: {len(whole)}  |  leftover of a kept section: {len(rows) - len(whole)}")
print("Token counts:", sorted(collections.Counter(r["n_tokens"] // 5 * 5 for r in rows).items()))
print("Top page groups:", collections.Counter(r["page"].split("-")[0] for r in rows).most_common(8))

random.seed(0)
pool = whole or rows
for i, r in enumerate(random.sample(pool, min(25, len(pool))), 1):
    print("=" * 80)
    print(f"[{i}] {r['page']} | {' > '.join(r['section_path'])} | {r['n_tokens']} tok | whole_lost={r['whole_section_lost']}")
    print(r["text"][:300])