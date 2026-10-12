import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "src")
import ingest as ing

CUR = ing.OUT_PATH
tmp = Path("data/tmp_rechunk/chunks.jsonl")
tmp.parent.mkdir(parents=True, exist_ok=True)
ing.OUT_PATH = tmp

def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def lines(p):
    return sum(1 for _ in p.open(encoding="utf-8"))

print("current  ", md5(CUR), lines(CUR))
for budget in (380, 400):
    ing.MAX_TOKENS = budget
    ing.chunk_all()
    print("MAX_TOKENS", budget, md5(tmp), lines(tmp), "MATCH" if md5(tmp) == md5(CUR) else "differs")