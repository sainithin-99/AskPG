import json
import threading
import time
from pathlib import Path

import requests

URL = "http://localhost:8000"
N = 3
results = [None] * N


def worker(i):
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/ask", json={"question": "what does work_mem control?"}, timeout=300)
    results[i] = (r.status_code, time.perf_counter() - t0, r.json().get("request_id"))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"all {N} done in {time.perf_counter() - t0:.1f} s")

ids = {r[2]: r for r in results}
trace = Path("data/traces/requests.jsonl")
for line in trace.read_text(encoding="utf-8").splitlines()[-N:]:
    rec = json.loads(line)
    print(rec["request_id"], rec["status"], f"queue_wait {rec['queue_wait_ms']:.0f} ms", f"wall {rec['wall_ms']:.0f} ms")
for i, (code, secs, rid) in enumerate(results):
    print(f"client {i}: HTTP {code}, {secs:.1f} s, id {rid}")