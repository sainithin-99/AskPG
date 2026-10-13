import sys
import time

sys.path.insert(0, "src")
import generate as g

print("LLM_TIMEOUT_S =", g.LLM_TIMEOUT_S)
llm = g.make_llm()
t0 = time.perf_counter()
try:
    text, u = llm([{"role": "user", "content": "Reply with the word OK."}])
    print("answered in", round(time.perf_counter() - t0, 1), "s:", text[:40])
except Exception as e:
    print("failed after", round(time.perf_counter() - t0, 1), "s:", type(e).__name__, str(e)[:100])