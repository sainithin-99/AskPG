import sys
import time

sys.path.insert(0, "src")
import generate as g
from openai import OpenAI

client = OpenAI(base_url=g.LLM_BASE_URL, api_key=g.LLM_API_KEY, timeout=1.0, max_retries=0)
t0 = time.perf_counter()
try:
    r = client.chat.completions.create(model=g.LLM_MODEL, temperature=0, max_tokens=g.MAX_OUTPUT_TOKENS,
                                       messages=[{"role": "user", "content": "Reply with the word OK."}])
    print("answered in", round(time.perf_counter() - t0, 1), "s")
except Exception as e:
    print("failed after", round(time.perf_counter() - t0, 1), "s:", type(e).__name__)