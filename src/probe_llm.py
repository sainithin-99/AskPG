r"""
One-off helper: find which hosted models your key can actually call (a model can be listed
by /models but still return 404 for your account).

Usage (cmd, after setting LLM_BASE_URL and LLM_API_KEY):
    python src\probe_llm.py nemotron-70b nemotron-51b qwen
Each argument is a substring matched against model ids. Up to 40 chat-like matches are tested
with a tiny request (max 8 output tokens), 2 s apart to stay under the rate limit.
reply='' on an OK line usually means a reasoning model: avoid it for this project.
"""
import os
import sys
import time

from openai import APIStatusError, OpenAI

SKIP = ("embed", "guard", "safety", "rerank", "retriev", "vlm", "vision", "parse", "topic-control")


def main(terms: list[str]) -> None:
    if not terms:
        raise SystemExit("Give at least one substring, e.g. python src\\probe_llm.py nemotron-70b qwen")
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                    timeout=60, max_retries=0)
    ids = sorted(m.id for m in client.models.list())
    picked = [i for i in ids if any(t.lower() in i.lower() for t in terms)]
    picked = [i for i in picked if not any(s in i.lower() for s in SKIP)][:40]
    if not picked:
        raise SystemExit("No model ids matched those substrings.")
    for mid in picked:
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=mid, messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=8, temperature=0)
            ms = (time.perf_counter() - t0) * 1000
            print(f"OK    {mid:<60} {ms:6.0f} ms  reply={(r.choices[0].message.content or '').strip()!r}")
        except APIStatusError as e:
            print(f"FAIL  {mid:<60} HTTP {e.status_code}")
        except Exception as e:
            print(f"FAIL  {mid:<60} {type(e).__name__}")
        time.sleep(2)


if __name__ == "__main__":
    main(sys.argv[1:])