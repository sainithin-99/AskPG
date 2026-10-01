r"""
Answer generation for AskPG: retrieve -> prompt with numbered sources -> LLM -> enforce citations.

Works with any OpenAI-compatible chat endpoint (Ollama, OpenAI, and most hosted providers).
Configure in E:\Projects\AskPG\.env (see .env.example) or with environment variables
(cmd: `set NAME=value`; a real environment variable overrides the .env value):

    LLM_BASE_URL         default http://localhost:11434/v1   (Ollama)
    LLM_API_KEY          default "ollama"                    (Ollama ignores it; hosted APIs need a real key)
    LLM_MODEL            default llama3.1:8b                 (must already be pulled: `ollama pull llama3.1:8b`)
    LLM_PRICE_IN_PER_M   USD per 1M input tokens,  default 0 (copy from your provider's pricing page)
    LLM_PRICE_OUT_PER_M  USD per 1M output tokens, default 0

Setup:
    pip install openai

Usage (from the repo root):
    python src/generate.py ask "what does work_mem control?" --mode hybrid_rerank -k 5

What "citation enforcement" means here (and what it does not):
  ENFORCED: the answer either is the refusal sentence, or has >= 1 citation and every [n] refers to a
            source that was actually shown to the model (1..k). One corrective retry; if it still fails,
            the caller gets status="citation_failed" and a safe fallback message instead of an uncited answer.
  NOT CHECKED: whether the cited source really supports the sentence. That is faithfulness, and needs the
            eval (LLM judge) to measure. `uncited_sentences` is a cheap proxy, not a guarantee.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

try:  # optional dependency: pip install python-dotenv
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# --------------------------------------------------------------------------- config
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY") or "ollama"
LLM_MODEL = os.environ.get("LLM_MODEL") or "llama3.1:8b"
PRICE_IN_PER_M = float(os.environ.get("LLM_PRICE_IN_PER_M") or 0)
PRICE_OUT_PER_M = float(os.environ.get("LLM_PRICE_OUT_PER_M") or 0)

# Reasoning models spend part of this budget on hidden thinking before the visible answer, so it must be
# generous; raise it (LLM_MAX_TOKENS in .env) if you see "Empty answer ... finish_reason=length".
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS") or 1500)
DEFAULT_K = 5

REFUSAL = "I can't find this in the PostgreSQL 17 documentation."
CITATION_FAILED = "I couldn't produce a properly cited answer. Please see the sources below or rephrase the question."

SYSTEM_PROMPT = f"""You answer questions about PostgreSQL 17 using ONLY the numbered sources provided.
Rules:
1. Use only facts stated in the sources. Do not use outside knowledge, even if you are sure.
2. After every sentence that states a fact, cite the supporting source(s) like [1] or [2][3], using plain ASCII square brackets (never 【1】).
3. Cite only source numbers that appear in the provided list.
4. Keep the answer concise. Include a short SQL or config snippet only if a source contains it.
5. If the sources do not contain the answer, reply with exactly: {REFUSAL}"""

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# --------------------------------------------------------------------------- prompt + citation logic
def build_messages(query: str, hits: list) -> list[dict]:
    sources = "\n\n".join(f"[{i}] {r['context']}\n{r['text']}" for i, (r, _) in enumerate(hits, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {query}"},
    ]


def normalize_citations(text: str) -> str:
    """Some models cite with full-width brackets (【5】). Same intent, different glyphs: rewrite to [5]
    so one parser handles everything (and the UI can link them)."""
    def fix(m):
        return "[" + re.sub(r"\s*[，、]\s*", ", ", m.group(1)) + "]"
    return re.sub(r"[【［]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[】］]", fix, text)


def _attach_citations(text: str) -> str:
    """Move a citation that follows the period ("... files. [5]") in front of it, so it stays with its sentence."""
    return re.sub(r"([.!?])\s+((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)",
                  lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)


def check_citations(answer: str, n_sources: int) -> dict:
    """Pure function (unit-testable without an LLM)."""
    text = answer.strip()
    is_refusal = REFUSAL.lower() in text.lower()
    nums = [int(n) for m in _CITE_RE.findall(text) for n in re.split(r"\s*,\s*", m)]
    valid = sorted({n for n in nums if 1 <= n <= n_sources})
    invalid = sorted({n for n in nums if not 1 <= n <= n_sources})
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", _attach_citations(text)) if len(s.split()) >= 4]
    uncited = 0 if is_refusal else sum(1 for s in sentences if not _CITE_RE.search(s))
    ok = is_refusal or (bool(valid) and not invalid)
    return {"ok": ok, "is_refusal": is_refusal, "cited": valid, "invalid": invalid,
            "uncited_sentences": uncited}


# --------------------------------------------------------------------------- LLM client
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models put inside the content field."""
    text = _THINK_RE.sub("", text)
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def make_llm():
    """Returns call(messages) -> (text, {"prompt_tokens", "completion_tokens", "llm_ms"})."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120, max_retries=2)

    def call(messages: list[dict]):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0, max_tokens=MAX_OUTPUT_TOKENS
        )
        ms = (time.perf_counter() - t0) * 1000
        u = r.usage
        choice = r.choices[0]
        text = strip_think(choice.message.content or "")
        if not text or text.startswith("<think>"):  # nothing visible, or the thinking was cut off
            raise RuntimeError(
                f"Empty answer from {LLM_MODEL} (finish_reason={choice.finish_reason}, "
                f"completion_tokens={getattr(u, 'completion_tokens', '?')}). A reasoning model probably used the "
                f"whole {MAX_OUTPUT_TOKENS}-token budget thinking: raise LLM_MAX_TOKENS or pick another model.")
        return text, {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "llm_ms": ms,
        }

    return call


# --------------------------------------------------------------------------- pipeline
def generate(query: str, retrieve, llm, k: int = DEFAULT_K, max_retries: int = 1) -> dict:
    t_start = time.perf_counter()
    hits, stage_ms = retrieve(query, k)
    messages = build_messages(query, hits)

    prompt_tok = completion_tok = 0
    llm_ms = 0.0
    attempts = 0
    answer, check = "", {"ok": False}
    while True:
        attempts += 1
        answer, u = llm(messages)
        answer = normalize_citations(answer)
        prompt_tok += u["prompt_tokens"]
        completion_tok += u["completion_tokens"]
        llm_ms += u["llm_ms"]
        check = check_citations(answer, len(hits))
        if check["ok"] or attempts > max_retries:
            break
        problem = (f"You cited source numbers that do not exist: {check['invalid']}."
                   if check["invalid"] else "Your answer has no citations.")
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"{problem} Rewrite the answer using only source numbers 1 to {len(hits)}, "
                                        f"citing after each factual sentence, or reply exactly: {REFUSAL}"},
        ]

    status = "refused" if check["is_refusal"] else ("ok" if check["ok"] else "citation_failed")
    cost = prompt_tok * PRICE_IN_PER_M / 1e6 + completion_tok * PRICE_OUT_PER_M / 1e6
    return {
        "query": query,
        "status": status,
        "answer": CITATION_FAILED if status == "citation_failed" else answer.strip(),
        "raw_answer": answer.strip(),
        "citations": [
            {"n": n, "chunk_id": hits[n - 1][0]["chunk_id"], "url": hits[n - 1][0]["url"],
             "context": hits[n - 1][0]["context"]}
            for n in check["cited"]
        ],
        "sources_shown": [{"n": i, "chunk_id": r["chunk_id"], "url": r["url"], "score": s}
                          for i, (r, s) in enumerate(hits, 1)],
        "uncited_sentences": check["uncited_sentences"],
        "attempts": attempts,
        "usage": {"prompt_tokens": prompt_tok, "completion_tokens": completion_tok, "cost_usd": cost},
        "latency_ms": {**stage_ms, "llm_ms": llm_ms, "total_ms": (time.perf_counter() - t_start) * 1000},
        "model": LLM_MODEL,
    }


# --------------------------------------------------------------------------- cli
def ask_cmd(query: str, mode: str, k: int, as_json: bool) -> None:
    import retrieve as rt

    retrieve = rt.build_retriever(mode)
    retrieve("warmup query", 1)  # keep model load out of the timings
    res = generate(query, retrieve, make_llm(), k=k)
    if as_json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"\nQ: {query}\nstatus={res['status']}  attempts={res['attempts']}  model={res['model']}  mode={mode}\n")
    print(res["answer"])
    if res["status"] == "citation_failed":
        print(f"\n[raw answer that failed enforcement]\n{res['raw_answer']}")
    print("\nSources shown to the model (* = cited):")
    cited = {c["n"] for c in res["citations"]}
    for s in res["sources_shown"]:
        print(f"  {'*' if s['n'] in cited else ' '} [{s['n']}] {s['url']}")
    u, lat = res["usage"], res["latency_ms"]
    print(f"\ntokens in/out: {u['prompt_tokens']}/{u['completion_tokens']}   cost: ${u['cost_usd']:.5f}")
    print("latency ms: " + " | ".join(f"{k} {v:.0f}" for k, v in lat.items()))
    print(f"uncited sentences (proxy): {res['uncited_sentences']}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("text")
    a.add_argument("--mode", default="hybrid_rerank", choices=("vector", "bm25", "hybrid", "hybrid_rerank"))
    a.add_argument("-k", type=int, default=DEFAULT_K)
    a.add_argument("--json", action="store_true", help="print the full result dict")
    args = ap.parse_args()
    ask_cmd(args.text, args.mode, args.k, args.json)