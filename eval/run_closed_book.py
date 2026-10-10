r"""
Closed-book baseline for AskPG: the same dataset questions go to the SAME generator model with NO retrieval,
so the gap to a retrieval run (g1) shows what the documentation adds.

Usage (from the repo root, cmd):
    python eval\run_closed_book.py --name cb1 --ids fact-01 unans-08     # smoke test on two questions
    python eval\run_closed_book.py --name cb1                            # all 30 questions
    python eval\run_closed_book.py --name cb1 --resume                   # redo only llm_error / missing questions

Outputs (eval\results\), same layout as run_gen_eval.py so judge.py can read them:
    gen-<name>_traces.jsonl   one record per question, appended as each question finishes (safe to interrupt)
    gen-<name>.json           meta + summary + compact per-question results

Design choices (read before comparing with g1):
  - Same model, temperature, max tokens and throttle as g1 (all come from generate.py / .env). A stale
    `set LLM_MODEL=...` in the cmd window silently changes the model: check meta["model"] before comparing.
  - The model MAY abstain, with the exact refusal sentence from generate.py, so the 9 unanswerable questions are
    not forced into an answer. The sentence mentions "documentation" although none is given: it is kept so the
    refusal detector is identical to g1. The system prompt tells the model to use its own knowledge.
  - The g1 prompt said "use ONLY the sources" and required citations. Here there are no sources and no citations,
    so the two prompts differ on purpose. Prompt md5 is stored in meta.
  - status "ok" here means ANSWERED (not refused). It says nothing about correctness: that needs the judge
    (correctness / completeness vs reference_answer). Faithfulness does not apply: there are no sources.
  - uuidv7() is not in PostgreSQL 17 but is in 18: a model that answers it may be using newer knowledge. Read that
    answer instead of counting it as a plain failure.
Untested when written (syntax-checked only).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (also puts <repo>/src on sys.path)
import generate as gen  # noqa: E402
import tracing as tr  # noqa: E402

STATUSES = ("ok", "refused", "llm_error")

SYSTEM_PROMPT = f"""You answer questions about PostgreSQL 17 from your own knowledge. No documentation is provided.
Rules:
1. Answer only if you are confident the answer is correct for PostgreSQL 17.
2. Keep the answer concise. Include a short SQL or config snippet only if it helps.
3. If the question is not about PostgreSQL 17, or you are not sure of the answer, reply with exactly: {gen.REFUSAL}"""


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ev.ROOT, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def answer(question: str, llm) -> dict:
    """Never raises for an LLM failure (same contract as generate.generate)."""
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"}]
    text, error = "", None
    u = {"prompt_tokens": 0, "completion_tokens": 0, "llm_ms": 0.0}
    try:
        text, u = llm(messages)
    except Exception as e:  # timeout, 404 model, 429, empty reasoning answer ...
        error = f"{type(e).__name__}: {e}"
    if error:
        status = "llm_error"
    elif gen.REFUSAL.lower() in text.lower():
        status = "refused"
    else:
        status = "ok"
    cost = u["prompt_tokens"] * gen.PRICE_IN_PER_M / 1e6 + u["completion_tokens"] * gen.PRICE_OUT_PER_M / 1e6
    return {
        "query": question, "status": status, "error": error,
        "answer": gen.LLM_ERROR if error else text.strip(), "raw_answer": text.strip(),
        "citations": [], "sources_shown": [], "uncited_sentences": 0, "attempts": 1,
        "usage": {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                  "cost_usd": cost},
        "latency_ms": {"llm_ms": u["llm_ms"], "total_ms": (time.perf_counter() - t0) * 1000},
        "model": gen.LLM_MODEL,
    }


def counts(recs: list) -> str:
    c = Counter(r["status"] for r in recs)
    return "  ".join(f"{s} {c.get(s, 0)}" for s in STATUSES)


def run(name: str, ids: list, limit: int, resume: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dataset = ev.load_dataset()
    if ids:
        unknown = [i for i in ids if i not in {q["id"] for q in dataset}]
        if unknown:
            raise SystemExit(f"Unknown question id(s): {unknown}")
        dataset = [q for q in dataset if q["id"] in ids]
    if limit:
        dataset = dataset[:limit]

    traces_path = ev.RESULTS_DIR / f"gen-{name}_traces.jsonl"
    out_path = ev.RESULTS_DIR / f"gen-{name}.json"
    done = {}
    if traces_path.exists():
        if not resume:
            raise SystemExit(f"{traces_path.name} already exists. Pass --resume to continue that run, "
                             f"or choose a new --name. (Runs are never mixed.)")
        done = tr.last_per_key(tr.read_traces(traces_path))
    todo = [q for q in dataset if q["id"] not in done or done[q["id"]]["status"] == "llm_error"]
    print(f"Closed-book run {name}: {len(dataset)} questions selected, {len(todo)} to run "
          f"({len(dataset) - len(todo)} already done), model={gen.LLM_MODEL}")

    if todo:
        llm = gen.make_llm()  # one client: the throttle state persists across questions
        for n, q in enumerate(todo, 1):
            res = answer(q["question"], llm)
            rec = tr.log_trace(traces_path, res, qid=q["id"], type=q["type"], run=name, mode="closed_book",
                               k=0, gold_in_prompt=None, reference_answer=q.get("reference_answer"))
            done[q["id"]] = rec
            print(f"  [{n}/{len(todo)}] {q['id']:<9} {res['status']:<10} llm {res['latency_ms']['llm_ms']:6.0f} ms"
                  f"  tok {res['usage']['prompt_tokens']}/{res['usage']['completion_tokens']}"
                  + (f"  ERROR {res['error'][:80]}" if res["error"] else ""))

    recs = [done[q["id"]] for q in dataset if q["id"] in done]
    ans = [r for r in recs if r["type"] != "unanswerable"]
    una = [r for r in recs if r["type"] == "unanswerable"]

    print(f"\n=== gen-{name} (closed book, {gen.LLM_MODEL}) ===")
    print("Status by type (ok = ANSWERED, not necessarily correct):")
    for t in ev.TYPES:
        sel = [r for r in recs if r["type"] == t]
        if sel:
            print(f"  {t:<13} n={len(sel):<3} {counts(sel)}")
    errors = [r for r in recs if r["status"] == "llm_error"]
    if errors:
        print(f"\nWARNING: {len(errors)} question(s) ended in llm_error; rates below exclude them. "
              f"Re-run with --resume. Reasons: {sorted({r['error'][:60] for r in errors})}")

    refused_a = [r for r in ans if r["status"] == "refused"]
    print(f"\nAnswerable (n={len(ans)}): {counts(ans)}")
    for r in refused_a:
        print(f"  ABSTAINED on an answerable question: {r['qid']:<9} {r['query']}")

    live_u = [r for r in una if r["status"] != "llm_error"]
    refused_u = [r for r in live_u if r["status"] == "refused"]
    print(f"\nUnanswerable (n={len(una)}, scored {len(live_u)}): refused {len(refused_u)} of {len(live_u)}")
    for r in live_u:
        if r["status"] != "refused":
            print(f"  ANSWERED instead of refusing: {r['qid']} {r['query']}")
            print(f"    {r['answer'][:400].replace(chr(10), ' ')}")

    summary = tr.summarize(recs)
    print()
    tr.print_summary(summary, "Trace summary (all questions in this run):")

    meta = {"name": name, "time": datetime.now().isoformat(timespec="seconds"), "mode": "closed_book", "k": 0,
            "model": gen.LLM_MODEL, "llm_host": urlparse(gen.LLM_BASE_URL).netloc,
            "max_output_tokens": gen.MAX_OUTPUT_TOKENS, "min_gap_s": gen.MIN_GAP_S,
            "price_in_per_m": gen.PRICE_IN_PER_M, "price_out_per_m": gen.PRICE_OUT_PER_M,
            "prompt_md5": hashlib.md5(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
            "dataset_md5": hashlib.md5(ev.DATASET_PATH.read_bytes()).hexdigest(),
            "git_commit": git_commit(), "n_questions": len(recs)}
    per_q = [{"id": r["qid"], "type": r["type"], "status": r["status"], "error": r.get("error"),
              "gold_in_prompt": None, "answer": r["answer"], "raw_answer": r["raw_answer"],
              "cited_chunks": [], "sources_shown": [], "uncited_sentences": 0, "attempts": r["attempts"],
              "usage": r["usage"], "latency_ms": r["latency_ms"]} for r in recs]
    out_path.write_text(json.dumps({"meta": meta, "summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out_path}\nTraces: {traces_path}")

    lat = summary["latency_ms"]
    answered = sum(1 for r in ans if r["status"] == "ok")
    print("\nRow for experiments.md (correctness comes from the judge, not from this script):")
    print("| run | model | answerable answered | answerable abstained | unanswerable refused | llm_error | "
          "p50 / p95 total ms | tokens in/out |")
    print(f"| gen-{name} (closed book) | {gen.LLM_MODEL} | {answered}/{len(ans)} | {len(refused_a)} | "
          f"{len(refused_u)}/{len(live_u)} | {len(errors)} | "
          f"{lat['total_ms']['p50']:.0f} / {lat['total_ms']['p95']:.0f} | "
          f"{summary['tokens']['prompt']}/{summary['tokens']['completion']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run label, e.g. cb1; also part of the result file names")
    ap.add_argument("--ids", nargs="*", default=[], help="run only these question ids (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N selected questions")
    ap.add_argument("--resume", action="store_true", help="continue a run: redo llm_error and missing questions")
    a = ap.parse_args()
    run(a.name, a.ids, a.limit, a.resume)
