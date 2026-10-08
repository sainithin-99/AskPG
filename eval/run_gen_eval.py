r"""
End-to-end generation run for AskPG: every dataset question goes through retrieve -> LLM -> citation check,
each result is traced, and the run is summarised.

Usage (from the repo root, cmd). The run name is any label you choose:
    python eval\run_gen_eval.py --name g1 --ids fact-01 unans-03      # smoke test on two questions
    python eval\run_gen_eval.py --name g1                             # all 30 questions
    python eval\run_gen_eval.py --name g1 --resume                    # redo only llm_error / missing questions

Outputs (eval\results\):
    gen-<name>_traces.jsonl   one record per question, appended AS EACH QUESTION FINISHES (safe to interrupt)
    gen-<name>.json           meta + summary + compact per-question results (read by the judge step later)

What it measures (and what it does NOT):
  MEASURED   status per question (ok / refused / citation_failed / llm_error), refusal behaviour on the
             9 unanswerable questions (7 are near-misses), false refusals on answerable questions, whether the
             gold section was in the prompt (gold_in_prompt, same rule as the retrieval eval: ALL golds for
             multi_hop, ANY otherwise), tokens, estimated cost, latency (retrieval / LLM / total, p50 / p95).
  NOT MEASURED  answer correctness and faithfulness (needs the LLM judge, next step) and the closed-book
             baseline. "status=ok" only means the citations are well-formed.
Read the printed answers: the script lists every unanswerable question that was ANSWERED and every answerable
question that was REFUSED, so you can read them instead of trusting counts.

A run mixes nothing: if gen-<name>_traces.jsonl exists you must pass --resume or choose a new name.
The model id, date, mode, k and the dataset hash are stored in the meta (the API key never is).
Untested when written.
"""
import argparse
import hashlib

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (also puts <repo>/src on sys.path)
import generate as gen  # noqa: E402
import retrieve as rt  # noqa: E402
import tracing as tr  # noqa: E402

STATUSES = ("ok", "refused", "citation_failed", "llm_error")

def git_commit():
    """Short hash of HEAD, or None (not a git repo / git missing). Uncommitted changes are NOT captured."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ev.ROOT, capture_output=True,
                                text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def gold_in_prompt(q: dict, shown_chunk_ids: list, by_id: dict):
    """Was the gold section among the chunks shown to the LLM? None for unanswerable questions."""
    if q["type"] == "unanswerable":
        return None
    keys = [ev.chunk_keys(by_id[c]) for c in shown_chunk_ids]
    found = [any(ev.gold_key(g) in ks for ks in keys) for g in q["gold"]]
    return all(found) if q["type"] == "multi_hop" else any(found)


def counts(recs: list) -> str:
    c = Counter(r["status"] for r in recs)
    return "  ".join(f"{s} {c.get(s, 0)}" for s in STATUSES)


def run(name: str, mode: str, k: int, ids: list, limit: int, resume: bool) -> None:
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
    print(f"Run {name}: {len(dataset)} questions selected, {len(todo)} to run "
          f"({len(dataset) - len(todo)} already done), mode={mode}, k={k}, model={gen.LLM_MODEL}")

    if todo:
        retrieve = rt.build_retriever(mode)
        retrieve("warmup query", 1)  # keep model load out of the timings
        llm = gen.make_llm()         # one client: the throttle state persists across questions
        by_id = rt.records_by_id()
        for n, q in enumerate(todo, 1):
            res = gen.generate(q["question"], retrieve, llm, k=k)
            gip = gold_in_prompt(q, [s["chunk_id"] for s in res["sources_shown"]], by_id)
            rec = tr.log_trace(traces_path, res, qid=q["id"], type=q["type"], run=name, mode=mode, k=k,
                               gold_in_prompt=gip, reference_answer=q.get("reference_answer"))
            done[q["id"]] = rec
            print(f"  [{n}/{len(todo)}] {q['id']:<9} {res['status']:<16} gold_in_prompt={gip!s:<5} "
                  f"llm {res['latency_ms']['llm_ms']:6.0f} ms  tok {res['usage']['prompt_tokens']}/"
                  f"{res['usage']['completion_tokens']}" + (f"  ERROR {res['error'][:80]}" if res["error"] else ""))

    recs = [done[q["id"]] for q in dataset if q["id"] in done]
    ans = [r for r in recs if r["type"] != "unanswerable"]
    una = [r for r in recs if r["type"] == "unanswerable"]

    print(f"\n=== gen-{name} ({mode}, k={k}, {gen.LLM_MODEL}) ===")
    print("Status by type:")
    for t in ev.TYPES:
        sel = [r for r in recs if r["type"] == t]
        if sel:
            print(f"  {t:<13} n={len(sel):<3} {counts(sel)}")
    errors = [r for r in recs if r["status"] == "llm_error"]
    if errors:
        print(f"\nWARNING: {len(errors)} question(s) ended in llm_error; rates below exclude them. "
              f"Re-run with --resume. Reasons: {sorted({r['error'][:60] for r in errors})}")

    live = [r for r in ans if r["status"] != "llm_error"]
    print(f"\nAnswerable questions (n={len(ans)}, scored {len(live)}):")
    print(f"  {counts(ans)}")
    for flag, label in ((True, "gold section WAS in the prompt"), (False, "gold section was NOT in the prompt")):
        sel = [r for r in live if r["gold_in_prompt"] is flag]
        if sel:
            print(f"  {label} (n={len(sel)}): {counts(sel)}")
    false_ref = [r for r in ans if r["status"] == "refused"]
    if false_ref:
        print("  FALSE REFUSALS (answerable but refused), read why:")
        for r in false_ref:
            print(f"    {r['qid']:<9} gold_in_prompt={r['gold_in_prompt']}  {r['query']}")

    live_u = [r for r in una if r["status"] != "llm_error"]
    refused_u = [r for r in live_u if r["status"] == "refused"]
    print(f"\nUnanswerable questions (n={len(una)}, scored {len(live_u)}): refused {len(refused_u)} of {len(live_u)}")
    answered_u = [r for r in live_u if r["status"] != "refused"]
    for r in answered_u:
        print(f"  ANSWERED instead of refusing: {r['qid']} ({r['status']}) {r['query']}")
        print(f"    {r['answer'][:400].replace(chr(10), ' ')}")

    summary = tr.summarize(recs)
    print()
    tr.print_summary(summary, "Trace summary (all questions in this run):")

    meta = {"name": name, "time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "k": k,
            "model": gen.LLM_MODEL, "llm_host": urlparse(gen.LLM_BASE_URL).netloc,
            "max_output_tokens": gen.MAX_OUTPUT_TOKENS, "min_gap_s": gen.MIN_GAP_S,
            "price_in_per_m": gen.PRICE_IN_PER_M, "price_out_per_m": gen.PRICE_OUT_PER_M,
            "dataset_md5": hashlib.md5(ev.DATASET_PATH.read_bytes()).hexdigest(),
            "chunks_md5": hashlib.md5(ev.CHUNKS_PATH.read_bytes()).hexdigest(),
            "git_commit": git_commit(),
            "n_questions": len(recs)}
    per_q = [{"id": r["qid"], "type": r["type"], "status": r["status"], "error": r.get("error"),
              "gold_in_prompt": r["gold_in_prompt"], "answer": r["answer"], "raw_answer": r["raw_answer"],
              "cited_chunks": [c["chunk_id"] for c in r["citations"]],
              "sources_shown": [s["chunk_id"] for s in r["sources_shown"]],
              "uncited_sentences": r["uncited_sentences"], "attempts": r["attempts"],
              "usage": r["usage"], "latency_ms": r["latency_ms"]} for r in recs]
    out_path.write_text(json.dumps({"meta": meta, "summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out_path}\nTraces: {traces_path}")

    lat = summary["latency_ms"]
    ok_a = sum(1 for r in ans if r["status"] == "ok")
    print("\nRow for experiments.md (LLM latency includes the network):")
    print("| run | model | answerable ok | false refusals | unanswerable refused | llm_error | "
          "p50 / p95 total ms | p50 / p95 llm ms | tokens in/out | est. cost |")
    print(f"| gen-{name} | {gen.LLM_MODEL} | {ok_a}/{len(ans)} | {len(false_ref)} | "
          f"{len(refused_u)}/{len(live_u)} | {len(errors)} | "
          f"{lat['total_ms']['p50']:.0f} / {lat['total_ms']['p95']:.0f} | "
          f"{lat.get('llm_ms', {}).get('p50', float('nan')):.0f} / {lat.get('llm_ms', {}).get('p95', float('nan')):.0f} | "
          f"{summary['tokens']['prompt']}/{summary['tokens']['completion']} | ${summary['cost_usd']:.4f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run label; also part of the result file names")
    ap.add_argument("--mode", default="hybrid_rerank", choices=rt.MODES)
    ap.add_argument("-k", type=int, default=gen.DEFAULT_K)
    ap.add_argument("--ids", nargs="*", default=[], help="run only these question ids (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N selected questions")
    ap.add_argument("--resume", action="store_true", help="continue a run: redo llm_error and missing questions")
    a = ap.parse_args()
    run(a.name, a.mode, a.k, a.ids, a.limit, a.resume)
