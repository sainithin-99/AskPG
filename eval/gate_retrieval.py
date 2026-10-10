r"""
Local regression gate for retrieval: re-run the eval on the current code/index and compare with a saved baseline.
No LLM calls, so it is free and deterministic (up to GPU float noise). Run it before every push that touches
ingest, embeddings, retrieval or the reranker.

Usage (from the repo root, cmd; Qdrant must be running and the index built):
    python eval\gate_retrieval.py                       # modes vector + hybrid_rerank vs eval\results\d3-<mode>.json
    python eval\gate_retrieval.py --modes vector hybrid hybrid_rerank bm25
    python eval\gate_retrieval.py --baseline d3 --tol-questions 1 --tol-mrr 0.03

What it compares (answerable questions only, labels as in the current eval\dataset.jsonl):
  - success@5 (counted in QUESTIONS: 1 question = 1/21 = 4.8 points) and MRR.
  - FAIL when success@5 drops by MORE than --tol-questions questions (default 1), or MRR drops by more than
    --tol-mrr (default 0.03). Both thresholds are my choice, not measured: they are there to absorb a near-tie
    flipping from GPU float noise (multi-01 is such a near-tie), not to hide a real loss. A drop of exactly the
    tolerance still prints WARN so it is not invisible.
  - Always printed: the questions that passed @5 in the baseline and fail now (LOST) and the reverse (GAINED).
    Read LOST even when the gate passes.

The baseline must match the current setup. The script aborts (it does not guess) when the baseline's chunk count
differs from chunks.jsonl, or its questions / gold counts differ from the current dataset: that means the chunks or
the labels changed, so the baseline is STALE and rows from different versions must not be compared as progress.
Create a new baseline with `python eval\run_eval.py run --name <label> --retriever <mode>` and log it in
experiments.md as its own version; then pass --baseline <label-prefix>.

Limits: the retrieval baseline file does not store a dataset hash, so a relabelling that keeps the same number of
golds per question is NOT detected. The gate says nothing about answer quality or latency.
Untested when written (syntax read only).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (also puts <repo>/src on sys.path)


def evaluate(mode: str, dataset: list[dict]) -> list[dict]:
    retrieve = ev.RETRIEVERS[mode]()
    retrieve("warmup query", 1)  # load models / warm CUDA outside the loop
    scored = []
    for q in dataset:
        if q["type"] == "unanswerable":
            continue
        hits, _ = retrieve(q["question"], ev.MAX_K)
        scored.append(ev.score_question(q, hits))
    return scored


def passes(r: dict, k: int = 5) -> bool:
    return r["success_rank"] is not None and r["success_rank"] <= k


def gate_mode(mode: str, prefix: str, dataset: list[dict], chunks: list[dict], tol_q: float, tol_mrr: float) -> bool:
    path = ev.RESULTS_DIR / f"{prefix}-{mode}.json"
    if not path.exists():
        print(f"[{mode}] ABORT: no baseline {path}")
        return False
    base = json.loads(path.read_text(encoding="utf-8"))
    if base.get("n_chunks") != len(chunks):
        print(f"[{mode}] ABORT: baseline has {base.get('n_chunks')} chunks, chunks.jsonl has {len(chunks)}. "
              f"Baseline is stale (re-chunked?). Make a new baseline and log it as a new version.")
        return False
    answerable = [q for q in dataset if q["type"] != "unanswerable"]
    base_q = {r["id"]: r for r in base["per_question"]}
    for q in answerable:
        r = base_q.get(q["id"])
        if r is None or len(r["gold_ranks"]) != len(q["gold"]):
            print(f"[{mode}] ABORT: baseline does not match the current dataset at {q['id']} (relabelled?). "
                  f"Baseline is stale.")
            return False

    now = evaluate(mode, dataset)
    n = len(now)
    b_sum = ev.summarize([base_q[q["id"]] for q in answerable])
    n_sum = ev.summarize(now)
    drop_q = (b_sum["success@5"] - n_sum["success@5"]) * n
    drop_mrr = b_sum["mrr"] - n_sum["mrr"]

    now_by_id = {r["id"]: r for r in now}
    lost = [q["id"] for q in answerable if passes(base_q[q["id"]]) and not passes(now_by_id[q["id"]])]
    gained = [q["id"] for q in answerable if not passes(base_q[q["id"]]) and passes(now_by_id[q["id"]])]

    print(f"[{mode}] baseline {path.name} (n={n})")
    print(f"   success@5  baseline {b_sum['success@5']:.3f} ({round(b_sum['success@5'] * n)}/{n})  "
          f"now {n_sum['success@5']:.3f} ({round(n_sum['success@5'] * n)}/{n})")
    print(f"   success@1  baseline {b_sum['success@1']:.3f}  now {n_sum['success@1']:.3f}   (not gated)")
    print(f"   MRR        baseline {b_sum['mrr']:.3f}  now {n_sum['mrr']:.3f}")
    print(f"   LOST (pass @5 in baseline, fail now): {lost or 'none'}")
    print(f"   GAINED (the reverse):                 {gained or 'none'}")

    fail = drop_q > tol_q + 1e-9 or drop_mrr > tol_mrr + 1e-9
    warn = not fail and (drop_q >= tol_q - 1e-9 and drop_q > 0 or drop_mrr > 0 and drop_mrr >= tol_mrr - 1e-9)
    print(f"   -> {'FAIL' if fail else ('WARN (at the tolerance)' if warn else 'PASS')}"
          f"   (success@5 drop {drop_q:+.2f} questions, tolerance {tol_q}; MRR drop {drop_mrr:+.3f}, tolerance {tol_mrr})\n")
    return not fail


def main(modes: list[str], prefix: str, tol_q: float, tol_mrr: float) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dataset = ev.load_dataset()
    chunks = ev.load_chunks()
    missing = ev.coverage_check(dataset, chunks)
    if missing:
        print(f"ABORT: {len(missing)} gold section(s) are in no chunk's `covers`: {missing[:5]}")
        return 1
    ok = True
    for mode in modes:
        ok = gate_mode(mode, prefix, dataset, chunks, tol_q, tol_mrr) and ok
    print("GATE PASSED" if ok else "GATE FAILED (or aborted): do not push retrieval changes until this is explained")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["vector", "hybrid_rerank"], choices=sorted(ev.RETRIEVERS))
    ap.add_argument("--baseline", default="d3", help="result-file prefix: eval\\results\\<prefix>-<mode>.json")
    ap.add_argument("--tol-questions", type=float, default=1.0)
    ap.add_argument("--tol-mrr", type=float, default=0.03)
    a = ap.parse_args()
    raise SystemExit(main(a.modes, a.baseline, a.tol_questions, a.tol_mrr))
