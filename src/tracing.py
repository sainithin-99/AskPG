r"""
Per-request tracing for AskPG: one JSON line per generate() result, plus p50/p95 aggregation.

(Named tracing.py, not trace.py: a file called trace.py in src/ would shadow Python's standard-library
`trace` module for every script run from src/.)

Usage from code:
    import tracing as tr
    tr.log_trace("eval/results/gen-x_traces.jsonl", result, qid="fact-01", type="factual")
    records = tr.read_traces(path)
    tr.print_summary(tr.summarize(records))

Usage from the command line (from the repo root, cmd):
    python src\tracing.py summary eval\results\<traces file>.jsonl

What one record holds: everything generate() returns (status, error, answer, raw_answer, citations,
sources_shown, usage, latency_ms, model, attempts, uncited_sentences) + a timestamp + whatever keyword
arguments the caller adds (qid, type, run, mode, k, gold_in_prompt, ...).

Latency groups in the summary:
    <stage>_ms   each retrieval stage (embed, vector, bm25, fuse, rerank) as reported by the retriever
    retrieval_ms sum of those stages (LOCAL time)
    llm_ms       the LLM call(s) only, errors excluded (HOSTED time: includes the network, report separately)
    total_ms     the whole generate() call (includes any throttle sleep)
Untested when written.
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


def log_trace(path, result: dict, **extra) -> dict:
    """Append one record (extra fields first, then the generate() result) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **extra, **result}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_traces(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_per_key(records: list[dict], key: str = "qid") -> dict:
    """Later records win: a question re-run after an llm_error replaces the failed record."""
    out = {}
    for r in records:
        out[r[key]] = r
    return out


def summarize(records: list[dict]) -> dict:
    stages = defaultdict(list)
    for r in records:
        lat = r.get("latency_ms") or {}
        retrieval = 0.0
        for s, v in lat.items():
            if s not in ("llm_ms", "total_ms"):
                stages[s].append(v)
                retrieval += v
        stages["retrieval_ms"].append(retrieval)
        stages["total_ms"].append(lat.get("total_ms", 0.0))
        if r.get("status") != "llm_error" and "llm_ms" in lat:
            stages["llm_ms"].append(lat["llm_ms"])
    order = [s for s in stages if s not in ("retrieval_ms", "llm_ms", "total_ms")] + \
            ["retrieval_ms", "llm_ms", "total_ms"]
    latency = {s: {"p50": pct(stages[s], 50), "p95": pct(stages[s], 95), "max": max(stages[s])}
               for s in order if stages.get(s)}

    usage = [r.get("usage") or {} for r in records]
    ok = [r for r in records if r.get("status") == "ok"]
    return {
        "n": len(records),
        "status": dict(Counter(r.get("status") for r in records)),
        "latency_ms": latency,
        "tokens": {"prompt": sum(u.get("prompt_tokens", 0) for u in usage),
                   "completion": sum(u.get("completion_tokens", 0) for u in usage)},
        "cost_usd": sum(u.get("cost_usd", 0.0) for u in usage),
        "retried": sum(1 for r in records if r.get("attempts", 1) > 1),
        "mean_uncited_sentences_ok": (statistics.mean(r.get("uncited_sentences", 0) for r in ok)
                                      if ok else None),
    }


def print_summary(s: dict, title: str = "") -> None:
    if title:
        print(title)
    print(f"  requests: {s['n']}   status: {s['status']}   retried (>1 attempt): {s['retried']}")
    print(f"  tokens in/out: {s['tokens']['prompt']}/{s['tokens']['completion']}   "
          f"estimated cost: ${s['cost_usd']:.5f}")
    if s["mean_uncited_sentences_ok"] is not None:
        print(f"  mean uncited sentences per ok answer (proxy): {s['mean_uncited_sentences_ok']:.2f}")
    print("  latency ms (p50 / p95 / max):")
    for stage, d in s["latency_ms"].items():
        print(f"    {stage:<13} {d['p50']:9.1f} {d['p95']:9.1f} {d['max']:9.1f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("path")
    args = ap.parse_args()
    recs = read_traces(args.path)
    if not recs:
        raise SystemExit(f"No records in {args.path}")
    print_summary(summarize(recs), f"{args.path}")
