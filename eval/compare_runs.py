r"""
Compare two saved eval runs per question: A = baseline, B = candidate (e.g. hybrid vs hybrid_rerank).

Usage (from the repo root, cmd):
    python eval\compare_runs.py d2-hybrid d2-hybrid_rerank
    python eval\compare_runs.py d2-vector d2-hybrid_rerank -k 5

Reads eval\results\<name>.json (fields used: per_question[].gold_ranks / success_rank / top1_score and
unanswerable[]) and the CURRENT eval\dataset.jsonl, so the runs must have been made with the current labels
(the script aborts if the number of gold sections per question no longer matches).

What it reports
  1. Question level: how many questions pass @k in A and B; LIFT (fail in A, pass in B), HARM (pass in A,
     fail in B), both pass, both fail.
  2. Gold-section level: a gold section is "in the top k" if gold_ranks[j] <= k.
     HARM = in A's top k but not in B's. LIFT = the reverse.
  3. Reranker exposure: for every gold section, the XLM-R pair length (query + breadcrumb + text, the bge-reranker
     tokenizer, no truncation) of EVERY chunk that covers that section. A pair over 512 tokens is cut by the
     cross-encoder, which drops the END of the chunk. Harm rate is split by exposed / not exposed.
     LIMITATION: a section can have many chunks and the script cannot tell which one held the answer, so
     "exposed" means >= 1 chunk of the section is over 512 (this over-flags big sections). The per-line
     "exposed chunks n/m" shows how much to trust the flag.
  4. Refusal scores: top-1 score of B for answerable vs unanswerable questions and a threshold sweep.
     Illustrative only: 9 unanswerable questions, tuned on the same data it is measured on.
Untested on the real files when written.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (provides load_dataset, load_chunks, chunk_keys, gold_key, RESULTS_DIR)

RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_LEN = 512


def load_run(name: str):
    path = ev.RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"No {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {r["id"]: r for r in data["per_question"]}


def label(g: dict) -> str:
    return g.get("heading") or g.get("anchor") or "?"


def section_exposure(answerable: list, chunks: list):
    """gold (qid, j) -> {'n_chunks', 'n_exposed', 'max_len'} using the reranker's own tokenizer."""
    from transformers import AutoTokenizer
    import embed_index as ei  # same doc_text the reranker is given (breadcrumb + body)

    tok = AutoTokenizer.from_pretrained(RERANK_MODEL, model_max_length=10**6)
    by_key = defaultdict(list)
    for c in chunks:
        for k in ev.chunk_keys(c):
            by_key[k].append(c)

    out = {}
    for q in answerable:
        for j, g in enumerate(q["gold"]):
            lens = [len(tok(q["question"], ei.doc_text(c), add_special_tokens=True,
                            truncation=False)["input_ids"]) for c in by_key.get(ev.gold_key(g), [])]
            out[(q["id"], j)] = {
                "n_chunks": len(lens),
                "n_exposed": sum(1 for n in lens if n > RERANK_MAX_LEN),
                "max_len": max(lens) if lens else 0,
            }
    return out


def main(a_name: str, b_name: str, k: int) -> None:
    dataset = ev.load_dataset()
    chunks = ev.load_chunks()
    answerable = [q for q in dataset if q["type"] != "unanswerable"]
    A, qa = load_run(a_name)
    B, qb = load_run(b_name)
    for q in answerable:
        for nm, pq in ((a_name, qa), (b_name, qb)):
            r = pq.get(q["id"])
            if r is None or len(r["gold_ranks"]) != len(q["gold"]):
                raise SystemExit(f"{nm}: result for {q['id']} does not match the current dataset "
                                 f"(relabelled since the run?). Re-run it.")

    def ok(r):
        return r["success_rank"] is not None and r["success_rank"] <= k

    def inside(rank):
        return rank is not None and rank <= k

    # ---- 1. question level
    lift = [q["id"] for q in answerable if not ok(qa[q["id"]]) and ok(qb[q["id"]])]
    harm = [q["id"] for q in answerable if ok(qa[q["id"]]) and not ok(qb[q["id"]])]
    both = [q["id"] for q in answerable if ok(qa[q["id"]]) and ok(qb[q["id"]])]
    neither = [q["id"] for q in answerable if not ok(qa[q["id"]]) and not ok(qb[q["id"]])]
    n = len(answerable)
    print(f"A = {a_name}   B = {b_name}   success@{k} question level (n={n})")
    print(f"  pass in A: {len(both) + len(harm)}   pass in B: {len(both) + len(lift)}   net {len(lift) - len(harm):+d}")
    print(f"  LIFT (fail in A, pass in B): {len(lift)}  {lift}")
    print(f"  HARM (pass in A, fail in B): {len(harm)}  {harm}")
    print(f"  both pass: {len(both)}   both fail: {len(neither)}  {neither}")

    # ---- 2 + 3. gold-section level with exposure
    print("\nComputing XLM-R pair lengths for gold sections (reranker tokenizer)...")
    expo = section_exposure(answerable, chunks)

    rows = []
    for q in answerable:
        for j, g in enumerate(q["gold"]):
            ra, rb = qa[q["id"]]["gold_ranks"][j], qb[q["id"]]["gold_ranks"][j]
            e = expo[(q["id"], j)]
            rows.append({"id": q["id"], "type": q["type"], "gold": label(g), "ra": ra, "rb": rb,
                         "in_a": inside(ra), "in_b": inside(rb), **e, "exposed": e["n_exposed"] > 0})

    def rate(sel, flag):
        base = [r for r in sel if r["in_a"]]
        hit = [r for r in base if not r["in_b"]]
        return len(hit), len(base)

    print(f"\nGold-section level @{k}  ({len(rows)} gold sections)")
    h, d = rate(rows, "harm")
    print(f"  HARM: in A's top {k} but not B's: {h} of {d} ({h / d:.0%})" if d else "  HARM: none in A's top k")
    for name, sel in (("exposed", [r for r in rows if r["exposed"]]),
                      ("not exposed", [r for r in rows if not r["exposed"]])):
        h, d = rate(sel, "harm")
        print(f"    {name:<12} {h} of {d}" + (f" ({h / d:.0%})" if d else ""))
    outside = [r for r in rows if not r["in_a"]]
    lifted = [r for r in outside if r["in_b"]]
    print(f"  LIFT: not in A's top {k} but in B's: {len(lifted)} of {len(outside)}"
          + (f" ({len(lifted) / len(outside):.0%})" if outside else ""))
    for name, sel in (("exposed", [r for r in outside if r["exposed"]]),
                      ("not exposed", [r for r in outside if not r["exposed"]])):
        print(f"    {name:<12} {sum(1 for r in sel if r['in_b'])} of {len(sel)}")

    print(f"\nChanged gold sections (A rank -> B rank; pair length is the max over the section's chunks):")
    for tag, cond in (("HARM", lambda r: r["in_a"] and not r["in_b"]),
                      ("LIFT", lambda r: not r["in_a"] and r["in_b"])):
        for r in rows:
            if cond(r):
                print(f"  {tag}  {r['id']:<9} {r['gold'][:48]:<48} {str(r['ra']):>4} -> {str(r['rb']):<4} "
                      f"max pair {r['max_len']:>4} tok  exposed chunks {r['n_exposed']}/{r['n_chunks']}")

    # ---- 4. refusal scores of B
    ans = sorted((qb[q["id"]]["top1_score"], q["id"]) for q in answerable
                 if qb[q["id"]].get("top1_score") is not None)
    una = sorted((u["top1_score"], u["id"]) for u in B.get("unanswerable", []) if u.get("top1_score") is not None)
    if ans and una:
        print(f"\nTop-1 score of {b_name}, answerable (lowest 6) vs unanswerable (all), ascending:")
        print("  answerable  : " + ", ".join(f"{i} {s:.3f}" for s, i in ans[:6]))
        print("  unanswerable: " + ", ".join(f"{i} {s:.3f}" for s, i in una))
        print(f"\n  threshold sweep (refuse when top-1 < t); {len(ans)} answerable, {len(una)} unanswerable")
        print(f"  {'t':>6}  {'answerable wrongly refused':>28}  {'unanswerable let through':>26}")
        for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98):
            wrong = sum(1 for s, _ in ans if s < t)
            thru = sum(1 for s, _ in una if s >= t)
            print(f"  {t:>6.2f}  {wrong:>20} of {len(ans):<5}  {thru:>18} of {len(una)}")
        print("  Illustrative only: tiny n, same data used to read the sweep. Sigmoid scores saturate near 1.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline run name, e.g. d2-hybrid")
    ap.add_argument("b", help="candidate run name, e.g. d2-hybrid_rerank")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    main(args.a, args.b, args.k)
