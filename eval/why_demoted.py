r"""
Diagnostic: for chosen eval questions, print the hybrid and hybrid_rerank top-k side by side so you can see
WHAT outranked a gold section after reranking. Gold-covering chunks are marked with G<index of the gold entry>.

Usage (from the repo root, cmd):
    python eval\why_demoted.py fact-05 multi-01 multi-04 amb-02
    python eval\why_demoted.py fact-05 -k 10

Reads the current eval\dataset.jsonl; builds both retrievers once. Prints the rank of each gold section within
the top 30 (the reranker only sees the top 30 fused candidates, so a gold chunk beyond hybrid rank 30 is
invisible to it). This is a READING aid for a handful of questions: do not change the pipeline from one query.
Untested on the real files when written.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (also puts <repo>/src on sys.path)
import retrieve as rt  # noqa: E402

POOL = 30


def show(mode: str, hits: list, gold_idx: dict, k: int) -> None:
    print(f"  --- {mode}")
    first = {}
    for rank, (rec, score) in enumerate(hits, 1):
        marks = sorted({j for key, j in gold_idx.items() if key in ev.chunk_keys(rec)})
        for j in marks:
            first.setdefault(j, rank)
        if rank <= k:
            tag = ("G" + ",".join(map(str, marks))) if marks else "  "
            print(f"   {rank:>2} {score:8.4f} {tag:<5} {rec['chunk_id']:<34} {rec['context'][:70]}")
    print("   gold first rank within top "
          f"{len(hits)}: " + (", ".join(f"G{j}={first.get(j, 'none')}" for j in sorted(set(gold_idx.values())))))


def main(ids: list, k: int) -> None:
    dataset = {q["id"]: q for q in ev.load_dataset()}
    unknown = [i for i in ids if i not in dataset]
    if unknown:
        raise SystemExit(f"Unknown question id(s): {unknown}")

    hyb, rer = rt.build_retriever("hybrid"), rt.build_retriever("hybrid_rerank")
    hyb("warmup query", 1)
    rer("warmup query", 1)

    for qid in ids:
        q = dataset[qid]
        gold_idx = {ev.gold_key(g): j for j, g in enumerate(q["gold"])}
        print("\n" + "=" * 100)
        print(f"{qid} ({q['type']}): {q['question']}")
        for j, g in enumerate(q["gold"]):
            print(f"  G{j} = {g['page']} / {g.get('heading') or g.get('anchor')}")
        show("hybrid", hyb(q["question"], POOL)[0], gold_idx, k)
        show("hybrid_rerank", rer(q["question"], POOL)[0], gold_idx, k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("-k", type=int, default=8, help="rows to print per mode (gold ranks use the top 30)")
    a = ap.parse_args()
    main(a.ids, a.k)
