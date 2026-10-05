r"""
One-off diagnostic: for a query and a phrase that the right answer must contain, show which chunk(s)
contain that phrase and where they rank in each retrieval mode.

Usage (from the repo root):
    python src\rank_of.py "is the COMMENT command part of the SQL standard?" "no COMMENT command"

Reading the output:
  - vector / bm25 / hybrid are searched down to rank 200 (hybrid: at most the fused candidates, <= 2 x N_CANDIDATES).
  - hybrid_rerank only re-scores the top RERANK_TOP (30) fused candidates. If the chunk's hybrid rank is
    worse than RERANK_TOP, the reranker never sees it, and "not in top N" for hybrid_rerank is expected.
  - "needle at char X of Y" shows how deep inside the chunk the phrase sits. A phrase near the end of a
    long chunk is diluted in the embedding and in the cross-encoder's view.

Untested: written for this session, run it once and check that it matches what retrieve.py prints.
"""
import sys

import retrieve as rt

K = 200


def main(query: str, needle: str) -> None:
    targets = {r["chunk_id"]: r for r in rt.load_records() if needle.lower() in r["text"].lower()}
    if not targets:
        raise SystemExit(f"No chunk contains {needle!r}.")

    print(f"Query: {query!r}")
    print(f"Chunks containing {needle!r}:")
    for cid, r in targets.items():
        pos = r["text"].lower().find(needle.lower())
        print(f"  {cid}  {r['n_tokens']} tok  {r['context']}")
        print(f"    needle at char {pos} of {len(r['text'])}")
    print()

    for mode in rt.MODES:
        retrieve = rt.build_retriever(mode)
        hits, _ = retrieve(query, K)
        ids = [r["chunk_id"] for r, _ in hits]
        found = [f"{cid} -> rank {ids.index(cid) + 1} of {len(ids)}" for cid in targets if cid in ids]
        print(f"{mode:14s} " + (", ".join(found) if found else f"not in top {len(ids)}"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python src\\rank_of.py "<query>" "<phrase the answer contains>"')
    main(sys.argv[1], sys.argv[2])