"""
Retrieval modes for AskPG, all behind one interface:

    retrieve(query, k) -> ([(payload, score), ...], {"stage_ms": float, ...})

Modes:
    vector          dense only (bge-small + Qdrant)          -- the baseline
    bm25            keyword only (bm25s, in memory)
    hybrid          vector + bm25 fused with Reciprocal Rank Fusion
    hybrid_rerank   hybrid candidates re-scored by a cross-encoder (bge-reranker-base)

Setup:
    pip install bm25s sentence-transformers qdrant-client

Usage (from the repo root):
    python src/retrieve.py query "what does work_mem control?" --mode hybrid -k 5
    python src/retrieve.py query "ALTER TABLE SET LOGGED" --mode hybrid_rerank -k 5

Score meaning differs per mode, so never compare scores across modes:
    vector = cosine, bm25 = BM25 score, hybrid = RRF score (tiny, rank-based),
    hybrid_rerank = cross-encoder logit. For an "unanswerable" threshold the reranker
    score is the most informative one; calibrate it on the eval set.

Payloads come from data/chunks/chunks.jsonl (loaded once), not from Qdrant, so BM25 and vector
results are identical dicts and the eval sees `covers` regardless of what is stored in Qdrant.
"""

import argparse
import functools
import json
import re
import time
from pathlib import Path

import bm25s

import embed_index as ei

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")

# --------------------------------------------------------------------------- config
N_CANDIDATES = 50        # per retriever, before fusion
RRF_K = 60               # standard constant from the RRF paper; damps the weight of top ranks
RERANK_TOP = 30          # how many fused candidates the cross-encoder scores
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_BATCH = 16
RERANK_MAX_LEN = 512

# Experiment knob (leave False for the baseline): also index the parts of snake_case identifiers,
# so "work mem" can match "work_mem". Change it as a separate, logged experiment.
SPLIT_IDENTIFIERS = False

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to was what when where which who "
    "why with does do can".split()
)


# --------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def load_records() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit("No chunks.jsonl. Run: python src/ingest.py chunk")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@functools.lru_cache(maxsize=1)
def records_by_id() -> dict[str, dict]:
    return {r["chunk_id"]: r for r in load_records()}


# --------------------------------------------------------------------------- BM25
def tokenize(text: str) -> list[str]:
    """Lowercase, keep snake_case identifiers whole (work_mem, pg_stat_activity), drop stopwords.
    No stemming: PG docs are full of exact identifiers, and stemming can blur them."""
    toks = _TOKEN_RE.findall(text.lower())
    if SPLIT_IDENTIFIERS:
        extra = [p for t in toks if "_" in t for p in t.split("_") if p]
        toks = toks + extra
    return [t for t in toks if t not in _STOP]


class BM25Index:
    def __init__(self, records: list[dict]):
        self.records = records
        # Same text the dense model sees (breadcrumb + body), so the two retrievers are comparable.
        self.index = bm25s.BM25()
        self.index.index([tokenize(ei.doc_text(r)) for r in records], show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        toks = [t for t in tokenize(query) if t in self.index.vocab_dict]  # unknown terms can't match anyway
        if not toks:
            return []
        k = min(k, len(self.records))
        idx, scores = self.index.retrieve([toks], k=k, show_progress=False)
        return [(self.records[i], float(s)) for i, s in zip(idx[0], scores[0]) if s > 0]


@functools.lru_cache(maxsize=1)
def get_bm25() -> BM25Index:
    return BM25Index(load_records())


# --------------------------------------------------------------------------- fusion
def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over lists of chunk_ids (best first). Score = sum 1/(k + rank).
    Uses ranks only, so it needs no score normalisation between cosine and BM25 (different scales)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- reranker
@functools.lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN)  # fp32 (GTX 16-series: no useful fp16)


def rerank(query: str, recs: list[dict]) -> list[tuple[dict, float]]:
    model = get_reranker()
    pairs = [(query, ei.doc_text(r)) for r in recs]
    scores = model.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    return sorted(zip(recs, map(float, scores)), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- retrievers
def build_retriever(mode: str):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    client = ei.get_client()
    by_id = records_by_id()
    bm25 = get_bm25() if mode != "vector" else None
    if mode == "hybrid_rerank":
        get_reranker()

    def dense(query: str, n: int, t: dict):
        t0 = time.perf_counter()
        qvec = ei.embed_query(query).tolist()
        t1 = time.perf_counter()
        pts = client.query_points(collection_name=ei.COLLECTION, query=qvec, limit=n, with_payload=True).points
        t2 = time.perf_counter()
        t["embed_ms"], t["vector_ms"] = (t1 - t0) * 1000, (t2 - t1) * 1000
        return [(by_id[p.payload["chunk_id"]], p.score) for p in pts]

    def sparse(query: str, n: int, t: dict):
        t0 = time.perf_counter()
        out = bm25.search(query, n)
        t["bm25_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def retrieve(query: str, k: int):
        t: dict = {}
        if mode == "vector":
            return [(r, s) for r, s in dense(query, k, t)], t
        if mode == "bm25":
            return sparse(query, k, t), t

        d, s = dense(query, N_CANDIDATES, t), sparse(query, N_CANDIDATES, t)
        t0 = time.perf_counter()
        fused = rrf_fuse([[r["chunk_id"] for r, _ in d], [r["chunk_id"] for r, _ in s]])
        t["fuse_ms"] = (time.perf_counter() - t0) * 1000

        if mode == "hybrid":
            return [(by_id[cid], sc) for cid, sc in fused[:k]], t

        cands = [by_id[cid] for cid, _ in fused[:RERANK_TOP]]
        t0 = time.perf_counter()
        ranked = rerank(query, cands)
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        return ranked[:k], t

    return retrieve


# --------------------------------------------------------------------------- cli
def query_cmd(query: str, mode: str, k: int) -> None:
    retrieve = build_retriever(mode)
    retrieve("warmup query", 1)  # load models / warm CUDA outside the timed region
    t0 = time.perf_counter()
    hits, stages = retrieve(query, k)
    total = (time.perf_counter() - t0) * 1000
    print(f"Query: {query!r}   mode={mode}")
    print("Latency: " + " | ".join(f"{s} {v:.0f}" for s, v in {**stages, 'total_ms': total}.items()) + "\n")
    for rank, (r, score) in enumerate(hits, 1):
        print(f"{rank}. score={score:.4f}  {r['chunk_id']}")
        print(f"   {r['context']}")
        print(f"   {r['url']}")
        print(f"   {r['text'].replace(chr(10), ' ')[:200]}...\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--mode", default="hybrid", choices=MODES)
    q.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    query_cmd(args.text, args.mode, args.k)