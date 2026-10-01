"""
Embed chunks and load them into Qdrant (dense vectors only; BM25 comes later).

Setup:
    pip install sentence-transformers qdrant-client numpy
    docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

Usage (from the repo root):
    python src/embed_index.py index --recreate    # (re)build the collection from data/chunks/chunks.jsonl
    python src/embed_index.py query "what does work_mem control?" -k 5

Re-running `index` without --recreate is safe: point IDs are deterministic, so existing chunks
are overwritten, not duplicated.

Other modules (retrieve.py, eval) import embed_query / get_client / COLLECTION from here so the
model, prefix and collection name are defined in exactly one place.
"""

import argparse
import functools
import json
import os
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"

PAYLOAD_FIELDS = ("chunk_id", "url", "page_title", "section_path", "covers", "context", "text", "n_tokens")

PG_VERSION = "17"
COLLECTION = f"askpg_pg{PG_VERSION}"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# bge recommends this instruction on the QUERY side only (passages get no prefix).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")  # ":memory:" works for tests
UPSERT_BATCH = 256
ENCODE_BATCH = 64

#PAYLOAD_FIELDS = ("chunk_id", "url", "page_title", "section_path", "context", "text", "n_tokens")


# --------------------------------------------------------------------------- shared helpers
def get_client() -> QdrantClient:
    if QDRANT_URL == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=QDRANT_URL)


@functools.lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBED_MODEL)  # picks CUDA / Apple MPS / CPU automatically
    model.max_seq_length = 512
    return model


def doc_text(rec: dict) -> str:
    """What actually gets embedded: breadcrumb + body. Headings carry a lot of the meaning."""
    return f"{rec['context']}\n{rec['text']}"


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be ints or UUIDs. uuid5 makes them stable across runs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"askpg/{PG_VERSION}/{chunk_id}"))


def embed_docs(texts: list[str]):
    return load_model().encode(
        texts, batch_size=ENCODE_BATCH, normalize_embeddings=True, show_progress_bar=False
    )


def embed_query(query: str):
    return load_model().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]


def vector_search(query: str, k: int = 5, client: QdrantClient | None = None):
    client = client or get_client()
    return client.query_points(
        collection_name=COLLECTION,
        query=embed_query(query).tolist(),
        limit=k,
        with_payload=True,
    ).points


# --------------------------------------------------------------------------- indexing
def count_truncated(texts: list[str]) -> int:
    """How many texts exceed the model's max length (they'd be silently cut off)."""
    model = load_model()
    over = 0
    for i in range(0, len(texts), 1000):
        ids = model.tokenizer(texts[i:i + 1000], add_special_tokens=True, truncation=False)["input_ids"]
        over += sum(len(x) > model.max_seq_length for x in ids)
    return over


def build_index(recreate: bool) -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit("No chunks found. Run: python src/ingest.py chunk")
    records = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    print(f"Loaded {len(records)} chunks from {CHUNKS_PATH}")

    client = get_client()
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}' ({EMBED_DIM}-dim, cosine)")

    load_model()
    print(f"Embedding device: {load_model().device}")
    truncated = count_truncated([doc_text(r) for r in records])
    print(f"Chunks longer than the model's {load_model().max_seq_length}-token limit: {truncated}"
          + ("  <- lower MAX_TOKENS in ingest.py" if truncated else ""))

    start_time = time.time()
    for start in range(0, len(records), UPSERT_BATCH):
        batch = records[start:start + UPSERT_BATCH]
        vectors = embed_docs([doc_text(r) for r in batch])
        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id(r["chunk_id"]),
                    vector=v.tolist(),
                    payload={k: r[k] for k in PAYLOAD_FIELDS},
                )
                for r, v in zip(batch, vectors)
            ],
        )
        done = min(start + UPSERT_BATCH, len(records))
        rate = done / (time.time() - start_time)
        print(f"  {done}/{len(records)} indexed  ({rate:.0f} chunks/s)")

    stored = client.count(COLLECTION, exact=True).count
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s. Collection holds {stored} points (expected {len(records)}).")
    if stored != len(records):
        print("WARNING: count mismatch. Duplicate chunk_ids? Check ingest.py output.")


# --------------------------------------------------------------------------- inspection
def query_cmd(query: str, k: int) -> None:
    client = get_client()
    embed_query("warmup")  # load the model (and warm CUDA) outside the timed region

    t0 = time.perf_counter()
    qvec = embed_query(query).tolist()
    t1 = time.perf_counter()
    hits = client.query_points(
        collection_name=COLLECTION, query=qvec, limit=k, with_payload=True
    ).points
    t2 = time.perf_counter()

    print(f"Query: {query!r}")
    print(f"Latency: embed {(t1 - t0) * 1000:.0f} ms | qdrant {(t2 - t1) * 1000:.0f} ms\n")
    for rank, h in enumerate(hits, 1):
        p = h.payload
        snippet = p["text"].replace("\n", " ")[:200]
        print(f"{rank}. score={h.score:.3f}  {p['chunk_id']}")
        print(f"   {p['context']}")
        print(f"   {p['url']}")
        print(f"   {snippet}...\n")

# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "index":
        build_index(args.recreate)
    else:
        query_cmd(args.text, args.k)
