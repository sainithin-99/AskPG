r"""
Make sure the Qdrant collection exists and has one point per chunk; build it if not. Used by the `indexer`
service in docker-compose.yml, and safe to run locally.

Usage (from the repo root, cmd; Qdrant must be running):
    python scripts\ensure_index.py

Behaviour
  - waits up to ~2 minutes for Qdrant (QDRANT_URL, default http://localhost:6333),
  - if the collection holds exactly as many points as chunks.jsonl has lines: does nothing,
  - otherwise rebuilds it with embed_index.build_index(recreate=True) (CPU: roughly 15 minutes for 6489 chunks,
    from the earlier 7 chunks/s CPU measurement; not re-measured in a container).

LIMIT: the check compares COUNTS only. If you re-chunk or change the embedder and the count stays the same, this
script will NOT notice. After such a change run the full rebuild yourself:
    python src\embed_index.py index --recreate
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import embed_index as ei  # noqa: E402


def wait_for_qdrant(client, tries: int = 60) -> None:
    for _ in range(tries):
        try:
            client.get_collections()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"Qdrant not reachable at {ei.QDRANT_URL} after {tries * 2} s.")


def main() -> None:
    if not ei.CHUNKS_PATH.exists():
        raise SystemExit(f"No {ei.CHUNKS_PATH}. Run: python src/ingest.py chunk")
    with ei.CHUNKS_PATH.open(encoding="utf-8") as f:
        n_chunks = sum(1 for line in f if line.strip())
    client = ei.get_client()
    wait_for_qdrant(client)
    have = client.count(ei.COLLECTION, exact=True).count if client.collection_exists(ei.COLLECTION) else None
    if have == n_chunks:
        print(f"Collection {ei.COLLECTION} already holds {have} points = {n_chunks} chunks. Nothing to do.")
        return
    print(f"Collection {ei.COLLECTION}: {have} points, chunks.jsonl has {n_chunks}. Rebuilding.")
    ei.build_index(recreate=True)


if __name__ == "__main__":
    main()
