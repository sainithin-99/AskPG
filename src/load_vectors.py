import json
from pathlib import Path

import numpy as np
from qdrant_client import models

import embed_index as ei

VECTORS_PATH = ei.ROOT / "data" / "vectors" / "vectors.npz"


def load_vectors_into(client) -> int:
    records = [json.loads(line) for line in ei.CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
    z = np.load(VECTORS_PATH)
    ids = [str(x) for x in z["chunk_ids"]]
    if ids != [r["chunk_id"] for r in records] or str(z["model"]) != ei.EMBED_MODEL:
        raise RuntimeError("vectors.npz does not match chunks.jsonl / embed model: re-run scripts/export_vectors.py")
    vecs = z["vectors"]
    if not client.collection_exists(ei.COLLECTION):
        client.create_collection(
            collection_name=ei.COLLECTION,
            vectors_config=models.VectorParams(size=ei.EMBED_DIM, distance=models.Distance.COSINE),
        )
    for s in range(0, len(records), ei.UPSERT_BATCH):
        batch = records[s:s + ei.UPSERT_BATCH]
        client.upsert(collection_name=ei.COLLECTION, points=[
            models.PointStruct(id=ei.point_id(r["chunk_id"]), vector=vecs[s + i].tolist(),
                               payload={k: r[k] for k in ei.PAYLOAD_FIELDS})
            for i, r in enumerate(batch)])
    return client.count(ei.COLLECTION, exact=True).count