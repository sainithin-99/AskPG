import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import embed_index as ei  # noqa: E402

records = [json.loads(line) for line in ei.CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
vecs = ei.embed_docs([ei.doc_text(r) for r in records]).astype("float32")
out = ROOT / "data" / "vectors" / "vectors.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, vectors=vecs, chunk_ids=np.array([r["chunk_id"] for r in records]),
                    model=np.array(ei.EMBED_MODEL))
print(f"Saved {out}: {vecs.shape}, {out.stat().st_size / 1e6:.1f} MB")