"""
Test setup. CI has no GPU, Qdrant, corpus or LLM, so heavy modules are stubbed
BEFORE src/retrieve.py is imported. Only pure functions are tested.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "eval"):
    sys.path.insert(0, str(p))

sys.modules["bm25s"] = types.ModuleType("bm25s")
_ei = types.ModuleType("embed_index")
_ei.doc_text = lambda rec: f"{rec['context']}\n{rec['text']}"
sys.modules["embed_index"] = _ei