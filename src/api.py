r"""
AskPG HTTP API (FastAPI): one /ask endpoint over retrieve + generate, with one JSONL trace line per request.

Setup:
    pip install fastapi uvicorn

Run (from the repo root; Qdrant must be running and .env must be set, same as for generate.py):
    uvicorn api:app --app-dir src --port 8000

Then open http://localhost:8000/docs and try POST /ask. Other endpoints:
    GET /health   Qdrant reachable? which model and retrieval mode?
    GET /stats    status counts and p50/p95 latency over the traces written so far

Config (environment variables, all optional):
    ASKPG_MODE        retrieval mode, default hybrid_rerank (vector | bm25 | hybrid | hybrid_rerank)
    ASKPG_TRACE_PATH  default data/traces/requests.jsonl (data/ is gitignored)

Design choices:
  - Models (embedder, reranker, BM25 index) load ONCE at startup, never per request.
  - ONE worker process and a lock around the pipeline. The models and the LLM throttle in generate.py keep
    in-process state and were not written to be thread-safe. The price is that concurrent requests queue;
    the wait is measured and stored as queue_wait_ms, so load shows up in the traces instead of hiding.
  - Sources are returned only when status == "ok", de-duplicated by URL. `citations` keeps the full
    n -> url map so a UI can link every [n] in the answer text.
  - Trace line = request_id, time, question, status, error, attempts, model, mode, k, per-stage latency,
    tokens, cost estimate, cited and shown chunk ids. It stores the user's question: a public deployment
    needs a retention policy (not decided yet).
  - This file does NOT use src/tracing.py: that file was not available when this was written, so the trace
    format here is self-contained. Unify the two later.

Status handling: ok / refused / citation_failed -> HTTP 200 with the status in the body; llm_error -> 502;
retrieval failure (Qdrant down, ...) -> 503.

NOT tested when written (syntax read only). The first run shows startup, one answer and one trace line;
it does not show behaviour under concurrent load.
"""
import json
import os
import statistics
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import embed_index as ei
import generate as gen
import retrieve as rt

ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = Path(os.environ.get("ASKPG_TRACE_PATH") or ROOT / "data" / "traces" / "requests.jsonl")
MODE = os.environ.get("ASKPG_MODE") or "hybrid_rerank"

_state: dict = {}
_pipeline_lock = threading.Lock()  # one request at a time through models + LLM throttle
_trace_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieve = rt.build_retriever(MODE)  # loads BM25 / reranker as the mode needs
    retrieve("warmup query", 1)  # load models and warm CUDA before the first real request
    _state["retrieve"] = retrieve
    _state["llm"] = gen.make_llm()  # one client: the throttle state persists across requests
    _state["n_chunks"] = len(rt.load_records())
    yield
    _state.clear()


app = FastAPI(title="AskPG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=gen.DEFAULT_K, ge=1, le=10)


# --------------------------------------------------------------------------- tracing
def write_trace(rec: dict) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _trace_lock, TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_traces(limit: int = 1000) -> list[dict]:
    if not TRACE_PATH.exists():
        return []
    with TRACE_PATH.open(encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line must not break /stats
    return out


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


# --------------------------------------------------------------------------- endpoints
@app.post("/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()
    rid = uuid.uuid4().hex[:12]
    t_req = time.perf_counter()
    base = {"request_id": rid, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question, "mode": MODE, "k": req.k}

    with _pipeline_lock:
        queue_ms = (time.perf_counter() - t_req) * 1000
        try:
            res = gen.generate(question, _state["retrieve"], _state["llm"], k=req.k)
        except Exception as e:  # generate() never raises for LLM failures, so this is retrieval / infrastructure
            wall_ms = (time.perf_counter() - t_req) * 1000
            write_trace({**base, "status": "retrieval_error", "error": f"{type(e).__name__}: {e}",
                         "latency_ms": {"queue_wait_ms": queue_ms, "wall_ms": wall_ms}})
            raise HTTPException(status_code=503, detail=f"Retrieval backend unavailable (request {rid}).")

    wall_ms = (time.perf_counter() - t_req) * 1000
    latency = {**res["latency_ms"], "queue_wait_ms": queue_ms, "wall_ms": wall_ms}
    write_trace({
        **base, "status": res["status"], "error": res["error"], "attempts": res["attempts"], "model": res["model"],
        "latency_ms": latency, "usage": res["usage"], "uncited_sentences": res["uncited_sentences"],
        "cited_chunks": [c["chunk_id"] for c in res["citations"]],
        "sources_shown": [s["chunk_id"] for s in res["sources_shown"]],
    })

    if res["status"] == "llm_error":
        raise HTTPException(status_code=502, detail=f"The language model did not answer (request {rid}). Try again.")

    sources, seen = [], set()
    if res["status"] == "ok":  # show sources only for a well-formed, cited answer; one entry per URL
        for c in res["citations"]:
            if c["url"] not in seen:
                seen.add(c["url"])
                sources.append({"n": c["n"], "url": c["url"], "context": c["context"]})
    return {
        "request_id": rid,
        "status": res["status"],
        "answer": res["answer"],
        "sources": sources,
        "citations": res["citations"] if res["status"] == "ok" else [],
        "model": res["model"],
        "latency_ms": {k: round(v, 1) for k, v in latency.items()},
        "usage": res["usage"],
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """The single-page frontend (web/index.html), served by the same process so no CORS setup is needed."""
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    try:
        ok = ei.get_client().collection_exists(ei.COLLECTION)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {type(e).__name__}")
    if not ok:
        raise HTTPException(status_code=503, detail=f"Collection {ei.COLLECTION} missing")
    return {"status": "ok", "model": gen.LLM_MODEL, "mode": MODE, "chunks": _state.get("n_chunks")}


@app.get("/stats")
def stats(last: int = 1000) -> dict:
    rows = read_traces(max(1, min(last, 10000)))
    if not rows:
        return {"requests": 0}
    counts: dict = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    def series(key: str) -> list[float]:
        return [r["latency_ms"][key] for r in rows if key in r.get("latency_ms", {})]

    lat = {key: {"p50": round(pct(xs, 50), 1), "p95": round(pct(xs, 95), 1)}
           for key in ("wall_ms", "total_ms", "llm_ms", "rerank_ms", "queue_wait_ms")
           if (xs := series(key))}
    toks_in = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rows)
    toks_out = sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows)
    cost = sum(r.get("usage", {}).get("cost_usd", 0.0) for r in rows)
    return {"requests": len(rows), "status": counts, "latency_ms": lat,
            "tokens": {"prompt": toks_in, "completion": toks_out}, "cost_usd_estimate": cost,
            "mean_wall_ms": round(statistics.mean(series("wall_ms")), 1) if series("wall_ms") else None}