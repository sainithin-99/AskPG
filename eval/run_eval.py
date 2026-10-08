"""
Retrieval evaluation for AskPG (generation metrics come later, in the same file).

Usage (from the repo root):
    python eval/run_eval.py find "work_mem"            # helper: list page + anchor of sections to label with
    python eval/run_eval.py find rollback syntax --text  # also search chunk bodies
    python eval/run_eval.py run --name v2-vector       # coverage check, then metrics, latency, JSON dump

Dataset: eval/dataset.jsonl, one JSON object per line:
    {"id": "q001",
     "question": "What does work_mem control?",
     "type": "factual",                  # factual | multi_hop | ambiguous | unanswerable
     "gold": [{"page": "runtime-config-resource", "anchor": "GUC-WORK-MEM"}],
     "reference_answer": "..."}          # used later by the answer-correctness judge
  - gold is a list of sections. Use {"page", "anchor"} or, if a section has no anchor, {"page", "heading"}.
  - multi_hop: several gold sections, ALL are needed. ambiguous: several gold sections, ANY is acceptable.
  - unanswerable: no gold (omit it or use []). These are excluded from recall/MRR; we only record top-1 scores
    so you can calibrate an "I don't know" threshold on real numbers.
  - Never label with chunk_id: chunk ids change whenever chunking changes.

A retrieved chunk is a hit for a gold section if that section is in the chunk's `covers` (page + anchor,
or page + heading text). Matching is section-level: every chunk of a section carries the same `covers`, so
"hit" means "a chunk from the right section was retrieved", not "the exact chunk with the answer".
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
DATASET_PATH = ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

KS = (1, 3, 5, 10)
MAX_K = max(KS)
TYPES = ("factual", "multi_hop", "ambiguous", "unanswerable")


# --------------------------------------------------------------------------- section keys
def norm_heading(t: str) -> str:
    """'F.29.3. Author' / '19.4 Resource Consumption' -> lowercase heading without numbering."""
    return re.sub(r"^(?:[A-Z]\.)?[\d.]+\s+", "", t).strip().lower()


def norm_page(p: str) -> str:
    """'https://.../17/sql-rollback.html' or 'sql-rollback.html' or 'sql-rollback' -> 'sql-rollback'"""
    p = p.split("#")[0].rsplit("/", 1)[-1]
    return p[:-5] if p.endswith(".html") else p


def chunk_keys(rec: dict) -> set:
    """Every (page, anchor) and (page, 'h:heading') this chunk covers."""
    page = norm_page(urlparse(rec["url"]).path)
    covers = rec.get("covers") or [{
        "anchor": urlparse(rec["url"]).fragment or None,
        "section_path": rec["section_path"],
    }]
    keys = set()
    for c in covers:
        if c.get("anchor"):
            keys.add((page, c["anchor"]))
        if c.get("section_path"):
            keys.add((page, "h:" + norm_heading(c["section_path"][-1])))
    return keys


def gold_key(g: dict) -> tuple:
    page = norm_page(g["page"])
    if g.get("anchor"):
        return (page, g["anchor"])
    if g.get("heading"):
        return (page, "h:" + norm_heading(g["heading"]))
    raise ValueError(f"gold entry needs 'anchor' or 'heading': {g}")


# --------------------------------------------------------------------------- loading + coverage check
def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit("No chunks.jsonl. Run: python src/ingest.py chunk")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"No dataset at {DATASET_PATH}. Label questions with the `find` helper first.")
    rows = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            for field in ("id", "question", "type"):
                if field not in r:
                    raise SystemExit(f"dataset line {n}: missing '{field}'")
            if r["type"] not in TYPES:
                raise SystemExit(f"dataset line {n} ({r['id']}): type must be one of {TYPES}")
            r["gold"] = r.get("gold") or []
            if r["type"] != "unanswerable" and not r["gold"]:
                raise SystemExit(f"dataset line {n} ({r['id']}): answerable question without gold")
            rows.append(r)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in dataset")
    return rows


def coverage_check(dataset: list[dict], chunks: list[dict]) -> list[tuple]:
    """Gold sections that no chunk covers. These would silently count as misses forever."""
    have = set()
    for c in chunks:
        have |= chunk_keys(c)
    missing = []
    for r in dataset:
        for g in r["gold"]:
            if gold_key(g) not in have:
                missing.append((r["id"], g))
    return missing


# --------------------------------------------------------------------------- retrievers
def vector_retriever():
    """Dense-only baseline, via the helpers embed_index.py exports."""
    import embed_index as ei

    client = ei.get_client()

    def retrieve(query: str, k: int):
        t0 = time.perf_counter()
        qvec = ei.embed_query(query).tolist()
        t1 = time.perf_counter()
        pts = client.query_points(
            collection_name=ei.COLLECTION, query=qvec, limit=k, with_payload=True
        ).points
        t2 = time.perf_counter()
        return [(p.payload, p.score) for p in pts], {
            "embed_ms": (t1 - t0) * 1000,
            "search_ms": (t2 - t1) * 1000,
        }

    return retrieve


# Later: "hybrid": ..., "hybrid_rerank": ... with the same signature:
#   retrieve(query, k) -> ([(payload, score), ...], {"stage_name_ms": float, ...})
# RETRIEVERS = {"vector": vector_retriever}

def make_retriever(mode: str):
    def factory():
        import retrieve as rt  # src/ is already on sys.path at the top of this file
        return rt.build_retriever(mode)
    return factory

RETRIEVERS = {m: make_retriever(m) for m in ("vector", "bm25", "hybrid", "hybrid_rerank")}


# --------------------------------------------------------------------------- metrics
def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


def score_question(q: dict, hits: list) -> dict:
    """Rank (1-based) of the first chunk covering each gold section, or None if not in the top MAX_K."""
    keys_per_hit = [chunk_keys(p) for p, _ in hits]
    ranks = []
    for g in q["gold"]:
        gk = gold_key(g)
        ranks.append(next((i + 1 for i, ks in enumerate(keys_per_hit) if gk in ks), None))
    found = [r for r in ranks if r is not None]
    first = min(found) if found else None
    # multi_hop needs ALL gold sections; factual/ambiguous need ANY (ambiguous lists alternatives).
    need_all = q["type"] == "multi_hop"
    if need_all:
        success_rank = max(ranks) if all(r is not None for r in ranks) else None
    else:
        success_rank = first
    top5_sections = {p["url"] for p, _ in hits[:5]}
    return {
        "id": q["id"], "type": q["type"], "gold_ranks": ranks, "first_rank": first,
        "success_rank": success_rank,
        "distinct_sections_top5": len(top5_sections),
        "top1_score": hits[0][1] if hits else None,
        "top5": [{"chunk_id": p["chunk_id"], "context": p["context"], "url": p["url"],
                "score": round(float(s), 4)} for p, s in hits[:5]],
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in KS:
        out[f"success@{k}"] = sum(1 for r in rows if r["success_rank"] and r["success_rank"] <= k) / n
    for k in (5,):
        recalls = []
        for r in rows:
            g = r["gold_ranks"]
            recalls.append(sum(1 for x in g if x and x <= k) / len(g))
        out[f"gold_recall@{k}"] = statistics.mean(recalls)
    out["mrr"] = statistics.mean(1 / r["success_rank"] if r["success_rank"] else 0.0 for r in rows)
    out["distinct_sections_top5"] = statistics.mean(r["distinct_sections_top5"] for r in rows)
    return out


# --------------------------------------------------------------------------- run
def run(name: str, retriever_name: str, allow_missing: bool) -> None:
    dataset = load_dataset()
    chunks = load_chunks()

    missing = coverage_check(dataset, chunks)
    if missing:
        print(f"COVERAGE CHECK FAILED: {len(missing)} gold section(s) are in no chunk's `covers`:")
        for qid, g in missing:
            print(f"  {qid}: {g}")
        print("Fix the label (use `find`) or the chunking. Otherwise these count as misses forever.")
        if not allow_missing:
            raise SystemExit(1)
    else:
        n_gold = sum(len(r["gold"]) for r in dataset)
        print(f"Coverage check OK: all {n_gold} gold sections exist in chunks.jsonl")

    retrieve = RETRIEVERS[retriever_name]()
    retrieve("warmup query", 1)  # load model / warm CUDA / open connections outside the timed region

    scored, unans, lat = [], [], defaultdict(list)
    for q in dataset:
        t0 = time.perf_counter()
        hits, stage_ms = retrieve(q["question"], MAX_K)
        total_ms = (time.perf_counter() - t0) * 1000
        for stage, ms in {**stage_ms, "total_ms": total_ms}.items():
            lat[stage].append(ms)
        if q["type"] == "unanswerable":
            unans.append({"id": q["id"], "top1_score": hits[0][1] if hits else None,
            "top1_context": hits[0][0]["context"] if hits else None})
        else:
            scored.append(score_question(q, hits))

    overall = summarize(scored)
    by_type = {t: summarize([r for r in scored if r["type"] == t]) for t in TYPES
               if any(r["type"] == t for r in scored)}

    print(f"\n=== {name}  ({retriever_name}, {len(dataset)} questions, "
          f"{len(scored)} answerable, {len(unans)} unanswerable) ===")
    cols = ["n", "success@1", "success@3", "success@5", "success@10", "gold_recall@5", "mrr",
            "distinct_sections_top5"]
    print(f"{'group':<12}" + "".join(f"{c:>14}" for c in cols))
    for label, m in [("ALL", overall), *by_type.items()]:
        print(f"{label:<12}" + "".join(
            f"{m[c]:>14.0f}" if c == "n" else f"{m[c]:>14.3f}" for c in cols))

    print("\nLatency (ms), single process, warm, one query at a time:")
    for stage, xs in lat.items():
        print(f"  {stage:<10} p50 {pct(xs, 50):7.1f}   p95 {pct(xs, 95):7.1f}   max {max(xs):7.1f}")

    if unans:
        a_scores = [r["top1_score"] for r in scored if r["top1_score"] is not None]
        u_scores = [r["top1_score"] for r in unans if r["top1_score"] is not None]
        print("\nTop-1 score, answerable vs unanswerable (for calibrating a refusal threshold):")
        print(f"  answerable   : min {min(a_scores):.3f}  median {statistics.median(a_scores):.3f}  max {max(a_scores):.3f}")
        print(f"  unanswerable : min {min(u_scores):.3f}  median {statistics.median(u_scores):.3f}  max {max(u_scores):.3f}")

    misses = [r for r in scored if not r["success_rank"] or r["success_rank"] > 5]
    print(f"\nQuestions that fail @5: {len(misses)}")
    for r in misses:
        print(f"  {r['id']} ({r['type']})  gold ranks {r['gold_ranks']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps({
        "name": name, "retriever": retriever_name, "time": datetime.now().isoformat(timespec="seconds"),
        "n_chunks": len(chunks), "overall": overall, "by_type": by_type,
        "latency_ms": {s: {"p50": pct(x, 50), "p95": pct(x, 95)} for s, x in lat.items()},
        "per_question": scored, "unanswerable": unans,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")

    o, t = overall, lat
    print("\nRow for experiments.md:")
    print("| run | success@1 | success@5 | success@10 | MRR | p50 total ms | p95 total ms |")
    print(f"| {name} | {o['success@1']:.3f} | {o['success@5']:.3f} | {o['success@10']:.3f} | {o['mrr']:.3f} "
          f"| {pct(t['total_ms'], 50):.0f} | {pct(t['total_ms'], 95):.0f} |")


# --------------------------------------------------------------------------- labelling helper
def find(terms: list[str], search_text: bool, limit: int = 40) -> None:
    """List (page, anchor, heading path) for sections whose page/heading (and optionally body) match all terms."""
    seen, shown = set(), 0
    for rec in load_chunks():
        page = norm_page(urlparse(rec["url"]).path)
        for c in rec.get("covers") or [{"anchor": None, "section_path": rec["section_path"]}]:
            key = (page, c.get("anchor"))
            if key in seen:
                continue
            hay = (page + " " + " ".join(c.get("section_path") or [])).lower()
            if search_text:
                hay += " " + rec["text"].lower()
            if all(t.lower() in hay for t in terms):
                seen.add(key)
                print(f'{{"page": "{page}", "anchor": {json.dumps(c.get("anchor"))}}}   '
                      f'# {" > ".join(c.get("section_path") or [])}')
                shown += 1
                if shown >= limit:
                    print(f"... stopped at {limit} results; add more terms to narrow down")
                    return
    if not shown:
        print("no matches (try --text, or fewer terms)")


# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True, help="run label, e.g. v2-vector; also the results filename")
    r.add_argument("--retriever", default="vector", choices=sorted(RETRIEVERS))
    r.add_argument("--allow-missing", action="store_true", help="continue despite gold sections missing from the index")
    f = sub.add_parser("find")
    f.add_argument("terms", nargs="+")
    f.add_argument("--text", action="store_true", help="also search chunk bodies, not just page name and headings")
    args = ap.parse_args()

    if args.cmd == "run":
        run(args.name, args.retriever, args.allow_missing)
    else:
        find(args.terms, args.text)