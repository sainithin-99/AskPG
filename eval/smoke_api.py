r"""
API smoke test for AskPG: a few known-answer questions sent to a RUNNING server (python eval\smoke_api.py).

Start the server first (separate cmd window, from the repo root):
    uvicorn api:app --app-dir src --port 8000
Then:
    python eval\smoke_api.py
    python eval\smoke_api.py --url http://localhost:8000

Each case checks, deterministically (no LLM judge):
  - HTTP 200 and status == expected ("ok" for answerable cases, "refused" for the unanswerable one)
  - for "ok": at least one returned source URL contains the expected page name, and the answer text contains
    a key fragment (case-insensitive). The fragment check is deliberately weak: it catches an empty or
    off-topic answer, not a subtly wrong one. Correctness is the judge's job, not this script's.
  - for "refused": no sources are returned.

KNOWN ISSUE cases (xfail): the COMMENT question is REFUSED in g1 because the reranker pushes the gold chunk out
of the top 5 (reranker truncation finding). It is kept in the set on purpose and reported as XFAIL, which does
not fail the run. If it ever passes you get XPASS: that is the signal that a fix worked, so look at it.

What this is and is not: a smoke test of the service (is it up, does it answer, are sources attached, does it
refuse). The hosted LLM is not reproducible at temperature 0, so one failure can be noise: rerun before you
believe it. It is NOT a quality gate. Exit code 1 if any non-xfail case fails.

Untested when written (syntax read only).
"""
import argparse
import sys
import time

import requests

# (name, question, expected status, expected page in a source URL, answer fragment, known_issue)
CASES = [
    ("comment-standard", "is the COMMENT command part of the SQL standard?", "ok", "sql-comment", "standard", True),
    ("rollback-syntax", "what is the syntax of ROLLBACK?", "ok", "sql-rollback", "rollback", False),
    ("work-mem", "what does work_mem control?", "ok", "runtime-config-resource", "memory", False),
    ("set-logged", "ALTER TABLE SET LOGGED", "ok", "sql-altertable", "logged", False),
    ("unanswerable", "what is the capital of France?", "refused", None, None, False),
]


def run_case(base: str, case: tuple) -> tuple[str, str]:
    """-> (verdict, detail). verdict is PASS / FAIL (the caller maps known issues to XFAIL / XPASS)."""
    name, question, want_status, page, fragment, _ = case
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base}/ask", json={"question": question, "k": 5}, timeout=180)
    except requests.RequestException as e:
        return "FAIL", f"request error: {type(e).__name__}"
    secs = time.perf_counter() - t0
    if r.status_code != 200:
        return "FAIL", f"HTTP {r.status_code}: {r.text[:120]}"
    body = r.json()
    problems = []
    if body.get("status") != want_status:
        problems.append(f"status {body.get('status')!r}, wanted {want_status!r}")
    if want_status == "ok":
        urls = [s["url"] for s in body.get("sources", [])]
        if not any(page in u for u in urls):
            problems.append(f"no source URL contains {page!r} (got {len(urls)} source(s))")
        if fragment and fragment.lower() not in body.get("answer", "").lower():
            problems.append(f"answer lacks {fragment!r}")
    elif body.get("sources"):
        problems.append("refused answer still returned sources")
    detail = f"{secs:.1f}s status={body.get('status')} sources={len(body.get('sources', []))}"
    return ("FAIL", "; ".join(problems) + f" ({detail})") if problems else ("PASS", detail)


def main(base: str) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        h = requests.get(f"{base}/health", timeout=15)
        h.raise_for_status()
    except requests.RequestException as e:
        print(f"Server not healthy at {base}: {type(e).__name__}. Start it first (see the docstring).")
        return 1
    print(f"/health: {h.json()}\n")

    hard_fails = 0
    for case in CASES:
        verdict, detail = run_case(base, case)
        known = case[5]
        if known:
            label = "XPASS" if verdict == "PASS" else "XFAIL"
        else:
            label = verdict
            hard_fails += verdict == "FAIL"
        print(f"{label:<6} {case[0]:<18} {detail}")
    print(f"\n{hard_fails} hard failure(s). XFAIL = known issue (does not fail the run); XPASS = known issue now passes: read it.")
    return 1 if hard_fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    raise SystemExit(main(ap.parse_args().url.rstrip("/")))
