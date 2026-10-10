import json

import pytest

import generate as gen
import retrieve as rt
import run_eval as ev
import tracing as tr
import check_gold as cg


# ---------------------------------------------------------------- generate.py
def test_refusal_is_ok():
    c = gen.check_citations(gen.REFUSAL, 5)
    assert c["ok"] and c["is_refusal"]


def test_valid_citation_ok():
    c = gen.check_citations("Foo bar baz qux. [1]", 2)
    assert c["ok"] and c["cited"] == [1] and c["uncited_sentences"] == 0


def test_comma_citation():
    c = gen.check_citations("Foo bar baz qux [1, 2].", 3)
    assert c["cited"] == [1, 2] and c["ok"]


def test_invalid_citation_number():
    c = gen.check_citations("Foo bar baz qux [3].", 2)
    assert not c["ok"] and c["invalid"] == [3]


def test_no_citation_fails():
    c = gen.check_citations("This has no citation at all.", 2)
    assert not c["ok"] and c["uncited_sentences"] == 1


def test_normalize_fullwidth():
    assert gen.normalize_citations("fact 【5】.") == "fact [5]."
    assert gen.normalize_citations("fact 【1，2】.") == "fact [1, 2]."


def test_strip_think():
    assert gen.strip_think("<think>abc</think>answer") == "answer"
    assert gen.strip_think("<think>cut off").startswith("<think>")


# ---------------------------------------------------------------- retrieve.py
def test_rrf_order():
    fused = rt.rrf_fuse([["a", "b"], ["b", "c"]])
    assert [cid for cid, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    assert rt.tokenize("What does work_mem control?") == ["work_mem", "control"]


# ---------------------------------------------------------------- run_eval.py
def make_rec(page, anchor, heading):
    return {"chunk_id": f"{page}:0000", "url": f"https://www.postgresql.org/docs/17/{page}.html#{anchor}",
            "section_path": [heading], "context": heading, "text": "x",
            "covers": [{"anchor": anchor, "section_path": [heading]}]}


def test_norm_heading_and_page():
    assert ev.norm_heading("19.4.1. Memory") == "memory"
    assert ev.norm_heading("F.29.3. Author") == "author"
    assert ev.norm_heading("Synopsis") == "synopsis"
    assert ev.norm_page("https://www.postgresql.org/docs/17/sql-rollback.html#x") == "sql-rollback"
    assert ev.norm_page("sql-rollback") == "sql-rollback"


def test_chunk_keys_and_gold_key():
    keys = ev.chunk_keys(make_rec("sql-rollback", "SQL-ROLLBACK", "ROLLBACK"))
    assert ("sql-rollback", "SQL-ROLLBACK") in keys
    assert ("sql-rollback", "h:rollback") in keys
    assert ev.gold_key({"page": "sql-rollback", "heading": "Synopsis"}) == ("sql-rollback", "h:synopsis")
    assert ev.gold_key({"page": "p", "anchor": "A", "heading": "H"}) == ("p", "A")
    with pytest.raises(ValueError):
        ev.gold_key({"page": "p"})


def _hits():
    return [(make_rec("a", "A1", "One"), 0.9), (make_rec("x", "X1", "Other"), 0.8),
            (make_rec("b", "B1", "Two"), 0.7)]


def test_score_multi_hop_needs_all():
    q = {"id": "m", "type": "multi_hop",
         "gold": [{"page": "a", "anchor": "A1"}, {"page": "b", "anchor": "B1"}]}
    r = ev.score_question(q, _hits())
    assert r["gold_ranks"] == [1, 3] and r["first_rank"] == 1 and r["success_rank"] == 3
    q["gold"][1] = {"page": "zzz", "anchor": "Z"}
    assert ev.score_question(q, _hits())["success_rank"] is None


def test_score_factual_needs_any():
    q = {"id": "f", "type": "ambiguous",
         "gold": [{"page": "zzz", "anchor": "Z"}, {"page": "b", "anchor": "B1"}]}
    assert ev.score_question(q, _hits())["success_rank"] == 3


def test_summarize():
    rows = [{"success_rank": 1, "gold_ranks": [1], "distinct_sections_top5": 3},
            {"success_rank": None, "gold_ranks": [None], "distinct_sections_top5": 5}]
    s = ev.summarize(rows)
    assert s["success@1"] == 0.5 and s["mrr"] == 0.5


def test_coverage_check():
    chunks = [make_rec("a", "A1", "One")]
    ds = [{"id": "q1", "gold": [{"page": "a", "anchor": "A1"}]},
          {"id": "q2", "gold": [{"page": "nope", "heading": "X"}]}]
    missing = ev.coverage_check(ds, chunks)
    assert [m[0] for m in missing] == ["q2"]


# ---------------------------------------------------------------- check_gold.py
def test_check_gold(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text(json.dumps({"chunk_id": "sql-comment:0000",
                             "covers": [{"anchor": "C1", "section_path": ["COMMENT", "Compatibility"]}]}) + "\n")
    headings, anchors = cg.load_index(p)
    assert cg.check_gold({"page": "sql-comment", "heading": "compatibility"}, headings, anchors) is None
    assert cg.check_gold({"page": "sql-comment", "anchor": "C1"}, headings, anchors) is None
    assert "not on page" in cg.check_gold({"page": "sql-comment", "heading": "Nope"}, headings, anchors)
    assert "not found" in cg.check_gold({"page": "zzz", "heading": "x"}, headings, anchors)


# ---------------------------------------------------------------- tracing.py
def test_pct_and_last_per_key():
    assert tr.pct([1, 2, 3, 4, 5], 50) == 3
    recs = [{"qid": "a", "status": "llm_error"}, {"qid": "a", "status": "ok"}]
    assert tr.last_per_key(recs)["a"]["status"] == "ok"


def test_trace_summarize():
    recs = [{"status": "ok", "latency_ms": {"embed_ms": 10, "llm_ms": 100, "total_ms": 120},
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost_usd": 0.0}}]
    s = tr.summarize(recs)
    assert s["n"] == 1 and s["status"] == {"ok": 1}
    assert s["latency_ms"]["retrieval_ms"]["p50"] == 10
    assert s["tokens"] == {"prompt": 5, "completion": 2}
    