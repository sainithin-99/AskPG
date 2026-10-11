"""More pure-function tests (no model, no Qdrant, no network). Expected values were worked out by hand and
NOT run when written: if one fails, decide whether my arithmetic or the code is wrong before touching either."""
import math

import generate as gen
import judge as jg
import run_eval as ev
import snippet_check as sc
import tracing as tr


# ---------------------------------------------------------------- snippet_check.py
def test_norm_unifies_quotes_and_whitespace():
    assert sc.norm("Hello  World") == "helloworld"
    assert sc.norm("\u2018a\u2019 \u2013 b") == "'a'-b"


def test_segments():
    assert sc.segments("SELECT * FROM t; -- comment") == ["select*fromt"]
    assert sc.segments("END;") == []          # under 8 characters is ignored
    assert sc.segments("ssl = on") == []      # known gap: short config lines are never checked


def test_extract_snippets():
    answer = "Text\n```sql\nSELECT 1;\n```\nand `GRANT SELECT ON t TO u;` and `work_mem`"
    assert sc.extract_snippets(answer) == [("fence", "SELECT 1;"), ("inline", "GRANT SELECT ON t TO u;")]


def test_check_snippet_verdicts():
    text = "GRANT SELECT ON phone_number TO assistant;"
    hay = {"c1": sc.norm(text)}
    assert sc.check_snippet("inline", text, hay, hay)["verdict"] == "supported"
    assert sc.check_snippet("inline", text, hay, {})["verdict"] == "supported (only uncited)"
    two = "line one is here\nsomething else entirely"
    part = {"c1": sc.norm("line one is here")}
    assert sc.check_snippet("fence", two, part, part)["verdict"] == "PARTIAL"
    assert sc.check_snippet("fence", two, {"c1": "zzz"}, {})["verdict"] == "UNSUPPORTED"
    assert sc.check_snippet("fence", two, {"c1": "zzz"}, {}, n_missing=1)["verdict"].startswith("UNKNOWN")
    assert sc.check_snippet("fence", "END;", hay, hay)["verdict"] == "trivial"


# ---------------------------------------------------------------- judge.py
def test_parse_json():
    assert jg.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert jg.parse_json('prefix {"a": 1} suffix') == {"a": 1}
    assert jg.parse_json("not json") is None
    assert jg.parse_json("[1, 2]") is None


def test_norm_claims():
    obj = {"claims": [
        {"claim": "x", "cited": [1, "a"], "verdict": "SUPPORTED", "found_in_other_source": True, "evidence": "e"},
        {"claim": "y", "verdict": "weird"},
        {"nope": 1},
    ]}
    out = jg.norm_claims(obj)
    assert len(out) == 2
    assert out[0]["cited"] == [1] and out[0]["verdict"] == "supported" and out[0]["found_in_other_source"]
    assert out[1]["verdict"] == "invalid" and out[1]["cited"] == []


def test_judge_summarize():
    usage = {"prompt_tokens": 1, "completion_tokens": 1}
    recs = [
        {"id": "a", "type": "factual", "error": None, "correctness": "correct", "completeness": "complete",
         "usage": usage, "claims": [{"verdict": "supported", "found_in_other_source": False},
                                    {"verdict": "not_supported", "found_in_other_source": False}]},
        {"id": "b", "type": "factual", "error": "boom", "correctness": None, "completeness": None,
         "usage": usage, "claims": []},
    ]
    s = jg.summarize(recs)
    assert s["answers_judged"] == 1 and s["judge_errors"] == 1
    assert s["claims"] == 2 and s["claim_supported_rate"] == 0.5
    assert s["answers_with_not_supported_claim"] == 1 and s["answers_fully_supported"] == 0


# ---------------------------------------------------------------- generate.py
def test_build_messages():
    msgs = gen.build_messages("q?", [({"context": "A > B", "text": "body"}, 0.5)])
    assert msgs[0]["role"] == "system" and gen.REFUSAL in msgs[0]["content"]
    assert msgs[1]["content"].startswith("Sources:\n\n[1] A > B\nbody")
    assert msgs[1]["content"].endswith("Question: q?")


# ---------------------------------------------------------------- tracing.py / run_eval.py
def test_trace_round_trip(tmp_path):
    p = tmp_path / "t.jsonl"
    rec = tr.log_trace(p, {"status": "ok"}, qid="q1")
    back = tr.read_traces(p)
    assert back == [rec] and back[0]["qid"] == "q1" and "ts" in back[0]
    assert tr.read_traces(tmp_path / "missing.jsonl") == []


def test_pct_empty_is_nan():
    assert math.isnan(ev.pct([], 50))
