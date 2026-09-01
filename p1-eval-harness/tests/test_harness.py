"""Unit tests for the harness schema, report generation, and summary glue."""

from harness import report, schema
from harness.metrics import engine


def test_schema_models():
    case = schema.GoldenCase(
        id="test-001",
        input="What is Apple's revenue?",
        expected=schema.ExpectedOutcome(answer="$416,161M", citations=["c1"], type="exact"),
        variation_rules=["numeric_tolerance:1%"],
        failure_category="table",
        domain="financial"
    )
    assert case.id == "test-001"
    assert case.expected.answer == "$416,161M"


def test_score_case_accepts_schema_dump():
    """GoldenCase dumps score directly through the engine."""
    case = schema.GoldenCase(
        id="test-002",
        input="What is revenue?",
        expected=schema.ExpectedOutcome(answer="$100 million", citations=["c1"], type="exact"),
        variation_rules=["numeric_tolerance:1%"],
        failure_category="lookup",
        domain="financial",
    ).model_dump()
    result = {"answer": "Revenue was $100 million.", "refused": False,
              "citations": ["c1"], "hits": []}
    scored = engine.score_case(case, result)
    assert scored["correct"] is True and scored["outcome"] == "answered"


def test_score_case_refusal_outcomes():
    unanswerable = schema.GoldenCase(
        id="test-003",
        input="What is revenue in 2030?",
        expected=schema.ExpectedOutcome(answer=None, citations=[], type="exact"),
        failure_category="unanswerable",
        domain="financial",
    ).model_dump()
    refused = {"answer": None, "refused": True, "refusal_reason": "not in corpus",
               "citations": [], "hits": []}
    scored = engine.score_case(unanswerable, refused)
    assert scored["correct"] is True and scored["outcome"] == "correct_refusal"


def test_summary_from_rows_shape():
    rows = [
        {"case_id": "a", "correct": True, "outcome": "answered",
         "input": "q1", "latency_ms": 100.0},
        {"case_id": "b", "correct": False, "outcome": "hallucination",
         "input": "q2", "latency_ms": 200.0},
    ]
    s = report.summary_from_rows("hybrid_rerank", rows)
    assert s["strategy"] == "hybrid_rerank"
    assert s["n"] == 2 and s["correct_count"] == 1
    assert s["accuracy"] == 0.5
    assert s["results"][0]["trace"]["query"] == "q1"


def test_report_generation(tmp_path):
    summary = {
        "strategy": "hybrid_rerank",
        "n": 2,
        "accuracy": 1.0,
        "correct_count": 2,
        "results": [
            {
                "case_id": "test-001",
                "correct": True,
                "outcome": "correct_answer",
                "trace": {"query": "What is revenue?", "latency_ms": 1500.0}
            }
        ]
    }
    html_p, md_p = report.generate_reports(summary, tmp_path)
    assert html_p.exists()
    assert md_p.exists()
