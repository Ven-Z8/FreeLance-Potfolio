"""Unit tests for Core Eval Harness Engine."""

import pytest
from harness import schema, report
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


def test_deterministic_metric_exact():
    case = schema.GoldenCase(
        id="test-002",
        input="What is revenue?",
        expected=schema.ExpectedOutcome(answer="$100M", citations=["c1"], type="exact"),
        variation_rules=["numeric_tolerance:1%"],
        failure_category="lookup",
        domain="financial"
    )
    trace = schema.AgentRunTrace(
        case_id="test-002",
        domain="financial",
        strategy="hybrid_rerank",
        query="What is revenue?",
        answer="Revenue is $100M",
        citations=["c1"]
    )
    correct, outcome, metrics = engine.Tier1DeterministicMetrics.evaluate(case, trace)
    assert correct is True
    assert outcome == "correct_answer"
    assert metrics["answer_accuracy"] == 1.0


def test_deterministic_metric_refusal():
    case = schema.GoldenCase(
        id="test-003",
        input="What is revenue in 2030?",
        expected=schema.ExpectedOutcome(answer=None, citations=[], type="exact"),
        failure_category="unanswerable",
        domain="financial"
    )
    trace = schema.AgentRunTrace(
        case_id="test-003",
        domain="financial",
        strategy="hybrid_rerank",
        query="What is revenue in 2030?",
        answer=None,
        refused=True,
        refusal_reason="unanswerable"
    )
    correct, outcome, metrics = engine.Tier1DeterministicMetrics.evaluate(case, trace)
    assert correct is True
    assert outcome == "correct_refusal"
    assert metrics["refusal_correctness"] == 1.0


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
