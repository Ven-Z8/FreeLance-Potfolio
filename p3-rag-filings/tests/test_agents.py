"""Tests for Specialized 6-Agent Swarm."""

from ragfilings.agents import (
    run_auditor,
    run_data_analyst,
    run_document_analyst,
    run_lead_orchestrator,
    run_researcher,
    run_synthesis_expert,
)

CFG = {
    "generation": {"model": "test/model", "max_tokens": 512, "verify_retries": 1},
    "verification": {"min_confidence": 0.35},
}

CHUNK = {
    "id": "AAPL_2025_10K:Item8:c007",
    "text": "Total net sales | $416,161 | $391,035\nGross margin percentage | 46.9% | 46.2%",
}


def test_auditor_verifies_correct_claim():
    res = run_auditor(
        answer="Net sales were $416,161 million.",
        cited_chunks=[CHUNK],
    )
    assert res["verified"]
    assert not res["failed_claims"]


def test_auditor_catches_hallucinated_claim():
    res = run_auditor(
        answer="Net sales were $999,999 million.",
        cited_chunks=[CHUNK],
    )
    assert not res["verified"]
    assert "$999,999 million" in res["failed_claims"]
    assert res["correction_guidance"] is not None


def test_data_analyst_ignores_simple_lookup():
    res = run_data_analyst(
        query="What is the CEO name?",
        chunks=[CHUNK],
        cfg=CFG,
    )
    assert res is None
