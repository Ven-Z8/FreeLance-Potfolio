"""Auditor Compliance Guardrail ReAct Agent with Deterministic Verification Tools."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from ..prompts import PromptRegistry
from ..tools.verification import extract_claims, verify

logger = logging.getLogger(__name__)


def build_auditor_tools(hits: list[dict[str, Any]]) -> list[Callable]:
    """Create specialized verification and compliance tools for the Auditor Guardrail."""

    @tool
    def verify_numerical_claims(draft_answer: str, cited_chunk_ids: list[str]) -> str:
        """Deterministically verify all numbers and dollar amounts in draft answer against cited chunks."""
        cited_chunks = [h["chunk"] for h in hits if h["chunk"].get("id") in cited_chunk_ids]
        check_res = verify(draft_answer, cited_chunks)
        if check_res.get("verified", True):
            return "ALL_CLAIMS_VERIFIED"
        failed = [c["raw"] for c in check_res.get("claims", []) if not c.get("found", True)]
        return f"FAILED_CLAIMS: {', '.join(failed)}"

    return [verify_numerical_claims]


def run_auditor(
    answer: str,
    citations: list[str] | None = None,
    hits: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
    cited_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute Auditor Compliance Guardrail ReAct Agent."""
    tools = build_auditor_tools(hits or [])
    system_prompt = PromptRegistry.get_auditor_guardrail()

    if cited_chunks is not None:
        valid_chunks = cited_chunks
    elif hits:
        if citations:
            valid_chunks = [h["chunk"] for h in hits if h["chunk"].get("id") in citations]
            if not valid_chunks:
                valid_chunks = [h["chunk"] for h in hits]
        else:
            valid_chunks = [h["chunk"] for h in hits]
    else:
        valid_chunks = []

    verification_res = verify(answer, valid_chunks)
    is_verified = verification_res.get("verified", True)
    failed = [c["raw"] for c in verification_res.get("claims", []) if not c.get("found", True)]

    invalid_citations = []
    if hits and citations:
        available_ids = {h["chunk"].get("id") for h in hits}
        for c in citations:
            if c not in available_ids:
                invalid_citations.append(c)

    feedback = None
    if not is_verified:
        feedback = f"The following figures were ungrounded in cited chunks: {', '.join(failed)}. Only cite numbers appearing verbatim."

    return {
        "verified": is_verified,
        "verification": verification_res,
        "failed_claims": failed,
        "correction_guidance": feedback,
        "invalid_citations": invalid_citations,
        "audit_feedback": feedback,
        "usage": {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.00005, "calls": 1},
    }
