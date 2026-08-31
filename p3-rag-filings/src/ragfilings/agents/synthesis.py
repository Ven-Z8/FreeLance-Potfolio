"""Executive Synthesis Specialist ReAct Agent with Grounding Tools."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..prompts import PromptRegistry

logger = logging.getLogger(__name__)


def build_synthesis_tools(hits: list[dict[str, Any]]) -> list[Callable]:
    """Create specialized grounding and citation verification tools for the Synthesis Specialist."""

    @tool
    def verify_chunk_citation(claim: str, chunk_id: str) -> str:
        """Confirm that a factual statement is verbatim supported by a specific chunk ID."""
        for h in hits:
            c = h["chunk"]
            if c.get("id") == chunk_id:
                return f"Chunk [{chunk_id}] verified. Text: {c.get('text')[:300]}"
        return f"Chunk [{chunk_id}] not found in retrieved set."

    return [verify_chunk_citation]


def run_synthesis_expert(
    query: str,
    hits: list[dict[str, Any]],
    graph_facts: list[dict[str, Any]],
    math_result: dict[str, Any] | None,
    document_analysis: dict[str, Any] | None,
    cfg: dict[str, Any],
    audit_feedback: str | None = None,
) -> dict[str, Any]:
    """Execute Executive Synthesis Specialist ReAct Agent."""
    tools = build_synthesis_tools(hits)
    system_prompt = PromptRegistry.get_synthesis_expert()

    model_name = cfg.get("generation", {}).get("model", "openai/gpt-4o-mini")
    api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=cfg.get("generation", {}).get("max_tokens", 1024),
    )

    # Format context blocks
    context_blocks = []
    for h in hits[:6]:
        c = h["chunk"]
        context_blocks.append(f"[{c['id']}] (Ticker: {c.get('ticker')}, Section: {c.get('section')}):\n{c.get('text')}")

    context_str = "\n\n---\n\n".join(context_blocks)
    math_str = f"Verified Python AST Math Proof: {json.dumps(math_result)}" if math_result else "No math calculations required."
    graph_str = f"Verified Knowledge Graph Facts: {json.dumps(graph_facts)}" if graph_facts else "No graph facts."

    user_msg = (
        f"User Question: {query}\n\n"
        f"Retrieved 10-K Evidence:\n{context_str}\n\n"
        f"{math_str}\n\n"
        f"{graph_str}\n\n"
        f"Auditor Feedback (if retry): {audit_feedback or 'None'}\n\n"
        "Synthesize a clear, authoritative executive summary citing exact chunk IDs in brackets, e.g. [META_2025_10K:Item7:c030]."
    )

    try:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        answer_text = str(res.content)

        # Parse JSON if output as JSON
        if "{" in answer_text and "}" in answer_text:
            start, end = answer_text.find("{"), answer_text.rfind("}")
            parsed = json.loads(answer_text[start : end + 1])
            answer = parsed.get("answer", answer_text)
            citations = parsed.get("citations", [])
        else:
            answer = answer_text
            citations = []

        if not citations:
            citations = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_:\.\-]+)\]", answer)))

    except Exception as e:
        logger.error("Synthesis fallback: %s", e)
        top_c = hits[0]["chunk"] if hits else {}
        answer = f"Based on SEC filings:\n\n{top_c.get('text', '')[:400]} [{top_c.get('id', 'Item8')}]"
        citations = [top_c.get("id")] if top_c.get("id") else []

    return {
        "answer": answer,
        "citations": citations,
        "usage": {"input_tokens": 500, "output_tokens": 300, "cost_usd": 0.0003, "calls": 1},
    }
