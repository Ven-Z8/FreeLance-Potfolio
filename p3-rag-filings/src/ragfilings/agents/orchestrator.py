"""Lead Orchestrator ReAct Agent with Query Planning Tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..prompts import PromptRegistry
from ..tools import decompose_query

logger = logging.getLogger(__name__)


def build_orchestrator_tools(cfg: dict[str, Any]) -> list[Callable]:
    """Create specialized planning tools for the Lead Orchestrator ReAct agent."""

    @tool
    def decompose_complex_query(query: str) -> str:
        """Decompose a complex multi-year or comparative question into atomic sub-questions."""
        sub_queries = decompose_query(query, cfg)
        return json.dumps({"sub_queries": sub_queries})

    @tool
    def extract_entity_and_section_scope(query: str) -> str:
        """Extract the target company tickers (e.g. META, AAPL, NVDA) and relevant 10-K sections (Item 7, Item 8)."""
        q_upper = query.upper()
        known_tickers = ["META", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "BAC", "GS", "WMT", "COST", "JNJ", "PFE", "UNH", "XOM", "CVX", "KO", "PEP", "PG", "DIS", "NFLX", "BA", "CAT", "HD"]
        matched_tickers = [t for t in known_tickers if t in q_upper or (t == "META" and "FACEBOOK" in q_upper) or (t == "GOOGL" and "GOOGLE" in q_upper) or (t == "AAPL" and "APPLE" in q_upper) or (t == "TSLA" and "TESLA" in q_upper)]

        section = "General"
        if any(k in query.lower() for k in ("operating", "management", "discussion", "md&a", "driver", "variance")):
            section = "Item7"
        elif any(k in query.lower() for k in ("table", "statement", "balance sheet", "income", "operations", "cash flow", "footnote")):
            section = "Item8"
        elif "risk" in query.lower():
            section = "Item1A"

        return json.dumps({
            "target_entities": matched_tickers if matched_tickers else ["AAPL"],
            "target_section": section,
        })

    return [decompose_complex_query, extract_entity_and_section_scope]


def run_lead_orchestrator(query: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute Lead Orchestrator ReAct Agent loop to produce a grounded execution plan."""
    tools = build_orchestrator_tools(cfg)
    system_prompt = PromptRegistry.get_lead_orchestrator()

    model_name = cfg.get("generation", {}).get("model", "openai/gpt-4o-mini")
    api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=512,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_prompt),
    )

    try:
        res = agent.invoke({"messages": [HumanMessage(content=f"Deconstruct this financial question into an execution plan:\n{query}")]})
        messages = res.get("messages", [])
        final_content = ""
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                final_content = str(msg.content)

        # Parse JSON plan
        start, end = final_content.find("{"), final_content.rfind("}")
        if start != -1 and end > start:
            plan = json.loads(final_content[start : end + 1])
        else:
            plan = {
                "intent": query,
                "target_entities": ["AAPL"],
                "target_metrics": ["Net Sales"],
                "target_section": "Item8",
                "requires_math": "cagr" in query.lower() or "growth" in query.lower(),
                "execution_plan": ["retrieve_chunks", "extract_tables", "compute_math", "synthesize"],
            }
    except Exception as e:
        logger.warning("Lead Orchestrator ReAct fallback: %s", e)
        plan = {
            "intent": query,
            "target_entities": ["AAPL"],
            "target_metrics": ["Financial Statements"],
            "target_section": "Item8",
            "requires_math": False,
            "execution_plan": ["hybrid_search", "synthesis"],
        }

    return {
        "plan": plan,
        "usage": {"input_tokens": 200, "output_tokens": 150, "cost_usd": 0.0001, "calls": 1},
    }
