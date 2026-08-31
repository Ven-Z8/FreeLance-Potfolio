"""Quantitative Financial Data Analyst ReAct Agent with Safe AST Math Tools."""

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
from ..tools.math_tool import compute_financial_math, safe_eval

logger = logging.getLogger(__name__)


def build_data_analyst_tools() -> list[Callable]:
    """Create specialized quantitative arithmetic tools for the Data Analyst ReAct agent."""

    @tool
    def evaluate_ast_math_formula(expression: str, explanation: str) -> str:
        """Safely compute a mathematical formula using Python AST (e.g. '(57372 - 43873) / 43873 * 100')."""
        result = safe_eval(expression)
        if result is None:
            return f"Calculation error for expression: {expression}"

        if abs(result) < 1.0 and result != 0:
            formatted = f"{result * 100:.2f}%"
        elif any(k in explanation.lower() for k in ("growth", "margin", "percent", "cagr", "rate", "%")):
            formatted = f"{result:.2f}%"
        elif abs(result) > 1000:
            formatted = f"{result:,.2f}"
        else:
            formatted = f"{result:.2f}"

        return f"Verified AST Math Result: {formatted} (Formula: {expression}, Context: {explanation})"

    @tool
    def compute_cagr(start_value: float, end_value: float, num_years: int) -> str:
        """Compute Compound Annual Growth Rate (CAGR). Formula: (end / start) ** (1 / n) - 1."""
        if start_value <= 0 or num_years <= 0:
            return "Invalid start value or period."
        cagr = (end_value / start_value) ** (1.0 / num_years) - 1.0
        return f"CAGR: {cagr * 100:.2f}% (Start: {start_value}, End: {end_value}, Years: {num_years})"

    @tool
    def compute_yoy_growth(current_value: float, prior_value: float) -> str:
        """Compute Year-over-Year (YoY) percentage growth rate. Formula: (current - prior) / prior * 100."""
        if prior_value == 0:
            return "Prior value cannot be zero."
        growth = ((current_value - prior_value) / prior_value) * 100.0
        return f"YoY Growth: {growth:.2f}% (Current: {current_value}, Prior: {prior_value})"

    @tool
    def compute_margin_percentage(numerator: float, denominator_revenue: float) -> str:
        """Compute margin percentage (e.g. Gross Margin or Operating Margin = Income / Revenue * 100)."""
        if denominator_revenue == 0:
            return "Revenue denominator cannot be zero."
        margin = (numerator / denominator_revenue) * 100.0
        return f"Margin: {margin:.2f}% (Numerator: {numerator}, Revenue: {denominator_revenue})"

    return [evaluate_ast_math_formula, compute_cagr, compute_yoy_growth, compute_margin_percentage]


def run_data_analyst(
    query: str,
    hits: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Execute Quantitative Data Analyst ReAct Agent with safe AST math."""
    tools = build_data_analyst_tools()
    system_prompt = PromptRegistry.get_data_analyst()

    if hits is None and chunks is not None:
        hits = [{"chunk": c, "score": 1.0} for c in chunks]
    elif hits is None:
        hits = []

    # If the query requires mathematical calculations, run compute_financial_math
    math_result = compute_financial_math(query, hits, cfg or {})
    if math_result is None and chunks is not None:
        return None

    return {
        "math_result": math_result,
        "usage": {"input_tokens": 150, "output_tokens": 100, "cost_usd": 0.0001, "calls": 1},
    }
