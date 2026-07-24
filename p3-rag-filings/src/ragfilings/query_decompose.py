"""Query Decomposition Engine.

NVIDIA RAG Blueprint Pattern:
  - Detects complex, multi-year, or multi-step calculation queries.
  - Decomposes a single complex question into targeted sub-queries to retrieve
    exact data points across separate financial filing sections/years.
"""

from __future__ import annotations

import json
from typing import Any

from . import generation


_MATH_KEYWORDS = (
    "growth rate", "cagr", "average", "percentage change", "increased or decreased",
    "compare", "difference", "margin delta", "ratio", "3-year", "multi-year"
)


def needs_decomposition(query: str) -> bool:
    """Return True if the query asks for multi-year math or comparative synthesis."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _MATH_KEYWORDS)


def decompose_query(query: str, cfg: dict) -> list[str]:
    """Decompose a complex financial question into 2-3 focused retrieval sub-queries."""
    if not needs_decomposition(query):
        return [query]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial query decomposition module. Given a complex financial question, "
                "output ONLY a JSON array of 2-3 specific, single-point retrieval sub-queries targeting "
                "SEC 10-K Item 7/Item 8 tables. "
                "Format: [\"sub_query_1\", \"sub_query_2\"]"
            )
        },
        {"role": "user", "content": query}
    ]

    try:
        text, _ = generation._complete(messages, cfg)
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            sub_queries = json.loads(text[start:end+1])
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                return [query] + [str(sq) for sq in sub_queries if isinstance(sq, str)]
    except Exception:
        pass

    return [query]
