"""Query Transformation & HyDE (Hypothetical Document Expansion) for RAG.

NVIDIA RAG Blueprint Pattern:
  - Generates hypothetical SEC 10-K response passages for complex queries (HyDE)
    so vector embedding matches filing language structure rather than raw question phrasing.
  - Expands queries into multi-query variations to catch table headers and footnote terms.
"""

from __future__ import annotations

import json
from typing import Any

from . import generation


def generate_hyde_passage(query: str, cfg: dict) -> str:
    """Generate a hypothetical 10-K excerpt that answers the question.
    
    The vector embedding of this hypothetical passage is then used for retrieval.
    """
    messages = [
        {"role": "system", "content": "You are a financial filing expert. Generate a hypothetical 10-K passage (2-3 sentences) with realistic tables/text that answers the following financial question. Use formal SEC 10-K financial terminology."},
        {"role": "user", "content": query}
    ]
    try:
        passage, _ = generation._complete(messages, cfg)
        return passage if passage else query
    except Exception:
        return query


def expand_query(query: str, cfg: dict) -> list[str]:
    """Generate 2-3 alternative search queries for multi-query retrieval."""
    expanded = [query]
    q_lower = query.lower()
    headline_terms = ["operating income", "net sales", "total revenue", "gross margin", "net interest income", "research and development", "r&d"]
    if any(term in q_lower for term in headline_terms):
        expanded.append(f"{query} Item 8 Consolidated Statements of Operations")
        expanded.append(f"{query} Consolidated Financial Statements Item 8")

    messages = [
        {"role": "system", "content": "Output a JSON array of 2 search query variations targeting SEC 10-K filings for the user prompt. Format: [\"query1\", \"query2\"]"},
        {"role": "user", "content": query}
    ]
    try:
        text, _ = generation._complete(messages, cfg)
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            variations = json.loads(text[start:end+1])
            if isinstance(variations, list) and len(variations) > 0:
                expanded.extend(str(v) for v in variations if isinstance(v, str))
    except Exception:
        pass
    return expanded

