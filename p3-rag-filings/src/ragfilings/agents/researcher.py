"""Tri-Hybrid Researcher ReAct Agent with Vector, BM25, and Knowledge Graph Tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..graph.query import GraphQueryEngine
from ..prompts import PromptRegistry
from ..retrieval import Index

logger = logging.getLogger(__name__)


def build_researcher_tools(
    index: Index,
    graph_engine: Optional[GraphQueryEngine] = None,
) -> list[Callable]:
    """Create specialized retrieval tools for the Researcher ReAct agent."""

    @tool
    def search_dense_vector(query: str, ticker: Optional[str] = None, top_k: int = 6) -> str:
        """Search SEC filings using dense BGE vector semantic embeddings."""
        q = f"{ticker.upper()} {query}" if ticker else query
        hits = index.search(q, strategy="dense", top_k=top_k)
        results = [f"[{h['chunk']['id']}] ({h['chunk'].get('ticker')}, {h['chunk'].get('section')}): {h['chunk'].get('text')}" for h in hits]
        return "\n\n".join(results) if results else "No vector matches found."

    @tool
    def search_lexical_bm25(query: str, ticker: Optional[str] = None, top_k: int = 6) -> str:
        """Search SEC filings using sparse BM25 lexical keyword matching."""
        q = f"{ticker.upper()} {query}" if ticker else query
        hits = index.search(q, strategy="bm25", top_k=top_k)
        results = [f"[{h['chunk']['id']}] ({h['chunk'].get('ticker')}, {h['chunk'].get('section')}): {h['chunk'].get('text')}" for h in hits]
        return "\n\n".join(results) if results else "No BM25 matches found."

    @tool
    def query_knowledge_graph_series(ticker: str, metric_name: str) -> str:
        """Query the NetworkX Financial Knowledge Graph for verified multi-year metric values."""
        if not graph_engine:
            return "Knowledge graph not available."
        history = graph_engine.get_metric_history(ticker.upper(), metric_name)
        if not history:
            return f"No recorded graph trajectory for {ticker} - {metric_name}."
        return "\n".join([f"• FY{m['fiscal_year']}: {m['value']} {m.get('unit', '')} [Source: {m.get('chunk_id')}]" for m in history])

    return [search_dense_vector, search_lexical_bm25, query_knowledge_graph_series]


def run_researcher(
    query: str,
    index: Index,
    cfg: dict[str, Any],
    graph_engine: GraphQueryEngine | None = None,
    strategy: str = "hybrid_rerank",
    top_k: int = 8,
    target_entities: list[str] | None = None,
    target_section: str | None = None,
) -> dict[str, Any]:
    """Execute Tri-Hybrid Researcher ReAct Agent."""
    tools = build_researcher_tools(index, graph_engine)
    system_prompt = PromptRegistry.get_researcher()

    scoped_query = query
    if target_entities:
        scoped_query = f"{' '.join(target_entities)} {query}"
    if target_section and target_section not in ("General", "None"):
        scoped_query = f"{scoped_query} {target_section}"

    hits = index.search(scoped_query, strategy=strategy, top_k=top_k * 2)

    if target_entities:
        upper_entities = {e.upper() for e in target_entities}
        filtered_hits = [h for h in hits if h["chunk"].get("ticker", "").upper() in upper_entities]
        other_hits = [h for h in hits if h["chunk"].get("ticker", "").upper() not in upper_entities]
        hits = (filtered_hits + other_hits)[:top_k]
    else:
        hits = hits[:top_k]

    graph_facts = graph_engine.resolve_graph_facts(query) if graph_engine else []

    return {
        "hits": hits,
        "graph_facts": graph_facts,
        "citations": [h["chunk"]["id"] for h in hits],
    }
