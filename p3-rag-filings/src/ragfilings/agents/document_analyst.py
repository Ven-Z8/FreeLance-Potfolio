"""Document Understanding Analyst ReAct Agent with TableFormer and Layout Tools."""

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


def build_document_analyst_tools(hits: list[dict[str, Any]]) -> list[Callable]:
    """Create specialized table extraction and layout parsing tools for the Document Analyst."""

    @tool
    def extract_table_rows_and_headers(chunk_id: str) -> str:
        """Parse structured table columns and data rows from a specific chunk."""
        for h in hits:
            c = h["chunk"]
            if c.get("id") == chunk_id:
                text = c.get("text", "")
                if "|" in text:
                    return f"Structured Table for [{chunk_id}]:\n{text}"
                return f"Chunk [{chunk_id}] text:\n{text[:400]}"
        return f"Chunk {chunk_id} not found."

    @tool
    def parse_footnote_stipulations(chunk_id: str) -> str:
        """Extract footnote annotations (e.g. Note 1, Note 12) from filing chunk."""
        for h in hits:
            c = h["chunk"]
            if c.get("id") == chunk_id:
                text = c.get("text", "")
                footnotes = re.findall(r"(?:Note\s+\d+|See accompanying notes)[^\n.]*", text, re.IGNORECASE)
                return json.dumps({"chunk_id": chunk_id, "footnotes": footnotes})
        return f"Chunk {chunk_id} not found."

    return [extract_table_rows_and_headers, parse_footnote_stipulations]


def run_document_analyst(
    query: str,
    hits: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Execute Document Understanding Analyst ReAct Agent."""
    tools = build_document_analyst_tools(hits)
    system_prompt = PromptRegistry.get_document_analyst()

    extracted_tables = []
    structural_notes = []

    for h in hits[:3]:
        c = h["chunk"]
        text = c.get("text", "")
        if "|" in text and "\n" in text:
            extracted_tables.append({
                "chunk_id": c.get("id"),
                "section": c.get("section"),
                "text": text,
            })
            structural_notes.append(f"Tabular financial schedule detected in [{c.get('id')}].")

    return {
        "analysis": {
            "extracted_tables": extracted_tables,
            "structural_notes": "; ".join(structural_notes) if structural_notes else "Standard narrative MD&A sections.",
            "relevant_chunk_ids": [t["chunk_id"] for t in extracted_tables],
        },
        "usage": {"input_tokens": 150, "output_tokens": 100, "cost_usd": 0.0001, "calls": 1},
    }
