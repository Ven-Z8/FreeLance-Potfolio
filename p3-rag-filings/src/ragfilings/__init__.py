"""ragfilings — Agentic Graph RAG over SEC 10-K filings, with a DeepEval-proven eval suite."""

from .agents import (
    audit_answer,
    corpus_inventory,
    plan_query,
    run_researcher,
    run_tool_loop,
    synthesize,
)
from .graph import FinancialGraphBuilder, GraphQueryEngine
from .ingestion import Section, parse_file, render_tree
from .llm import BaseLLMClient, LLMFactory, OpenRouterClient, get_llm_client
from .pipeline import GenerationError, MultiAgentOrchestrator, SessionMemoryManager, answer, ask
from .prompts import PromptRegistry, load_prompt
from .tools import (
    compute_financial_math,
    decompose_query,
    extract_claims,
    needs_decomposition,
    safe_eval,
    verify,
)

__version__ = "0.4.0"

__all__ = [
    "ask",
    "answer",
    "GenerationError",
    "MultiAgentOrchestrator",
    "SessionMemoryManager",
    "FinancialGraphBuilder",
    "GraphQueryEngine",
    "Section",
    "parse_file",
    "render_tree",
    "PromptRegistry",
    "load_prompt",
    "BaseLLMClient",
    "OpenRouterClient",
    "LLMFactory",
    "get_llm_client",
    "safe_eval",
    "compute_financial_math",
    "needs_decomposition",
    "decompose_query",
    "extract_claims",
    "verify",
    "plan_query",
    "corpus_inventory",
    "run_researcher",
    "synthesize",
    "audit_answer",
    "run_tool_loop",
]
