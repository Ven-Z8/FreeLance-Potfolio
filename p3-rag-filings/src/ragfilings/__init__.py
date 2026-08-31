"""ragfilings — Enterprise Agentic Multimodal Graph RAG Platform."""

from .agents import (
    run_auditor,
    run_data_analyst,
    run_document_analyst,
    run_lead_orchestrator,
    run_researcher,
    run_synthesis_expert,
)
from .graph import FinancialGraphBuilder, GraphQueryEngine
from .ingestion import DoclingParser, Section, parse_file, render_tree
from .llm import BaseLLMClient, LLMFactory, OpenRouterClient, get_llm_client
from .pipeline import GenerationError, MultiAgentOrchestrator, SessionMemoryManager, SoloMetaOrchestrator, answer, ask
from .prompts import PromptRegistry, load_prompt
from .tools import compute_financial_math, decompose_query, extract_claims, safe_eval, verify

__version__ = "0.3.0"

__all__ = [
    "ask",
    "answer",
    "GenerationError",
    "MultiAgentOrchestrator",
    "SoloMetaOrchestrator",
    "SessionMemoryManager",
    "FinancialGraphBuilder",
    "GraphQueryEngine",
    "DoclingParser",
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
    "run_lead_orchestrator",
    "run_document_analyst",
    "run_researcher",
    "run_data_analyst",
    "run_synthesis_expert",
    "run_auditor",
]
