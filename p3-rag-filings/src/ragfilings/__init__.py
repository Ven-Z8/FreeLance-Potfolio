"""ragfilings — domain-agnostic Agentic Graph RAG engine + domain skill packs.

The engine (retrieval, grounded synthesis loop, verification, refusal gating)
is domain-agnostic; each domain supplies a skill pack under
`ragfilings.domains.<name>` — prompts, deterministic fact layer, scope agent
(rescue / clarification), claim semantics, and derivation tools. Shipped
pack: `financial` (SEC 10-K filings).

Evaluation (golden sets, scoring, calibrated judge, regression runs) lives in
the sibling p1-eval-harness project.
"""

from .agents import (
    audit_answer,
    corpus_inventory,
    plan_query,
    run_researcher,
    run_tool_loop,
    synthesize,
)
from .domains import DomainPack, available_packs, get_pack
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

__version__ = "0.5.0"

__all__ = [
    "ask",
    "answer",
    "GenerationError",
    "MultiAgentOrchestrator",
    "SessionMemoryManager",
    "DomainPack",
    "get_pack",
    "available_packs",
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
