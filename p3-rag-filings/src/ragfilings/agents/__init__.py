"""Agent package — planner, researcher, synthesis, auditor.

Each agent is a real LLM step: instructor-validated structured outputs or a
native tool-calling loop, all with API-reported usage accounting.
"""

from .auditor import audit_answer
from .planner import corpus_inventory, plan_query
from .researcher import run_researcher
from .synthesis import synthesize
from .tool_loop import add_usage, run_tool_loop

__all__ = [
    "plan_query",
    "corpus_inventory",
    "run_researcher",
    "synthesize",
    "audit_answer",
    "run_tool_loop",
    "add_usage",
]
