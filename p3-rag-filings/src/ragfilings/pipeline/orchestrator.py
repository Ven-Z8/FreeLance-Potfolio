"""Production 6-Agent LangGraph Swarm with Individual ReAct Agents & SQLite Memory."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import (
    run_auditor,
    run_data_analyst,
    run_document_analyst,
    run_lead_orchestrator,
    run_researcher,
    run_synthesis_expert,
)
from ..graph.builder import FinancialGraphBuilder
from ..graph.query import GraphQueryEngine
from ..retrieval import Index, confidence
from .memory import SessionMemoryManager

logger = logging.getLogger(__name__)


class LangGraphState(TypedDict):
    session_id: str
    query: str
    cfg: Dict[str, Any]
    index: Any
    graph_engine: Optional[Any]
    strategy: str
    plan: Dict[str, Any]
    hits: List[dict]
    graph_facts: List[dict]
    document_analysis: Dict[str, Any]
    math_result: Optional[dict]
    answer: Optional[str]
    citations: List[str]
    invalid_citations: List[str]
    verified: bool
    verification: Dict[str, Any]
    audit_feedback: Optional[str]
    retry_count: int
    usage: Dict[str, Any]
    latency_ms: float
    refused: bool
    refusal_reason: Optional[str]


def _build_graph_engine_if_needed(cfg: dict[str, Any], index: Index | None) -> GraphQueryEngine | None:
    """Build or load in-memory NetworkX knowledge graph."""
    try:
        builder = FinancialGraphBuilder.load("corpus/graph/financial_graph.json")
        if builder.graph.number_of_nodes() == 0 and index and hasattr(index, "chunks"):
            builder.build_from_chunks(index.chunks)
            builder.save("corpus/graph/financial_graph.json")
        return GraphQueryEngine(builder=builder)
    except Exception:
        return None


def build_langgraph_workflow() -> StateGraph:
    memory = SessionMemoryManager()

    def lead_orchestrator_node(state: LangGraphState) -> dict:
        orch_res = run_lead_orchestrator(state["query"], state["cfg"])
        plan = orch_res.get("plan", {})
        u = orch_res.get("usage", {})

        memory.log_step(
            state["session_id"],
            1,
            "LeadOrchestrator",
            "plan_workflow",
            plan,
        )

        return {
            "plan": plan,
            "usage": {
                "input_tokens": state["usage"]["input_tokens"] + u.get("input_tokens", 0),
                "output_tokens": state["usage"]["output_tokens"] + u.get("output_tokens", 0),
                "cost_usd": state["usage"]["cost_usd"] + u.get("cost_usd", 0.0),
                "calls": state["usage"]["calls"] + 1,
            },
        }

    def researcher_node(state: LangGraphState) -> dict:
        plan = state.get("plan", {})
        res = run_researcher(
            query=state["query"],
            index=state["index"],
            cfg=state["cfg"],
            graph_engine=state.get("graph_engine"),
            strategy=state.get("strategy", "hybrid_rerank"),
            top_k=state["cfg"].get("retrieval", {}).get("top_k", 8),
            target_entities=plan.get("target_entities"),
            target_section=plan.get("target_section"),
        )

        hits = res.get("hits", [])
        conf = confidence(hits)
        min_conf = state["cfg"].get("verification", {}).get("min_confidence", 0.35)

        refused = False
        refusal_reason = None
        if conf < min_conf:
            refused = True
            refusal_reason = f"low retrieval confidence: {conf:.3f} < {min_conf}"

        memory.log_step(
            state["session_id"],
            2,
            "Researcher",
            "retrieve_hybrid_and_graph",
            {"n_hits": len(hits), "confidence": conf, "graph_facts": len(res.get("graph_facts", []))},
        )

        return {
            "hits": hits,
            "graph_facts": res.get("graph_facts", []),
            "citations": res.get("citations", []),
            "refused": refused,
            "refusal_reason": refusal_reason,
        }

    def document_analyst_node(state: LangGraphState) -> dict:
        if state.get("refused"):
            return {}

        doc_res = run_document_analyst(state["query"], state["hits"], state["cfg"])
        analysis = doc_res.get("analysis", {})
        u = doc_res.get("usage", {})

        memory.log_step(
            state["session_id"],
            3,
            "DocumentAnalyst",
            "analyze_layout_and_tables",
            {
                "extracted_tables": len(analysis.get("extracted_tables", [])),
                "structure_notes": analysis.get("structural_notes", "")[:120],
            },
        )

        return {
            "document_analysis": analysis,
            "usage": {
                "input_tokens": state["usage"]["input_tokens"] + u.get("input_tokens", 0),
                "output_tokens": state["usage"]["output_tokens"] + u.get("output_tokens", 0),
                "cost_usd": state["usage"]["cost_usd"] + u.get("cost_usd", 0.0),
                "calls": state["usage"]["calls"] + 1,
            },
        }

    def data_analyst_node(state: LangGraphState) -> dict:
        if state.get("refused"):
            return {}

        data_res = run_data_analyst(state["query"], state["hits"], state["cfg"])
        math_res = data_res.get("math_result")
        u = data_res.get("usage", {})

        memory.log_step(
            state["session_id"],
            4,
            "DataAnalyst",
            "ast_math_calculation",
            math_res or {"calculated": False},
        )

        return {
            "math_result": math_res,
            "usage": {
                "input_tokens": state["usage"]["input_tokens"] + u.get("input_tokens", 0),
                "output_tokens": state["usage"]["output_tokens"] + u.get("output_tokens", 0),
                "cost_usd": state["usage"]["cost_usd"] + u.get("cost_usd", 0.0),
                "calls": state["usage"]["calls"] + 1,
            },
        }

    def synthesis_node(state: LangGraphState) -> dict:
        if state.get("refused"):
            return {
                "answer": None,
                "citations": [],
            }

        synth_res = run_synthesis_expert(
            query=state["query"],
            hits=state["hits"],
            graph_facts=state.get("graph_facts", []),
            math_result=state.get("math_result"),
            document_analysis=state.get("document_analysis"),
            cfg=state["cfg"],
            audit_feedback=state.get("audit_feedback"),
        )

        ans = synth_res.get("answer")
        cits = synth_res.get("citations", [])
        u = synth_res.get("usage", {})

        memory.log_step(
            state["session_id"],
            5,
            "SynthesisExpert",
            "grounded_synthesis",
            {"citations_count": len(cits), "answer_len": len(ans) if ans else 0},
        )

        return {
            "answer": ans,
            "citations": cits,
            "usage": {
                "input_tokens": state["usage"]["input_tokens"] + u.get("input_tokens", 0),
                "output_tokens": state["usage"]["output_tokens"] + u.get("output_tokens", 0),
                "cost_usd": state["usage"]["cost_usd"] + u.get("cost_usd", 0.0),
                "calls": state["usage"]["calls"] + 1,
            },
        }

    def auditor_node(state: LangGraphState) -> dict:
        if state.get("refused") or not state.get("answer"):
            return {
                "verified": False,
                "verification": {"verified": False, "claims": []},
            }

        audit_res = run_auditor(
            answer=state["answer"],
            citations=state.get("citations", []),
            hits=state["hits"],
            cfg=state["cfg"],
        )

        verified = audit_res.get("verified", False)
        ver_details = audit_res.get("verification", {})
        invalid_cits = audit_res.get("invalid_citations", [])
        feedback = audit_res.get("audit_feedback")
        u = audit_res.get("usage", {})

        memory.log_step(
            state["session_id"],
            6,
            "AuditorGuardrail",
            "audit_verification",
            {
                "verified": verified,
                "claims_audited": len(ver_details.get("claims", [])),
                "invalid_citations": invalid_cits,
                "feedback": feedback,
            },
        )

        return {
            "verified": verified,
            "verification": ver_details,
            "invalid_citations": invalid_cits,
            "audit_feedback": feedback,
            "retry_count": state.get("retry_count", 0) + (0 if verified else 1),
            "usage": {
                "input_tokens": state["usage"]["input_tokens"] + u.get("input_tokens", 0),
                "output_tokens": state["usage"]["output_tokens"] + u.get("output_tokens", 0),
                "cost_usd": state["usage"]["cost_usd"] + u.get("cost_usd", 0.0),
                "calls": state["usage"]["calls"] + 1,
            },
        }

    def should_self_correct(state: LangGraphState) -> str:
        if state.get("refused"):
            return END
        if state.get("verified"):
            return END
        if state.get("retry_count", 0) >= 1:
            return END
        return "synthesis_node"

    # Build Directed Cyclic StateGraph in LangGraph
    workflow = StateGraph(LangGraphState)

    workflow.add_node("lead_orchestrator_node", lead_orchestrator_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("document_analyst_node", document_analyst_node)
    workflow.add_node("data_analyst_node", data_analyst_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("auditor_node", auditor_node)

    workflow.set_entry_point("lead_orchestrator_node")
    workflow.add_edge("lead_orchestrator_node", "researcher_node")
    workflow.add_edge("researcher_node", "document_analyst_node")
    workflow.add_edge("document_analyst_node", "data_analyst_node")
    workflow.add_edge("data_analyst_node", "synthesis_node")
    workflow.add_edge("synthesis_node", "auditor_node")

    workflow.add_conditional_edges("auditor_node", should_self_correct, {
        "synthesis_node": "synthesis_node",
        END: END,
    })

    return workflow


class MultiAgentOrchestrator:
    """Production 6-Agent Swarm Orchestrator with StateGraph."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.workflow = build_langgraph_workflow().compile()
        self.memory = SessionMemoryManager()

    def run(
        self,
        query: str,
        index: Index,
        strategy: str = "agent_react",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        graph_engine = _build_graph_engine_if_needed(self.cfg, index)

        initial_state: LangGraphState = {
            "session_id": session_id,
            "query": query,
            "cfg": self.cfg,
            "index": index,
            "graph_engine": graph_engine,
            "strategy": strategy,
            "plan": {},
            "hits": [],
            "graph_facts": [],
            "document_analysis": {},
            "math_result": None,
            "answer": None,
            "citations": [],
            "invalid_citations": [],
            "verified": False,
            "verification": {},
            "audit_feedback": None,
            "retry_count": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0},
            "latency_ms": 0.0,
            "refused": False,
            "refusal_reason": None,
        }

        final_state = self.workflow.invoke(initial_state)
        latency = (time.perf_counter() - t0) * 1000.0
        final_state["latency_ms"] = latency

        # Checkpoint session in SQLite
        self.memory.save_session(
            session_id=session_id,
            query=query,
            final_answer=final_state.get("answer") or "",
            verified=final_state.get("verified", False),
            strategy=strategy,
            cost_usd=final_state.get("usage", {}).get("cost_usd", 0.0),
            latency_ms=latency,
        )

        return {
            "session_id": session_id,
            "refused": final_state.get("refused", False),
            "refusal_reason": final_state.get("refusal_reason"),
            "answer": final_state.get("answer"),
            "citations": final_state.get("citations", []),
            "invalid_citations": final_state.get("invalid_citations", []),
            "verified": final_state.get("verified", False),
            "verification": final_state.get("verification", {}),
            "confidence": confidence(final_state.get("hits", [])),
            "hits": final_state.get("hits", []),
            "math_result": final_state.get("math_result"),
            "graph_facts": final_state.get("graph_facts", []),
            "usage": final_state.get("usage", {}),
            "latency_ms": latency,
        }


# Alias for backward compatibility
SoloMetaOrchestrator = MultiAgentOrchestrator
