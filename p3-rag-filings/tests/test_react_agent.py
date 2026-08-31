"""Tests for Individual ReAct Agent Tools."""

from ragfilings.agents.auditor import build_auditor_tools
from ragfilings.agents.data_analyst import build_data_analyst_tools
from ragfilings.agents.document_analyst import build_document_analyst_tools
from ragfilings.agents.orchestrator import build_orchestrator_tools
from ragfilings.agents.researcher import build_researcher_tools
from ragfilings.agents.synthesis import build_synthesis_tools
from ragfilings.graph.builder import FinancialGraphBuilder
from ragfilings.graph.query import GraphQueryEngine


class DummyIndex:
    def __init__(self):
        self.chunks = [
            {
                "id": "META_2025_10K:Item7:c030",
                "ticker": "META",
                "fiscal_year": 2025,
                "section": "Item7",
                "text": "Research and development expense was $57,372 million in 2025, representing 29% of revenue.",
            }
        ]

    def search(self, query, strategy="hybrid_rerank", top_k=8):
        return [{"chunk": self.chunks[0], "score": 0.92}]


def test_individual_react_agent_tools():
    builder = FinancialGraphBuilder()
    builder.add_metric_value(
        ticker="META",
        fiscal_year=2025,
        metric_name="R&D Expense",
        value=57372.0,
        chunk_id="META_2025_10K:Item7:c030",
    )
    engine = GraphQueryEngine(builder=builder)
    index = DummyIndex()
    hits = [{"chunk": index.chunks[0], "score": 0.92}]

    # 1. Orchestrator tools
    orch_tools = build_orchestrator_tools({})
    assert len(orch_tools) >= 2

    # 2. Researcher tools
    res_tools = build_researcher_tools(index, engine)
    assert any(t.name == "search_dense_vector" for t in res_tools)
    assert any(t.name == "query_knowledge_graph_series" for t in res_tools)

    # 3. Document Analyst tools
    doc_tools = build_document_analyst_tools(hits)
    assert any(t.name == "extract_table_rows_and_headers" for t in doc_tools)

    # 4. Data Analyst tools
    data_tools = build_data_analyst_tools()
    math_tool = next(t for t in data_tools if t.name == "evaluate_ast_math_formula")
    math_res = math_tool.invoke({"expression": "(57372 - 43873) / 43873 * 100", "explanation": "YoY R&D Growth"})
    assert "30.77%" in math_res

    # 5. Synthesis tools
    synth_tools = build_synthesis_tools(hits)
    assert any(t.name == "verify_chunk_citation" for t in synth_tools)

    # 6. Auditor tools
    audit_tools = build_auditor_tools(hits)
    assert any(t.name == "verify_numerical_claims" for t in audit_tools)
