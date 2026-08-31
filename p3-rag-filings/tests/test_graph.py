"""Tests for NetworkX Financial Knowledge Graph."""

import pytest
import networkx as nx

from ragfilings.graph import FinancialGraphBuilder, GraphQueryEngine


def test_graph_builder_adds_nodes_and_edges(tmp_path):
    builder = FinancialGraphBuilder()
    builder.add_company("AAPL", "Apple Inc.")
    builder.add_metric_value(
        ticker="AAPL",
        fiscal_year=2025,
        metric_name="Net Sales",
        value=416161.0,
        unit="USD_M",
        chunk_id="AAPL_2025_10K:Item8:c007",
    )

    assert builder.graph.number_of_nodes() >= 4
    assert builder.graph.number_of_edges() >= 3

    # Test save and load
    save_path = tmp_path / "test_graph.json"
    builder.save(save_path)
    assert save_path.exists()

    loaded = FinancialGraphBuilder.load(save_path)
    assert loaded.graph.number_of_nodes() == builder.graph.number_of_nodes()


def test_graph_builder_ingests_chunk_table():
    builder = FinancialGraphBuilder()
    chunk = {
        "id": "AAPL_2025_10K:Item8:c007",
        "ticker": "AAPL",
        "fiscal_year": 2025,
        "section": "Item8",
        "text": "Total net sales | $416,161 | $391,035\nGross margin percentage | 46.9% | 46.2%",
    }
    builder.ingest_chunk(chunk)
    engine = GraphQueryEngine(builder=builder)

    hist = engine.get_metric_history("AAPL", "Net Sales")
    assert len(hist) >= 1
    assert hist[0]["value"] == 416161.0
    assert hist[0]["chunk_id"] == "AAPL_2025_10K:Item8:c007"


def test_graph_query_engine_resolves_query_facts():
    builder = FinancialGraphBuilder()
    builder.add_metric_value(
        ticker="AAPL",
        fiscal_year=2025,
        metric_name="Net Sales",
        value=416161.0,
        chunk_id="AAPL_2025_10K:Item8:c007",
    )
    engine = GraphQueryEngine(builder=builder)

    facts = engine.resolve_graph_facts("What was AAPL net sales in 2025?")
    assert len(facts) >= 1
    assert facts[0]["ticker"] == "AAPL"
    assert facts[0]["value"] == 416161.0
