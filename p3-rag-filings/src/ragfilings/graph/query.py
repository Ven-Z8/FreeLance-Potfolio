"""Knowledge Graph Query Engine for Multi-Hop Financial Traversal."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from .builder import FinancialGraphBuilder, KNOWN_METRICS

logger = logging.getLogger(__name__)


class GraphQueryEngine:
    """High-level query interface over the NetworkX Financial Knowledge Graph."""

    def __init__(self, builder: FinancialGraphBuilder | None = None, graph: nx.DiGraph | None = None) -> None:
        if builder:
            self.graph = builder.graph
        elif graph is not None:
            self.graph = graph
        else:
            self.graph = nx.DiGraph()

    def get_metric_history(self, ticker: str, metric_name: str) -> list[dict[str, Any]]:
        """Retrieve multi-year trajectory for a given company metric."""
        t_upper = ticker.upper()
        clean_metric = KNOWN_METRICS.get(metric_name.lower().strip(), metric_name.strip().title())
        results = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") == "MetricValue":
                if data.get("ticker") == t_upper:
                    if clean_metric.lower() in data.get("metric", "").lower():
                        results.append({
                            "ticker": t_upper,
                            "metric": data.get("metric"),
                            "fiscal_year": data.get("fiscal_year"),
                            "value": data.get("value"),
                            "unit": data.get("unit", "USD_M"),
                            "chunk_id": data.get("chunk_id"),
                        })

        return sorted(results, key=lambda x: str(x.get("fiscal_year", "")))

    def compare_metrics(self, tickers: list[str], metric_name: str, fiscal_year: str | int | None = None) -> list[dict[str, Any]]:
        """Compare multiple companies on a specific financial metric."""
        results = []
        for t in tickers:
            hist = self.get_metric_history(t, metric_name)
            if fiscal_year:
                hist = [h for h in hist if str(h.get("fiscal_year")) == str(fiscal_year)]
            results.extend(hist)
        return results

    def find_entity_subgraph(self, ticker: str, radius: int = 2) -> dict[str, Any]:
        """Extract ego-subgraph centered around a company entity node."""
        cid = f"company:{ticker.upper()}"
        if not self.graph.has_node(cid):
            return {"nodes": [], "edges": []}

        sub = nx.ego_graph(self.graph.to_undirected(), cid, radius=radius)
        nodes = [{"id": n, **self.graph.nodes[n]} for n in sub.nodes()]
        edges = [
            {"source": u, "target": v, **self.graph.get_edge_data(u, v, default={})}
            for u, v in sub.edges()
            if self.graph.has_edge(u, v)
        ]
        return {"nodes": nodes, "edges": edges}

    def resolve_graph_facts(self, query: str) -> list[dict[str, Any]]:
        """Identify entities mentioned in the query and extract verified graph facts."""
        q_lower = query.lower()
        extracted_facts = []

        # Detect tickers or company names
        found_tickers = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") == "Company":
                t = data.get("ticker", "")
                name = data.get("name", "")
                if t.lower() in q_lower or (name and name.lower() in q_lower):
                    if t not in found_tickers:
                        found_tickers.append(t)

        # Detect metrics
        found_metrics = []
        for k_metric, clean_name in KNOWN_METRICS.items():
            if k_metric in q_lower:
                found_metrics.append(clean_name)

        for t in found_tickers:
            for m in found_metrics:
                hist = self.get_metric_history(t, m)
                extracted_facts.extend(hist)

        return extracted_facts
