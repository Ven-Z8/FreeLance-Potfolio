"""NetworkX Knowledge Graph Builder for SEC Filings."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx

from .schema import EntityNode, RelationEdge

logger = logging.getLogger(__name__)

# Common financial metrics and aliases to extract into the knowledge graph
KNOWN_METRICS = {
    "net sales": "Net Sales",
    "revenue": "Total Revenue",
    "total revenue": "Total Revenue",
    "net income": "Net Income",
    "operating income": "Operating Income",
    "gross margin": "Gross Margin",
    "gross profit": "Gross Profit",
    "operating expenses": "Operating Expenses",
    "research and development": "R&D Expense",
    "r&d": "R&D Expense",
    "total assets": "Total Assets",
    "total liabilities": "Total Liabilities",
    "cash and cash equivalents": "Cash & Cash Equivalents",
    "diluted earnings per share": "Diluted EPS",
    "diluted eps": "Diluted EPS",
}


class FinancialGraphBuilder:
    """Constructs and manages an in-memory NetworkX entity-relation graph."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.graph = graph if graph is not None else nx.DiGraph()

    def add_company(self, ticker: str, name: str | None = None) -> str:
        cid = f"company:{ticker.upper()}"
        if not self.graph.has_node(cid):
            self.graph.add_node(
                cid,
                label="Company",
                ticker=ticker.upper(),
                name=name or ticker.upper(),
            )
        return cid

    def add_filing(self, ticker: str, fiscal_year: int | str, form: str = "10-K") -> str:
        comp_id = self.add_company(ticker)
        fid = f"filing:{ticker.upper()}_{fiscal_year}_{form}"
        if not self.graph.has_node(fid):
            self.graph.add_node(
                fid,
                label="Filing",
                ticker=ticker.upper(),
                fiscal_year=str(fiscal_year),
                form=form,
            )
            self.graph.add_edge(comp_id, fid, relation="REPORTED_IN")
        return fid

    def add_section(self, filing_id: str, section_id: str, title: str | None = None) -> str:
        sid = f"section:{filing_id}:{section_id}"
        if not self.graph.has_node(sid):
            self.graph.add_node(
                sid,
                label="Section",
                section_code=section_id,
                title=title or section_id,
            )
            self.graph.add_edge(filing_id, sid, relation="HAS_SECTION")
        return sid

    def add_metric_value(
        self,
        ticker: str,
        fiscal_year: int | str,
        metric_name: str,
        value: float | str,
        unit: str = "USD_M",
        chunk_id: str | None = None,
        section_id: str | None = None,
    ) -> str:
        """Add a financial metric measurement node with graph links."""
        filing_id = self.add_filing(ticker, fiscal_year)
        clean_metric = KNOWN_METRICS.get(metric_name.lower().strip(), metric_name.strip().title())
        metric_id = f"metric:{clean_metric.lower().replace(' ', '_')}"

        if not self.graph.has_node(metric_id):
            self.graph.add_node(
                metric_id,
                label="FinancialMetric",
                name=clean_metric,
            )

        val_node_id = f"val:{ticker.upper()}:{clean_metric.lower().replace(' ', '_')}:{fiscal_year}"
        self.graph.add_node(
            val_node_id,
            label="MetricValue",
            ticker=ticker.upper(),
            metric=clean_metric,
            fiscal_year=str(fiscal_year),
            value=value,
            unit=unit,
            chunk_id=chunk_id,
        )

        # Edges
        self.graph.add_edge(filing_id, val_node_id, relation="RECORDED_VALUE")
        self.graph.add_edge(metric_id, val_node_id, relation="CONTAINS_METRIC")

        if chunk_id:
            cid = f"chunk:{chunk_id}"
            if not self.graph.has_node(cid):
                self.graph.add_node(cid, label="Chunk", chunk_id=chunk_id)
            self.graph.add_edge(val_node_id, cid, relation="CITES_CHUNK")

        return val_node_id

    def ingest_chunk(self, chunk: dict[str, Any]) -> None:
        """Parse table rows and structured lines in a chunk into graph triples."""
        cid = chunk.get("id", "")
        ticker = chunk.get("ticker", "")
        fy = chunk.get("fiscal_year", "")
        sec = chunk.get("section", "")
        text = chunk.get("text", "")

        if not ticker or not fy:
            return

        filing_id = self.add_filing(ticker, fy)
        if sec:
            self.add_section(filing_id, sec)

        # Parse pipe-separated table rows or key-value metric lines
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Case A: Pipe table line: "Total net sales | $416,161 | $391,035"
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    metric_candidate = parts[0].lower()
                    for k_metric in KNOWN_METRICS:
                        if k_metric in metric_candidate:
                            # Try parsing the first number
                            num_match = re.search(r"[\$]?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)", parts[1])
                            if num_match:
                                raw_val = num_match.group(1).replace(",", "")
                                try:
                                    f_val = float(raw_val)
                                    self.add_metric_value(
                                        ticker=ticker,
                                        fiscal_year=fy,
                                        metric_name=metric_candidate,
                                        value=f_val,
                                        chunk_id=cid,
                                        section_id=sec,
                                    )
                                except ValueError:
                                    pass
                            break

    def build_from_chunks(self, chunks: list[dict[str, Any]]) -> nx.DiGraph:
        """Batch construct the graph from a collection of corpus chunks."""
        for c in chunks:
            self.ingest_chunk(c)
        logger.info("Knowledge Graph built with %d nodes and %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        return self.graph

    def save(self, path: str | Path) -> None:
        """Persist graph to JSON node-link format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FinancialGraphBuilder:
        """Load graph from JSON node-link format."""
        p = Path(path)
        if not p.exists():
            return cls(nx.DiGraph())
        data = json.loads(p.read_text(encoding="utf-8"))
        graph = nx.node_link_graph(data, directed=True)
        return cls(graph)
