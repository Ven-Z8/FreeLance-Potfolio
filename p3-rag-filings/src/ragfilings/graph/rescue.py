"""Deterministic graph rescue for refused lookups.

When the generator refuses because it cannot find a figure in the retrieved
chunks, this layer attempts a fully deterministic lookup against the fact
graph: it extracts (ticker, metric, fiscal year) scope from the question
text alone — no LLM involved — and if the graph holds the exact fact, the
pipeline retries synthesis with the fact and its provenance chunk added to
the context.

Conservative by design, because a wrong rescue turns a correct refusal into
a hallucination:

- rescue fires only on complete (ticker, metric, year) triples;
- metric scope comes from unambiguous multi-word statement phrases only
  (never bare "revenue", which would match "WhatsApp revenue");
- the question must be fully explained by ticker + metric + year: any
  residual qualifier ("first quarter", "data center", "Waymo") aborts;
- sub-period questions (quarterly, half-year) abort;
- facts flagged as mis-extracted (corpus/graph/excluded_facts.json) are
  never surfaced.

The rescue can therefore only ever inject a real, provenance-backed fact;
whether it answers the question is still the synthesis model's call, and
every figure it cites is re-verified against the source chunk downstream.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config as cfg_mod
from .builder import KNOWN_METRICS
from .query import GraphQueryEngine

_YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-4]\d)\b")
_PERIOD_RE = re.compile(
    r"\bquarter(?:ly)?\b|\bq[1-4]\b|\b(?:first|second|third|fourth) quarter\b"
    r"|\b(?:three|six|nine) months\b|\bhalf[- ]?year\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]*")

# Interrogative scaffolding that carries no topical scope. Anything left
# over after removing companies, metric phrases, years, and these words is a
# qualifier the graph cannot vouch for, so rescue aborts.
_FILLERS = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "for", "in",
    "on", "as", "by", "or", "and", "to", "from", "its", "did", "do", "does",
    "how", "much", "many", "which", "company", "companies", "report",
    "reported", "higher", "lower", "change", "changed", "increase",
    "decrease", "fiscal", "year", "years", "fy", "s", "10-k", "between",
    "over", "vs", "compared", "with", "expense", "amount",
}

# First words of company names that are also common English words; these
# match only via the full company name, never via the first-word fallback.
_GENERIC_FIRST_WORDS = {"home", "bank"}

# Multi-word statement phrases are unambiguous metric references; single-word
# keys ("revenue", "r&d") are too broad for rescue scope.
_RESCUE_PHRASES = sorted(
    (p for p in KNOWN_METRICS if " " in p and KNOWN_METRICS[p] != "Gross Margin"),
    key=len,
    reverse=True,
)

_UNIT_SUFFIX = {"USD_M": " million", "USD_B": " billion", "USD_TH": " thousand"}


@dataclass(frozen=True)
class RescueQuery:
    ticker: str
    metric: str
    fiscal_year: int

    @property
    def fact_id(self) -> str:
        return (f"val:{self.ticker}:{self.metric.lower().replace(' ', '_')}"
                f":{self.fiscal_year}")


@dataclass
class RescueOutcome:
    queries: list[RescueQuery]
    facts: list[dict[str, Any]]
    chunk_ids: list[str]
    chunks: list[dict[str, Any]]
    facts_block: str
    derived_values: list[float] = field(default_factory=list)


def load_excluded_facts(path: str | Path | None = None) -> frozenset[str]:
    """Fact ids verified wrong by human review; never surfaced by rescue."""
    p = Path(path) if path else cfg_mod.ROOT / "corpus" / "graph" / "excluded_facts.json"
    if not p.exists():
        return frozenset()
    data = json.loads(p.read_text(encoding="utf-8"))
    return frozenset(data.get("excluded", {}))


def load_company_aliases(manifest_path: str | Path | None = None) -> dict[str, list[str]]:
    """Ticker -> matchable name aliases from the corpus manifest."""
    p = Path(manifest_path) if manifest_path else cfg_mod.ROOT / "corpus" / "manifest.csv"
    aliases: dict[str, list[str]] = {}
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"].strip().upper()
            names = {ticker.lower(), _normalize(row["company"])}
            first = _normalize(row["company"]).split()[0]
            if len(first) >= 4 and first not in _GENERIC_FIRST_WORDS:
                names.add(first)
            aliases[ticker] = sorted(names, key=len, reverse=True)
    return aliases


def _normalize(text: str) -> str:
    low = text.lower().replace("&", " and ").replace("'", " ")
    return re.sub(r"[^a-z0-9\s-]", " ", low)


def _format_fact(fact: dict[str, Any]) -> str:
    num = f"{fact['value']:,.2f}".rstrip("0").rstrip(".")
    if fact.get("unit") == "PCT":
        return f"{num}%"
    if str(fact.get("metric", "")).endswith("EPS"):
        return f"${num}"
    return f"${num}{_UNIT_SUFFIX.get(fact.get('unit', ''), '')}"


def _derived_values(facts: list[dict[str, Any]]) -> list[float]:
    """Grounded deltas / percent changes over the rescued facts.

    Change questions (same ticker + metric, two years) yield the absolute
    delta and absolute percent change; comparison questions (same metric +
    year, two tickers) yield the absolute difference. Magnitudes only —
    answers phrase these as positive figures with an up/down qualifier.
    """
    out: list[float] = []
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for f in facts:
        by_metric.setdefault(str(f.get("metric", "")), []).append(f)
    for group in by_metric.values():
        if len(group) < 2:
            continue
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for f in group:
            by_ticker.setdefault(str(f.get("ticker", "")), []).append(f)
        for tf in by_ticker.values():
            if len(tf) >= 2:
                ordered = sorted(tf, key=lambda x: int(x.get("fiscal_year", 0)))
                for a, b in zip(ordered, ordered[1:]):
                    va, vb = float(a["value"]), float(b["value"])
                    out.append(abs(vb - va))
                    if va:
                        out.append(abs((vb - va) / va * 100.0))
        by_year: dict[str, list[float]] = {}
        for f in group:
            by_year.setdefault(str(f.get("fiscal_year", "")), []).append(float(f["value"]))
        for values in by_year.values():
            if len(values) >= 2:
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        out.append(abs(values[i] - values[j]))
    return out


class GraphRescue:
    """Deterministic rescue lookups against the fact graph."""

    def __init__(self, engine: GraphQueryEngine, chunks_by_id: dict[str, dict[str, Any]],
                 company_aliases: dict[str, list[str]] | None = None,
                 excluded: frozenset[str] | None = None) -> None:
        self.engine = engine
        self.chunks_by_id = chunks_by_id
        self.company_aliases = company_aliases or {}
        self.excluded = excluded if excluded is not None else load_excluded_facts()
        self._alias_re = sorted(
            ((alias, ticker)
             for ticker, names in self.company_aliases.items()
             for alias in names),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    # ------------------------------------------------------------- extraction

    def extract_queries(self, query: str) -> list[RescueQuery] | None:
        """Deterministic (ticker, metric, year) scope, or None if the
        question carries any qualifier the graph cannot vouch for."""
        if _PERIOD_RE.search(query):
            return None
        text = _normalize(query)

        tickers: list[str] = []
        for alias, ticker in self._alias_re:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                if ticker not in tickers:
                    tickers.append(ticker)
                text = re.sub(rf"\b{re.escape(alias)}\b", " ", text)
        if not tickers:
            return None

        metrics: list[str] = []
        for phrase in _RESCUE_PHRASES:
            if re.search(rf"\b{re.escape(_normalize(phrase))}\b", text):
                canon = KNOWN_METRICS[phrase]
                if canon not in metrics:
                    metrics.append(canon)
                text = re.sub(rf"\b{re.escape(_normalize(phrase))}\b", " ", text)
        if not metrics:
            return None

        years = [int(y) for y in _YEAR_RE.findall(text)]
        if not years:
            return None
        text = _YEAR_RE.sub(" ", text)

        residual = [t for t in _TOKEN_RE.findall(text) if t not in _FILLERS]
        if residual:
            return None

        return [RescueQuery(t, m, y)
                for t in tickers for m in metrics for y in dict.fromkeys(years)]

    # ---------------------------------------------------------------- lookup

    def rescue(self, query: str) -> RescueOutcome | None:
        """Facts answering the question's scope, or None when rescue abstains."""
        queries = self.extract_queries(query)
        if not queries:
            return None

        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for q in queries:
            if q.fact_id in seen or q.fact_id in self.excluded:
                continue
            row = self.engine.get_metric_value(q.ticker, q.metric, q.fiscal_year)
            if row is None:
                continue
            seen.add(q.fact_id)
            facts.append(row)
        if not facts:
            return None

        chunks = [self.chunks_by_id[f["chunk_id"]] for f in facts
                  if f.get("chunk_id") in self.chunks_by_id]
        if not chunks:
            return None
        chunk_ids = [c["id"] for c in chunks]

        lines = [
            f"- {f['ticker']} {f['metric']} FY{f['fiscal_year']}: "
            f"{_format_fact(f)} (source chunk: {f['chunk_id']})"
            for f in facts
        ]
        block = (
            "[GRAPH_FACTS — deterministic extractions from 10-K tables]\n"
            "Each line carries the source chunk ID of the table it was parsed "
            "from. If you use one of these figures, cite that source chunk ID, "
            "not this block.\n" + "\n".join(lines)
        )
        return RescueOutcome(queries=queries, facts=facts, chunk_ids=chunk_ids,
                             chunks=chunks, facts_block=block,
                             derived_values=_derived_values(facts))
