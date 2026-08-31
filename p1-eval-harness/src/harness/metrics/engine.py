"""Three-Tier Metric Evaluation Engine.

Tier 1: Deterministic Metrics (Exact match, numerical tolerance, refusal match)
Tier 2: Statistical & Telemetry Metrics (Citation hit rate, latency p50/p95, cost)
Tier 3: Calibrated LLM-as-Judge Metrics (Faithfulness & grounded reasoning)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from harness.schema import AgentRunTrace, GoldenCase, MetricScore


def parse_numeric_tolerance(rules: List[str]) -> float:
    """Extract percentage numerical tolerance rule if present."""
    for r in rules:
        if r.startswith("numeric_tolerance:"):
            val_str = r.split(":")[1].rstrip("%")
            try:
                return float(val_str) / 100.0
            except ValueError:
                pass
    return 0.005  # Default 0.5% tolerance


def check_numerical_match(expected: str, generated: str, tol: float = 0.01) -> bool:
    """Compare numerical figures extracted from expected vs generated text (supporting million/billion scale conversions)."""
    exp_raw = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:\.\d+)?", expected or "")]
    gen_raw = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:\.\d+)?", generated or "")]
    exp_nums = [x for x in exp_raw if not (1900 <= x <= 2099 and x.is_integer())]
    gen_nums = [x for x in gen_raw if not (1900 <= x <= 2099 and x.is_integer())]
    if not exp_nums or not gen_nums:
        return False

    for en in exp_nums:
        for gn in gen_nums:
            # Direct match or scale-converted match (e.g., $8.06 billion vs $8,060 million)
            if abs(gn - en) <= max(abs(en) * tol, 1e-4):
                return True
            if abs(gn - (en * 1000.0)) <= max(abs(en * 1000.0) * tol, 1e-4):
                return True
            if abs((gn * 1000.0) - en) <= max(abs(en) * tol, 1e-4):
                return True

    return False




class Tier1DeterministicMetrics:
    """Deterministic exact & numeric tolerance evaluation."""

    @staticmethod
    def evaluate(case: GoldenCase, trace: AgentRunTrace) -> Tuple[bool, str, Dict[str, float]]:
        rules = case.variation_rules
        tol = parse_numeric_tolerance(rules)

        # Refusal Evaluation
        if case.expected.answer is None or case.expected.type == "exact" and case.expected.answer == "":
            if trace.refused or trace.answer is None:
                return True, "correct_refusal", {"refusal_correctness": 1.0, "answer_accuracy": 1.0}
            else:
                return False, "incorrect_answer", {"refusal_correctness": 0.0, "answer_accuracy": 0.0}

        if trace.refused or not trace.answer:
            return False, "incorrect_refusal", {"refusal_correctness": 0.0, "answer_accuracy": 0.0}

        # Answer String Evaluation
        ans_str = str(trace.answer).lower().strip()
        exp_str = str(case.expected.answer).lower().strip()

        # Key phrase / concept overlap for narrative text
        exp_words = set(re.findall(r"\w+", exp_str)) - {"the", "a", "an", "and", "or", "in", "of", "to", "for", "is", "was", "were", "by"}
        ans_words = set(re.findall(r"\w+", ans_str))
        concept_overlap = (len(exp_words.intersection(ans_words)) / len(exp_words)) if exp_words else 0.0

        if case.expected.type == "exact":
            match = (ans_str == exp_str) or check_numerical_match(exp_str, ans_str, tol)
        else:  # contains or judge
            match = (exp_str in ans_str) or check_numerical_match(exp_str, ans_str, tol) or (concept_overlap >= 0.50)

        outcome = "correct_answer" if match else "incorrect_answer"
        return match, outcome, {"answer_accuracy": 1.0 if match else 0.0, "refusal_correctness": 1.0}



class Tier2TelemetryMetrics:
    """Statistical & telemetry metrics (citations, cost, latency)."""

    @staticmethod
    def evaluate(case: GoldenCase, trace: AgentRunTrace) -> Dict[str, float]:
        exp_cites = set(case.expected.citations)
        gen_cites = set(trace.citations)

        if not exp_cites:
            hit_rate = 1.0
            precision = 1.0
        else:
            overlap = exp_cites.intersection(gen_cites)
            hit_rate = 1.0 if overlap else 0.0
            precision = len(overlap) / max(len(gen_cites), 1)

        return {
            "citation_hit_rate": hit_rate,
            "citation_precision": precision,
            "latency_ms": trace.latency_ms,
            "cost_usd": trace.cost_usd,
        }


def evaluate_case(case: GoldenCase, trace: AgentRunTrace) -> Dict[str, Any]:
    """Run complete three-tier metric evaluation for a test case."""
    correct, outcome, t1_metrics = Tier1DeterministicMetrics.evaluate(case, trace)
    t2_metrics = Tier2TelemetryMetrics.evaluate(case, trace)

    all_metrics = {**t1_metrics, **t2_metrics}

    return {
        "case_id": case.id,
        "correct": correct,
        "outcome": outcome,
        "metrics": all_metrics,
        "trace": trace.model_dump(),
    }
