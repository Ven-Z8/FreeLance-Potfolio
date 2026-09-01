"""Two-tier scoring engine.

Tier 1 — deterministic: numeric matching with variation rules, the
refusal/unanswerable matrix, and citation/retrieval prefix hits. Ground
truth for exact/contains cases.
Tier 2 — calibrated LLM-judge (harness/judge.py): G-Eval correctness for
judge-type and ambiguous cases, plus faithfulness / answer_relevancy /
contextual_precision on every answered case. Judge scores complement the
deterministic tier; they never replace ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from harness.metrics.claims import extract_claims
from harness.schema import GoldenCase

_UNIT_RATIOS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_TOL_RE = re.compile(r"numeric_tolerance:([\d.]+)%")


def load_cases(golden_dir: str | Path) -> list[dict[str, Any]]:
    """Load golden test cases and enforce the frozen case schema.

    In directory mode only canonical golden_set_*.jsonl files are used
    (skeleton templates are excluded). Every line is validated against
    GoldenCase before scoring, so a malformed case fails loudly at load
    time instead of corrupting a run.
    """
    cases: list[dict[str, Any]] = []
    golden_path = Path(golden_dir)
    paths = [golden_path] if golden_path.is_file() else sorted(
        p for p in golden_path.glob("golden_set_*.jsonl") if "skeleton" not in p.name
    )
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(GoldenCase.model_validate(json.loads(line)).model_dump())
    return cases


def _parse_rules(rules: list[str]) -> tuple[float, bool]:
    tol, unit_eq = 0.0, False
    for r in rules:
        if m := _TOL_RE.fullmatch(r.strip()):
            tol = float(m.group(1)) / 100.0
        elif r.strip() == "unit_equivalence":
            unit_eq = True
    return tol, unit_eq


def _scale_word(raw: str) -> str | None:
    for w in ("trillion", "billion", "million", "thousand"):
        if w in raw.lower():
            return w
    return None


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip(" .")).strip()


def _figures_match(expected_text: str, actual_text: str, tol: float, unit_eq: bool) -> bool:
    """Does the actual answer state the expected figure within rules?"""
    exp_claims = extract_claims(expected_text)
    if not exp_claims:
        return _norm_text(expected_text) in _norm_text(actual_text)

    exp = exp_claims[0]
    for act in extract_claims(actual_text):
        if act["is_pct"] != exp["is_pct"]:
            continue
        ratios = _UNIT_RATIOS if (unit_eq and not exp["is_pct"]) else (1.0,)
        for r in ratios:
            a, e = act["value"] * r, exp["value"]
            if e and abs(a - e) / abs(e) <= max(tol, 1e-9):
                if unit_eq or _scale_word(act["raw"]) == _scale_word(exp["raw"]):
                    return True
    return False


def _prefix_hit(produced: list[str], expected: list[str]) -> bool | None:
    if not expected:
        return None
    return any(p == e or p.startswith(e + ":") or p.startswith(e) for p in produced for e in expected)


def score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    scorer: Any = None,
    include_deepeval_metrics: bool = True,
) -> dict[str, Any]:
    """Score one case. `scorer` is a DeepEvalScorer (judge.py); it is
    required for judge-type and ambiguous cases.

    `include_deepeval_metrics=False` skips the complementary faithfulness /
    relevancy / contextual-precision judge calls (accuracy scoring — the
    deterministic checks plus G-Eval correctness — is unaffected). This is the
    fast path for accuracy-focused regression runs on rate-limited providers.
    """
    expected = case["expected"]
    tol, unit_eq = _parse_rules(case.get("variation_rules", []))
    hit_ids = [h["chunk"]["id"] for h in result.get("hits", [])]
    retrieval_hit = _prefix_hit(hit_ids, expected["citations"])
    scored: dict[str, Any] = {
        "case_id": case["id"],
        "category": case["failure_category"],
        "difficulty": case.get("difficulty"),
        "refused": result.get("refused", False),
        "citation_hit": None,
        "retrieval_hit": retrieval_hit,
        "judge_reason": None,
        "judge_score": None,
        "deepeval": None,
    }

    if expected["answer"] is None:
        if result.get("refused"):
            scored.update(correct=True, outcome="correct_refusal", retrieval_hit=None)
            return scored
        if case["failure_category"] == "ambiguous":
            # answered without refusing: correct only if the judge agrees the
            # response surfaces the ambiguity instead of guessing
            if scorer is None:
                raise ValueError(f"{case['id']}: ambiguous case needs the DeepEval scorer")
            verdict = scorer.correctness(case, result)
            scored.update(
                correct=bool(verdict["correct"]),
                outcome="answered",
                judge_reason=verdict.get("reason"),
                judge_score=verdict.get("score"),
            )
            return scored
        scored.update(correct=False, outcome="hallucination")
        return scored

    if result.get("refused"):
        scored.update(correct=False, outcome="incorrect_refusal")
        return scored

    scored["citation_hit"] = _prefix_hit(result.get("citations", []), expected["citations"])
    ctype = expected["type"]
    if ctype in ("exact", "contains"):
        correct = _figures_match(expected["answer"], result.get("answer", ""), tol, unit_eq)
    elif ctype == "judge":
        if scorer is None:
            raise ValueError(f"{case['id']}: judge-type case needs the DeepEval scorer")
        verdict = scorer.correctness(case, result)
        correct = bool(verdict["correct"])
        scored["judge_reason"] = verdict.get("reason")
        scored["judge_score"] = verdict.get("score")
    else:
        raise ValueError(f"unknown expected.type: {ctype!r}")

    scored.update(correct=correct, outcome="answered")
    if scorer is not None and include_deepeval_metrics:
        scored["deepeval"] = scorer.metrics(case, result)
    return scored


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    refusals = [r for r in rows if r["refused"]]
    lat = [r["latency_ms"] for r in rows]
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += bool(r["correct"])
    for c in by_cat.values():
        c["accuracy"] = c["correct"] / c["n"]

    de = [r.get("deepeval") or {} for r in rows]
    return {
        "n": len(rows),
        "accuracy": _mean([r["correct"] for r in rows]),
        "citation_faithfulness": _mean([r["citation_hit"] for r in rows]),
        "retrieval_hit_rate": _mean([r["retrieval_hit"] for r in rows]),
        "hallucination_rate": (
            _mean([r["outcome"] == "hallucination" for r in unanswerable]) if unanswerable else None
        ),
        "refusal_rate": len(refusals) / len(rows) if rows else None,
        "refusal_correctness": _mean([r["correct"] for r in refusals]),
        "verified_rate": _mean([r.get("verified") for r in rows]),
        "deepeval_faithfulness": _mean([d.get("faithfulness") for d in de]),
        "deepeval_answer_relevancy": _mean([d.get("answer_relevancy") for d in de]),
        "deepeval_contextual_precision": _mean([d.get("contextual_precision") for d in de]),
        "deepeval_correctness": _mean([r.get("judge_score") for r in rows]),
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None,
        "cost_per_query_usd": _mean([r["cost_usd"] for r in rows]),
        "by_category": by_cat,
    }
