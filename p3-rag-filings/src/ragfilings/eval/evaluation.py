"""Evaluation Runner: Golden cases × retrieval strategies → scorecard + traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..llm import complete_with_resilience
from ..pipeline import ask
from ..tools import extract_claims
from .ragas_eval import evaluate_ragas

_UNIT_RATIOS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_TOL_RE = re.compile(r"numeric_tolerance:([\d.]+)%")


def load_cases(golden_dir: str | Path) -> list[dict[str, Any]]:
    """Load golden test cases from directory or file."""
    cases = []
    golden_path = Path(golden_dir)
    if golden_path.is_file():
        with golden_path.open(encoding="utf-8") as f:
            cases.extend(json.loads(line) for line in f if line.strip())
        return cases

    for path in sorted(golden_path.glob("**/*.jsonl")):
        if "skeleton" in path.name:
            continue
        with path.open(encoding="utf-8") as f:
            cases.extend(json.loads(line) for line in f if line.strip())
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


_JUDGE_SYSTEM = """\
You are a strict grader for a question-answering system over SEC filings.
Decide whether the system's answer is factually equivalent to the expected
answer for the given question: same key figures (tolerate rounding and unit
re-expression), same direction/explanation where the question asks for one.
Wording differences are fine; missing or contradicting the key facts is not.
Reply with ONLY JSON: {"correct": true/false, "reason": "<one sentence>"}"""


def _judge(case: dict[str, Any], result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {case['input']}\n\n"
                f"Expected answer: {case['expected']['answer']}\n\n"
                f"System answer: {result.get('answer', '')}"
            ),
        },
    ]
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    judge_model = cfg.get("eval", {}).get("judge_model", cfg.get("generation", {}).get("model"))

    for attempt in range(2):
        text, u = complete_with_resilience(messages, cfg, model=judge_model)
        for k in usage:
            usage[k] += u.get(k, 0)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data.get("correct"), bool):
                    return {"correct": data["correct"], "reason": str(data.get("reason", "")), "usage": usage}
            except json.JSONDecodeError:
                pass
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "Reply with ONLY the JSON object."})

    return {"correct": False, "reason": "judge returned unparseable output", "usage": usage}


def score_case(case: dict[str, Any], result: dict[str, Any], cfg: dict[str, Any] | None) -> dict[str, Any]:
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
        "judge_usage": None,
    }

    if expected["answer"] is None:
        if result.get("refused"):
            scored.update(correct=True, outcome="correct_refusal", retrieval_hit=None)
        else:
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
        verdict = _judge(case, result, cfg or {})
        correct = verdict["correct"]
        scored["judge_reason"] = verdict["reason"]
        scored["judge_usage"] = verdict["usage"]
    else:
        raise ValueError(f"unknown expected.type: {ctype!r}")

    scored.update(correct=correct, outcome="answered")
    ragas_m = evaluate_ragas(
        question=case["input"],
        answer=result.get("answer", ""),
        hits=result.get("hits", []),
        case=case,
        cfg=cfg or {},
    )
    scored["ragas"] = ragas_m
    return scored


def build_trace(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Emit trace dictionary in standard format."""
    hits = result.get("hits", [])
    ver = result.get("verification", {"verified": True, "claims": []})
    final = result.get("answer") if not result.get("refused") else f"[REFUSED] {result.get('refusal_reason')}"
    steps = [
        {
            "index": 0,
            "kind": "tool_call",
            "content": None,
            "retry_of": None,
            "tool_call": {
                "name": "retrieve",
                "arguments": {
                    "query": case["input"],
                    "strategy": result.get("strategy", ""),
                    "top_k": len(hits),
                },
                "result": [
                    {"id": h["chunk"]["id"], "score": h["score"], "dense_sim": h["dense_sim"]}
                    for h in hits
                ],
                "error": None,
                "latency_ms": None,
            },
        },
        {
            "index": 1,
            "kind": "tool_call",
            "content": None,
            "retry_of": None,
            "tool_call": {
                "name": "verify_claims",
                "arguments": {"n_claims": len(ver.get("claims", []))},
                "result": {
                    "verified": ver.get("verified", True),
                    "failed": [c["raw"] for c in ver.get("claims", []) if not c.get("found", True)],
                },
                "error": None,
                "latency_ms": None,
            },
        },
        {
            "index": 2,
            "kind": "response",
            "content": final,
            "tool_call": None,
            "retry_of": None,
        },
    ]
    u = result.get("usage", {})
    return {
        "case_id": case["id"],
        "input": case["input"],
        "final_output": final or "",
        "steps": steps,
        "citations": list(result.get("citations", [])),
        "usage": {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "cost_usd": u.get("cost_usd", 0.0),
        },
        "latency_ms": result.get("latency_ms", 0.0),
        "model": result.get("model", ""),
        "metadata": {
            "adapter": "ragfilings-v0",
            "strategy": result.get("strategy", ""),
            "refused": result.get("refused", False),
            "refusal_reason": result.get("refusal_reason"),
            "confidence": result.get("confidence"),
            "invalid_citations": result.get("invalid_citations", []),
            "verification_verified": ver.get("verified", True),
            "golden_verification": "pending",
        },
    }


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
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None,
        "cost_per_query_usd": _mean([r["cost_usd"] for r in rows]),
        "by_category": by_cat,
    }


def run_eval(
    cfg: dict[str, Any],
    golden_dir: str | Path,
    strategies: list[str],
    out_dir: str | Path = "reports",
    limit: int | None = None,
    index=None,
) -> dict[str, Any]:
    """Run golden evaluation suite across retrieval strategies."""
    from ..retrieval import load_index

    cases = load_cases(golden_dir)[:limit]
    if index is None:
        index = load_index(cfg["embedding"]["index_dir"], cfg["embedding"]["model"])

    out_dir = Path(out_dir)
    all_results: dict[str, Any] = {}

    for strategy in strategies:
        trace_dir = out_dir / "traces" / strategy
        trace_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        judge_cost = 0.0
        results_path = out_dir / f"results_{strategy}.jsonl"

        with results_path.open("w", encoding="utf-8") as results_f:
            for i, case in enumerate(cases, 1):
                result = ask(
                    case["input"],
                    cfg,
                    index=index,
                    strategy=strategy,
                    refusal_log=out_dir / "refusals.jsonl",
                )
                scored = score_case(case, result, cfg)
                if scored.get("judge_usage"):
                    judge_cost += scored["judge_usage"]["cost_usd"]

                trace = build_trace(case, result)
                (trace_dir / f"{case['id']}.json").write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False, default=str)
                )

                row = {
                    **scored,
                    "latency_ms": result.get("latency_ms", 0.0),
                    "cost_usd": result.get("usage", {}).get("cost_usd", 0.0),
                    "verified": result.get("verification", {}).get("verified", True),
                    "answer": result.get("answer"),
                    "refusal_reason": result.get("refusal_reason"),
                    "citations": result.get("citations", []),
                }
                rows.append(row)
                results_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                mark = "+" if scored["correct"] else "-"
                print(f"[{strategy} {i:>2}/{len(cases)}] {mark} {case['id']} {scored['outcome']}", flush=True)

        metrics = aggregate(rows)
        metrics["judge_cost_usd"] = judge_cost
        metrics["model"] = cfg.get("generation", {}).get("model", "")
        metrics["judge_model"] = cfg.get("eval", {}).get("judge_model", "")
        all_results[strategy] = {"metrics": metrics, "rows": rows}
        acc = metrics.get("accuracy") or 0.0
        print(f"[{strategy}] accuracy={acc:.1%} n={metrics['n']}", flush=True)

    return all_results
