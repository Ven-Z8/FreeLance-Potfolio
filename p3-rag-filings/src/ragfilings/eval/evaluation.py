"""Evaluation Runner: golden cases × retrieval strategies → scorecard + traces.

Scoring is two-tier:
1. Deterministic — numeric matching with variation rules, refusal matrix,
   citation/retrieval prefix hits. Ground truth for exact/contains cases.
2. DeepEval layer (deepeval_judge.py) — G-Eval correctness for judge-type
   and ambiguous cases, plus faithfulness / answer_relevancy /
   contextual_precision on every answered case. All judge calls go through
   the OpenRouter judge client with real cost accounting.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline import ask
from ..tools import extract_claims

_UNIT_RATIOS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
_TOL_RE = re.compile(r"numeric_tolerance:([\d.]+)%")


def load_cases(golden_dir: str | Path) -> list[dict[str, Any]]:
    """Load golden test cases. In directory mode only canonical
    golden_set_*.jsonl files are used (candidates/handcrafted drafts and the
    skeleton template are excluded)."""
    cases = []
    golden_path = Path(golden_dir)
    if golden_path.is_file():
        with golden_path.open(encoding="utf-8") as f:
            cases.extend(json.loads(line) for line in f if line.strip())
        return cases

    for path in sorted(golden_path.glob("golden_set_*.jsonl")):
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


def score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    scorer: Any = None,
) -> dict[str, Any]:
    """Score one case. `scorer` is a DeepEvalScorer (deepeval_judge.py); it is
    required for judge-type and ambiguous cases."""
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
    if scorer is not None:
        scored["deepeval"] = scorer.metrics(case, result)
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
            "adapter": "ragfilings-v1",
            "strategy": result.get("strategy", ""),
            "refused": result.get("refused", False),
            "refusal_reason": result.get("refusal_reason"),
            "confidence": result.get("confidence"),
            "invalid_citations": result.get("invalid_citations", []),
            "verification_verified": ver.get("verified", True),
            "golden_verification": "v1 proven",
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


def _run_meta(cfg: dict[str, Any], golden_dir: str | Path, strategies: list[str],
              judge_model: str) -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": sha,
        "golden_set": str(golden_dir),
        "strategies": strategies,
        "generation_model": cfg.get("generation", {}).get("model"),
        "judge_model": judge_model,
        "retrieval": cfg.get("retrieval", {}),
        "verification": cfg.get("verification", {}),
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
    from .deepeval_judge import DeepEvalScorer

    cases = load_cases(golden_dir)[:limit]
    if index is None:
        index = load_index(cfg["embedding"]["index_dir"], cfg["embedding"]["model"])

    scorer = DeepEvalScorer(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            _run_meta(cfg, golden_dir, strategies, scorer.judge.get_model_name()),
            indent=2,
        ),
        encoding="utf-8",
    )
    all_results: dict[str, Any] = {}

    for strategy in strategies:
        trace_dir = out_dir / "traces" / strategy
        trace_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        results_path = out_dir / f"results_{strategy}.jsonl"

        with results_path.open("w", encoding="utf-8") as results_f:
            consecutive_errors = 0
            for i, case in enumerate(cases, 1):
                try:
                    result = ask(
                        case["input"],
                        cfg,
                        index=index,
                        strategy=strategy,
                        refusal_log=out_dir / "refusals.jsonl",
                    )
                    scored = score_case(case, result, cfg, scorer=scorer)
                except Exception as e:
                    # one failed case must not kill a long run; record it
                    scored = {
                        "case_id": case["id"],
                        "category": case["failure_category"],
                        "difficulty": case.get("difficulty"),
                        "refused": False,
                        "citation_hit": None,
                        "retrieval_hit": None,
                        "judge_reason": None,
                        "judge_score": None,
                        "deepeval": None,
                        "correct": False,
                        "outcome": "error",
                        "error": f"{type(e).__name__}: {e}"[:400],
                    }
                    result = {
                        "latency_ms": 0.0,
                        "usage": {"cost_usd": 0.0},
                        "verification": {"verified": False},
                        "answer": None,
                        "refusal_reason": None,
                        "citations": [],
                    }
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        raise RuntimeError(
                            "5 consecutive case failures — aborting run "
                            f"(last error: {scored.get('error')})"
                        )
                else:
                    consecutive_errors = 0

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
                results_f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                mark = "+" if scored["correct"] else "-"
                print(f"[{strategy} {i:>2}/{len(cases)}] {mark} {case['id']} {scored['outcome']}", flush=True)

        metrics = aggregate(rows)
        metrics.update(scorer.judge.ledger.to_dict())
        metrics["model"] = cfg.get("generation", {}).get("model", "")
        metrics["judge_model"] = scorer.judge.get_model_name()
        metrics["error_count"] = sum(1 for r in rows if r["outcome"] == "error")
        all_results[strategy] = {"metrics": metrics, "rows": rows}
        acc = metrics.get("accuracy") or 0.0
        suffix = f" errors={metrics['error_count']}" if metrics["error_count"] else ""
        print(f"[{strategy}] accuracy={acc:.1%} n={metrics['n']}{suffix}", flush=True)

    return all_results
