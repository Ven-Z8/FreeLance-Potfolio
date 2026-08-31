"""Evaluation Harness Runner.

Executes test cases against agent adapters, records full JSON trajectory traces,
computes three-tier metrics, and aggregates results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from harness.metrics import engine
from harness.schema import AgentRunTrace, GoldenCase


def load_golden_cases(jsonl_path: str | Path) -> List[GoldenCase]:
    """Load JSONL golden set into list of GoldenCase Pydantic models."""
    path = Path(jsonl_path)
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cases.append(GoldenCase.model_validate(data))
    return cases


def run_evaluation_suite(
    golden_path: str | Path,
    adapter: Any,
    strategy: str = "hybrid_rerank",
    out_dir: str | Path = "reports",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Execute evaluation suite across golden dataset and output traces."""
    cases = load_golden_cases(golden_path)
    if limit:
        cases = cases[:limit]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces" / strategy
    trace_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rows = []

    print(f"\n🚀 Running Harness Evaluation: {len(cases)} Golden Cases | Strategy: {strategy}")
    print("=" * 70)

    for i, case in enumerate(cases, 1):
        try:
            trace = adapter.run_case(case, strategy=strategy)
        except Exception as exc:
            trace = AgentRunTrace(
                case_id=case.id,
                domain=case.domain,
                strategy=strategy,
                query=case.input,
                refused=True,
                refusal_reason=f"API Error: {exc}",
                answer=None,
                citations=[],
                invalid_citations=[],
                verification={"verified": False, "claims": []},
                steps=[],
                latency_ms=0.0,
                cost_usd=0.0,
                confidence=0.0,
            )

        # Save trajectory trace JSON
        (trace_dir / f"{case.id}.json").write_text(
            json.dumps(trace.model_dump(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )

        res = engine.evaluate_case(case, trace)
        results.append(res)
        rows.append(res)

        mark = "✅" if res["correct"] else "❌"
        print(f"[{i:>3}/{len(cases)}] {mark} {case.id} ({case.failure_category}): {res['outcome']}", flush=True)



    # Compute Summary Metrics
    total = len(results)
    correct_cnt = sum(1 for r in results if r["correct"])
    accuracy = correct_cnt / max(total, 1)

    by_category = {}
    for r in results:
        cat = r.get("trace", {}).get("query")
        # Extract from case
    
    summary = {
        "strategy": strategy,
        "n": total,
        "accuracy": accuracy,
        "correct_count": correct_cnt,
        "results": results
    }

    print("=" * 70)
    print(f"📊 Final Scorecard: Overall Accuracy = {accuracy:.1%} ({correct_cnt}/{total})\n")

    return summary
