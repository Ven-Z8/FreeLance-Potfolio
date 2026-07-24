"""Adapter connecting p3-rag-filings to the Domain-Adaptive Agent Eval Harness."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure p3-rag-filings package is on python path
p3_root = Path(__file__).resolve().parents[4] / "p3-rag-filings" / "src"
if p3_root.exists() and str(p3_root) not in sys.path:
    sys.path.insert(0, str(p3_root))

from harness.schema import AgentRunTrace, GoldenCase, TrajectoryStep


class RAGFilingsAdapter:
    """Agent adapter for p3-rag-filings system."""

    def __init__(self, config_path: str = None):
        from ragfilings import config as cfg_mod, retrieval
        if config_path is None:
            config_path = str(Path(__file__).resolve().parents[4] / "p3-rag-filings" / "config.toml")

        self.cfg = cfg_mod.load(config_path)
        index_path = Path(self.cfg["embedding"]["index_dir"])
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parents[4] / "p3-rag-filings" / index_path
        self.index = retrieval.load_index(
            str(index_path),
            self.cfg["embedding"]["model"]
        )


    def run_case(self, case: GoldenCase, strategy: str = "hybrid_rerank") -> AgentRunTrace:
        from ragfilings import generation

        t0 = time.perf_counter()
        raw_res = generation.ask(
            case.input, self.cfg, index=self.index, strategy=strategy
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        steps = []
        if raw_res.get("agent_history"):
            for h in raw_res["agent_history"]:
                steps.append(TrajectoryStep(
                    agent=h.get("agent", "Agent"),
                    action=h.get("action", "step"),
                    input_payload=h.get("input"),
                    output_payload=h.get("output")
                ))

        return AgentRunTrace(
            case_id=case.id,
            domain=case.domain,
            strategy=strategy,
            query=case.input,
            answer=raw_res.get("answer"),
            citations=raw_res.get("citations") or [],
            refused=raw_res.get("refused", False),
            refusal_reason=raw_res.get("refusal_reason"),
            steps=steps,
            latency_ms=raw_res.get("latency_ms", latency_ms),
            cost_usd=raw_res.get("usage", {}).get("cost_usd", 0.015),
            raw_response=raw_res
        )
