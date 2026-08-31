"""Unit tests for Project 5 Cost & Latency Optimization Framework."""

import pytest
from optimization.optimizer import SystemOptimizerEngine


def test_optimization_benchmark_protocol():
    engine = SystemOptimizerEngine()
    report = engine.run_full_benchmark()

    assert report.baseline.cost_per_100_runs_usd == 14.50
    assert len(report.optimized_steps) == 4

    final_step = report.optimized_steps[-1]
    assert final_step.cost_reduction_pct >= 75.0
    assert final_step.latency_reduction_pct >= 60.0
    assert final_step.eval_accuracy == report.baseline.eval_accuracy
    assert report.final_summary["quality_held"] is True
