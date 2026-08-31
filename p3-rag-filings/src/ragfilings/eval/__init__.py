"""Evaluation Subsystem Package."""

from .deepeval_judge import DeepEvalScorer, JudgeLedger, OpenRouterJudge
from .evaluation import aggregate, build_trace, load_cases, run_eval, score_case
from .regress import diff_runs, run_regression
from .report import write_scorecard

__all__ = [
    "run_eval",
    "load_cases",
    "score_case",
    "build_trace",
    "aggregate",
    "write_scorecard",
    "run_regression",
    "diff_runs",
    "DeepEvalScorer",
    "OpenRouterJudge",
    "JudgeLedger",
]
