"""Domain-Adaptive Agent Eval Harness Schema Definitions.

Defines schemas for:
  - Golden Case (v0 frozen schema)
  - Agent Run Trajectory Trace (recording every step, tool call, latency, cost)
  - Metric Results & Evaluation Summary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExpectedOutcome(BaseModel):
    """Ground truth expected outcome for a test case."""
    answer: Optional[str] = Field(default=None, description="Expected answer string or null if unanswerable.")
    citations: List[str] = Field(default_factory=list, description="Expected citation chunk IDs.")
    type: str = Field(default="judge", description="Evaluation type: exact, contains, or judge.")


class GoldenCase(BaseModel):
    """Frozen v0 schema for evaluation test cases."""
    id: str = Field(description="Unique case identifier (e.g. fin-0001, b20-001).")
    input: str = Field(description="User prompt or question.")
    expected: ExpectedOutcome = Field(description="Expected ground truth outcome.")
    variation_rules: List[str] = Field(default_factory=list, description="Rules like numeric_tolerance:0.5%, unit_equivalence.")
    difficulty: str = Field(default="medium", description="Case difficulty: easy, medium, hard.")
    failure_category: str = Field(default="lookup", description="Target failure category: lookup, synthesis, table, unanswerable, ambiguous.")
    domain: str = Field(default="financial", description="Domain: financial, biomedical, legal, healthcare.")
    notes: Optional[str] = Field(default=None, description="Case notes or design objective.")


class TrajectoryStep(BaseModel):
    """A single step or tool call in an agent trajectory."""
    agent: str = Field(description="Sub-agent or role name.")
    action: str = Field(description="Action name or tool invoked.")
    input_payload: Optional[Any] = Field(default=None, description="Input parameters passed to step.")
    output_payload: Optional[Any] = Field(default=None, description="Output returned from step.")
    latency_ms: float = Field(default=0.0, description="Step latency in milliseconds.")


class AgentRunTrace(BaseModel):
    """Structured JSON trajectory trace of a full agent run."""
    case_id: str = Field(description="ID of the executed golden case.")
    domain: str = Field(description="Domain of the test case.")
    strategy: str = Field(description="Agent strategy executed.")
    query: str = Field(description="Query string.")
    answer: Optional[str] = Field(default=None, description="Generated answer.")
    citations: List[str] = Field(default_factory=list, description="Citations produced.")
    refused: bool = Field(default=False, description="Whether agent refused to answer.")
    refusal_reason: Optional[str] = Field(default=None, description="Reason for refusal.")
    steps: List[TrajectoryStep] = Field(default_factory=list, description="Trajectory steps.")
    latency_ms: float = Field(default=0.0, description="Total run latency in milliseconds.")
    cost_usd: float = Field(default=0.0, description="Estimated total run cost in USD.")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw agent output dictionary.")


class MetricScore(BaseModel):
    """Individual metric score result."""
    name: str = Field(description="Metric name.")
    score: float = Field(description="Metric score (0.0 to 1.0 or raw scalar).")
    tier: str = Field(description="Metric tier: deterministic, telemetry, judge.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metric execution metadata.")


class CaseEvalResult(BaseModel):
    """Complete evaluation result for a single case execution."""
    case_id: str = Field(description="Case ID.")
    correct: bool = Field(description="Overall pass/fail result.")
    outcome: str = Field(description="Outcome tag: correct_answer, incorrect_answer, correct_refusal, incorrect_refusal.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Map of metric names to numeric scores.")
    trace: AgentRunTrace = Field(description="Recorded agent run trajectory.")
