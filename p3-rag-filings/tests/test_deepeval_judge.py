"""DeepEval judge layer: cost accounting, prompt routing, scorer glue.

No real API calls: the OpenRouter completion path is monkeypatched, so these
tests pin the wiring (ledger accounting, schema prompting, verdict mapping)
rather than model behavior.
"""

from __future__ import annotations

import json

import pytest

from ragfilings.eval import deepeval_judge as dj

CFG = {
    "eval": {"judge_model": "test/judge-model"},
    "generation": {"model": "test/gen-model", "max_tokens": 100},
}


class FakeResponse:
    pass


@pytest.fixture()
def patched_llm(monkeypatch):
    """Returns a recorder standing in for complete_with_resilience."""
    calls = []

    def fake_complete(messages, cfg, model=None, client=None, max_tokens=None,
                      temperature=0.0, role="generation"):
        calls.append({"messages": messages, "model": model, "role": role})
        return json.dumps({"verdict": "correct", "score": 0.9}), {
            "input_tokens": 100, "output_tokens": 25, "cost_usd": 0.002,
        }

    monkeypatch.setattr(dj, "complete_with_resilience", fake_complete)
    return calls


def test_judge_resolves_model_and_ledger(patched_llm):
    judge = dj.OpenRouterJudge(CFG)
    assert judge.get_model_name() == "test/judge-model"
    out = judge.generate("grade this")
    assert json.loads(out)["verdict"] == "correct"
    assert judge.ledger.to_dict() == {
        "judge_calls": 1, "judge_input_tokens": 100,
        "judge_output_tokens": 25, "judge_cost_usd": 0.002,
    }
    judge.generate("grade again")
    assert judge.ledger.to_dict()["judge_calls"] == 2
    assert judge.ledger.to_dict()["judge_cost_usd"] == pytest.approx(0.004)
    # routed through the judge role + configured model
    assert all(c["role"] == "judge" for c in patched_llm)
    assert all(c["model"] == "test/judge-model" for c in patched_llm)


def test_judge_falls_back_to_generation_role_model(patched_llm):
    cfg = {"generation": {"model": "test/gen-model"}}
    judge = dj.OpenRouterJudge(cfg)
    assert judge.get_model_name() == "test/gen-model"


def test_judge_injects_schema_system_prompt(patched_llm):
    from pydantic import BaseModel

    class Verdict(BaseModel):
        score: float
        reason: str

    judge = dj.OpenRouterJudge(CFG)
    judge.generate("grade", schema=Verdict)
    msgs = patched_llm[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "JSON object" in msgs[0]["content"]
    assert "score" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "grade"}


def test_build_metrics_wired_to_judge(patched_llm):
    judge = dj.OpenRouterJudge(CFG)
    metrics = dj.build_metrics(judge)
    assert set(metrics) == {"correctness", "faithfulness", "answer_relevancy",
                            "contextual_precision"}
    for m in metrics.values():
        assert getattr(m, "model", None) is judge


def test_scorer_correctness_maps_verdict_and_threshold(patched_llm):
    scorer = dj.DeepEvalScorer(CFG)
    verdict_metric = scorer._metrics["correctness"]

    class FakeMetric:
        def __init__(self, score):
            self._score = score
            self.reason = "because"

        def measure(self, tc):
            self.score = self._score

    scorer._metrics["correctness"] = FakeMetric(0.9)
    case = {
        "id": "fin-x", "input": "q?",
        "expected": {"answer": "$1 million", "citations": [], "type": "judge"},
        "failure_category": "synthesis", "notes": "",
    }
    result = {"answer": "about $1 million", "hits": []}
    v = scorer.correctness(case, result)
    assert v == {"correct": True, "score": 0.9, "reason": "because"}

    scorer._metrics["correctness"] = FakeMetric(0.2)
    v2 = scorer.correctness(case, result)
    assert v2["correct"] is False

    scorer._metrics["correctness"] = verdict_metric  # restore


def test_scorer_ambiguous_expected_output_describes_clarification(patched_llm):
    scorer = dj.DeepEvalScorer(CFG)
    case = {
        "id": "fin-y", "input": "What was the net income?",
        "expected": {"answer": None, "citations": [], "type": "judge"},
        "failure_category": "ambiguous", "notes": "company and year unspecified",
    }
    text = scorer._expected_output(case)
    assert "clarification" in text and "company and year unspecified" in text

    answerable = {**case, "expected": {"answer": "$1M", "citations": [], "type": "judge"},
                  "failure_category": "synthesis"}
    assert scorer._expected_output(answerable) == "$1M"


def test_scorer_metrics_skips_empty_answers(patched_llm):
    scorer = dj.DeepEvalScorer(CFG)
    case = {
        "id": "fin-z", "input": "q?",
        "expected": {"answer": "$1 million", "citations": [], "type": "exact"},
        "failure_category": "lookup", "notes": "",
    }
    assert scorer.metrics(case, {"answer": None, "hits": []}) == {}


def test_scorer_correctness_survives_metric_failure(patched_llm):
    scorer = dj.DeepEvalScorer(CFG)

    class ExplodingMetric:
        def measure(self, tc):
            raise RuntimeError("provider down")

    scorer._metrics["correctness"] = ExplodingMetric()
    case = {
        "id": "fin-w", "input": "q?",
        "expected": {"answer": "$1 million", "citations": [], "type": "judge"},
        "failure_category": "synthesis", "notes": "",
    }
    v = scorer.correctness(case, {"answer": "x", "hits": []})
    assert v["correct"] is False and "provider down" in v["reason"]


def test_score_with_deepeval_reports_metric_errors_as_none(patched_llm):
    class ExplodingMetric:
        def measure(self, tc):
            raise ValueError("no retrieval context")

    out = dj.score_with_deepeval(
        {"id": "x", "input": "q", "expected": {"answer": "a", "citations": []}},
        {"answer": "a", "hits": []},
        {"faithfulness": ExplodingMetric()},
    )
    assert out["faithfulness"] is None
    assert "no retrieval context" in out["faithfulness_error"]
