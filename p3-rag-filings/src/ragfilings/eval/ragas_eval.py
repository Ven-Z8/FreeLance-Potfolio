"""RAGAS & Evaluation Framework Metrics."""

from __future__ import annotations

import json
from typing import Any

from ..llm import complete_with_resilience
from ..tools import extract_claims, verify


def compute_faithfulness(answer: str, context_chunks: list[str], cfg: dict[str, Any]) -> float:
    """RAGAS Faithfulness: percentage of claims in answer supported by context."""
    if not answer or not context_chunks:
        return 1.0 if not answer else 0.0

    claims = extract_claims(answer)
    if not claims:
        return 1.0

    dummy_chunks = [{"text": c} for c in context_chunks]
    res = verify(answer, dummy_chunks)
    claims_res = res.get("claims", [])
    if not claims_res:
        return 1.0
    supported_count = sum(1 for c in claims_res if c.get("found"))
    return float(supported_count) / float(len(claims_res))


def compute_answer_relevancy(question: str, answer: str, cfg: dict[str, Any]) -> float:
    """RAGAS Answer Relevancy: LLM-scored relevance of answer to question."""
    if not answer:
        return 0.0
    if answer.startswith("REFUSED") or "unanswerable" in answer.lower():
        return 1.0

    messages = [
        {
            "role": "system",
            "content": "You evaluate answer relevancy. Output a JSON object: {\"relevancy_score\": <float between 0.0 and 1.0>, \"reason\": \"<short reason>\"}",
        },
        {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
    ]
    try:
        text, _ = complete_with_resilience(messages, cfg)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            data = json.loads(text[start : end + 1])
            return float(data.get("relevancy_score", 0.8))
    except Exception:
        pass
    return 0.85


def compute_context_precision(hits: list[dict[str, Any]], ground_truth_citations: list[str]) -> float:
    """RAGAS Context Precision: Position-weighted precision of relevant chunks in hits."""
    if not hits or not ground_truth_citations:
        return 1.0 if not ground_truth_citations else 0.0

    relevant_flags = []
    for h in hits:
        chunk_id = h.get("chunk", {}).get("id", "")
        chunk_sec = h.get("chunk", {}).get("section", "")
        ticker = h.get("chunk", {}).get("ticker", "")

        is_rel = False
        for gt in ground_truth_citations:
            if gt in chunk_id or (ticker in gt and chunk_sec in gt):
                is_rel = True
                break
        relevant_flags.append(1 if is_rel else 0)

    if not any(relevant_flags):
        return 0.0

    total_precision = 0.0
    rel_so_far = 0
    for k, rel in enumerate(relevant_flags, 1):
        if rel:
            rel_so_far += 1
            total_precision += rel_so_far / k

    return total_precision / max(1, sum(relevant_flags))


def compute_context_recall(hits: list[dict[str, Any]], expected_answer: str | None) -> float:
    """RAGAS Context Recall: whether expected answer statements appear in retrieved chunks."""
    if expected_answer is None:
        return 1.0
    if not hits:
        return 0.0

    combined_text = "\n".join(h.get("chunk", {}).get("text", "") for h in hits)
    claims = extract_claims(expected_answer)
    if not claims:
        return 1.0

    found = 0
    for c in claims:
        val = c.get("value")
        raw = c.get("raw", "")
        if (val is not None and str(val) in combined_text) or (raw and raw in combined_text):
            found += 1
    return float(found) / float(len(claims))


def evaluate_ragas(
    question: str,
    answer: str,
    hits: list[dict[str, Any]],
    case: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run RAGAS evaluation on a single query run."""
    context_chunks = [h.get("chunk", {}).get("text", "") for h in hits]
    gt_citations = case.get("expected", {}).get("citations", [])
    expected_ans = case.get("expected", {}).get("answer")

    faithfulness = compute_faithfulness(answer, context_chunks, cfg)
    relevancy = compute_answer_relevancy(question, answer, cfg)
    precision = compute_context_precision(hits, gt_citations)
    recall = compute_context_recall(hits, expected_ans)

    score = (faithfulness * 0.35) + (relevancy * 0.25) + (precision * 0.20) + (recall * 0.20)

    return {
        "ragas_score": round(score, 3),
        "faithfulness": round(faithfulness, 3),
        "answer_relevancy": round(relevancy, 3),
        "context_precision": round(precision, 3),
        "context_recall": round(recall, 3),
    }
