"""Core Grounded RAG Pipeline Engine.

Single responsibility: orchestrates confidence gating, context assembly,
prompt dispatch via LLM clients, JSON response extraction, and deterministic
verification with corrective retries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..llm import BaseLLMClient, complete_with_resilience, get_llm_client
from ..prompts import PromptRegistry
from ..retrieval import Index, confidence, load_index
from ..tools import compute_financial_math, decompose_query, needs_decomposition, verify

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """The model never produced parseable JSON, even after a reformat retry."""


def _parse_json(text: str) -> dict[str, Any] | None:
    """Tolerantly extract and parse a JSON object from model prose."""
    clean_text = text
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0]
    elif "```" in clean_text:
        clean_text = clean_text.split("```")[1].split("```")[0]

    start, end = clean_text.find("{"), clean_text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(clean_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "answer" in data else None


def _complete(
    messages: list[dict[str, str]],
    cfg: dict[str, Any],
    model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Chat completion entry point with multi-provider failover."""
    return complete_with_resilience(messages, cfg, model=model)


def answer(
    query: str,
    hits: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    """Execute grounded 10-K answer synthesis with verification and confidence gating."""
    conf = confidence(hits)
    min_confidence = cfg.get("verification", {}).get("min_confidence", 0.35)

    usage: dict[str, float | int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "calls": 0,
    }

    if conf < min_confidence:
        return {
            "refused": True,
            "refusal_reason": f"low retrieval confidence: {conf:.3f} < {min_confidence}",
            "answer": None,
            "citations": [],
            "invalid_citations": [],
            "verification": {"verified": True, "claims": []},
            "confidence": conf,
            "usage": usage,
        }

    math_res = None
    if needs_decomposition(query):
        math_res = compute_financial_math(query, [h["chunk"] for h in hits], cfg, client=client)

    context = "\n\n".join(f"[{h['chunk']['id']}]\n{h['chunk']['text']}" for h in hits)
    if math_res:
        context += (
            f"\n\n[PYTHON_MATH_TOOL_VERIFIED_RESULT]\n"
            f"Calculated {math_res['explanation']}: {math_res['formatted']} (Formula: {math_res['expression']})"
        )

    by_id = {h["chunk"]["id"]: h["chunk"] for h in hits}
    msgs = [
        {"role": "system", "content": PromptRegistry.get_system_synthesis()},
        {"role": "user", "content": f"Context chunks:\n\n{context}\n\nQuestion: {query}"},
    ]

    llm_client = client or get_llm_client(cfg=cfg)

    def _call_model() -> dict[str, Any]:
        nonlocal msgs
        for attempt in range(2):
            if client is not None:
                text, u = complete_with_resilience(messages=msgs, cfg=cfg, client=client)
            else:
                import sys
                mod = sys.modules.get(__name__)
                mod_complete = getattr(mod, "_complete", None) if mod else None
                if mod_complete and mod_complete != complete_with_resilience and mod_complete != _complete:
                    text, u = mod_complete(msgs, cfg)
                elif _complete != complete_with_resilience:
                    text, u = _complete(msgs, cfg)
                else:
                    text, u = complete_with_resilience(messages=msgs, cfg=cfg, client=llm_client)
            for k in ("input_tokens", "output_tokens", "cost_usd"):
                usage[k] += u.get(k, 0)
            usage["calls"] += 1

            data = _parse_json(text)
            if data is not None:
                msgs = msgs + [{"role": "assistant", "content": text}]
                return data

            if attempt == 0:
                msgs = msgs + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Reply with ONLY the JSON object."},
                ]

        raise GenerationError("model did not return parseable JSON after retry")

    data = _call_model()
    retries = cfg.get("generation", {}).get("verify_retries", 1)

    while True:
        # Check model self-reported refusal
        if data.get("answer") is None:
            return {
                "refused": True,
                "refusal_reason": f"model: {data.get('reason') or 'not answerable'}",
                "answer": None,
                "citations": [],
                "invalid_citations": [],
                "verification": {"verified": True, "claims": []},
                "confidence": conf,
                "usage": usage,
                "math_result": math_res,
            }

        raw_citations = [c for c in data.get("citations") or [] if isinstance(c, str)]
        citations = [c for c in raw_citations if c in by_id]
        invalid = [c for c in raw_citations if c not in by_id]
        cited = [by_id[c] for c in citations] or [h["chunk"] for h in hits]

        checked = verify(str(data["answer"]), cited, math_result=math_res)
        if checked["verified"] or retries <= 0:
            break

        retries -= 1
        failed_claims = [c["raw"] for c in checked["claims"] if not c["found"]]
        retry_prompt = PromptRegistry.get_verification_retry(failed_claims)

        msgs = msgs + [{"role": "user", "content": retry_prompt}]
        data = _call_model()

    return {
        "refused": False,
        "refusal_reason": None,
        "answer": str(data["answer"]),
        "citations": citations,
        "invalid_citations": invalid,
        "verification": checked,
        "confidence": conf,
        "usage": usage,
        "math_result": math_res,
    }


def log_refusal(path: str | Path, query: str, result: dict[str, Any], strategy: str) -> None:
    """Persist refusal event for analysis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "query": query,
                    "strategy": strategy,
                    "reason": result.get("refusal_reason"),
                    "confidence": result.get("confidence"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def ask(
    query: str,
    cfg: dict[str, Any],
    index: Index | None = None,
    strategy: str | None = None,
    refusal_log: str | Path = "reports/refusals.jsonl",
) -> dict[str, Any]:
    """End-to-end RAG pipeline execution."""
    if index is None:
        index = load_index(cfg["embedding"]["index_dir"], cfg["embedding"]["model"])

    strat = strategy or cfg.get("retrieval", {}).get("strategy", "dense")

    if strat == "agent_react":
        from .orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator(cfg)
        res = orch.run(query, index, strategy="hybrid_rerank")
        res["strategy"] = "agent_react"
        res["model"] = cfg.get("generation", {}).get("model", "")
        if res.get("refused"):
            log_refusal(refusal_log, query, res, strat)
        return res

    t0 = time.perf_counter()
    top_k = cfg.get("retrieval", {}).get("top_k", 8)

    if needs_decomposition(query):
        sub_queries = decompose_query(query, cfg)
        hits: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for sq in sub_queries:
            sq_hits = index.search(sq, strat, top_k)
            for h in sq_hits:
                cid = h["chunk"]["id"]
                if cid not in seen_ids:
                    hits.append(h)
                    seen_ids.add(cid)
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:top_k]
    else:
        hits = index.search(query, strat, top_k)

    result = answer(query, hits, cfg)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    result["strategy"] = strat
    result["hits"] = hits
    result["model"] = cfg.get("generation", {}).get("model", "")

    if result.get("refused"):
        log_refusal(refusal_log, query, result, strat)

    return result
