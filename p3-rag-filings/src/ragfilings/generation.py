"""Grounded generation via OpenRouter — every claim cites chunk IDs.

Provider: OpenRouter's OpenAI-compatible endpoint, so the model is a config
value (`[generation] model`), not a code decision. Structured output is
prompt-enforced JSON with tolerant parsing + one reformat retry — deliberately
NOT a provider-specific structured-outputs feature, so any OpenRouter slug
works. Cost comes from OpenRouter's `usage.include` accounting, not a local
price table.

Pipeline per query (see also ask() at the bottom):
  1. Confidence gate BEFORE any model call: top dense cosine below
     config[verification][min_confidence] -> refuse, with logged reason.
  2. Generate: chunks rendered as [chunk_id]-tagged context, JSON answer.
  3. Verify: numerical claims re-found in the cited chunks
     (verification.verify); one corrective retry naming the failed claims,
     then ship flagged if still unverified — flagged, never silently wrong.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import math_tool, query_decompose, retrieval, verification


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GenerationError(RuntimeError):
    """The model never produced parseable JSON, even after a reformat retry."""


_SYSTEM = """\
You answer questions about SEC 10-K filings using ONLY the context chunks
provided. Each chunk starts with its ID in [brackets].

Reply with ONLY a JSON object, no other text:
{"answer": "<concise answer with exact figures as stated in the chunks>",
 "citations": ["<id of every chunk your answer relies on>"],
 "reason": null}

Rules:
- Use only facts from the chunks. Copy figures exactly (units and all).
- Cite every chunk you used; never cite a chunk you did not use.
- If the chunks do not contain the answer, or the question is too ambiguous
  to answer precisely, set "answer" to null and explain briefly in "reason".
- Table rows read: label | most recent period | prior period | ... unless the
  header row says otherwise. Match the fiscal year the question asks about.
"""


def _get_provider_info() -> tuple[str, str, str]:
    from . import config
    config._load_env()
    provider_override = os.environ.get("LLM_PROVIDER", "").lower()
    if provider_override == "nvidia" and os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"], "https://integrate.api.nvidia.com/v1", "nvidia"
    if provider_override == "openrouter" and os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"], "https://openrouter.ai/api/v1", "openrouter"

    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"], "https://openrouter.ai/api/v1", "openrouter"
    if os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"], "https://integrate.api.nvidia.com/v1", "nvidia"
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], "https://api.openai.com/v1", "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"], "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini"
    raise RuntimeError(
        "No API key found in environment or .env — set OPENROUTER_API_KEY, NVIDIA_API_KEY, "
        "OPENAI_API_KEY, or GEMINI_API_KEY."
    )




def _client():
    key, base_url, _ = _get_provider_info()
    from openai import OpenAI  # lazy: retrieval-only paths shouldn't import it
    return OpenAI(base_url=base_url, api_key=key)


def _complete(messages: list[dict], cfg: dict, model: str | None = None
              ) -> tuple[str, dict[str, Any]]:
    """One chat completion. Returns (text, usage) — the only network call site."""
    key, base_url, provider = _get_provider_info()
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=key)

    target_model = model or cfg["generation"].get("model", "anthropic/claude-sonnet-4.5")
    if provider == "nvidia" and ("claude" in target_model or "gpt" in target_model):
        target_model = "meta/llama-3.3-70b-instruct"
    elif provider == "gemini" and ("claude" in target_model or "meta/" in target_model):
        target_model = "gemini-2.5-flash"
    elif provider == "openai" and ("claude" in target_model or "meta/" in target_model):
        target_model = "gpt-4o-mini"

    kwargs: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "max_tokens": cfg["generation"].get("max_tokens", 1200),
    }
    if provider == "openrouter":
        kwargs["extra_body"] = {"usage": {"include": True}}

    resp = client.chat.completions.create(**kwargs)
    u = resp.usage
    cost = getattr(u, "cost", None) if u else 0.0
    if cost is None and u and getattr(u, "model_extra", None):
        cost = u.model_extra.get("cost")
    return resp.choices[0].message.content or "", {
        "input_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "cost_usd": float(cost or 0.0),
    }



def _parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "answer" in data else None


def answer(query: str, hits: list[dict[str, Any]], cfg: dict) -> dict[str, Any]:
    """Grounded generation pass over the retrieved hits.

    Gate: top dense cosine must clear min_confidence. Otherwise refuse.
    """
    conf = retrieval.confidence(hits)
    threshold = cfg["verification"]["min_confidence"]
    usage: dict[str, float | int] = {
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0,
    }
    if conf < threshold:
        return {
            "refused": True,
            "refusal_reason": f"low retrieval confidence: {conf:.3f} < {threshold}",
            "answer": None, "citations": [], "invalid_citations": [],
            "verification": {"verified": True, "claims": []},
            "confidence": conf, "usage": usage,
        }

    math_res = None
    if query_decompose.needs_decomposition(query):
        math_res = math_tool.compute_financial_math(query, [h["chunk"] for h in hits], cfg)

    context = "\n\n".join(f"[{h['chunk']['id']}]\n{h['chunk']['text']}" for h in hits)
    if math_res:
        context += (
            f"\n\n[PYTHON_MATH_TOOL_VERIFIED_RESULT]\n"
            f"Calculated {math_res['explanation']}: {math_res['formatted']} (Formula: {math_res['expression']})"
        )

    by_id = {h["chunk"]["id"]: h["chunk"] for h in hits}
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Context chunks:\n\n{context}\n\nQuestion: {query}"},
    ]

    def call() -> dict:
        nonlocal messages
        for attempt in range(2):  # one reformat retry for non-JSON replies
            text, u = _complete(messages, cfg)
            for k in ("input_tokens", "output_tokens", "cost_usd"):
                usage[k] += u[k]
            usage["calls"] += 1
            data = _parse_json(text)
            if data is not None:
                messages = messages + [{"role": "assistant", "content": text}]
                return data
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Reply with ONLY the JSON object."},
                ]
        raise GenerationError("model did not return parseable JSON after retry")

    data = call()
    retries = cfg["generation"].get("verify_retries", 1)
    while True:
        if data.get("answer") is None:
            return {
                "refused": True,
                "refusal_reason": f"model: {data.get('reason') or 'not answerable'}",
                "answer": None, "citations": [], "invalid_citations": [],
                "verification": {"verified": True, "claims": []},
                "confidence": conf, "usage": usage,
            }
        raw_citations = [c for c in data.get("citations") or [] if isinstance(c, str)]
        citations = [c for c in raw_citations if c in by_id]
        invalid = [c for c in raw_citations if c not in by_id]
        cited = [by_id[c] for c in citations] or [h["chunk"] for h in hits]
        checked = verification.verify(str(data["answer"]), cited)
        if checked["verified"] or retries <= 0:
            break
        retries -= 1
        failed = ", ".join(c["raw"] for c in checked["claims"] if not c["found"])
        messages = messages + [{
            "role": "user",
            "content": (
                f"Verification failed: the figure(s) {failed} do not appear in the "
                "chunks you cited. Re-read the cited chunks carefully (check which "
                "column is the requested period) and reply with corrected JSON, or "
                "set answer to null if the context truly lacks the figure."),
        }]
        data = call()

    return {
        "refused": False, "refusal_reason": None,
        "answer": str(data["answer"]),
        "citations": citations, "invalid_citations": invalid,
        "verification": checked, "confidence": conf, "usage": usage,
        "math_result": math_res,
    }


def log_refusal(path: str | Path, query: str, result: dict, strategy: str) -> None:
    """Every refusal is logged with its reason — the eval measures correctness."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query, "strategy": strategy,
            "reason": result["refusal_reason"],
            "confidence": result.get("confidence"),
        }, ensure_ascii=False) + "\n")


def ask(query: str, cfg: dict, index: retrieval.Index | None = None,
        strategy: str | None = None,
        refusal_log: str | Path = "reports/refusals.jsonl") -> dict[str, Any]:
    """Full pipeline: decompose -> retrieve -> gate -> math -> generate -> verify."""
    if index is None:
        index = retrieval.load_index(cfg["embedding"]["index_dir"],
                                     cfg["embedding"]["model"])
    strategy = strategy or cfg["retrieval"]["strategy"]

    if strategy == "agent_react":
        from . import orchestrator
        orch = orchestrator.MultiAgentOrchestrator(cfg)
        res = orch.run(query, index, strategy="hybrid_rerank")
        res["strategy"] = "agent_react"
        res["model"] = cfg["generation"]["model"]
        if res["refused"]:
            log_refusal(refusal_log, query, res, strategy)
        return res

    t0 = time.perf_counter()

    if query_decompose.needs_decomposition(query):
        sub_queries = query_decompose.decompose_query(query, cfg)
        hits = []
        seen_ids = set()
        for sq in sub_queries:
            sq_hits = index.search(sq, strategy, cfg["retrieval"]["top_k"])
            for h in sq_hits:
                if h["chunk"]["id"] not in seen_ids:
                    hits.append(h)
                    seen_ids.add(h["chunk"]["id"])
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:cfg["retrieval"]["top_k"]]
    else:
        hits = index.search(query, strategy, cfg["retrieval"]["top_k"])


    result = answer(query, hits, cfg)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    result["strategy"] = strategy
    result["hits"] = hits
    result["model"] = cfg["generation"]["model"]
    if result["refused"]:
        log_refusal(refusal_log, query, result, strategy)
    return result

