"""LLM Client Factory and Completion Dispatcher (OpenRouter)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .base import BaseLLMClient
from .openrouter import OpenRouterClient
from .types import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating OpenRouter LLM clients."""

    @classmethod
    def create_client(
        cls,
        provider_name: str | None = None,
        cfg: dict[str, Any] | None = None,
        default_model: str | None = None,
    ) -> BaseLLMClient:
        """Create an OpenRouter client instance."""
        from .. import config as cfg_mod
        cfg_mod._load_env()

        model = default_model
        if model is None and cfg and "generation" in cfg:
            model = cfg["generation"].get("model")

        return OpenRouterClient(default_model=model)


def get_llm_client(
    provider_name: str | None = None,
    cfg: dict[str, Any] | None = None,
    default_model: str | None = None,
) -> BaseLLMClient:
    """Convenience functional factory."""
    return LLMFactory.create_client(provider_name=provider_name, cfg=cfg, default_model=default_model)


def complete_with_resilience(
    messages: list[ChatMessage | dict[str, str]],
    cfg: dict[str, Any],
    model: str | None = None,
    client: BaseLLMClient | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    """Complete a prompt via OpenRouter with automatic retry."""
    from .. import config as cfg_mod
    cfg_mod._load_env()

    tokens = max_tokens or cfg.get("generation", {}).get("max_tokens", 1200)
    target_model = model or cfg.get("generation", {}).get("model")

    active_client = client or get_llm_client(cfg=cfg, default_model=target_model)

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = active_client.complete(
                messages=messages,
                model=target_model,
                max_tokens=tokens,
                temperature=temperature,
            )
            return resp.content, resp.usage.to_dict()
        except Exception as e:
            last_exc = e
            logger.warning("OpenRouter call attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1.0)

    raise last_exc or RuntimeError("OpenRouter LLM completion failed.")
