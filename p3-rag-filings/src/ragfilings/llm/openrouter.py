"""OpenRouter Provider Client."""

from __future__ import annotations

import os
from typing import Any

from .base import BaseLLMClient
from .types import ChatMessage, LLMResponse, TokenUsage


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client with dynamic usage cost parsing."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "minimax/minimax-m3:free"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 60.0,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        url = base_url or os.environ.get("OPENROUTER_BASE_URL", self.DEFAULT_BASE_URL)
        model = default_model or self.DEFAULT_MODEL
        super().__init__(api_key=key, base_url=url, default_model=model)
        self.timeout = timeout
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-or-"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    def complete(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        target_model = model or self.default_model
        raw_msgs = self.normalize_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": raw_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"usage": {"include": True}},
        }
        request_kwargs.update(kwargs)

        resp = client.chat.completions.create(**request_kwargs)
        choice = resp.choices[0]
        content = choice.message.content or getattr(choice.message, "reasoning", "") or ""

        u = resp.usage
        cost = getattr(u, "cost", None) if u else 0.0
        if cost is None and u and getattr(u, "model_extra", None):
            cost = u.model_extra.get("cost")

        usage = TokenUsage(
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            cost_usd=float(cost or 0.0),
        )

        return LLMResponse(content=content, usage=usage, model=target_model, raw_response=resp)
