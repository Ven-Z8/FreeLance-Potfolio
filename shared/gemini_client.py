"""Shared Google Gemini LLM API Client for Freelance Portfolio."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def load_env():
    """Load keys from .env file into os.environ if not already set."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v and not os.environ.get(k):
                    os.environ[k] = v


def get_gemini_key() -> str:
    load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is missing in .env file. Please add your key to /Volumes/VeN/FreeLance-Potfolio/.env")
    return key


class GeminiClient:
    """Production client executing real LLM completion calls via Google Gemini REST API."""

    def __init__(self, default_model: str = "gemini-flash-latest"):
        self.default_model = default_model

    def call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        json_output: bool = False,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        api_key = get_gemini_key()

        # Tried-and-tested working model aliases for Gemini API
        candidate_models = [model] if model else [self.default_model, "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash"]

        last_err = None
        for model_name in candidate_models:
            if not model_name:
                continue
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                }
            }

            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            if json_output:
                payload["generationConfig"]["responseMimeType"] = "application/json"

            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                headers={"Content-Type": "application/json"},
                data=data_bytes
            )

            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req) as resp:
                    raw_res = json.loads(resp.read().decode("utf-8"))
                latency_ms = (time.perf_counter() - t0) * 1000.0

                text = ""
                if "candidates" in raw_res and raw_res["candidates"]:
                    cand = raw_res["candidates"][0]
                    if "content" in cand and "parts" in cand["content"]:
                        parts = cand["content"]["parts"]
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"]

                usage_metadata = raw_res.get("usageMetadata", {})
                prompt_tokens = usage_metadata.get("promptTokenCount", 0)
                candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
                total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + candidates_tokens)

                cost_usd = (prompt_tokens * 0.000000075) + (candidates_tokens * 0.00000030)

                return {
                    "text": text.strip(),
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "prompt_tokens": prompt_tokens,
                    "candidates_tokens": candidates_tokens,
                    "total_tokens": total_tokens,
                    "model": model_name,
                    "raw": raw_res
                }
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Gemini API call failed across models: {last_err}")
