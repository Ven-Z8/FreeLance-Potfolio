"""Prompt Loader Module.

Centralized Manager for prompt templates following promptingguide.ai enterprise standards.
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Load a prompt template from src/ragfilings/prompts/<name>.prompt."""
    if name in _CACHE:
        return _CACHE[name]

    file_path = _PROMPT_DIR / f"{name}.prompt"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template missing: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    _CACHE[name] = content
    return content
