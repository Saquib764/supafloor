"""Thin OpenAI client for structured JSON completions."""

from __future__ import annotations

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings
from config.pipeline import get_pipeline_config

T = TypeVar("T", bound=BaseModel)


def openai_model() -> str:
    return settings.openai_model


def _client() -> OpenAI:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    timeout = float(get_pipeline_config()["llm"].get("timeout_seconds") or 30)
    return OpenAI(api_key=api_key, timeout=timeout)


def parse_structured(system: str, user: str, response_model: type[T]) -> T:
    """Call OpenAI chat completions.parse with a Pydantic response schema."""
    completion = _client().beta.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=response_model,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal or "LLM returned no parsed response")
    return message.parsed
