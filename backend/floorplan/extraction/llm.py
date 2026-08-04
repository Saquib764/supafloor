"""Vision LLM client for structured floor plan extraction."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from extraction.constants import EXTRACTION_MODEL, LLM_TIMEOUT_SECONDS

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)

T = TypeVar("T", bound=BaseModel)


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)


def image_part(png_bytes: bytes) -> dict[str, Any]:
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def parse_structured_messages(
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    model: str = EXTRACTION_MODEL,
    prompt_cache_key: str | None = None,
) -> T:
    """Call OpenAI chat completions.parse with a message history and Pydantic response schema."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_model,
    }
    if prompt_cache_key is not None:
        kwargs["prompt_cache_key"] = prompt_cache_key
    completion = _client().beta.chat.completions.parse(**kwargs)
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal or "LLM returned no parsed response")
    return message.parsed
