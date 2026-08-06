"""Vision LLM client for structured floor plan extraction."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar

import replicate
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from extraction.constants import (
    EXTRACTION_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    REPLICATE_MAX_NEW_TOKENS,
    REPLICATE_MODEL,
    REPLICATE_TEMPERATURE,
    REPLICATE_TOP_P,
)

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)

T = TypeVar("T", bound=BaseModel)


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)


def _replicate_client() -> replicate.Client:
    api_token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set")
    return replicate.Client(api_token=api_token, timeout=LLM_TIMEOUT_SECONDS)


def image_part(png_bytes: bytes) -> dict[str, Any]:
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _messages_text(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system = ""
    user = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system" and isinstance(content, str):
            system = content
        elif role == "user":
            if isinstance(content, str):
                user = content
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user = str(part.get("text", ""))
    return system, user


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    for candidate in (stripped, _strip_json_fence(stripped)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fence_match is not None:
        return json.loads(fence_match.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("LLM returned no JSON object")


def _replicate_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        return "".join(str(part) for part in output)
    return str(output)


def _parse_openai(
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    model: str,
    prompt_cache_key: str | None,
) -> T:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_model,
    }
    if prompt_cache_key is not None:
        kwargs["prompt_cache_key"] = prompt_cache_key
    completion = _openai_client().beta.chat.completions.parse(**kwargs)
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal or "LLM returned no parsed response")
    return message.parsed


def _parse_replicate(
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    model: str,
    image_png: bytes,
) -> T:
    system, user = _messages_text(messages)
    output = _replicate_client().run(
        model,
        input={
            "image": io.BytesIO(image_png),
            "prompt": user,
            "system_prompt": system,
            "temperature": REPLICATE_TEMPERATURE,
            "top_p": REPLICATE_TOP_P,
            "max_new_tokens": REPLICATE_MAX_NEW_TOKENS,
        },
    )
    payload = _extract_json(_replicate_text(output))
    return response_model.model_validate(payload)


def parse_structured_messages(
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    model: str = EXTRACTION_MODEL,
    prompt_cache_key: str | None = None,
    image_png: bytes | None = None,
) -> T:
    """Call the configured vision LLM and parse a Pydantic response schema."""
    if LLM_PROVIDER == "replicate":
        if image_png is None:
            raise ValueError("image_png is required when LLM_PROVIDER is replicate")
        return _parse_replicate(
            messages,
            response_model,
            model=REPLICATE_MODEL if model == EXTRACTION_MODEL else model,
            image_png=image_png,
        )
    return _parse_openai(
        messages,
        response_model,
        model=model,
        prompt_cache_key=prompt_cache_key,
    )
