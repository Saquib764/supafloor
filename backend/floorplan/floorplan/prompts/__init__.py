"""Central prompt library for LLM-backed pipeline substeps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent
_REGISTRY_PATH = _PROMPTS_DIR / "registry.yaml"


@lru_cache
def _load_registry() -> dict[str, Any]:
    data = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    return data.get("prompts") or {}


@lru_cache
def _load_prompt_file(relative_path: str) -> dict[str, Any]:
    path = _PROMPTS_DIR / relative_path
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_prompt(prompt_id: str) -> dict[str, Any]:
    entry = _load_registry().get(prompt_id)
    if not entry:
        raise KeyError(f"Unknown prompt id: {prompt_id}")
    data = _load_prompt_file(str(entry["file"]))
    data.setdefault("id", prompt_id)
    return data


def render_prompt(prompt_id: str, **variables: Any) -> tuple[str, str]:
    data = load_prompt(prompt_id)
    system = str(data.get("system", "")).format(**variables)
    user = str(data.get("user", "")).format(**variables)
    return system, user
