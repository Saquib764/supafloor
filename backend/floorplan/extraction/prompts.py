"""Load extraction prompt templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache
def _load_prompt(name: str) -> dict[str, Any]:
    path = _PROMPTS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def render_prompt(name: str = "v1", **variables: Any) -> tuple[str, str]:
    data = _load_prompt(name)
    system = str(data.get("system", "")).format(**variables)
    user = str(data.get("user", "")).format(**variables)
    return system, user
