"""Load engine settings from config.yaml at the service root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "score_level": "basic",
    "score_threshold": 8.0,
    "generate_max_attempts": 10,
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return dict(_DEFAULTS)
    return {**_DEFAULTS, **raw}
