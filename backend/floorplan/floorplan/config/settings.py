"""Load and update engine settings from config.yaml at the service root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "score_level": "basic",
    "score_threshold": 8.0,
    "generate_max_attempts": 10,
    "layout_mode": "cpsat",
}

_SCORE_LEVELS = frozenset({"basic", "intermediate", "advance"})
_LAYOUT_MODES = frozenset({"cpsat", "llm"})


def load_config() -> dict[str, Any]:
    """Read config.yaml fresh each call so edits apply without restart."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return dict(_DEFAULTS)
    return {**_DEFAULTS, **raw}


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    level = str(cfg.get("score_level", _DEFAULTS["score_level"]))
    if level not in _SCORE_LEVELS:
        raise ValueError(f"score_level must be one of {sorted(_SCORE_LEVELS)}")

    mode = str(cfg.get("layout_mode", _DEFAULTS["layout_mode"]))
    if mode not in _LAYOUT_MODES:
        raise ValueError(f"layout_mode must be one of {sorted(_LAYOUT_MODES)}")

    threshold = float(cfg.get("score_threshold", _DEFAULTS["score_threshold"]))
    if not 0.0 <= threshold <= 10.0:
        raise ValueError("score_threshold must be between 0 and 10")

    attempts = int(cfg.get("generate_max_attempts", _DEFAULTS["generate_max_attempts"]))
    if attempts < 1:
        raise ValueError("generate_max_attempts must be >= 1")

    return {
        **cfg,
        "score_level": level,
        "score_threshold": threshold,
        "generate_max_attempts": attempts,
        "layout_mode": mode,
    }


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge score/layout updates into config.yaml and return the applied config."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}
    merged = _validate({**load_config(), **updates})
    for key in _DEFAULTS:
        raw[key] = merged[key]
    _CONFIG_PATH.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return load_config()
