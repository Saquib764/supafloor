"""Load and update engine settings from config.yaml at the service root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

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
    """Merge updates into config.yaml and return the applied config."""
    merged = _validate({**load_config(), **updates})
    # Keep only known keys plus any extras already present.
    known = set(_DEFAULTS)
    out = {k: merged[k] for k in merged if k in known or k in updates}
    for key, default in _DEFAULTS.items():
        out.setdefault(key, default)

    lines = [
        "# FloorPlan engine settings",
        "",
        "# Active scoring tier: basic | intermediate | advance",
        f"score_level: {out['score_level']}",
        "",
        "# Accept a generated plan when this level's overall score is >= threshold",
        f"score_threshold: {out['score_threshold']}",
        "",
        "# Max generation attempts when the level score is below threshold",
        f"generate_max_attempts: {out['generate_max_attempts']}",
        "",
        "# Layout generation mode: cpsat | llm",
        f"layout_mode: {out['layout_mode']}",
        "",
    ]
    _CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")
    return load_config()
