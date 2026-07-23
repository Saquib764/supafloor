"""Load and update engine settings from config.yaml at the service root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from floorplan.config.pipeline import clear_pipeline_config_cache, get_pipeline_config

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "score_level": "basic",
    "score_threshold": 8.0,
    "generate_max_attempts": 10,
}

_PIPELINE_SUBSTEPS = ("target_areas", "connection_graph", "centers")
_PROVIDERS = frozenset({"deterministic", "llm"})
_SCORE_LEVELS = frozenset({"basic", "intermediate", "advance"})


def _pipeline_substeps() -> dict[str, str]:
    sample = get_pipeline_config()["pipeline"]["sample_into_document"]
    return {key: str(sample[key]) for key in _PIPELINE_SUBSTEPS}


def load_config() -> dict[str, Any]:
    """Read config.yaml fresh each call so edits apply without restart."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}
    cfg = {key: raw.get(key, _DEFAULTS[key]) for key in _DEFAULTS}
    return {**_validate(cfg), **_pipeline_substeps()}


def _validate_provider(value: Any, field: str) -> str:
    mode = str(value)
    if mode not in _PROVIDERS:
        raise ValueError(f"{field} must be one of {sorted(_PROVIDERS)}")
    return mode


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    level = str(cfg.get("score_level", _DEFAULTS["score_level"]))
    if level not in _SCORE_LEVELS:
        raise ValueError(f"score_level must be one of {sorted(_SCORE_LEVELS)}")

    threshold = float(cfg.get("score_threshold", _DEFAULTS["score_threshold"]))
    if not 0.0 <= threshold <= 10.0:
        raise ValueError("score_threshold must be between 0 and 10")

    attempts = int(cfg.get("generate_max_attempts", _DEFAULTS["generate_max_attempts"]))
    if attempts < 1:
        raise ValueError("generate_max_attempts must be >= 1")

    return {
        "score_level": level,
        "score_threshold": threshold,
        "generate_max_attempts": attempts,
    }


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge score and pipeline updates into config.yaml and return the applied config."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}

    merged = {**load_config(), **updates}
    validated = _validate({key: merged[key] for key in _DEFAULTS})
    for key in _DEFAULTS:
        raw[key] = validated[key]

    pipeline_updates = {key: _validate_provider(merged[key], key) for key in _PIPELINE_SUBSTEPS if key in updates}
    if pipeline_updates:
        raw.setdefault("pipeline", {})
        raw["pipeline"].setdefault("sample_into_document", {})
        raw["pipeline"]["sample_into_document"].update(pipeline_updates)
        clear_pipeline_config_cache()

    _CONFIG_PATH.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return load_config()
