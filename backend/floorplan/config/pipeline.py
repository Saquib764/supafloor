"""Load pipeline step/substep providers from config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

Provider = Literal["deterministic", "llm"]

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _SERVICE_ROOT / "config.yaml"

_SAMPLE_INTO_DOCUMENT_SUBSTEPS = (
    "boundary_params",
    "space_names",
    "target_areas",
    "entry_point",
    "connection_graph",
    "centers",
    "rectangles",
)


def _ensure_env_loaded() -> None:
    load_dotenv(_SERVICE_ROOT / ".env")


@lru_cache
def get_pipeline_config() -> dict[str, Any]:
    _ensure_env_loaded()
    if not _CONFIG_PATH.is_file():
        return _default_config()
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return _merge_defaults(data)


def clear_pipeline_config_cache() -> None:
    get_pipeline_config.cache_clear()


def _default_config() -> dict[str, Any]:
    return {
        "pipeline": {
            "sample_into_document": {name: "deterministic" for name in _SAMPLE_INTO_DOCUMENT_SUBSTEPS},
            "generate_geometry": "deterministic",
            "place_openings": "deterministic",
        },
        "llm": {
            "provider": "openai",
            "fallback": "deterministic",
            "timeout_seconds": 30,
            "centers_post_process": True,
            "prompts": {"target_areas": "target_areas/v1"},
        },
    }


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    base = _default_config()
    pipeline = data.get("pipeline") or {}
    sample = pipeline.get("sample_into_document") or {}
    merged_sample = dict(base["pipeline"]["sample_into_document"])
    merged_sample.update(sample)
    llm = dict(base["llm"])
    llm.update(data.get("llm") or {})
    prompts = dict(base["llm"]["prompts"])
    prompts.update((data.get("llm") or {}).get("prompts") or {})
    llm["prompts"] = prompts
    return {
        "pipeline": {
            "sample_into_document": merged_sample,
            "generate_geometry": pipeline.get("generate_geometry", base["pipeline"]["generate_geometry"]),
            "place_openings": pipeline.get("place_openings", base["pipeline"]["place_openings"]),
        },
        "llm": llm,
        "config_path": str(_CONFIG_PATH),
    }


def substep_provider(step: str, substep: str) -> Provider:
    cfg = get_pipeline_config()
    if step != "sample_into_document":
        value = cfg["pipeline"].get(step, "deterministic")
        return "llm" if value == "llm" else "deterministic"
    value = cfg["pipeline"]["sample_into_document"].get(substep, "deterministic")
    return "llm" if value == "llm" else "deterministic"


def prompt_id_for(substep: str) -> str:
    return str(get_pipeline_config()["llm"]["prompts"].get(substep, ""))


def pipeline_settings_snapshot() -> dict[str, Any]:
    cfg = get_pipeline_config()
    return {
        "config_path": cfg.get("config_path", str(_CONFIG_PATH)),
        "pipeline": cfg["pipeline"],
        "llm": {
            "provider": cfg["llm"].get("provider"),
            "fallback": cfg["llm"].get("fallback"),
            "timeout_seconds": cfg["llm"].get("timeout_seconds"),
            "centers_post_process": cfg["llm"].get("centers_post_process"),
            "prompts": cfg["llm"].get("prompts"),
        },
    }
