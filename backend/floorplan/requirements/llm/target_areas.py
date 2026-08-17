"""Resolve per-room target areas via deterministic sampling or LLM."""

from __future__ import annotations

import random
from typing import Any

from config.pipeline import get_pipeline_config, prompt_id_for, substep_provider
from requirements.llm.client import openai_model, parse_structured
from requirements.llm.schemas import TargetAreasResponse
from requirements.prompts import render_prompt
from requirements.llm._context import format_user_metadata, spaces_table
from requirements.areas import area_options_for
from requirements.program import pick_selected_areas, scale_selected_areas


def _area_constraints_table(names: list[str]) -> str:
    rows = []
    for name in names:
        opts = area_options_for(name)
        rows.append(f"- {name}: {opts[0]:.0f}–{opts[2]:.0f} (presets {opts[0]:.0f}, {opts[1]:.0f}, {opts[2]:.0f})")
    return "\n".join(rows)


def _clamp_llm_areas(raw: dict[str, float], names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        if name not in raw:
            raise ValueError(f"LLM response missing space: {name}")
        opts = area_options_for(name)
        lo, hi = min(opts), max(opts)
        value = float(raw[name])
        if value <= 0:
            raise ValueError(f"Invalid area for {name}: {value}")
        out[name] = float(max(1, int(max(lo, min(hi, value)))))
    return out


def _llm_pick_areas(
    names: list[str],
    boundary_area: float,
    doc: dict[str, Any],
    boundary_params: dict[str, Any],
    prompt_id: str,
) -> dict[str, float]:
    req = doc["requirements"]
    sampling = doc["sampling"]
    summary = doc["summary"]
    requested = req.get("target_area_sqft")
    system, user = render_prompt(
        prompt_id,
        layout_total_area_sqft=round(float(boundary_area), 1),
        requested_target_area_sqft=requested if requested is not None else "unspecified",
        boundary_aspect=round(float(boundary_params.get("aspect", 1.0)), 3),
        notch_count=len(boundary_params.get("notches") or []),
        bedroom_count=req.get("bedroom_count"),
        seed=sampling.get("seed", 0),
        plan_id=summary.get("id", ""),
        user_metadata_block=format_user_metadata(req),
        spaces_table=spaces_table(names),
        area_constraints_table=_area_constraints_table(names),
    )
    parsed = parse_structured(system, user, TargetAreasResponse)
    raw = {entry.name: entry.target_area_sqft for entry in parsed.spaces}
    return _clamp_llm_areas(raw, names)


def resolve_selected_areas(
    names: list[str],
    boundary_area: float,
    doc: dict[str, Any],
    rng: random.Random,
    *,
    boundary_params: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Pick target areas and return scaled values plus execution metadata."""
    configured = substep_provider("sample_into_document", "target_areas")
    prompt_id = prompt_id_for("target_areas")
    model = openai_model()
    cfg = get_pipeline_config()
    fallback = cfg["llm"].get("fallback", "deterministic")

    meta: dict[str, Any] = {
        "prompt_id": prompt_id if configured == "llm" else None,
        "model": model if configured == "llm" else None,
        "source": "deterministic",
        "fallback": False,
        "layout_total_area_sqft": round(float(boundary_area), 3),
    }

    if configured != "llm":
        raw = pick_selected_areas(names, rng)
        meta["pre_scale_areas"] = dict(raw)
        scaled = scale_selected_areas(raw, boundary_area)
        meta["post_scale_areas"] = dict(scaled)
        return scaled, meta

    try:
        raw = _llm_pick_areas(names, boundary_area, doc, boundary_params, prompt_id)
        meta["source"] = "llm"
        meta["pre_scale_areas"] = dict(raw)
    except Exception as exc:
        if fallback != "deterministic":
            raise
        meta["source"] = "deterministic_fallback"
        meta["fallback"] = True
        meta["error"] = str(exc)
        raw = pick_selected_areas(names, rng)
        meta["pre_scale_areas"] = dict(raw)

    scaled = scale_selected_areas(raw, boundary_area)
    meta["post_scale_areas"] = dict(scaled)
    return scaled, meta
