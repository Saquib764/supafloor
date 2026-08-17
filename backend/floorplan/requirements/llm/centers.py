"""Resolve room seed centers via deterministic sampling or LLM."""

from __future__ import annotations

import random
from typing import Any

from floorplan.config.pipeline import get_pipeline_config, prompt_id_for, substep_provider
from floorplan.llm._context import (
    connection_edges_table,
    format_user_metadata,
    spaces_table,
    target_areas_summary,
)
from floorplan.llm.client import openai_model, parse_structured
from floorplan.llm.schemas import CentersResponse
from floorplan.prompts import render_prompt
from floorplan.requirements.centers import (
    center_coverage_score,
    clamp_center_inside,
    improve_center_coverage,
    sample_centers,
)

_MIN_COVERAGE = 0.5


def _centers_post_process_enabled() -> bool:
    return bool(get_pipeline_config()["llm"].get("centers_post_process", True))


def _validate_llm_centers(
    parsed: CentersResponse,
    names: list[str],
    boundary_poly: Polygon,
) -> dict[str, list[float]]:
    name_set = set(names)
    seen: set[str] = set()
    out: dict[str, list[float]] = {}
    for entry in parsed.centers:
        if entry.name not in name_set:
            raise ValueError(f"LLM center for unknown space: {entry.name}")
        if entry.name in seen:
            raise ValueError(f"duplicate LLM center for: {entry.name}")
        seen.add(entry.name)
        x, y = clamp_center_inside((float(entry.x), float(entry.y)), boundary_poly)
        out[entry.name] = [round(x, 3), round(y, 3)]
    missing = name_set - seen
    if missing:
        raise ValueError(f"LLM centers missing spaces: {sorted(missing)}")
    return out


def _llm_pick_centers(
    names: list[str],
    doc: dict[str, Any],
    *,
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
    entry_point: list[float],
    entry_edge: str | None,
    connection_graph: dict[str, Any],
    layout_total_area_sqft: float,
    prompt_id: str,
) -> dict[str, list[float]]:
    req = doc["requirements"]
    sampling = doc["sampling"]
    summary = doc["summary"]
    minx, miny, maxx, maxy = boundary_poly.bounds
    system, user = render_prompt(
        prompt_id,
        layout_total_area_sqft=round(float(layout_total_area_sqft), 1),
        boundary_minx=round(float(minx), 1),
        boundary_miny=round(float(miny), 1),
        boundary_maxx=round(float(maxx), 1),
        boundary_maxy=round(float(maxy), 1),
        entry_x=round(float(entry_point[0]), 1),
        entry_y=round(float(entry_point[1]), 1),
        entry_edge=entry_edge or "unknown",
        bedroom_count=req.get("bedroom_count"),
        seed=sampling.get("seed", 0),
        plan_id=summary.get("id", ""),
        user_metadata_block=format_user_metadata(req),
        spaces_table=spaces_table(names),
        target_areas_summary=target_areas_summary(selected_areas),
        connection_edges_table=connection_edges_table(connection_graph),
    )
    parsed = parse_structured(system, user, CentersResponse)
    return _validate_llm_centers(parsed, names, boundary_poly)


def resolve_centers(
    names: list[str],
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
    entry_point: list[float],
    connection_graph: dict[str, Any],
    doc: dict[str, Any],
    rng: random.Random,
    *,
    entry_edge: str | None = None,
    layout_total_area_sqft: float | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Pick seed centers and return values plus execution metadata."""
    configured = substep_provider("sample_into_document", "centers")
    prompt_id = prompt_id_for("centers")
    model = openai_model()
    cfg = get_pipeline_config()
    fallback = cfg["llm"].get("fallback", "deterministic")
    post_process = _centers_post_process_enabled()

    meta: dict[str, Any] = {
        "prompt_id": prompt_id if configured == "llm" else None,
        "model": model if configured == "llm" else None,
        "source": "deterministic",
        "fallback": False,
        "post_processed": False,
        "centers_post_process": post_process,
    }

    if configured != "llm":
        centers = sample_centers(
            names,
            selected_areas,
            boundary_poly,
            entry_point,
            connection_graph,
            rng,
        )
        meta["coverage_score"] = round(center_coverage_score(centers, boundary_poly), 3)
        return centers, meta

    area = layout_total_area_sqft if layout_total_area_sqft is not None else float(boundary_poly.area)

    try:
        centers = _llm_pick_centers(
            names,
            doc,
            selected_areas=selected_areas,
            boundary_poly=boundary_poly,
            entry_point=entry_point,
            entry_edge=entry_edge,
            connection_graph=connection_graph,
            layout_total_area_sqft=area,
            prompt_id=prompt_id,
        )
        coverage_before = center_coverage_score(centers, boundary_poly)
        meta["coverage_score_before"] = round(coverage_before, 3)
        if coverage_before < _MIN_COVERAGE:
            raise ValueError(f"LLM centers coverage too low: {coverage_before:.3f}")

        if post_process:
            centers = improve_center_coverage(centers, boundary_poly)
            meta["post_processed"] = True

        meta.update(
            source="llm",
            coverage_score=round(center_coverage_score(centers, boundary_poly), 3),
        )
        return centers, meta
    except Exception as exc:
        if fallback != "deterministic":
            raise
        meta.update(
            source="deterministic_fallback",
            fallback=True,
            error=str(exc),
        )
        centers = sample_centers(
            names,
            selected_areas,
            boundary_poly,
            entry_point,
            connection_graph,
            rng,
        )
        meta["coverage_score"] = round(center_coverage_score(centers, boundary_poly), 3)
        return centers, meta
