"""FastAPI routes for the floor plan engine."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from extraction.constants import CANVAS_SIZE
from extraction.extract import extract_rooms
from extraction.to_floorplan import rooms_to_floorplan
from floorplan.config import load_config, save_config
from dataset import delete_saved_plan, list_saved_plans, load_sample
from export.render import export_realistic
from floorplan.schemas.requests import (
    EvaluateRequest,
    GenerateRequest,
    RealisticExportRequest,
    RenderRequest,
    UpdateConfigRequest,
)
from floorplan.services.dataset_corpus import image_path, list_items, load_json
from floorplan.services.engine import FloorPlanEngine

router = APIRouter(prefix="/floorplan", tags=["floorplan"])
engine = FloorPlanEngine()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "floorplan-v1"}


@router.get("/config")
def get_config() -> dict[str, Any]:
    """Expose engine settings from config.yaml."""
    return load_config()


@router.put("/config")
def put_config(body: UpdateConfigRequest) -> dict[str, Any]:
    """Update config.yaml; takes effect immediately without restart."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return load_config()
    try:
        return save_config(updates)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"Failed to write config.yaml: {exc}") from exc


@router.post("/evaluate")
def evaluate(body: EvaluateRequest) -> dict:
    scores = engine.score(body.plan, weights=body.weights)
    return scores.model_dump()


@router.post("/render")
def render(body: RenderRequest) -> dict[str, Any]:
    """POST /floorplan/render → svg, png, metadata."""
    return engine.render(
        body.plan,
        size=body.size,
        layers=body.layers,
        highlight_room_id=body.highlight_room_id,
    )


@router.post("/export/realistic")
def export_realistic_route(body: RealisticExportRequest) -> dict[str, Any]:
    """POST /floorplan/export/realistic → 1024 PNG as base64."""
    if body.plan is None and not body.rooms:
        raise HTTPException(400, "Provide either plan or rooms")
    try:
        return export_realistic(
            plan=body.plan,
            rooms=body.rooms,
            canvas_size=body.canvas_size,
            size=body.size,
            room_outline_thick=body.room_outline_thick,
            boundary_thick=body.boundary_thick,
            opening_thick=body.opening_thick,
            symbol_probability=body.symbol_probability,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/extract")
async def extract_image(
    file: UploadFile = File(...),
    canvas_size: int = CANVAS_SIZE,
) -> dict[str, Any]:
    data = await file.read()
    try:
        rooms, meta, plot_png, debug_png = extract_rooms(data, canvas_size=canvas_size)
        floor_plan = rooms_to_floorplan(rooms, canvas_size=canvas_size, meta=meta)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "rooms": rooms,
        "meta": meta,
        "floor_plan": floor_plan.model_dump(),
        "plot_png_b64": base64.b64encode(plot_png).decode("ascii"),
        "debug_png_b64": base64.b64encode(debug_png).decode("ascii"),
    }


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    sample = engine.generate_sample(
        bedroom_count=body.bedroom_count,
        target_area=body.target_area,
        seed=body.seed,
        include_renders=body.include_renders,
        save=body.save,
    )
    return sample.model_dump()


@router.get("/plans")
def plans() -> list[dict]:
    return list_saved_plans()


@router.post("/plans/{plan_id}/load")
def plan_load(plan_id: str, include_renders: bool = True) -> dict:
    """Load saved plan JSON into FloorPlan + scores for the viewer."""
    try:
        return load_sample(plan_id, include_renders=include_renders).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Plan not found: {plan_id}") from exc


@router.delete("/plans/{plan_id}")
def plan_delete(plan_id: str) -> dict[str, str]:
    try:
        delete_saved_plan(plan_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Plan not found: {plan_id}") from exc
    return {"deleted": plan_id}


@router.get("/dataset/basic5000")
def dataset_basic5000_list(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    return list_items(offset, limit)


@router.get("/dataset/basic5000/{stem}")
def dataset_basic5000_json(stem: str) -> dict[str, Any]:
    try:
        return {"stem": stem, "json": load_json(stem)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Dataset item not found: {stem}") from exc


@router.get("/dataset/basic5000/{stem}/image")
def dataset_basic5000_image(stem: str) -> FileResponse:
    try:
        path = image_path(stem)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Dataset image not found: {stem}") from exc
    return FileResponse(path, media_type="image/png")
