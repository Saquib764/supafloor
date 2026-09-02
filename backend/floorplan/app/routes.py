"""FastAPI routes for the floor plan engine."""

from __future__ import annotations

import json

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from dataset import delete_saved_plan, generate_sample, list_saved_plans, load_sample
from dataset.dataset_corpus import image_path, list_items, load_json
from export.render import export_realistic
from config import load_config, save_config
from evaluator import score_plan
from extraction.florence import (
    FlorenceUnavailableError,
    infer_automatic_image,
    infer_points_image,
    list_samples,
    sample_image_path,
)
from renderer import render_plan
from schemas.requests import (
    EvaluateRequest,
    GenerateRequest,
    RealisticExportRequest,
    RenderRequest,
    UpdateConfigRequest,
)

router = APIRouter(prefix="/floorplan", tags=["floorplan"])


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
    scores = score_plan(body.plan, weights=body.weights)
    return scores.model_dump()


@router.post("/render")
def render(body: RenderRequest) -> dict[str, Any]:
    """POST /floorplan/render → svg, png, metadata."""
    return render_plan(
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
            size=body.size,
            room_outline_thick=body.room_outline_thick,
            boundary_thick=body.boundary_thick,
            opening_thick=body.opening_thick,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    sample = generate_sample(
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


async def _read_image_source(
    file: UploadFile | None,
    sample: str | None,
) -> bytes:
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded file is empty")
        return data
    if sample:
        try:
            return sample_image_path(sample).read_bytes()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Sample not found: {sample}") from exc
    raise HTTPException(400, "Provide either file upload or sample name")


@router.get("/florence/samples")
def florence_samples() -> dict[str, Any]:
    return {"samples": list_samples()}


@router.get("/florence/samples/{name}")
def florence_sample_image(name: str) -> FileResponse:
    try:
        path = sample_image_path(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Sample not found: {name}") from exc
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media)


@router.post("/florence/automatic")
async def florence_automatic(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
) -> dict[str, Any]:
    data = await _read_image_source(file, sample)
    try:
        return infer_automatic_image(data)
    except FlorenceUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/florence/points")
async def florence_points(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    points: str = Form(...),
    rooms: str = Form("[]"),
) -> dict[str, Any]:
    try:
        point_list = json.loads(points)
        room_list = json.loads(rooms)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}") from exc

    if not isinstance(point_list, list) or not point_list:
        raise HTTPException(400, "points must be a non-empty JSON array")
    if not isinstance(room_list, list):
        raise HTTPException(400, "rooms must be a JSON array")

    data = await _read_image_source(file, sample)
    try:
        return infer_points_image(data, point_list, room_list)
    except FlorenceUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
