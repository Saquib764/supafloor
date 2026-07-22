"""FastAPI routes for the floor plan engine."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from floorplan.dataset import delete_saved_plan, list_saved_plans, load_sample
from floorplan.schemas.requests import EvaluateRequest, GenerateRequest, RenderRequest
from floorplan.services.engine import FloorPlanEngine

router = APIRouter(prefix="/floorplan", tags=["floorplan"])
engine = FloorPlanEngine()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "floorplan-v1"}


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


@router.post("/import")
async def import_image(
    file: UploadFile = File(...),
    feet_per_pixel: float = 0.05,
) -> dict:
    data = await file.read()
    try:
        plan = engine.import_image(data, feet_per_pixel=feet_per_pixel)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return plan.model_dump()


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    sample = engine.generate_sample(
        total_rooms=body.total_rooms,
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
