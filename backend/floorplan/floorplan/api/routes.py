"""FastAPI routes for the floor plan engine."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from floorplan.dataset import get_saved_plan, list_saved_plans, load_sample, recreate_sample
from floorplan.schemas.requests import (
    AdaptRequest,
    DatasetRequest,
    EvaluateRequest,
    GenerateRequest,
    InspectRequest,
    InstantiateRequest,
    MutateRequest,
    RecreateRequest,
    RenderRequest,
)
from floorplan.services.engine import FloorPlanEngine
from floorplan.topology import build_adjacency

router = APIRouter(prefix="/floorplan", tags=["floorplan"])
engine = FloorPlanEngine()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "floorplan-v1"}


@router.get("/templates")
def templates() -> list[dict]:
    return engine.list_templates()


@router.get("/templates/{template_id}")
def template_detail(template_id: str) -> dict:
    try:
        return engine.get_template(template_id).model_dump()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/instantiate")
def instantiate(body: InstantiateRequest) -> dict:
    try:
        plan = engine.instantiate(body.template_id, width=body.width, depth=body.depth)
        return plan.model_dump()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/adapt")
def adapt(body: AdaptRequest) -> dict:
    return engine.adapt(body.plan, body.boundary).model_dump()


@router.post("/mutate")
def mutate(body: MutateRequest) -> dict:
    try:
        plan, meta = engine.mutate(body.plan, body.op, seed=body.seed)
        return {"plan": plan.model_dump(), "meta": meta}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
    meters_per_pixel: float = 0.05,
) -> dict:
    data = await file.read()
    try:
        plan = engine.import_image(data, meters_per_pixel=meters_per_pixel)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return plan.model_dump()


@router.post("/generate")
def generate(body: GenerateRequest) -> dict:
    sample = engine.generate_sample(
        total_rooms=body.total_rooms,
        target_area=body.target_area,
        seed=body.seed,
        mutate=body.mutate,
        include_renders=body.include_renders,
        save=body.save,
    )
    return sample.model_dump()


@router.get("/plans")
def plans() -> list[dict]:
    return list_saved_plans()


@router.get("/plans/{plan_id}")
def plan_detail(plan_id: str) -> dict:
    try:
        return get_saved_plan(plan_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Plan not found: {plan_id}") from exc


@router.post("/plans/{plan_id}/load")
def plan_load(plan_id: str, include_renders: bool = True) -> dict:
    """Materialize FloorPlan from saved generated section (for listing/view)."""
    try:
        return load_sample(plan_id, include_renders=include_renders).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Plan not found: {plan_id}") from exc


@router.post("/plans/{plan_id}/recreate")
def plan_recreate(plan_id: str, body: RecreateRequest | None = None) -> dict:
    """Re-run generation from saved parameters (algorithm test path)."""
    body = body or RecreateRequest()
    try:
        sample = recreate_sample(plan_id, include_renders=body.include_renders, save=body.save)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Plan not found: {plan_id}") from exc
    return sample.model_dump()


@router.post("/dataset")
def dataset(body: DatasetRequest) -> dict:
    samples = engine.generate_dataset(
        n=body.n,
        total_rooms=body.total_rooms,
        seed=body.seed,
        include_renders=body.include_renders,
    )
    return {
        "count": len(samples),
        "samples": [
            {
                "id": s.id,
                "overall": s.scores.overall if s.scores else None,
                "valid": s.metadata.get("valid"),
                "total_rooms": (s.requirements.get("parameters") or {}).get("total_rooms"),
                "topology_hash": s.metadata.get("topology_hash"),
                "plan": s.floor_plan.model_dump(),
                "scores": s.scores.model_dump() if s.scores else None,
            }
            for s in samples
        ],
    }


@router.post("/inspect")
def inspect(body: InspectRequest) -> dict:
    try:
        return engine.inspect_room(body.plan, body.room_id)
    except KeyError as exc:
        raise HTTPException(404, f"Room not found: {exc}") from exc


@router.post("/demo")
def demo(total_rooms: int = 3, seed: int = 0) -> dict:
    """Convenience: generate from scratch + score + render for the viewer."""
    sample = engine.generate_sample(
        total_rooms=total_rooms,
        seed=seed,
        mutate=False,
        include_renders=True,
        save=False,
    )
    plan = sample.floor_plan
    scores = sample.scores
    rendered = engine.render(plan, layers=["boundary", "rooms", "labels", "doors", "windows", "topology"])
    return {
        "plan": plan.model_dump(),
        "scores": scores.model_dump() if scores else None,
        "render": rendered,
        "requirements": sample.requirements,
        "adjacency": build_adjacency(plan),
    }
