"""FastAPI routes for the floor plan engine."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from floorplan.schemas.requests import (
    AdaptRequest,
    DatasetRequest,
    EvaluateRequest,
    GenerateRequest,
    InspectRequest,
    InstantiateRequest,
    MutateRequest,
    RenderRequest,
)
from floorplan.services.engine import FloorPlanEngine

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
        template_id=body.template_id,
        width=body.width,
        depth=body.depth,
        boundary_shape=body.boundary_shape,
        seed=body.seed,
        mutate=body.mutate,
        include_renders=body.include_renders,
    )
    return sample.model_dump()


@router.post("/dataset")
def dataset(body: DatasetRequest) -> dict:
    samples = engine.generate_dataset(
        n=body.n,
        template_ids=body.template_ids,
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
                "template": s.requirements.get("template_id"),
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
def demo(template_id: str = "compact_2bed") -> dict:
    """Convenience: instantiate + score + render for the viewer."""
    plan = engine.instantiate(template_id)
    scores = engine.score(plan)
    rendered = engine.render(plan, layers=["boundary", "rooms", "labels", "doors", "windows", "topology"])
    return {
        "plan": plan.model_dump(),
        "scores": scores.model_dump(),
        "render": rendered,
        "adjacency": __import__("floorplan.topology", fromlist=["build_adjacency"]).build_adjacency(plan),
    }
