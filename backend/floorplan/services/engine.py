"""High-level FloorPlanEngine orchestrating V1 capabilities."""

from __future__ import annotations

from typing import Any, Optional

from floorplan.evaluator import evaluate_rules, score_plan
from floorplan.importer import import_bw_image
from floorplan.mutations import apply_mutation, mutate_pipeline
from floorplan.renderer import render_plan
from floorplan.schemas.enums import MutationOp
from floorplan.schemas.models import Boundary, FloorPlan
from floorplan.services.boundary_adapt import adapt_to_boundary, generate_boundary
from floorplan.services.reconcile import reconcile_plan
from floorplan.templates import get_template, instantiate_template, list_templates
from floorplan.topology import build_adjacency, room_metrics


class FloorPlanEngine:
    """Deterministic residential floor plan engine (no LLM)."""

    def list_templates(self) -> list[dict]:
        return list_templates()

    def get_template(self, template_id: str):
        return get_template(template_id)

    def instantiate(self, template_id: str, width: float | None = None, depth: float | None = None) -> FloorPlan:
        return instantiate_template(template_id, width=width, depth=depth)

    def adapt(self, plan: FloorPlan, boundary: Boundary) -> FloorPlan:
        return adapt_to_boundary(plan, boundary)

    def reconcile(self, plan: FloorPlan) -> FloorPlan:
        return reconcile_plan(plan)

    def make_boundary(self, width: float = 32.81, depth: float = 26.25, shape: str = "rect") -> Boundary:
        return generate_boundary(width=width, depth=depth, shape=shape)

    def mutate(self, plan: FloorPlan, op: str, seed: int = 0) -> tuple[FloorPlan, dict]:
        return apply_mutation(plan, MutationOp(op), seed=seed)

    def mutate_pipeline(self, plan: FloorPlan, seed: int = 0):
        return mutate_pipeline(plan, seed=seed)

    def evaluate(self, plan: FloorPlan) -> list:
        return evaluate_rules(plan)

    def score(self, plan: FloorPlan, weights: Optional[dict[str, float]] = None):
        return score_plan(plan, weights=weights)

    def render(self, plan: FloorPlan, **kwargs) -> dict[str, Any]:
        return render_plan(plan, **kwargs)

    def import_image(self, image_bytes: bytes, **kwargs) -> FloorPlan:
        return import_bw_image(image_bytes, **kwargs)

    def generate_sample(self, **kwargs):
        from floorplan.dataset import generate_sample

        return generate_sample(**kwargs)

    def generate_dataset(self, n: int = 10, **kwargs):
        from floorplan.dataset import generate_dataset

        return generate_dataset(n=n, **kwargs)

    def inspect_room(self, plan: FloorPlan, room_id: str) -> dict[str, Any]:
        room = plan.room_by_id(room_id)
        if room is None:
            raise KeyError(room_id)
        adj = build_adjacency(plan)
        scores = score_plan(plan)
        room_rules = [r for r in scores.rules if room.name in r.reason or room_id in str(r.details)]
        return {
            **room_metrics(room),
            "adjacent_rooms": adj.get(room_id, []),
            "doors": [d.model_dump() for d in plan.doors if room_id in d.connects],
            "windows": [w.model_dump() for w in plan.windows],
            "individual_scores": [r.model_dump() for r in room_rules],
            "plan_overall": scores.overall,
        }
