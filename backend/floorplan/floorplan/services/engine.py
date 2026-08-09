"""High-level FloorPlanEngine orchestrating V1 capabilities."""

from __future__ import annotations

from typing import Any, Optional

from floorplan.evaluator import score_plan
from floorplan.renderer import render_plan
from floorplan.schemas.models import FloorPlan


class FloorPlanEngine:
    """Residential floor plan engine."""

    def score(self, plan: FloorPlan, weights: Optional[dict[str, float]] = None):
        return score_plan(plan, weights=weights)

    def render(self, plan: FloorPlan, **kwargs) -> dict[str, Any]:
        return render_plan(plan, **kwargs)

    def generate_sample(self, **kwargs):
        from dataset import generate_sample

        return generate_sample(**kwargs)

    def list_saved_plans(self):
        from dataset import list_saved_plans

        return list_saved_plans()
