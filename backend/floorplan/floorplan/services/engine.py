"""High-level FloorPlanEngine orchestrating V1 capabilities."""

from __future__ import annotations

from typing import Any, Optional

from floorplan.evaluator import score_plan
from floorplan.importer import import_bw_image
from floorplan.renderer import render_plan
from floorplan.schemas.models import FloorPlan


class FloorPlanEngine:
    """Deterministic residential floor plan engine (no LLM)."""

    def score(self, plan: FloorPlan, weights: Optional[dict[str, float]] = None):
        return score_plan(plan, weights=weights)

    def render(self, plan: FloorPlan, **kwargs) -> dict[str, Any]:
        return render_plan(plan, **kwargs)

    def import_image(self, image_bytes: bytes, **kwargs) -> FloorPlan:
        return import_bw_image(image_bytes, **kwargs)

    def generate_sample(self, **kwargs):
        from app.config import settings
        from floorplan.dataset import generate_sample

        kwargs.setdefault("score_level", settings.score_level)
        kwargs.setdefault("score_threshold", settings.score_threshold)
        kwargs.setdefault("max_attempts", settings.generate_max_attempts)
        return generate_sample(**kwargs)

    def list_saved_plans(self):
        from floorplan.dataset import list_saved_plans

        return list_saved_plans()
