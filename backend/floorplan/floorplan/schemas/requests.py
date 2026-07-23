"""HTTP request bodies for the floor plan API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from floorplan.schemas.models import FloorPlan


class RenderRequest(BaseModel):
    plan: FloorPlan
    size: int = 800
    layers: Optional[list[str]] = None
    highlight_room_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    plan: FloorPlan
    weights: Optional[dict[str, float]] = None


class GenerateRequest(BaseModel):
    bedroom_count: int = Field(default=3, ge=1, le=8)
    target_area: Optional[float] = Field(default=None, gt=0)
    seed: int = 0
    include_renders: bool = True
    save: bool = True


class UpdateConfigRequest(BaseModel):
    score_level: Optional[str] = None
    score_threshold: Optional[float] = Field(default=None, ge=0, le=10)
    generate_max_attempts: Optional[int] = Field(default=None, ge=1)
    target_areas: Optional[str] = None
    connection_graph: Optional[str] = None
    centers: Optional[str] = None
