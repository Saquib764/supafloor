"""HTTP request bodies for the floor plan API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from floorplan.schemas.models import Boundary, FloorPlan


class RenderRequest(BaseModel):
    plan: FloorPlan
    size: int = 800
    layers: Optional[list[str]] = None
    highlight_room_id: Optional[str] = None


class InstantiateRequest(BaseModel):
    template_id: str = "compact_2bed"
    width: float = 10.0
    depth: float = 8.0


class AdaptRequest(BaseModel):
    plan: FloorPlan
    boundary: Boundary


class MutateRequest(BaseModel):
    plan: FloorPlan
    op: str = "insert_closet"
    seed: int = 0


class EvaluateRequest(BaseModel):
    plan: FloorPlan
    weights: Optional[dict[str, float]] = None


class GenerateRequest(BaseModel):
    total_rooms: int = Field(default=3, ge=1, le=8)
    target_area: Optional[float] = Field(default=None, gt=0)
    seed: int = 0
    mutate: bool = False
    include_renders: bool = True
    save: bool = True


class RecreateRequest(BaseModel):
    include_renders: bool = True
    save: bool = True


class DatasetRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=100)
    total_rooms: Optional[int] = Field(default=None, ge=1, le=8)
    seed: int = 0
    include_renders: bool = False


class InspectRequest(BaseModel):
    plan: FloorPlan
    room_id: str
