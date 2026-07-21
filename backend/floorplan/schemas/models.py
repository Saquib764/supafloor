"""Immutable floor plan domain models."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .enums import RoomType, ZoneType, Severity, MutationOp


def _uid() -> str:
    return uuid4().hex[:12]


class Point2D(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class Boundary(BaseModel):
    """Exterior building/unit boundary polygon (CCW preferred)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    exterior: list[Point2D]
    holes: list[list[Point2D]] = Field(default_factory=list)
    units: str = "m"

    def coords(self) -> list[tuple[float, float]]:
        return [p.as_tuple() for p in self.exterior]


class Room(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    name: str
    room_type: RoomType
    polygon: list[Point2D]
    zone: ZoneType = ZoneType.PUBLIC
    level: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def coords(self) -> list[tuple[float, float]]:
        return [p.as_tuple() for p in self.polygon]


class Wall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    start: Point2D
    end: Point2D
    thickness: float = 0.15
    is_exterior: bool = False
    room_ids: list[str] = Field(default_factory=list)


class Door(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    wall_id: str
    position: Point2D
    width: float = 0.9
    swing: str = "in"
    connects: list[str] = Field(default_factory=list)
    is_exterior: bool = False


class Window(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    wall_id: str
    position: Point2D
    width: float = 1.2
    height: float = 1.4
    sill_height: float = 0.9


class Zone(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    zone_type: ZoneType
    room_ids: list[str] = Field(default_factory=list)
    label: Optional[str] = None


class ConstraintGraph(BaseModel):
    """Logical adjacency / preference graph (topology, not coordinates)."""

    model_config = ConfigDict(frozen=True)

    nodes: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    required_exterior: list[str] = Field(default_factory=list)
    preferred_adjacency: list[tuple[str, str]] = Field(default_factory=list)


class FloorPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    name: str = "untitled"
    boundary: Boundary
    rooms: list[Room] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    constraint_graph: Optional[ConstraintGraph] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def room_by_id(self, room_id: str) -> Optional[Room]:
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None

    def rooms_of_type(self, room_type: RoomType) -> list[Room]:
        return [r for r in self.rooms if r.room_type == room_type]


class RuleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    name: str
    score: float
    reason: str
    severity: Severity
    repair_suggestion: Optional[str] = None
    hard: bool = False
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    geometry: float = 0.0
    topology: float = 0.0
    construction: float = 0.0
    privacy: float = 0.0
    circulation: float = 0.0
    furniture: float = 0.0
    daylight: float = 0.0
    architecture: float = 0.0
    overall: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    rules: list[RuleResult] = Field(default_factory=list)


class RoomRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    room_type: RoomType
    name: str
    required: bool = True
    min_area: float = 0.0
    preferred_area: Optional[float] = None
    preferred_zone: ZoneType = ZoneType.PUBLIC
    requires_exterior: bool = False
    min_width: float = 0.0


class TemplateSpec(BaseModel):
    """Logical apartment template — topology, not coordinates."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    name: str
    description: str = ""
    rooms: list[RoomRequirement] = Field(default_factory=list)
    preferred_adjacency: list[tuple[str, str]] = Field(default_factory=list)
    required_exterior: list[str] = Field(default_factory=list)
    wet_core_rooms: list[str] = Field(default_factory=list)
    expansion_points: list[str] = Field(default_factory=list)
    mutation_rules: list[MutationOp] = Field(default_factory=list)
    default_layout: Optional[dict[str, Any]] = None


class DatasetSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    boundary: Boundary
    requirements: dict[str, Any] = Field(default_factory=dict)
    topology: Optional[ConstraintGraph] = None
    floor_plan: FloorPlan
    scores: Optional[ScoreBreakdown] = None
    rule_breakdown: list[RuleResult] = Field(default_factory=list)
    mutation_history: list[dict[str, Any]] = Field(default_factory=list)
    svg: Optional[str] = None
    png_b64: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    approved: Optional[bool] = None
