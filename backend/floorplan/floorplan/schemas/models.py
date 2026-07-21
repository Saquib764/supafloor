"""Immutable floor plan domain models.

A FloorPlan is primarily three separate lists:
  - rooms: each has a name and a closed, geometrically valid polygon
  - doors: line segments
  - windows: line segments
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from .enums import RoomType, ZoneType, Severity, MutationOp


def _uid() -> str:
    return uuid4().hex[:12]


class Point2D(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


def _close_ring(points: list[Point2D]) -> list[Point2D]:
    """Ensure ring is explicitly closed (first point repeated at end)."""
    if len(points) < 3:
        raise ValueError("Polygon requires at least 3 vertices")
    if points[0].as_tuple() != points[-1].as_tuple():
        return list(points) + [points[0]]
    return list(points)


def _open_ring(points: list[Point2D]) -> list[Point2D]:
    """Drop duplicate closing vertex for Shapely construction helpers."""
    if len(points) >= 2 and points[0].as_tuple() == points[-1].as_tuple():
        return list(points[:-1])
    return list(points)


def _validate_closed_polygon(points: list[Point2D]) -> list[Point2D]:
    closed = _close_ring(points)
    ring = [p.as_tuple() for p in _open_ring(closed)]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.geom_type != "Polygon" or poly.is_empty or poly.area <= 0:
            raise ValueError("Room polygon is not a geometrically valid closed polygon")
        ring = list(poly.exterior.coords)
        closed = [Point2D(x=float(x), y=float(y)) for x, y in ring]
        if closed[0].as_tuple() != closed[-1].as_tuple():
            closed = closed + [closed[0]]
    elif poly.is_empty or poly.area <= 0:
        raise ValueError("Room polygon must have positive area")
    return closed


class Boundary(BaseModel):
    """Exterior building/unit boundary polygon (closed, CCW preferred)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    exterior: list[Point2D]
    holes: list[list[Point2D]] = Field(default_factory=list)
    units: str = "m"

    @field_validator("exterior", mode="before")
    @classmethod
    def _close_exterior(cls, value: Any) -> list[Point2D]:
        pts = [Point2D.model_validate(p) if not isinstance(p, Point2D) else p for p in value]
        return _validate_closed_polygon(pts)

    @field_validator("holes", mode="before")
    @classmethod
    def _close_holes(cls, value: Any) -> list[list[Point2D]]:
        if not value:
            return []
        out: list[list[Point2D]] = []
        for hole in value:
            pts = [Point2D.model_validate(p) if not isinstance(p, Point2D) else p for p in hole]
            out.append(_validate_closed_polygon(pts))
        return out

    def coords(self) -> list[tuple[float, float]]:
        """Open ring coords (no duplicate closing vertex) for Shapely."""
        return [p.as_tuple() for p in _open_ring(self.exterior)]


class Room(BaseModel):
    """A room is a named closed polygon. Creators must always produce a valid ring."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    name: str
    polygon: list[Point2D]
    room_type: RoomType = RoomType.OTHER
    zone: ZoneType = ZoneType.PUBLIC
    level: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("polygon", mode="before")
    @classmethod
    def _ensure_closed_valid(cls, value: Any) -> list[Point2D]:
        pts = [Point2D.model_validate(p) if not isinstance(p, Point2D) else p for p in value]
        return _validate_closed_polygon(pts)

    def coords(self) -> list[tuple[float, float]]:
        """Open ring coords for Shapely (closing vertex stripped)."""
        return [p.as_tuple() for p in _open_ring(self.polygon)]


class LineSegment(BaseModel):
    """Geometric line used by doors and windows."""

    model_config = ConfigDict(frozen=True)

    start: Point2D
    end: Point2D

    @model_validator(mode="after")
    def _non_degenerate(self) -> LineSegment:
        if self.start.as_tuple() == self.end.as_tuple():
            raise ValueError("Line segment must have non-zero length")
        return self

    def coords(self) -> list[tuple[float, float]]:
        return [self.start.as_tuple(), self.end.as_tuple()]

    def as_linestring(self) -> LineString:
        return LineString(self.coords())

    def length(self) -> float:
        return float(self.as_linestring().length)

    def midpoint(self) -> Point2D:
        return Point2D(
            x=(self.start.x + self.end.x) / 2.0,
            y=(self.start.y + self.end.y) / 2.0,
        )


class Door(BaseModel):
    """Door opening modeled as a line segment on a wall edge."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    start: Point2D
    end: Point2D
    connects: list[str] = Field(default_factory=list)
    is_exterior: bool = False
    swing: str = "in"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_degenerate(self) -> Door:
        if self.start.as_tuple() == self.end.as_tuple():
            raise ValueError("Door line must have non-zero length")
        return self

    def as_line(self) -> LineSegment:
        return LineSegment(start=self.start, end=self.end)

    def midpoint(self) -> Point2D:
        return Point2D(x=(self.start.x + self.end.x) / 2.0, y=(self.start.y + self.end.y) / 2.0)

    def length(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float((dx * dx + dy * dy) ** 0.5)


class Window(BaseModel):
    """Window opening modeled as a line segment on an exterior edge."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    start: Point2D
    end: Point2D
    room_ids: list[str] = Field(default_factory=list)
    height: float = 1.4
    sill_height: float = 0.9
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_degenerate(self) -> Window:
        if self.start.as_tuple() == self.end.as_tuple():
            raise ValueError("Window line must have non-zero length")
        return self

    def as_line(self) -> LineSegment:
        return LineSegment(start=self.start, end=self.end)

    def midpoint(self) -> Point2D:
        return Point2D(x=(self.start.x + self.end.x) / 2.0, y=(self.start.y + self.end.y) / 2.0)

    def length(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float((dx * dx + dy * dy) ** 0.5)


class Wall(BaseModel):
    """Optional derived wall segment (not required for FloorPlan authorship)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    start: Point2D
    end: Point2D
    thickness: float = 0.15
    is_exterior: bool = False
    room_ids: list[str] = Field(default_factory=list)


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
    """Floor plan = separate lists of rooms, doors, and windows.

    Rooms own closed polygons. Doors and windows are independent line objects.
    Boundary / walls / zones are optional supporting data.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_uid)
    name: str = "untitled"
    rooms: list[Room] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    boundary: Optional[Boundary] = None
    walls: list[Wall] = Field(default_factory=list)
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

    def ensure_boundary(self) -> Boundary:
        """Return boundary, deriving from room union bbox if missing."""
        if self.boundary is not None:
            return self.boundary
        if not self.rooms:
            raise ValueError("FloorPlan has no boundary and no rooms")
        xs: list[float] = []
        ys: list[float] = []
        for room in self.rooms:
            for x, y in room.coords():
                xs.append(x)
                ys.append(y)
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        return Boundary(
            exterior=[
                Point2D(x=minx, y=miny),
                Point2D(x=maxx, y=miny),
                Point2D(x=maxx, y=maxy),
                Point2D(x=minx, y=maxy),
                Point2D(x=minx, y=miny),
            ]
        )


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
    aesthetics: float = 0.0
    overall: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    rules: list[RuleResult] = Field(default_factory=list)
    summary: str = ""
    report: str = ""


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
    boundary: Optional[Boundary] = None
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
