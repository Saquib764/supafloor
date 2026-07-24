"""Natural-language layout prompts, schemas, and room JSON from floor plan geometry."""

from __future__ import annotations

import json
from typing import Any

from floorplan.geometry import area, bounding_box, centroid
from floorplan.schemas.models import FloorPlan

_REGION_BAND = 1 / 3


def _region_label(nx: float, ny: float) -> str:
    """Map normalized bbox position to a word region (no numbers)."""

    def band(value: float) -> int:
        if value < _REGION_BAND:
            return -1
        if value > 1 - _REGION_BAND:
            return 1
        return 0

    yb, xb = band(ny), band(nx)
    ns = {-1: "north", 1: "south"}.get(yb)
    ew = {-1: "west", 1: "east"}.get(xb)

    if ns and ew:
        return f"{ns}-{ew}"
    if ns:
        return "north-center" if xb == 0 else ns
    if ew:
        return "west-center" if yb == 0 else ew
    return "center"


def _join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _door_neighbors(plan: FloorPlan) -> dict[str, set[str]]:
    id_to_name = {room.id: room.name for room in plan.rooms}
    names = {room.name for room in plan.rooms}
    neighbors: dict[str, set[str]] = {room.name: set() for room in plan.rooms}
    for door in plan.doors:
        linked = [
            id_to_name.get(value, value)
            for value in door.connects
            if id_to_name.get(value, value) in names
        ]
        for i, left in enumerate(linked):
            for right in linked[i + 1 :]:
                neighbors[left].add(right)
                neighbors[right].add(left)
    return neighbors


def _round_coord(value: float) -> float:
    return round(float(value), 3)


def _point_row(x: float, y: float) -> dict[str, float]:
    return {"x": _round_coord(x), "y": _round_coord(y)}


def _selected_areas(plan: FloorPlan) -> dict[str, float]:
    raw = plan.metadata.get("selected_areas")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, area_value in raw.items():
        try:
            value = float(area_value)
        except (TypeError, ValueError):
            continue
        out[str(name)] = value
    return out


def build_layout_prompt(plan: FloorPlan) -> str:
    """Describe each room name and relative location in words only."""
    if not plan.rooms:
        return ""

    minx, miny, maxx, maxy = bounding_box(plan.ensure_boundary().coords())
    width = max(maxx - minx, 1e-9)
    height = max(maxy - miny, 1e-9)
    neighbors = _door_neighbors(plan)

    lines: list[str] = []
    for room in sorted(plan.rooms, key=lambda r: (centroid(r.coords())[1], centroid(r.coords())[0])):
        cx, cy = centroid(room.coords())
        region = _region_label((cx - minx) / width, (cy - miny) / height)
        line = f"{room.name} is in the {region}"
        linked = sorted(neighbors.get(room.name, ()))
        if linked:
            line += f", next to {_join_names(linked)}"
        lines.append(f"{line}.")

    return "\n".join(lines)


def build_layout_schema(plan: FloorPlan) -> str:
    """JSON schema: room target areas + boundary polygon (pretty-printed)."""
    if not plan.rooms:
        return ""

    targets = _selected_areas(plan)
    rooms = [
        {
            "name": room.name,
            "target_area_sqft": (
                _round_coord(targets[room.name]) if room.name in targets else None
            ),
        }
        for room in sorted(plan.rooms, key=lambda r: r.name)
    ]
    layout_polygon = [
        _point_row(p.x, p.y) for p in (plan.boundary.exterior if plan.boundary else [])
    ]
    return json.dumps({"rooms": rooms, "layout_polygon": layout_polygon}, indent=2)


def build_room_output(plan: FloorPlan) -> str:
    """JSON room polygons the model should emit (pretty-printed)."""
    if not plan.rooms:
        return "[]"

    rooms: list[dict[str, Any]] = []
    for room in sorted(plan.rooms, key=lambda r: r.name):
        row: dict[str, Any] = {
            "name": room.name,
            "actual_area_sqft": _round_coord(area(room.coords())),
            "polygon": [_point_row(p.x, p.y) for p in room.polygon],
        }
        if room.room_type is not None:
            row["room_type"] = room.room_type.value
        if room.zone is not None:
            row["zone"] = room.zone.value
        rooms.append(row)
    return json.dumps(rooms, indent=2)
