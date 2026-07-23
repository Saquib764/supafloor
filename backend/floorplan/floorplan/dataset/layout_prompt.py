"""Natural-language layout prompts from floor plan geometry."""

from __future__ import annotations

from floorplan.geometry import bounding_box, centroid
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
    ns = { -1: "north", 1: "south" }.get(yb)
    ew = { -1: "west", 1: "east" }.get(xb)

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
