"""Convert extraction room polygons to FloorPlan domain JSON."""

from __future__ import annotations

import re
from typing import Any

from floorplan.geometry.ops import area, aspect_ratio, bounding_box, to_points
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, Room
from floorplan.topology import build_adjacency


def _room_type_from_name(name: str) -> RoomType:
    n = name.lower()
    if "master" in n and "bed" in n:
        return RoomType.MASTER_BEDROOM
    if "bed" in n:
        return RoomType.BEDROOM
    if "powder" in n:
        return RoomType.POWDER
    if "bath" in n:
        return RoomType.BATHROOM
    if "kitchen" in n:
        return RoomType.KITCHEN
    if "living" in n:
        return RoomType.LIVING
    if "dining" in n:
        return RoomType.DINING
    if "closet" in n:
        return RoomType.CLOSET
    if "hall" in n or "corridor" in n:
        return RoomType.HALLWAY
    if "laundry" in n:
        return RoomType.LAUNDRY
    if "entry" in n or "foyer" in n:
        return RoomType.ENTRY
    if "study" in n or "office" in n:
        return RoomType.STUDY
    if "balcony" in n or "patio" in n:
        return RoomType.BALCONY
    if "storage" in n or "utility" in n:
        return RoomType.STORAGE
    return RoomType.OTHER


def _zone_for_type(room_type: RoomType) -> ZoneType:
    if room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.CLOSET):
        return ZoneType.PRIVATE
    if room_type in (RoomType.BATHROOM, RoomType.POWDER, RoomType.LAUNDRY):
        return ZoneType.WET
    if room_type == RoomType.HALLWAY:
        return ZoneType.CIRCULATION
    return ZoneType.PUBLIC


def _canvas_to_plan(cx: float, cy: float, canvas_size: int) -> tuple[float, float]:
    """Map extraction canvas coords (Y-down) to plan coords (Y-up)."""
    return float(cx), float(canvas_size + 1 - cy)


def _slug_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"room_{index + 1}"


def rooms_to_floorplan(
    rooms_raw: list[dict[str, Any]],
    *,
    canvas_size: int,
    meta: dict[str, Any] | None = None,
) -> FloorPlan:
    """Build a FloorPlan from extracted room polygons."""
    rooms: list[Room] = []
    for i, raw in enumerate(rooms_raw):
        coords = [_canvas_to_plan(p["x"], p["y"], canvas_size) for p in raw["polygon"]]
        room_type = _room_type_from_name(raw["name"])
        zone = _zone_for_type(room_type)
        rooms.append(
            Room(
                name=_slug_name(raw["name"], i),
                room_type=room_type,
                polygon=to_points(coords),
                zone=zone,
                metadata={
                    "label": raw["name"],
                    "area": area(coords),
                    "bbox": bounding_box(coords),
                    "aspect_ratio": aspect_ratio(coords),
                },
            )
        )

    boundary = Boundary(
        exterior=to_points(
            [
                (1, canvas_size),
                (canvas_size, canvas_size),
                (canvas_size, 1),
                (1, 1),
                (1, canvas_size),
            ]
        )
    )

    plan = FloorPlan(
        name="extracted",
        boundary=boundary,
        rooms=rooms,
        metadata={
            "source": "extraction",
            "canvas_size": canvas_size,
            "extraction": meta or {},
        },
    )

    adj = build_adjacency(plan)
    return plan.model_copy(update={"metadata": {**plan.metadata, "neighbours": adj}})
