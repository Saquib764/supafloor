"""Furniture placement simulator — decreases score when placement is impossible.

Furniture sizes are in feet.
"""

from __future__ import annotations

from dataclasses import dataclass

from floorplan.geometry import area, bounding_box, min_width
from floorplan.schemas.enums import RoomType, Severity
from floorplan.schemas.models import FloorPlan, RuleResult, Room


@dataclass(frozen=True)
class FurnitureItem:
    name: str
    width: float
    depth: float
    clearance: float = 1.3


FURNITURE_BY_ROOM: dict[RoomType, list[FurnitureItem]] = {
    RoomType.BEDROOM: [
        FurnitureItem("Bed", 5.25, 6.5, 1.6),
        FurnitureItem("Wardrobe", 4.0, 2.0, 1.3),
    ],
    RoomType.MASTER_BEDROOM: [
        FurnitureItem("Bed", 6.0, 6.5, 2.0),
        FurnitureItem("Wardrobe", 5.25, 2.0, 1.3),
    ],
    RoomType.LIVING: [
        FurnitureItem("Sofa", 7.2, 3.0, 2.0),
        FurnitureItem("TV", 4.0, 1.3, 5.0),
    ],
    RoomType.DINING: [
        FurnitureItem("Dining", 4.6, 3.0, 2.3),
    ],
    RoomType.KITCHEN: [
        FurnitureItem("Kitchen counter", 8.0, 2.0, 3.3),
    ],
    RoomType.BATHROOM: [
        FurnitureItem("Shower", 3.0, 3.0, 1.0),
        FurnitureItem("Toilet", 1.6, 2.3, 1.3),
        FurnitureItem("Sink", 2.0, 1.5, 1.3),
    ],
    RoomType.UTILITY: [
        FurnitureItem("Washing machine", 2.0, 2.0, 1.6),
    ],
}


def can_place(room: Room, item: FurnitureItem) -> bool:
    """Conservative fit check: item bbox + clearance must fit in room bbox / min width."""
    coords = room.coords()
    minx, miny, maxx, maxy = bounding_box(coords)
    rw, rh = maxx - minx, maxy - miny
    need_w = item.width + item.clearance
    need_d = item.depth + item.clearance
    fits_orient = (need_w <= rw and need_d <= rh) or (need_w <= rh and need_d <= rw)
    if not fits_orient:
        return False
    if min_width(coords) < min(item.width, item.depth) + item.clearance * 0.5:
        return False
    # Area proxy
    if area(coords) < item.width * item.depth * 2.2:
        return False
    return True


def score_furniture(plan: FloorPlan) -> RuleResult:
    total = 0
    ok = 0
    failed: list[str] = []
    for room in plan.rooms:
        items = FURNITURE_BY_ROOM.get(room.room_type, [])
        for item in items:
            total += 1
            if can_place(room, item):
                ok += 1
            else:
                failed.append(f"{item.name}@{room.name}")
    if total == 0:
        return RuleResult(
            rule_id="furniture",
            name="Furniture Usability",
            score=6.0,
            reason="No furniture templates applicable",
            severity=Severity.INFO,
            hard=False,
            passed=True,
        )
    score = 10.0 * ok / total
    return RuleResult(
        rule_id="furniture",
        name="Furniture Usability",
        score=round(score, 2),
        reason=f"Placed {ok}/{total} furniture items",
        severity=Severity.WARNING if score < 5 else Severity.INFO,
        repair_suggestion=("Enlarge rooms failing furniture: " + ", ".join(failed[:5])) if failed else None,
        hard=False,
        passed=score >= 4.0,
        details={"failed": failed},
    )
