"""Furniture placement simulator — decreases score when placement is impossible."""

from __future__ import annotations

from dataclasses import dataclass

from thefluxtrain.floorplan.geometry import area, bounding_box, min_width
from thefluxtrain.floorplan.schemas.enums import RoomType, Severity
from thefluxtrain.floorplan.schemas.models import FloorPlan, RuleResult, Room


@dataclass(frozen=True)
class FurnitureItem:
    name: str
    width: float
    depth: float
    clearance: float = 0.4


FURNITURE_BY_ROOM: dict[RoomType, list[FurnitureItem]] = {
    RoomType.BEDROOM: [
        FurnitureItem("Bed", 1.6, 2.0, 0.5),
        FurnitureItem("Wardrobe", 1.2, 0.6, 0.4),
    ],
    RoomType.MASTER_BEDROOM: [
        FurnitureItem("Bed", 1.8, 2.0, 0.6),
        FurnitureItem("Wardrobe", 1.6, 0.6, 0.4),
    ],
    RoomType.LIVING: [
        FurnitureItem("Sofa", 2.2, 0.9, 0.6),
        FurnitureItem("TV", 1.2, 0.4, 1.5),
    ],
    RoomType.DINING: [
        FurnitureItem("Dining", 1.4, 0.9, 0.7),
    ],
    RoomType.KITCHEN: [
        FurnitureItem("Kitchen counter", 2.4, 0.6, 1.0),
    ],
    RoomType.BATHROOM: [
        FurnitureItem("Shower", 0.9, 0.9, 0.3),
        FurnitureItem("Toilet", 0.5, 0.7, 0.4),
        FurnitureItem("Sink", 0.6, 0.45, 0.4),
    ],
    RoomType.UTILITY: [
        FurnitureItem("Washing machine", 0.6, 0.6, 0.5),
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
