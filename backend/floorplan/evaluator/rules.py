"""Architectural rule engine — every rule returns score, reason, severity, repair."""

from __future__ import annotations

from typing import Callable

from floorplan.geometry import (
    area,
    aspect_ratio,
    contains,
    min_width,
    perimeter,
    polygon_from_boundary,
    polygon_from_room,
    rectangularity,
)
from floorplan.schemas.enums import RoomType, Severity, ZoneType
from floorplan.schemas.models import FloorPlan, RuleResult
from floorplan.topology import (
    adjacency_via_doors,
    build_adjacency,
    exterior_touching_rooms,
    reachable,
)

RuleFn = Callable[[FloorPlan], RuleResult]


def _hard(rule_id: str, name: str, passed: bool, reason: str, repair: str | None = None, score: float | None = None) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        name=name,
        score=10.0 if passed else (score if score is not None else 0.0),
        reason=reason,
        severity=Severity.CRITICAL if not passed else Severity.INFO,
        repair_suggestion=None if passed else repair,
        hard=True,
        passed=passed,
    )


def _soft(rule_id: str, name: str, score: float, reason: str, repair: str | None = None, severity: Severity = Severity.INFO) -> RuleResult:
    s = max(0.0, min(10.0, score))
    if s < 4:
        severity = Severity.WARNING
    return RuleResult(
        rule_id=rule_id,
        name=name,
        score=s,
        reason=reason,
        severity=severity,
        repair_suggestion=repair if s < 7 else None,
        hard=False,
        passed=s >= 4.0,
    )


# ── Hard rules ──────────────────────────────────────────────────────────────


def rule_polygon_validity(plan: FloorPlan) -> RuleResult:
    bad = []
    for room in plan.rooms:
        poly = polygon_from_room(room)
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            bad.append(room.name)
    b = polygon_from_boundary(plan.boundary)
    if not b.is_valid or b.is_empty:
        bad.append("boundary")
    return _hard(
        "polygon_validity",
        "Polygon Validity",
        not bad,
        "All polygons valid" if not bad else f"Invalid polygons: {', '.join(bad)}",
        "Repair invalid polygons / remove self-intersections",
    )


def rule_room_overlap(plan: FloorPlan) -> RuleResult:
    overlaps = []
    rooms = plan.rooms
    for i, a in enumerate(rooms):
        pa = polygon_from_room(a).buffer(-0.02)
        for b in rooms[i + 1 :]:
            pb = polygon_from_room(b).buffer(-0.02)
            if pa.is_empty or pb.is_empty:
                continue
            inter = pa.intersection(pb)
            if not inter.is_empty and inter.area > 0.05:
                overlaps.append(f"{a.name}/{b.name}")
    return _hard(
        "room_overlap",
        "Room Overlap",
        not overlaps,
        "No significant overlaps" if not overlaps else f"Overlaps: {', '.join(overlaps)}",
        "Move walls to eliminate overlapping room areas",
    )


def rule_boundary_containment(plan: FloorPlan) -> RuleResult:
    boundary = polygon_from_boundary(plan.boundary).buffer(0.05)
    outside = []
    for room in plan.rooms:
        poly = polygon_from_room(room)
        if not boundary.covers(poly):
            # allow tiny numerical overflow
            diff = poly.difference(boundary)
            if diff.area > 0.1:
                outside.append(room.name)
    return _hard(
        "boundary_containment",
        "Boundary Containment",
        not outside,
        "All rooms inside boundary" if not outside else f"Outside boundary: {', '.join(outside)}",
        "Scale or shift rooms to fit within boundary",
    )


def rule_minimum_area(plan: FloorPlan) -> RuleResult:
    mins = {
        RoomType.LIVING: 12.0,
        RoomType.BEDROOM: 7.0,
        RoomType.MASTER_BEDROOM: 10.0,
        RoomType.KITCHEN: 4.0,
        RoomType.BATHROOM: 3.0,
        RoomType.DINING: 6.0,
    }
    fails = []
    for room in plan.rooms:
        m = mins.get(room.room_type)
        if m and area(room.coords()) < m:
            fails.append(f"{room.name} ({area(room.coords()):.1f}<{m})")
    return _hard(
        "minimum_area",
        "Minimum Area",
        not fails,
        "Areas OK" if not fails else f"Too small: {', '.join(fails)}",
        "Expand undersized rooms",
    )


def rule_minimum_width(plan: FloorPlan) -> RuleResult:
    mins = {
        RoomType.LIVING: 2.8,
        RoomType.BEDROOM: 2.2,
        RoomType.MASTER_BEDROOM: 2.6,
        RoomType.BATHROOM: 1.2,
        RoomType.KITCHEN: 1.6,
        RoomType.HALLWAY: 0.9,
    }
    fails = []
    for room in plan.rooms:
        m = mins.get(room.room_type)
        if m and min_width(room.coords()) < m * 0.85:
            fails.append(f"{room.name}")
    return _hard(
        "minimum_width",
        "Minimum Width",
        not fails,
        "Widths OK" if not fails else f"Too narrow: {', '.join(fails)}",
        "Widen narrow rooms by moving shared walls",
    )


def rule_reachability(plan: FloorPlan) -> RuleResult:
    seen = reachable(plan)
    missing = [r.name for r in plan.rooms if r.id not in seen and r.room_type != RoomType.BALCONY]
    return _hard(
        "reachability",
        "Reachability",
        not missing,
        "All rooms reachable" if not missing else f"Unreachable: {', '.join(missing)}",
        "Add doors connecting unreachable rooms to circulation",
    )


def rule_required_rooms(plan: FloorPlan) -> RuleResult:
    types = {r.room_type for r in plan.rooms}
    needed = [RoomType.LIVING]
    # bathroom or powder for non-studio-like
    if len(plan.rooms) >= 3:
        needed.append(RoomType.BATHROOM)
    missing = [t.value for t in needed if t not in types]
    # kitchen optional for pure bedroom suites but preferred
    return _hard(
        "required_rooms",
        "Required Rooms Exist",
        not missing,
        "Required rooms present" if not missing else f"Missing: {', '.join(missing)}",
        "Insert missing required rooms",
    )


def rule_doors_on_walls(plan: FloorPlan) -> RuleResult:
    if not plan.doors:
        return _hard("doors_on_walls", "Doors On Walls", False, "No doors defined", "Add doors on shared walls")
    # Soft check: doors should have connects
    bad = [d.id for d in plan.doors if not d.connects]
    return _hard(
        "doors_on_walls",
        "Doors On Walls",
        not bad,
        "Doors annotated" if not bad else f"Doors missing connections: {bad}",
        "Assign door-room connections",
    )


def rule_windows_on_exterior(plan: FloorPlan) -> RuleResult:
    exterior = set(exterior_touching_rooms(plan))
    bedrooms = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.LIVING)]
    need_light = [r for r in bedrooms if r.id not in exterior]
    # Windows presence
    has_windows = len(plan.windows) > 0
    passed = has_windows and len(need_light) <= len(bedrooms) // 2
    reason = "Exterior exposure OK" if passed else f"Poor exterior access for: {[r.name for r in need_light]}"
    return _hard(
        "windows_on_exterior",
        "Windows On Exterior",
        passed or len(bedrooms) == 0,
        reason if has_windows else "No windows defined",
        "Add windows on exterior walls of living/bedrooms",
    )


def rule_bathroom_accessible(plan: FloorPlan) -> RuleResult:
    baths = [r for r in plan.rooms if r.room_type in (RoomType.BATHROOM, RoomType.POWDER)]
    if not baths:
        return _hard("bathroom_accessible", "Bathroom Accessible", False, "No bathroom", "Insert bathroom")
    adj = adjacency_via_doors(plan)
    ok = True
    for b in baths:
        if not adj.get(b.id):
            ok = False
    return _hard(
        "bathroom_accessible",
        "Bathroom Accessible",
        ok,
        "Bathrooms connected" if ok else "Bathroom lacks door connections",
        "Connect bathroom to hall or bedroom",
    )


def rule_kitchen_accessible(plan: FloorPlan) -> RuleResult:
    kitchens = [r for r in plan.rooms if r.room_type == RoomType.KITCHEN]
    if not kitchens:
        return _hard("kitchen_accessible", "Kitchen Accessible", True, "No kitchen in program", None)
    adj = adjacency_via_doors(plan)
    ok = all(adj.get(k.id) for k in kitchens)
    return _hard(
        "kitchen_accessible",
        "Kitchen Accessible",
        ok,
        "Kitchen connected" if ok else "Kitchen not connected",
        "Connect kitchen to living/dining",
    )


# ── Soft rules ──────────────────────────────────────────────────────────────


def rule_living_size(plan: FloorPlan) -> RuleResult:
    living = [r for r in plan.rooms if r.room_type == RoomType.LIVING]
    if not living:
        return _soft("living_size", "Living Size", 3.0, "No living room", "Add living room")
    a = max(area(r.coords()) for r in living)
    score = min(10.0, a / 2.0)  # 20m2 -> 10
    return _soft("living_size", "Living Size", score, f"Living area {a:.1f} m²", "Expand living room")


def rule_bedroom_ratio(plan: FloorPlan) -> RuleResult:
    beds = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    living = [r for r in plan.rooms if r.room_type == RoomType.LIVING]
    if not beds or not living:
        return _soft("bedroom_ratio", "Bedroom Ratio", 5.0, "Incomplete program for ratio")
    bed_area = sum(area(r.coords()) for r in beds)
    liv_area = sum(area(r.coords()) for r in living)
    ratio = bed_area / max(liv_area, 1e-9)
    # Prefer bedrooms collectively ~0.8–1.8x living
    if 0.7 <= ratio <= 1.8:
        score = 9.0
    elif 0.5 <= ratio <= 2.2:
        score = 6.5
    else:
        score = 4.0
    return _soft("bedroom_ratio", "Bedroom Ratio", score, f"Bedroom/living area ratio {ratio:.2f}", "Rebalance bedroom vs living areas")


def rule_natural_light(plan: FloorPlan) -> RuleResult:
    exterior = set(exterior_touching_rooms(plan))
    need = [r for r in plan.rooms if r.room_type in (RoomType.LIVING, RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.DINING)]
    if not need:
        return _soft("natural_light", "Natural Light", 5.0, "No daylight-critical rooms")
    lit = sum(1 for r in need if r.id in exterior)
    score = 10.0 * lit / len(need)
    return _soft("natural_light", "Natural Light", score, f"{lit}/{len(need)} critical rooms on exterior", "Give bedrooms/living exterior walls")


def rule_privacy(plan: FloorPlan) -> RuleResult:
    public = {RoomType.LIVING, RoomType.DINING, RoomType.ENTRY, RoomType.KITCHEN}
    private = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
    adj = build_adjacency(plan)
    id_type = {r.id: r.room_type for r in plan.rooms}
    direct = 0
    for a, neighbors in adj.items():
        if id_type.get(a) in private:
            for b in neighbors:
                if id_type.get(b) in public and id_type.get(b) != RoomType.HALLWAY:
                    # living adjacent bedroom is mild privacy issue unless hall buffers
                    if id_type.get(b) in (RoomType.LIVING, RoomType.DINING):
                        direct += 1
    score = max(2.0, 10.0 - direct * 2.5)
    return _soft("privacy", "Privacy", score, f"Direct public-private adjacencies: {direct}", "Buffer bedrooms with hallway")


def rule_circulation(plan: FloorPlan) -> RuleResult:
    circ = [r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY)]
    total = sum(area(r.coords()) for r in plan.rooms) or 1.0
    circ_area = sum(area(r.coords()) for r in circ)
    pct = circ_area / total
    # Ideal 5–12%
    if 0.04 <= pct <= 0.14:
        score = 9.0
    elif pct < 0.02:
        score = 5.0
    elif pct > 0.20:
        score = 4.0
    else:
        score = 6.5
    return _soft("circulation", "Circulation Efficiency", score, f"Circulation {pct*100:.1f}% of area", "Reduce hallway or add clear entry path")


def rule_wet_core_compactness(plan: FloorPlan) -> RuleResult:
    wet = [r for r in plan.rooms if r.zone == ZoneType.WET or r.room_type in (RoomType.BATHROOM, RoomType.POWDER, RoomType.KITCHEN, RoomType.UTILITY)]
    if len(wet) < 2:
        return _soft("wet_core", "Wet Core Compactness", 7.0, "Single wet room")
    from floorplan.geometry import centroid, distance

    centers = [centroid(r.coords()) for r in wet]
    # average pairwise distance
    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            ax, ay = centers[i]
            bx, by = centers[j]
            dists.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    avg = sum(dists) / len(dists)
    score = max(2.0, 10.0 - avg * 1.5)
    return _soft("wet_core", "Wet Core Compactness", score, f"Avg wet-room distance {avg:.2f} m", "Cluster wet rooms for shared plumbing")


def rule_rectangularity(plan: FloorPlan) -> RuleResult:
    if not plan.rooms:
        return _soft("rectangularity", "Rectangularity", 0.0, "No rooms")
    scores = [rectangularity(r.coords()) for r in plan.rooms]
    avg = sum(scores) / len(scores)
    return _soft("rectangularity", "Construction Rectangularity", avg * 10, f"Avg rectangularity {avg:.2f}", "Simplify room shapes")


def rule_aspect_quality(plan: FloorPlan) -> RuleResult:
    bad = 0
    for room in plan.rooms:
        if room.room_type in (RoomType.CLOSET, RoomType.STORAGE, RoomType.HALLWAY, RoomType.BALCONY):
            continue
        ar = aspect_ratio(room.coords())
        if ar > 2.8:
            bad += 1
    score = max(2.0, 10.0 - bad * 2.0)
    return _soft("aspect_quality", "Room Proportions", score, f"Over-elongated rooms: {bad}", "Improve aspect ratios toward 1–2")


def rule_storage_adequacy(plan: FloorPlan) -> RuleResult:
    storage = [r for r in plan.rooms if r.room_type in (RoomType.CLOSET, RoomType.STORAGE)]
    beds = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    if not beds:
        return _soft("storage", "Storage Adequacy", 6.0, "No bedrooms to size storage against")
    score = min(10.0, 4.0 + len(storage) * 3.0)
    return _soft("storage", "Storage Adequacy", score, f"{len(storage)} storage spaces for {len(beds)} bedrooms", "Insert closet near bedrooms")


def rule_balcony_usability(plan: FloorPlan) -> RuleResult:
    bals = [r for r in plan.rooms if r.room_type == RoomType.BALCONY]
    if not bals:
        return _soft("balcony", "Balcony Usability", 6.0, "No balcony in plan")
    a = max(area(r.coords()) for r in bals)
    score = min(10.0, a * 1.5)
    return _soft("balcony", "Balcony Usability", score, f"Balcony {a:.1f} m²", "Enlarge balcony to ≥4 m²")


HARD_RULES: list[RuleFn] = [
    rule_polygon_validity,
    rule_room_overlap,
    rule_boundary_containment,
    rule_minimum_area,
    rule_minimum_width,
    rule_reachability,
    rule_required_rooms,
    rule_doors_on_walls,
    rule_windows_on_exterior,
    rule_bathroom_accessible,
    rule_kitchen_accessible,
]

SOFT_RULES: list[RuleFn] = [
    rule_living_size,
    rule_bedroom_ratio,
    rule_natural_light,
    rule_privacy,
    rule_circulation,
    rule_wet_core_compactness,
    rule_rectangularity,
    rule_aspect_quality,
    rule_storage_adequacy,
    rule_balcony_usability,
]


def evaluate_rules(plan: FloorPlan) -> list[RuleResult]:
    results = []
    for fn in HARD_RULES + SOFT_RULES:
        results.append(fn(plan))
    return results


def hard_failures(results: list[RuleResult]) -> list[RuleResult]:
    return [r for r in results if r.hard and not r.passed]


def is_valid(plan: FloorPlan) -> bool:
    return not hard_failures(evaluate_rules(plan))
