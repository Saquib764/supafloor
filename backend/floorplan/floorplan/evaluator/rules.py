"""Architectural rule engine — every rule returns score, reason, severity, repair."""

from __future__ import annotations

from collections import deque
from typing import Callable

from floorplan.geometry import (
    area,
    aspect_ratio,
    centroid,
    min_width,
    polygon_from_boundary,
    polygon_from_room,
    rectangularity,
)
from floorplan.schemas.enums import RoomType, Severity, ZoneType
from floorplan.schemas.models import FloorPlan, Room, RuleResult
from floorplan.topology import (
    adjacency_via_doors,
    build_adjacency,
    exterior_touching_rooms,
    reachable,
)

RuleFn = Callable[[FloorPlan], RuleResult]

_PUBLIC = {RoomType.LIVING, RoomType.DINING, RoomType.KITCHEN, RoomType.ENTRY}
_PRIVATE = {RoomType.BEDROOM, RoomType.MASTER_BEDROOM}
_CIRCULATION = {RoomType.HALLWAY, RoomType.ENTRY}
_SKIP_HUB = {RoomType.CLOSET, RoomType.STORAGE, RoomType.BALCONY, RoomType.OTHER}


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
    # Rooms are validated closed/positive on create; only re-check boundary here.
    bad: list[str] = []
    try:
        b = polygon_from_boundary(plan.ensure_boundary())
        if not b.is_valid or b.is_empty:
            bad.append("boundary")
    except ValueError:
        bad.append("boundary")
    return _hard(
        "polygon_validity",
        "Polygon Validity",
        not bad,
        "Boundary polygon valid" if not bad else f"Invalid polygons: {', '.join(bad)}",
        "Repair invalid boundary polygon",
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
    boundary = polygon_from_boundary(plan.ensure_boundary()).buffer(0.05)
    outside = []
    for room in plan.rooms:
        poly = polygon_from_room(room)
        if not boundary.covers(poly):
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
    """Doors must exist and connect rooms (rule id kept for score category map)."""
    if not plan.doors:
        return _hard("doors_on_walls", "Door Lines", False, "No doors defined", "Add door line segments")
    bad = [d.id for d in plan.doors if not d.connects]
    return _hard(
        "doors_on_walls",
        "Door Lines",
        not bad,
        "Doors have room connections" if not bad else f"Doors missing connects: {bad}",
        "Ensure each door has connects[]",
    )


def rule_windows_on_exterior(plan: FloorPlan) -> RuleResult:
    exterior = set(exterior_touching_rooms(plan))
    bedrooms = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.LIVING)]
    need_light = [r for r in bedrooms if r.id not in exterior]
    has_windows = bool(plan.windows)
    passed = has_windows and len(need_light) <= len(bedrooms) // 2
    reason = "Exterior exposure OK" if passed else f"Poor exterior access for: {[r.name for r in need_light]}"
    return _hard(
        "windows_on_exterior",
        "Window Lines",
        passed or len(bedrooms) == 0,
        reason if has_windows else "No window lines defined",
        "Add window line segments on exterior edges of living/bedrooms",
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


# ── Soft aesthetics (spatial layout conventions) ─────────────────────────────


def _largest(rooms: list[Room]) -> Room | None:
    return max(rooms, key=lambda r: area(r.coords()), default=None)


def _type_map(plan: FloorPlan) -> dict[str, RoomType]:
    return {r.id: r.room_type for r in plan.rooms}


def _hop_distances(adj: dict[str, list[str]], start: str) -> dict[str, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _shortest_hop(adj: dict[str, list[str]], starts: list[str], target: str) -> int | None:
    best: int | None = None
    for s in starts:
        d = _hop_distances(adj, s).get(target)
        if d is None:
            continue
        if best is None or d < best:
            best = d
    return best


def rule_living_centrality(plan: FloorPlan) -> RuleResult:
    """Living near plan center and within short door-hops of most rooms."""
    living = [r for r in plan.rooms if r.room_type == RoomType.LIVING]
    liv = _largest(living)
    if liv is None:
        return _soft("living_centrality", "Living Centrality", 3.0, "No living room", "Add a central living room")

    # Area-weighted plan centroid vs living centroid
    total_a = sum(area(r.coords()) for r in plan.rooms) or 1.0
    pcx = sum(centroid(r.coords())[0] * area(r.coords()) for r in plan.rooms) / total_a
    pcy = sum(centroid(r.coords())[1] * area(r.coords()) for r in plan.rooms) / total_a
    lcx, lcy = centroid(liv.coords())
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for r in plan.rooms:
        for x, y in r.coords():
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5 or 1.0
    geo_dist = ((lcx - pcx) ** 2 + (lcy - pcy) ** 2) ** 0.5
    geo_score = 10.0 * (1.0 - min(1.0, geo_dist / (0.45 * diag)))

    adj = adjacency_via_doors(plan)
    hops = _hop_distances(adj, liv.id)
    targets = [r for r in plan.rooms if r.id != liv.id and r.room_type not in _SKIP_HUB]
    reachable_hops = [hops[r.id] for r in targets if r.id in hops]
    if reachable_hops:
        mean_hops = sum(reachable_hops) / len(reachable_hops)
        hub_score = max(2.0, 10.0 - max(0.0, mean_hops - 1.0) * 3.5)
    else:
        mean_hops = 0.0
        hub_score = 5.0

    score = 0.45 * geo_score + 0.55 * hub_score
    return _soft(
        "living_centrality",
        "Living Centrality",
        score,
        f"Living {geo_dist:.1f} m from plan center; mean {mean_hops:.1f} hops to other rooms",
        "Place living nearer the center and connect rooms via hall/living hub",
    )


def rule_kitchen_public_link(plan: FloorPlan) -> RuleResult:
    """Kitchen should connect directly to dining or living (or one short hop via hall)."""
    kitchens = [r for r in plan.rooms if r.room_type == RoomType.KITCHEN]
    if not kitchens:
        return _soft("kitchen_public_link", "Kitchen–Public Link", 7.0, "No kitchen in program")

    adj = adjacency_via_doors(plan)
    types = _type_map(plan)
    living_ids = [r.id for r in plan.rooms if r.room_type == RoomType.LIVING]
    dining_ids = [r.id for r in plan.rooms if r.room_type == RoomType.DINING]
    scores: list[float] = []
    notes: list[str] = []

    for k in kitchens:
        nbrs = {types[n] for n in adj.get(k.id, []) if n in types}
        if RoomType.DINING in nbrs or RoomType.LIVING in nbrs:
            scores.append(10.0)
            notes.append(f"{k.name} opens to dining/living")
            continue
        # Path to dining then living
        best = None
        for tid in dining_ids + living_ids:
            h = _hop_distances(adj, k.id).get(tid)
            if h is not None and (best is None or h < best):
                best = h
        if best is None:
            scores.append(2.0)
            notes.append(f"{k.name} unreachable from living/dining")
        elif best <= 1:
            scores.append(9.0)
            notes.append(f"{k.name} adjacent to public room")
        elif best == 2 and RoomType.HALLWAY in nbrs:
            scores.append(7.0)
            notes.append(f"{k.name} 2 hops via hall to public")
        elif best == 2:
            scores.append(5.5)
            notes.append(f"{k.name} 2 hops from living/dining")
        else:
            scores.append(3.0)
            notes.append(f"{k.name} {best} hops from living/dining")

    score = sum(scores) / len(scores)
    return _soft(
        "kitchen_public_link",
        "Kitchen–Public Link",
        score,
        "; ".join(notes),
        "Connect kitchen directly to dining or living",
    )


def rule_entry_sequence(plan: FloorPlan) -> RuleResult:
    """Prefer exterior → entry → (hall) → living; avoid exterior into private rooms."""
    entries = [r for r in plan.rooms if r.room_type == RoomType.ENTRY]
    living = _largest([r for r in plan.rooms if r.room_type == RoomType.LIVING])
    adj = adjacency_via_doors(plan)
    types = _type_map(plan)

    exterior_doors = [d for d in plan.doors if d.is_exterior]
    exterior_room_ids: set[str] = set()
    for d in exterior_doors:
        exterior_room_ids.update(d.connects)
    if not exterior_room_ids:
        exterior_room_ids = set(exterior_touching_rooms(plan))

    private_hits = 0
    if exterior_doors:
        for d in exterior_doors:
            if any(types.get(c) in _PRIVATE for c in d.connects):
                private_hits += 1
    else:
        private_hits = sum(1 for rid in exterior_room_ids if types.get(rid) in _PRIVATE)

    if living is None:
        return _soft("entry_sequence", "Entry Sequence", 4.0, "No living room to sequence toward", "Add living after entry")

    score = 8.0
    reasons: list[str] = []

    if private_hits:
        score -= private_hits * 2.5
        reasons.append(f"exterior opens toward private room ({private_hits})")

    if entries:
        entry = _largest(entries) or entries[0]
        # Entry should touch exterior or receive an exterior door
        entry_on_exit = entry.id in exterior_room_ids or any(
            d.is_exterior and entry.id in d.connects for d in plan.doors
        )
        if entry_on_exit:
            score += 1.0
            reasons.append("entry at unit exit")
        else:
            score -= 1.5
            reasons.append("entry not at exterior")

        hops = _hop_distances(adj, entry.id).get(living.id)
        if hops is None:
            score -= 3.0
            reasons.append("living unreachable from entry")
        elif hops == 1:
            score += 1.0
            reasons.append("living adjacent to entry")
        elif hops == 2:
            # Prefer hall between entry and living
            nbrs = {types[n] for n in adj.get(entry.id, []) if n in types}
            if RoomType.HALLWAY in nbrs or RoomType.LIVING in nbrs:
                reasons.append("living 2 hops from entry via circulation")
            else:
                score -= 0.5
                reasons.append("living 2 hops from entry")
        else:
            score -= min(3.0, (hops - 2) * 1.5)
            reasons.append(f"living {hops} hops from entry")
    else:
        # No foyer: living itself at exit is acceptable; deep living is not
        if living.id in exterior_room_ids or any(d.is_exterior and living.id in d.connects for d in plan.doors):
            score -= 0.5
            reasons.append("no entry; living at exterior")
        else:
            starts = list(exterior_room_ids) or ([plan.rooms[0].id] if plan.rooms else [])
            hops = _shortest_hop(adj, starts, living.id)
            if hops is None:
                score -= 2.0
                reasons.append("no entry; living far from exit")
            elif hops <= 1:
                score -= 1.0
                reasons.append("no entry buffer before living")
            else:
                score -= 2.5
                reasons.append(f"no entry; living {hops} hops from exit")

    reason = "; ".join(reasons) if reasons else "Entry sequence ok"
    return _soft(
        "entry_sequence",
        "Entry Sequence",
        score,
        reason,
        "Add a small entry/foyer so exit leads to living, not bedrooms",
    )


def rule_hall_organization(plan: FloorPlan) -> RuleResult:
    """Private rooms should hang off hallway/entry rather than using living as a corridor."""
    beds = [r for r in plan.rooms if r.room_type in _PRIVATE]
    if not beds:
        return _soft("hall_organization", "Hall Organization", 7.0, "No bedrooms to organize")

    adj = adjacency_via_doors(plan)
    types = _type_map(plan)
    circ_ids = {r.id for r in plan.rooms if r.room_type in _CIRCULATION}
    living_ids = {r.id for r in plan.rooms if r.room_type == RoomType.LIVING}

    buffered = 0
    living_only = 0
    for b in beds:
        nbrs = set(adj.get(b.id, []))
        if nbrs & circ_ids:
            buffered += 1
        elif nbrs & living_ids and not (nbrs & circ_ids):
            living_only += 1

    # Living-as-corridor: path between two beds goes only through living (no hall on path)
    corridor_pairs = 0
    pair_count = 0
    for i, a in enumerate(beds):
        for b in beds[i + 1 :]:
            pair_count += 1
            # BFS recording parent
            parent: dict[str, str | None] = {a.id: None}
            q = deque([a.id])
            found = False
            while q and not found:
                cur = q.popleft()
                for nxt in adj.get(cur, []):
                    if nxt in parent:
                        continue
                    parent[nxt] = cur
                    if nxt == b.id:
                        found = True
                        break
                    q.append(nxt)
            if not found:
                continue
            path = []
            cur: str | None = b.id
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            mid = path[1:-1]
            mid_types = [types[x] for x in mid if x in types]
            if mid and all(t == RoomType.LIVING for t in mid_types) and RoomType.HALLWAY not in mid_types:
                corridor_pairs += 1

    frac_buffered = buffered / len(beds)
    score = 3.0 + frac_buffered * 6.0
    score -= living_only * 1.2
    if pair_count:
        score -= (corridor_pairs / pair_count) * 2.5
    if not circ_ids and len(beds) >= 2:
        score -= 1.5

    return _soft(
        "hall_organization",
        "Hall Organization",
        score,
        f"{buffered}/{len(beds)} bedrooms on hall/entry; {living_only} living-only; "
        f"{corridor_pairs} bed pairs routed through living",
        "Serve bedrooms from a hallway instead of crossing the living room",
    )


def rule_public_near_exit(plan: FloorPlan) -> RuleResult:
    """Living/public zone should sit closer to the exit than private bedrooms."""
    living = _largest([r for r in plan.rooms if r.room_type == RoomType.LIVING])
    beds = [r for r in plan.rooms if r.room_type in _PRIVATE]
    entries = [r for r in plan.rooms if r.room_type == RoomType.ENTRY]
    adj = adjacency_via_doors(plan)

    starts: list[str] = [e.id for e in entries]
    if not starts:
        exterior_doors = [d for d in plan.doors if d.is_exterior]
        for d in exterior_doors:
            starts.extend(d.connects)
    if not starts:
        starts = exterior_touching_rooms(plan)
    if not starts and plan.rooms:
        starts = [plan.rooms[0].id]

    if living is None:
        return _soft("public_near_exit", "Public Near Exit", 4.0, "No living room", "Place living near the exit")

    liv_hops = _shortest_hop(adj, starts, living.id)
    if liv_hops is None:
        return _soft(
            "public_near_exit",
            "Public Near Exit",
            3.0,
            "Living unreachable from exit/entry",
            "Connect living to the entry path",
        )

    bed_hops = []
    for b in beds:
        h = _shortest_hop(adj, starts, b.id)
        if h is not None:
            bed_hops.append(h)

    # Living ideally 0–1 hops from exit; bedrooms equal or deeper
    if liv_hops <= 1:
        score = 9.5
    elif liv_hops == 2:
        score = 7.0
    else:
        score = max(2.0, 6.0 - (liv_hops - 2) * 1.5)

    if bed_hops:
        min_bed = min(bed_hops)
        avg_bed = sum(bed_hops) / len(bed_hops)
        if liv_hops > min_bed:
            score -= min(3.5, (liv_hops - min_bed) * 2.0)
        elif liv_hops <= avg_bed:
            score = min(10.0, score + 0.5)
        reason = (
            f"Living {liv_hops} hops from exit; bedrooms min {min_bed} / avg {avg_bed:.1f} hops"
        )
    else:
        reason = f"Living {liv_hops} hops from exit (no bedrooms)"

    return _soft(
        "public_near_exit",
        "Public Near Exit",
        score,
        reason,
        "Keep living closer to the exit than bedrooms",
    )


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
    rule_living_centrality,
    rule_kitchen_public_link,
    rule_entry_sequence,
    rule_hall_organization,
    rule_public_near_exit,
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
