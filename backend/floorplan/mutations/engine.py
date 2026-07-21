"""Mutation engine — diversify floor plans, then validate/accept.

Creators always emit closed room polygons and door/window line segments.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

from shapely import affinity
from shapely.geometry import Point

from floorplan.evaluator import hard_failures, score_plan
from floorplan.geometry.authors import make_door_line, make_room, make_window_line
from floorplan.geometry.ops import bounding_box, rect, rotate, scale, to_points, translate
from floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, Point2D, Room, _uid
from floorplan.services.reconcile import reconcile_plan


def _new_plan(**kwargs) -> FloorPlan:
    kwargs.pop("id", None)
    return FloorPlan(id=_uid(), **kwargs)


def _with_rooms(plan: FloorPlan, rooms: list[Room], **extra) -> FloorPlan:
    return plan.model_copy(update={"rooms": rooms, "id": _uid(), **extra})


def _mirror_point(p: Point2D, cx: float) -> Point2D:
    return Point2D(x=2 * cx - p.x, y=p.y)


def _rotate_point(p: Point2D, angle: float, origin: tuple[float, float]) -> Point2D:
    pt = affinity.rotate(Point(p.x, p.y), angle, origin=origin)
    return Point2D(x=float(pt.x), y=float(pt.y))


def _scale_room(
    plan: FloorPlan,
    rng: random.Random,
    *,
    types: set[RoomType] | None,
    factor_range: tuple[float, float],
) -> FloorPlan:
    if not plan.rooms:
        return plan
    candidates = [r for r in plan.rooms if types is None or r.room_type in types]
    if not candidates:
        return plan
    target = rng.choice(candidates)
    new_poly = scale(target.coords(), rng.uniform(*factor_range), origin="centroid")
    rooms = [
        r.model_copy(update={"polygon": to_points(new_poly)}) if r.id == target.id else r
        for r in plan.rooms
    ]
    return _with_rooms(plan, rooms)


def mutate_mirror(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    boundary = plan.ensure_boundary()
    minx, _, maxx, _ = bounding_box(boundary.coords())
    cx = (minx + maxx) / 2

    def mirror_coords(coords):
        return [(2 * cx - x, y) for x, y in coords]

    return _new_plan(
        name=plan.name,
        rooms=[r.model_copy(update={"polygon": to_points(mirror_coords(r.coords()))}) for r in plan.rooms],
        doors=[
            d.model_copy(update={"start": _mirror_point(d.start, cx), "end": _mirror_point(d.end, cx)})
            for d in plan.doors
        ],
        windows=[
            w.model_copy(update={"start": _mirror_point(w.start, cx), "end": _mirror_point(w.end, cx)})
            for w in plan.windows
        ],
        boundary=Boundary(exterior=to_points(mirror_coords(boundary.coords()))),
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "mirror"},
    )


def mutate_rotate(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    angle = rng.choice([90, 180, 270])
    boundary = plan.ensure_boundary()
    bx0, by0, bx1, by1 = bounding_box(boundary.coords())
    origin = ((bx0 + bx1) / 2, (by0 + by1) / 2)

    def rot(coords):
        return rotate(coords, angle, origin=origin)

    return _new_plan(
        name=plan.name,
        rooms=[r.model_copy(update={"polygon": to_points(rot(r.coords()))}) for r in plan.rooms],
        doors=[
            d.model_copy(
                update={
                    "start": _rotate_point(d.start, angle, origin),
                    "end": _rotate_point(d.end, angle, origin),
                }
            )
            for d in plan.doors
        ],
        windows=[
            w.model_copy(
                update={
                    "start": _rotate_point(w.start, angle, origin),
                    "end": _rotate_point(w.end, angle, origin),
                }
            )
            for w in plan.windows
        ],
        boundary=Boundary(exterior=to_points(rot(boundary.coords()))),
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "rotate"},
    )


def mutate_resize_room(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    return _scale_room(
        plan,
        rng,
        types={t for t in RoomType if t not in (RoomType.HALLWAY, RoomType.ENTRY)},
        factor_range=(0.92, 1.08),
    )


def mutate_expand_living(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    return _scale_room(plan, rng, types={RoomType.LIVING}, factor_range=(1.03, 1.12))


def mutate_reduce_hallway(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    return _scale_room(plan, rng, types={RoomType.HALLWAY}, factor_range=(0.85, 0.95))


def mutate_swap_rooms(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    swappable = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.STUDY)]
    if len(swappable) < 2:
        return plan
    a, b = rng.sample(swappable, 2)
    rooms = []
    for r in plan.rooms:
        if r.id == a.id:
            rooms.append(r.model_copy(update={"polygon": b.polygon}))
        elif r.id == b.id:
            rooms.append(r.model_copy(update={"polygon": a.polygon}))
        else:
            rooms.append(r)
    return _with_rooms(plan, rooms)


def mutate_insert_closet(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    beds = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    if not beds:
        return plan
    bed = rng.choice(beds)
    minx, miny, _, _ = bounding_box(bed.coords())
    closet = make_room(
        f"Closet-{bed.name}",
        rect(minx - 2.8, miny + 0.3, 2.6, 2.0),
        room_type=RoomType.CLOSET,
        zone=ZoneType.PRIVATE,
    )
    door = make_door_line((minx - 0.16, miny + 0.7), (minx - 0.16, miny + 3.3), connects=[bed.id, closet.id])
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [closet],
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        boundary=plan.boundary,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_closet"},
    )


def mutate_insert_storage(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    _, _, maxx, maxy = bounding_box(plan.ensure_boundary().coords())
    storage = make_room(
        "Storage",
        rect(maxx - 3.3, maxy - 3.9, 3.0, 3.3),
        room_type=RoomType.STORAGE,
        zone=ZoneType.SERVICE,
    )
    doors = list(plan.doors)
    circ = next((r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY, RoomType.LIVING)), None)
    if circ:
        doors.append(
            make_door_line((maxx - 3.3, maxy - 3.3), (maxx - 3.3, maxy - 0.7), connects=[circ.id, storage.id])
        )
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [storage],
        doors=doors,
        windows=plan.windows,
        boundary=plan.boundary,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_storage"},
    )


def mutate_insert_balcony(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    if any(r.room_type == RoomType.BALCONY for r in plan.rooms):
        return plan
    living = next((r for r in plan.rooms if r.room_type == RoomType.LIVING), None)
    if not living:
        return plan
    minx, _, maxx, maxy = bounding_box(living.coords())
    depth = 4.0
    balcony = make_room(
        "Balcony",
        rect(minx + 1.0, maxy, max(5.0, (maxx - minx) * 0.5), depth),
        room_type=RoomType.BALCONY,
        zone=ZoneType.OUTDOOR,
    )
    bminx, bminy, bmaxx, bmaxy = bounding_box(plan.ensure_boundary().coords())
    boundary = Boundary(exterior=to_points(rect(bminx, bminy, bmaxx - bminx, max(bmaxy, maxy + depth) - bminy)))
    cx = (minx + maxx) / 2
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [balcony],
        doors=list(plan.doors)
        + [make_door_line((cx - 1.5, maxy), (cx + 1.5, maxy), connects=[living.id, balcony.id])],
        windows=list(plan.windows)
        + [make_window_line((cx - 2.5, maxy + depth), (cx + 2.5, maxy + depth), room_ids=[balcony.id])],
        boundary=boundary,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_balcony"},
    )


def mutate_insert_toilet(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    if any(r.room_type == RoomType.POWDER for r in plan.rooms):
        return plan
    hall = next((r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY)), None)
    if not hall:
        return plan
    _, miny, maxx, _ = bounding_box(hall.coords())
    powder = make_room("Powder", rect(maxx, miny, 4.0, 4.6), room_type=RoomType.POWDER, zone=ZoneType.WET)
    door = make_door_line((maxx, miny + 1.0), (maxx, miny + 3.6), connects=[hall.id, powder.id])
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [powder],
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        boundary=plan.boundary,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_toilet"},
    )


def mutate_move_wall(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    """Nudge a non-entry room (legacy name; not a true wall edit)."""
    candidates = [r for r in plan.rooms if r.room_type != RoomType.ENTRY]
    if not candidates:
        return plan
    target = rng.choice(candidates)
    dx, dy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    new_poly = translate(target.coords(), dx, dy)
    rooms = [
        r.model_copy(update={"polygon": to_points(new_poly)}) if r.id == target.id else r
        for r in plan.rooms
    ]
    return _with_rooms(plan, rooms)


MUTATORS: dict[MutationOp, Callable[[FloorPlan, random.Random], FloorPlan]] = {
    MutationOp.MIRROR: mutate_mirror,
    MutationOp.ROTATE: mutate_rotate,
    MutationOp.RESIZE_ROOM: mutate_resize_room,
    MutationOp.EXPAND_LIVING: mutate_expand_living,
    MutationOp.REDUCE_HALLWAY: mutate_reduce_hallway,
    MutationOp.SWAP_ROOMS: mutate_swap_rooms,
    MutationOp.INSERT_CLOSET: mutate_insert_closet,
    MutationOp.INSERT_STORAGE: mutate_insert_storage,
    MutationOp.INSERT_BALCONY: mutate_insert_balcony,
    MutationOp.INSERT_TOILET: mutate_insert_toilet,
    MutationOp.MOVE_WALL: mutate_move_wall,
    MutationOp.INSERT_UTILITY: mutate_insert_storage,
}


def apply_mutation(plan: FloorPlan, op: MutationOp, seed: Optional[int] = None) -> tuple[FloorPlan, dict[str, Any]]:
    rng = random.Random(seed)
    fn = MUTATORS.get(op)
    if fn is None:
        return plan, {"op": op.value, "applied": False, "reason": "unsupported"}
    try:
        mutated = fn(plan, rng)
        if mutated.id == plan.id:
            mutated = mutated.model_copy(update={"id": _uid()})
        return mutated, {"op": op.value, "applied": True, "seed": seed}
    except Exception as exc:  # noqa: BLE001
        return plan, {"op": op.value, "applied": False, "reason": str(exc)}


def mutate_pipeline(
    plan: FloorPlan,
    ops: Optional[list[MutationOp]] = None,
    *,
    seed: int = 0,
    max_attempts: int = 8,
    require_valid: bool = True,
) -> tuple[FloorPlan, list[dict[str, Any]], bool]:
    """Template → Mutation → Validation → Accept/Reject."""
    rng = random.Random(seed)
    ops = ops or [
        MutationOp.MIRROR,
        MutationOp.RESIZE_ROOM,
        MutationOp.EXPAND_LIVING,
        MutationOp.SWAP_ROOMS,
        MutationOp.MOVE_WALL,
    ]
    history: list[dict[str, Any]] = []
    current = plan
    for attempt in range(max_attempts):
        op = rng.choice(ops)
        mutated, meta = apply_mutation(current, op, seed=seed + attempt)
        meta["attempt"] = attempt
        if not meta.get("applied"):
            history.append(meta)
            continue
        mutated = reconcile_plan(mutated)
        scores = score_plan(mutated)
        valid = not hard_failures(scores.rules)
        meta["valid"] = valid
        meta["score"] = scores.overall
        meta["reconciled"] = True
        history.append(meta)
        if require_valid and not valid:
            continue
        return mutated, history, True
    return current, history, current.id != plan.id
