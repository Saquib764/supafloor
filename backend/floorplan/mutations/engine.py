"""Mutation engine — diversify floor plans, then validate/accept.

Creators always emit closed room polygons and door/window line segments.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

from shapely import affinity
from shapely.geometry import Point

from floorplan.evaluator import is_valid, score_plan
from floorplan.geometry.authors import make_door_line, make_room, make_window_line
from floorplan.geometry.ops import (
    bounding_box,
    rect,
    rotate,
    scale,
    to_points,
    translate,
)
from floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from floorplan.schemas.models import Boundary, Door, FloorPlan, Point2D, Room, Window, _uid


def _new_plan(**kwargs) -> FloorPlan:
    data = {k: v for k, v in kwargs.items() if k != "id"}
    data["id"] = _uid()
    return FloorPlan(**data)


def _with_rooms(plan: FloorPlan, rooms: list[Room], **extra) -> FloorPlan:
    data = plan.model_dump()
    data["rooms"] = [r.model_dump() for r in rooms]
    data.update(extra)
    data["id"] = _uid()
    return FloorPlan(**data)


def _boundary(plan: FloorPlan) -> Boundary:
    return plan.ensure_boundary()


def _mirror_point(p: Point2D, cx: float) -> Point2D:
    return Point2D(x=2 * cx - p.x, y=p.y)


def _rotate_point(p: Point2D, angle: float, origin: str | tuple[float, float] = "centroid") -> Point2D:
    # origin as centroid of a dummy needs plan context; callers pass absolute origin coords
    if origin == "centroid":
        raise ValueError("rotate_point requires concrete origin coordinates")
    pt = affinity.rotate(Point(p.x, p.y), angle, origin=origin)
    return Point2D(x=float(pt.x), y=float(pt.y))


def mutate_mirror(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    """Mirror across vertical centerline of boundary."""
    boundary = _boundary(plan)
    minx, miny, maxx, maxy = bounding_box(boundary.coords())
    cx = (minx + maxx) / 2

    def mirror_coords(coords):
        return [(2 * cx - x, y) for x, y in coords]

    rooms = [
        room.model_copy(update={"polygon": to_points(mirror_coords(room.coords()))})
        for room in plan.rooms
    ]
    new_boundary = Boundary(exterior=to_points(mirror_coords(boundary.coords())))
    doors = [
        d.model_copy(update={"start": _mirror_point(d.start, cx), "end": _mirror_point(d.end, cx)})
        for d in plan.doors
    ]
    windows = [
        w.model_copy(update={"start": _mirror_point(w.start, cx), "end": _mirror_point(w.end, cx)})
        for w in plan.windows
    ]
    return _new_plan(
        name=plan.name,
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=new_boundary,
        walls=plan.walls,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "mirror"},
    )


def mutate_rotate(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    angle = rng.choice([90, 180, 270])
    boundary = _boundary(plan)
    bx0, by0, bx1, by1 = bounding_box(boundary.coords())
    origin = ((bx0 + bx1) / 2, (by0 + by1) / 2)

    def rot(coords):
        return rotate(coords, angle, origin=origin)

    rooms = [r.model_copy(update={"polygon": to_points(rot(r.coords()))}) for r in plan.rooms]
    new_boundary = Boundary(exterior=to_points(rot(boundary.coords())))
    doors = [
        d.model_copy(
            update={
                "start": _rotate_point(d.start, angle, origin=origin),
                "end": _rotate_point(d.end, angle, origin=origin),
            }
        )
        for d in plan.doors
    ]
    windows = [
        w.model_copy(
            update={
                "start": _rotate_point(w.start, angle, origin=origin),
                "end": _rotate_point(w.end, angle, origin=origin),
            }
        )
        for w in plan.windows
    ]
    return _new_plan(
        name=plan.name,
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=new_boundary,
        walls=plan.walls,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "rotate"},
    )


def mutate_resize_room(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    if not plan.rooms:
        return plan
    candidates = [r for r in plan.rooms if r.room_type not in (RoomType.HALLWAY, RoomType.ENTRY)]
    if not candidates:
        candidates = list(plan.rooms)
    target = rng.choice(candidates)
    factor = rng.uniform(0.92, 1.08)
    new_poly = scale(target.coords(), factor, origin="centroid")
    rooms = []
    for r in plan.rooms:
        if r.id == target.id:
            rooms.append(r.model_copy(update={"polygon": to_points(new_poly)}))
        else:
            rooms.append(r)
    return _with_rooms(plan, rooms)


def mutate_expand_living(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    living = [r for r in plan.rooms if r.room_type == RoomType.LIVING]
    if not living:
        return plan
    target = living[0]
    new_poly = scale(target.coords(), rng.uniform(1.03, 1.12), origin="centroid")
    rooms = [r.model_copy(update={"polygon": to_points(new_poly)}) if r.id == target.id else r for r in plan.rooms]
    return _with_rooms(plan, rooms)


def mutate_reduce_hallway(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    halls = [r for r in plan.rooms if r.room_type == RoomType.HALLWAY]
    if not halls:
        return plan
    target = halls[0]
    new_poly = scale(target.coords(), rng.uniform(0.85, 0.95), origin="centroid")
    rooms = [r.model_copy(update={"polygon": to_points(new_poly)}) if r.id == target.id else r for r in plan.rooms]
    return _with_rooms(plan, rooms)


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
    minx, miny, maxx, maxy = bounding_box(bed.coords())
    cw, ch = 0.8, 0.6
    alt = rect(minx - cw - 0.05, miny + 0.1, cw, ch)
    closet = make_room(f"Closet-{bed.name}", alt, room_type=RoomType.CLOSET, zone=ZoneType.PRIVATE)
    door = make_door_line((minx - 0.05, miny + 0.2), (minx - 0.05, miny + 0.2 + 0.8), connects=[bed.id, closet.id])
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [closet],
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        boundary=plan.boundary,
        walls=plan.walls,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_closet"},
    )


def mutate_insert_storage(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    boundary = _boundary(plan)
    minx, miny, maxx, maxy = bounding_box(boundary.coords())
    coords = rect(maxx - 1.0, maxy - 1.2, 0.9, 1.0)
    storage = make_room("Storage", coords, room_type=RoomType.STORAGE, zone=ZoneType.SERVICE)
    circ = next((r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY, RoomType.LIVING)), None)
    doors = list(plan.doors)
    if circ:
        doors.append(
            make_door_line((maxx - 1.0, maxy - 1.0), (maxx - 1.0, maxy - 0.2), connects=[circ.id, storage.id])
        )
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [storage],
        doors=doors,
        windows=plan.windows,
        boundary=plan.boundary,
        walls=plan.walls,
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
    minx, miny, maxx, maxy = bounding_box(living.coords())
    bcoords = rect(minx + 0.3, maxy, max(1.5, (maxx - minx) * 0.5), 1.2)
    balcony = make_room("Balcony", bcoords, room_type=RoomType.BALCONY, zone=ZoneType.OUTDOOR)
    boundary = _boundary(plan)
    bminx, bminy, bmaxx, bmaxy = bounding_box(boundary.coords())
    new_b = rect(bminx, bminy, bmaxx - bminx, max(bmaxy, maxy + 1.2) - bminy)
    new_boundary = Boundary(exterior=to_points(new_b))
    cx = (minx + maxx) / 2
    door = make_door_line((cx - 0.45, maxy), (cx + 0.45, maxy), connects=[living.id, balcony.id])
    win = make_window_line((cx - 0.75, maxy + 1.2), (cx + 0.75, maxy + 1.2), room_ids=[balcony.id])
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [balcony],
        doors=list(plan.doors) + [door],
        windows=list(plan.windows) + [win],
        boundary=new_boundary,
        walls=plan.walls,
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
    minx, miny, maxx, maxy = bounding_box(hall.coords())
    coords = rect(maxx, miny, 1.2, 1.4)
    powder = make_room("Powder", coords, room_type=RoomType.POWDER, zone=ZoneType.WET)
    door = make_door_line((maxx, miny + 0.3), (maxx, miny + 1.1), connects=[hall.id, powder.id])
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [powder],
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        boundary=plan.boundary,
        walls=plan.walls,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_toilet"},
    )


def mutate_move_wall(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    """Nudge a non-entry room by a small translation (proxy for wall move)."""
    candidates = [r for r in plan.rooms if r.room_type not in (RoomType.ENTRY,)]
    if not candidates:
        return plan
    target = rng.choice(candidates)
    dx, dy = rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)
    new_poly = translate(target.coords(), dx, dy)
    rooms = [r.model_copy(update={"polygon": to_points(new_poly)}) if r.id == target.id else r for r in plan.rooms]
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
            data = mutated.model_dump()
            data["id"] = _uid()
            mutated = FloorPlan(**data)
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
        valid = is_valid(mutated)
        meta["valid"] = valid
        meta["score"] = score_plan(mutated).overall
        history.append(meta)
        if require_valid and not valid:
            continue
        current = mutated
        return current, history, True
    return current, history, current.id != plan.id
