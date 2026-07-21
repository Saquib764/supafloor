"""Mutation engine — diversify floor plans, then validate/repair/accept."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Optional

from thefluxtrain.floorplan.evaluator import hard_failures, is_valid, score_plan
from thefluxtrain.floorplan.geometry.ops import (
    area,
    bounding_box,
    rect,
    rotate,
    scale,
    to_points,
    translate,
)
from thefluxtrain.floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from thefluxtrain.floorplan.schemas.models import Boundary, Door, FloorPlan, Point2D, Room, Window, _uid


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


def mutate_mirror(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    """Mirror across vertical centerline of boundary."""
    minx, miny, maxx, maxy = bounding_box(plan.boundary.coords())
    cx = (minx + maxx) / 2

    def mirror_coords(coords):
        return [(2 * cx - x, y) for x, y in coords]

    rooms = [
        room.model_copy(update={"polygon": to_points(mirror_coords(room.coords()))})
        for room in plan.rooms
    ]
    boundary = plan.boundary.model_copy(update={"exterior": to_points(mirror_coords(plan.boundary.coords()))})
    doors = [
        d.model_copy(update={"position": Point2D(x=2 * cx - d.position.x, y=d.position.y)})
        for d in plan.doors
    ]
    windows = [
        w.model_copy(update={"position": Point2D(x=2 * cx - w.position.x, y=w.position.y)})
        for w in plan.windows
    ]
    return _new_plan(
        name=plan.name,
        boundary=boundary,
        rooms=rooms,
        walls=plan.walls,
        doors=doors,
        windows=windows,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "mirror"},
    )


def mutate_rotate(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    angle = rng.choice([90, 180, 270])
    origin = "centroid"

    def rot(coords):
        return rotate(coords, angle, origin=origin)

    rooms = [r.model_copy(update={"polygon": to_points(rot(r.coords()))}) for r in plan.rooms]
    boundary = plan.boundary.model_copy(update={"exterior": to_points(rot(plan.boundary.coords()))})
    # Approximate door/window rotation via shapely point rotate would be better; skip precise for V1
    return _new_plan(
        name=plan.name,
        boundary=boundary,
        rooms=rooms,
        walls=plan.walls,
        doors=plan.doors,
        windows=plan.windows,
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
    # Place in corner of bedroom by carving conceptual overlay (non-destructive insert nearby)
    closet_coords = rect(minx + 0.1, miny + 0.1, cw, ch)
    # Shrink bedroom slightly conceptually: keep both (may overlap slightly — validator may reject)
    closet = Room(
        name=f"Closet-{bed.name}",
        room_type=RoomType.CLOSET,
        polygon=to_points(closet_coords),
        zone=ZoneType.PRIVATE,
    )
    # Offset closet outside bedroom if possible along -x
    alt = rect(minx - cw - 0.05, miny + 0.1, cw, ch)
    closet = Room(
        name=f"Closet-{bed.name}",
        room_type=RoomType.CLOSET,
        polygon=to_points(alt),
        zone=ZoneType.PRIVATE,
    )
    door = Door(
        wall_id="internal",
        position=Point2D(x=minx - 0.02, y=miny + 0.4),
        connects=[bed.id, closet.id],
    )
    return _new_plan(
        name=plan.name,
        boundary=plan.boundary,
        rooms=list(plan.rooms) + [closet],
        walls=plan.walls,
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_closet"},
    )


def mutate_insert_storage(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    minx, miny, maxx, maxy = bounding_box(plan.boundary.coords())
    coords = rect(maxx - 1.0, maxy - 1.2, 0.9, 1.0)
    storage = Room(name="Storage", room_type=RoomType.STORAGE, polygon=to_points(coords), zone=ZoneType.SERVICE)
    # Connect to nearest circulation if any
    circ = next((r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY, RoomType.LIVING)), None)
    doors = list(plan.doors)
    if circ:
        doors.append(
            Door(wall_id="internal", position=Point2D(x=maxx - 1.0, y=maxy - 0.6), connects=[circ.id, storage.id])
        )
    return _new_plan(
        name=plan.name,
        boundary=plan.boundary,
        rooms=list(plan.rooms) + [storage],
        walls=plan.walls,
        doors=doors,
        windows=plan.windows,
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
    # Attach outside living along max-y
    bcoords = rect(minx + 0.3, maxy, max(1.5, (maxx - minx) * 0.5), 1.2)
    balcony = Room(name="Balcony", room_type=RoomType.BALCONY, polygon=to_points(bcoords), zone=ZoneType.OUTDOOR)
    # Expand boundary to include balcony
    bminx, bminy, bmaxx, bmaxy = bounding_box(plan.boundary.coords())
    new_b = rect(bminx, bminy, bmaxx - bminx, max(bmaxy, maxy + 1.2) - bminy)
    boundary = Boundary(exterior=to_points(new_b))
    door = Door(
        wall_id="internal",
        position=Point2D(x=(minx + maxx) / 2, y=maxy),
        connects=[living.id, balcony.id],
    )
    win = Window(wall_id="ext", position=Point2D(x=(minx + maxx) / 2, y=maxy + 1.2), width=1.5)
    return _new_plan(
        name=plan.name,
        boundary=boundary,
        rooms=list(plan.rooms) + [balcony],
        walls=plan.walls,
        doors=list(plan.doors) + [door],
        windows=list(plan.windows) + [win],
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
    powder = Room(name="Powder", room_type=RoomType.POWDER, polygon=to_points(coords), zone=ZoneType.WET)
    door = Door(wall_id="internal", position=Point2D(x=maxx, y=miny + 0.7), connects=[hall.id, powder.id])
    return _new_plan(
        name=plan.name,
        boundary=plan.boundary,
        rooms=list(plan.rooms) + [powder],
        walls=plan.walls,
        doors=list(plan.doors) + [door],
        windows=plan.windows,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "mutation": "insert_toilet"},
    )


def mutate_move_wall(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    """Nudge a non-circulation room by a small translation (proxy for wall move)."""
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
            # reject, keep current
            continue
        current = mutated
        # accept first successful valid mutation
        return current, history, True
    return current, history, current.id != plan.id
