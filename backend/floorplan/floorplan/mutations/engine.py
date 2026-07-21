"""Mutation engine — diversify floor plans, then validate/accept.

Creators always emit closed room polygons and door/window line segments.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

from floorplan.evaluator import hard_failures, score_plan
from floorplan.geometry.authors import make_door_line, make_room, make_window_line
from floorplan.geometry.ops import bounding_box, rect, to_points
from floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, _uid
from floorplan.services.reconcile import reconcile_plan


def _new_plan(**kwargs) -> FloorPlan:
    kwargs.pop("id", None)
    return FloorPlan(id=_uid(), **kwargs)


def mutate_insert_closet(plan: FloorPlan, rng: random.Random) -> FloorPlan:
    beds = [r for r in plan.rooms if r.room_type in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM)]
    if not beds:
        return plan
    bed = rng.choice(beds)
    minx, miny, _, _ = bounding_box(bed.coords())
    closet = make_room(
        f"Closet-{bed.name}",
        rect(minx - 0.85, miny + 0.1, 0.8, 0.6),
        room_type=RoomType.CLOSET,
        zone=ZoneType.PRIVATE,
    )
    door = make_door_line((minx - 0.05, miny + 0.2), (minx - 0.05, miny + 1.0), connects=[bed.id, closet.id])
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
        rect(maxx - 1.0, maxy - 1.2, 0.9, 1.0),
        room_type=RoomType.STORAGE,
        zone=ZoneType.SERVICE,
    )
    doors = list(plan.doors)
    circ = next((r for r in plan.rooms if r.room_type in (RoomType.HALLWAY, RoomType.ENTRY, RoomType.LIVING)), None)
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
    balcony = make_room(
        "Balcony",
        rect(minx + 0.3, maxy, max(1.5, (maxx - minx) * 0.5), 1.2),
        room_type=RoomType.BALCONY,
        zone=ZoneType.OUTDOOR,
    )
    bminx, bminy, bmaxx, bmaxy = bounding_box(plan.ensure_boundary().coords())
    boundary = Boundary(exterior=to_points(rect(bminx, bminy, bmaxx - bminx, max(bmaxy, maxy + 1.2) - bminy)))
    cx = (minx + maxx) / 2
    return _new_plan(
        name=plan.name,
        rooms=list(plan.rooms) + [balcony],
        doors=list(plan.doors)
        + [make_door_line((cx - 0.45, maxy), (cx + 0.45, maxy), connects=[living.id, balcony.id])],
        windows=list(plan.windows)
        + [make_window_line((cx - 0.75, maxy + 1.2), (cx + 0.75, maxy + 1.2), room_ids=[balcony.id])],
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
    powder = make_room("Powder", rect(maxx, miny, 1.2, 1.4), room_type=RoomType.POWDER, zone=ZoneType.WET)
    door = make_door_line((maxx, miny + 0.3), (maxx, miny + 1.1), connects=[hall.id, powder.id])
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


MUTATORS: dict[MutationOp, Callable[[FloorPlan, random.Random], FloorPlan]] = {
    MutationOp.INSERT_CLOSET: mutate_insert_closet,
    MutationOp.INSERT_STORAGE: mutate_insert_storage,
    MutationOp.INSERT_BALCONY: mutate_insert_balcony,
    MutationOp.INSERT_TOILET: mutate_insert_toilet,
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
        MutationOp.INSERT_CLOSET,
        MutationOp.INSERT_STORAGE,
        MutationOp.INSERT_BALCONY,
        MutationOp.INSERT_TOILET,
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
