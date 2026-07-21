"""Tests for plan reconciliation."""

from floorplan.geometry.authors import make_door_line, make_room, make_window_line
from floorplan.geometry.ops import area, rect
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, Point2D
from floorplan.services.reconcile import reconcile_plan
from floorplan.evaluator import hard_failures, score_plan
from floorplan.dataset import generate_sample


def _boundary(w=10.0, d=8.0) -> Boundary:
    return Boundary(
        exterior=[
            Point2D(x=0, y=0),
            Point2D(x=w, y=0),
            Point2D(x=w, y=d),
            Point2D(x=0, y=d),
            Point2D(x=0, y=0),
        ]
    )


def test_reconcile_clips_room_outside_boundary_and_removes_empty():
    boundary = _boundary(10, 8)
    inside = make_room("Living", rect(0, 0, 6, 8), room_type=RoomType.LIVING)
    outside = make_room("Lost", rect(20, 20, 2, 2), room_type=RoomType.STORAGE, zone=ZoneType.SERVICE)
    plan = FloorPlan(name="t", rooms=[inside, outside], doors=[], windows=[], boundary=boundary)
    fixed = reconcile_plan(plan)
    assert len(fixed.rooms) == 1
    assert fixed.rooms[0].name == "Living"
    assert all(0 <= x <= 10 and 0 <= y <= 8 for x, y in fixed.rooms[0].coords())


def test_reconcile_carves_overlap_from_larger_room():
    boundary = _boundary(10, 8)
    big = make_room("Big", rect(0, 0, 6, 6), room_type=RoomType.LIVING)
    small = make_room("Small", rect(4, 0, 3, 3), room_type=RoomType.BEDROOM, zone=ZoneType.PRIVATE)
    # overlap area = 2x3 = 6
    plan = FloorPlan(
        name="t",
        rooms=[big, small],
        doors=[
            make_door_line((0.5, 0), (1.4, 0), connects=[big.id], is_exterior=True),
            make_door_line((5, 0), (5, 0.9), connects=[big.id, small.id]),
        ],
        windows=[make_window_line((0, 3), (0, 4.5), room_ids=[big.id])],
        boundary=boundary,
    )
    fixed = reconcile_plan(plan)
    assert len(fixed.rooms) == 2
    by_name = {r.name: r for r in fixed.rooms}
    # Bigger gave away overlap: big area should shrink; small keeps ~9
    assert area(by_name["Small"].coords()) == area(small.coords())
    assert area(by_name["Big"].coords()) < area(big.coords())
    # No significant overlap remains
    from floorplan.geometry.ops import intersection

    inter = intersection(by_name["Big"].coords(), by_name["Small"].coords())
    assert inter.area < 0.05


def test_reconcile_ensures_door_per_room_and_drops_hanging():
    boundary = _boundary(10, 8)
    a = make_room("A", rect(0, 0, 5, 8), room_type=RoomType.LIVING)
    b = make_room("B", rect(5, 0, 5, 8), room_type=RoomType.BEDROOM, zone=ZoneType.PRIVATE)
    hanging = make_door_line((50, 50), (51, 50), connects=["gone"])
    plan = FloorPlan(name="t", rooms=[a, b], doors=[hanging], windows=[], boundary=boundary)
    fixed = reconcile_plan(plan)
    assert all(d.length() > 0 for d in fixed.doors)
    assert hanging.id not in {d.id for d in fixed.doors} or all(
        "gone" not in d.connects for d in fixed.doors
    )
    covered = {c for d in fixed.doors for c in d.connects}
    for room in fixed.rooms:
        assert room.id in covered


def test_generate_sample_reconciles_boundary_and_overlap():
    sample = generate_sample(3, seed=1, include_renders=False, save=False)
    failures = {r.rule_id for r in hard_failures(sample.scores.rules)}
    assert "room_overlap" not in failures
    assert "boundary_containment" not in failures
    assert sample.boundary is not None
    assert sample.requirements["parameters"]["total_rooms"] == 3
    assert sample.floor_plan.doors == []
    assert sample.floor_plan.windows == []
    assert sample.requirements["generated"]["spaces"]
