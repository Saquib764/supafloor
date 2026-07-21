"""Tests for closed room polygons and door/window line geometry."""

import pytest
from pydantic import ValidationError

from floorplan.geometry import make_door_line, make_room, make_window_line
from floorplan.schemas.enums import RoomType
from floorplan.schemas.models import Door, FloorPlan, Point2D, Room, Window
from floorplan.templates import instantiate_template


def test_room_polygon_is_closed_and_valid():
    room = make_room("Living", [(0, 0), (4, 0), (4, 3), (0, 3)], room_type=RoomType.LIVING)
    assert room.polygon[0].as_tuple() == room.polygon[-1].as_tuple()
    assert len(room.polygon) == 5
    assert room.name == "Living"


def test_room_rejects_degenerate_polygon():
    with pytest.raises(ValidationError):
        Room(name="Bad", polygon=[Point2D(x=0, y=0), Point2D(x=1, y=0)])


def test_door_and_window_are_lines():
    door = make_door_line((0, 0), (0.9, 0), connects=["a", "b"])
    window = make_window_line((0, 3), (1.2, 3), room_ids=["a"])
    assert door.length() == pytest.approx(0.9)
    assert window.length() == pytest.approx(1.2)
    assert door.midpoint().x == pytest.approx(0.45)
    with pytest.raises(ValidationError):
        Door(start=Point2D(x=1, y=1), end=Point2D(x=1, y=1))
    with pytest.raises(ValidationError):
        Window(start=Point2D(x=0, y=0), end=Point2D(x=0, y=0))


def test_instantiated_plan_structure():
    plan = instantiate_template("compact_2bed")
    assert plan.rooms
    assert plan.doors
    assert plan.windows
    for room in plan.rooms:
        assert room.polygon[0].as_tuple() == room.polygon[-1].as_tuple()
        assert len(room.polygon) >= 4
    for door in plan.doors:
        assert door.length() > 0
        assert {"start", "end"}.issubset(door.model_dump())
        assert "position" not in door.model_dump()
        assert "wall_id" not in door.model_dump()
    for win in plan.windows:
        assert win.length() > 0
        assert "position" not in win.model_dump()
        assert "wall_id" not in win.model_dump()
    # FloorPlan is rooms + doors + windows (+ optional boundary)
    dumped = plan.model_dump()
    assert "rooms" in dumped and "doors" in dumped and "windows" in dumped
