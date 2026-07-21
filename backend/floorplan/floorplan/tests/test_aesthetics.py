"""Aesthetics soft-rule scoring tests."""

from floorplan.evaluator import score_plan
from floorplan.evaluator.rules import (
    evaluate_rules,
    rule_entry_sequence,
    rule_hall_organization,
    rule_kitchen_public_link,
    rule_living_centrality,
    rule_public_near_exit,
)
from floorplan.geometry import make_door_line, make_room, rect
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import FloorPlan
from floorplan.templates import instantiate_template


AESTHETIC_IDS = {
    "living_centrality",
    "kitchen_public_link",
    "entry_sequence",
    "hall_organization",
    "public_near_exit",
}


def _good_layout() -> FloorPlan:
    """Entry → living hub; kitchen on living; bedrooms off hallway."""
    entry = make_room("Entry", rect(0, 3, 2, 2), room_type=RoomType.ENTRY, zone=ZoneType.CIRCULATION)
    living = make_room("Living", rect(2, 2, 5, 4), room_type=RoomType.LIVING, zone=ZoneType.PUBLIC)
    kitchen = make_room("Kitchen", rect(7, 2, 3, 2), room_type=RoomType.KITCHEN, zone=ZoneType.SERVICE)
    hall = make_room("Hall", rect(2, 0, 3, 2), room_type=RoomType.HALLWAY, zone=ZoneType.CIRCULATION)
    bed1 = make_room("Bed1", rect(0, 0, 2, 3), room_type=RoomType.BEDROOM, zone=ZoneType.PRIVATE)
    bed2 = make_room("Bed2", rect(5, 0, 3, 2), room_type=RoomType.BEDROOM, zone=ZoneType.PRIVATE)
    rooms = [entry, living, kitchen, hall, bed1, bed2]
    doors = [
        make_door_line((0.2, 5), (1.2, 5), connects=[entry.id], is_exterior=True),
        make_door_line((2, 4), (2, 4.8), connects=[entry.id, living.id]),
        make_door_line((7, 3), (7, 3.8), connects=[kitchen.id, living.id]),
        make_door_line((3.5, 2), (4.5, 2), connects=[hall.id, living.id]),
        make_door_line((2, 1), (2, 1.8), connects=[hall.id, bed1.id]),
        make_door_line((5, 1), (5, 1.8), connects=[hall.id, bed2.id]),
    ]
    return FloorPlan(name="good_aesthetics", rooms=rooms, doors=doors, windows=[])


def _bad_layout() -> FloorPlan:
    """Exterior into bedroom; kitchen isolated behind bed; living deep."""
    bed = make_room("Bed", rect(0, 0, 4, 4), room_type=RoomType.BEDROOM, zone=ZoneType.PRIVATE)
    living = make_room("Living", rect(8, 0, 4, 4), room_type=RoomType.LIVING, zone=ZoneType.PUBLIC)
    kitchen = make_room("Kitchen", rect(4, 4, 3, 3), room_type=RoomType.KITCHEN, zone=ZoneType.SERVICE)
    rooms = [bed, living, kitchen]
    doors = [
        make_door_line((0.5, 0), (1.5, 0), connects=[bed.id], is_exterior=True),
        make_door_line((4, 2), (4, 2.8), connects=[bed.id, living.id]),
        make_door_line((5.5, 4), (6.5, 4), connects=[kitchen.id, bed.id]),
    ]
    return FloorPlan(name="bad_aesthetics", rooms=rooms, doors=doors, windows=[])


def test_aesthetic_rules_registered_in_evaluate():
    plan = instantiate_template("compact_2bed")
    ids = {r.rule_id for r in evaluate_rules(plan)}
    assert AESTHETIC_IDS <= ids


def test_score_plan_includes_aesthetics_category():
    plan = instantiate_template("compact_2bed")
    scores = score_plan(plan)
    assert "aesthetics" in scores.weights
    assert 0.0 <= scores.aesthetics <= 10.0
    aesthetic_rules = [r for r in scores.rules if r.rule_id in AESTHETIC_IDS]
    assert len(aesthetic_rules) == 5
    assert "Aesthetics" in scores.report


def test_good_layout_scores_higher_than_bad():
    good = score_plan(_good_layout())
    bad = score_plan(_bad_layout())
    assert good.aesthetics > bad.aesthetics


def test_kitchen_public_link_prefers_living_adjacency():
    good = rule_kitchen_public_link(_good_layout())
    bad = rule_kitchen_public_link(_bad_layout())
    assert good.score > bad.score


def test_entry_sequence_penalizes_exterior_into_bedroom():
    good = rule_entry_sequence(_good_layout())
    bad = rule_entry_sequence(_bad_layout())
    assert good.score > bad.score


def test_hall_and_centrality_and_public_exit_run():
    plan = _good_layout()
    assert rule_living_centrality(plan).score >= 4.0
    assert rule_hall_organization(plan).score >= 5.0
    assert rule_public_near_exit(plan).score >= 6.0
