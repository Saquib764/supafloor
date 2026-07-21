"""Notched boundary + parameters/generated document tests."""

import random

from floorplan.geometry.boundary import generate_notched_boundary, is_manhattan
from floorplan.geometry.ops import bounding_box, points_to_polygon
from floorplan.requirements import (
    build_document,
    build_requirements,
    document_to_plan,
    recreate_document,
    run_generation,
    sample_parameters,
    space_names_for,
)
from floorplan.requirements.areas import AREA_OPTIONS, area_options_for, name_root


def test_notched_boundary_is_manhattan_integer():
    rng = random.Random(7)
    target = 80.0
    boundary, meta = generate_notched_boundary(rng, target_area=target)
    coords = boundary.coords()
    assert all(float(x).is_integer() and float(y).is_integer() for x, y in coords)
    assert is_manhattan(coords)
    poly = points_to_polygon(coords)
    assert poly.is_valid
    assert poly.area > 0
    minx, miny, maxx, maxy = bounding_box(coords)
    bbox_area = (maxx - minx) * (maxy - miny)
    assert abs(bbox_area - meta["bbox_area"]) < 1e-6
    assert abs(bbox_area - target) / target < 0.35


def test_notched_boundary_deterministic():
    a, _ = generate_notched_boundary(random.Random(42), target_area=60)
    b, _ = generate_notched_boundary(random.Random(42), target_area=60)
    assert a.coords() == b.coords()


def test_name_root_and_area_library():
    assert name_root("Room 2") == "Room"
    assert name_root("Toilet") == "Toilet"
    assert len(area_options_for("Room 1")) == 3
    assert "Room" in AREA_OPTIONS


def test_space_names_include_room_n():
    names = space_names_for(3, random.Random(1))
    assert "Living" in names
    assert "Room 1" in names
    assert "Room 2" in names
    assert "Room 3" in names
    assert "Bathroom" in names


def test_build_requirements_thin():
    req = build_requirements(2, usable_area=85.0, rng=random.Random(2))
    assert req["total_rooms"] == 2
    assert "space_names" in req


def test_document_parameters_generated_split(tmp_path):
    doc = build_document(3, target_area=90, seed=11)
    assert "parameters" in doc and "generated" in doc
    assert "spaces" in doc["generated"]
    assert "rectangles" in doc["parameters"]
    assert "boundary" in doc["generated"]
    assert "spaces" not in doc["parameters"]

    again = run_generation(doc["parameters"])
    assert again["boundary"]["polygon"] == doc["generated"]["boundary"]["polygon"]
    assert again["spaces"] == doc["generated"]["spaces"]

    recreated = recreate_document(doc)
    assert recreated["generated"] == doc["generated"]

    plan = document_to_plan(doc)
    assert plan.doors == []
    assert plan.windows == []
    assert len(plan.rooms) >= 1
    for space in doc["generated"]["spaces"]:
        if space["polygon"]:
            assert is_manhattan([(p[0], p[1]) for p in space["polygon"]])


def test_sample_parameters_stable():
    a = sample_parameters(2, target_area=70, seed=5)
    b = sample_parameters(2, target_area=70, seed=5)
    assert a == b
