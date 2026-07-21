"""Geometry operation tests."""

from thefluxtrain.floorplan.geometry import (
    area,
    aspect_ratio,
    bounding_box,
    centroid,
    contains,
    difference,
    distance,
    intersection,
    offset,
    rotate,
    scale,
    shared_edge,
    simplify,
    touches,
    translate,
    union,
)
from thefluxtrain.floorplan.geometry.ops import rect


def test_area_and_centroid():
    coords = rect(0, 0, 4, 3)
    assert abs(area(coords) - 12.0) < 1e-6
    cx, cy = centroid(coords)
    assert abs(cx - 2.0) < 1e-6
    assert abs(cy - 1.5) < 1e-6


def test_bbox_aspect():
    coords = rect(1, 2, 6, 2)
    assert bounding_box(coords) == (1.0, 2.0, 7.0, 4.0)
    assert abs(aspect_ratio(coords) - 3.0) < 1e-6


def test_boolean_ops():
    a = rect(0, 0, 4, 4)
    b = rect(2, 2, 4, 4)
    u = union(a, b)
    assert u.area > 15
    inter = intersection(a, b)
    assert abs(inter.area - 4.0) < 0.1
    diff = difference(a, b)
    assert abs(diff.area - 12.0) < 0.1


def test_shared_edge_and_touches():
    a = rect(0, 0, 2, 2)
    b = rect(2, 0, 2, 2)
    assert touches(a, b)
    edge = shared_edge(a, b)
    assert edge is not None
    assert edge.length >= 1.5


def test_contains_distance():
    outer = rect(0, 0, 10, 10)
    inner = rect(2, 2, 2, 2)
    outside = rect(20, 20, 1, 1)
    assert contains(outer, inner)
    assert not contains(inner, outer)
    assert distance(inner, outside) > 10


def test_affine_and_offset():
    coords = rect(0, 0, 2, 2)
    moved = translate(coords, 3, 4)
    assert abs(bounding_box(moved)[0] - 3) < 1e-6
    scaled = scale(coords, 2, origin=(0, 0))
    assert abs(area(scaled) - 16) < 0.2
    rotated = rotate(coords, 90, origin=(0, 0))
    assert len(rotated) >= 3
    grown = offset(coords, 0.5)
    assert area(grown) > area(coords)
    simple = simplify([(0, 0), (1, 0.001), (2, 0), (2, 2), (0, 2)], tolerance=0.01)
    assert len(simple) <= 5
