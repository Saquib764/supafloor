"""Sample and transform wall-aligned shelf rectangles."""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from shapely.geometry import Point, Polygon as ShapelyPolygon

Coord = tuple[float, float]

_WALL_RECT_ROOM_PROBABILITY = 0.15
_WALL_RECT_MAX_PER_ROOM = 2
_WALL_RECT_MIN_WALL_LEN_IN = 36.0
_WALL_RECT_EDGE_MARGIN_IN = 6.0
_WALL_RECT_MIN_LENGTH_IN = 12.0
_WALL_RECT_LENGTH_FRAC_MIN = 0.35
_WALL_RECT_LENGTH_FRAC_MAX = 0.75
_WALL_RECT_WIDTH_FRAC_MIN = 0.1
_WALL_RECT_WIDTH_FRAC_MAX = 0.7
_SEGMENT_MATCH_TOL = 0.5


def _open_ring(polygon: list[dict[str, Any]]) -> list[Coord]:
    ring = [(float(point["x"]), float(point["y"])) for point in polygon]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _coord(point: dict[str, Any]) -> Coord:
    return (float(point["x"]), float(point["y"]))


def _segment_length(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_on_segment_distance(point: Coord, a: Coord, b: Coord) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / len2))
    q = (a[0] + t * dx, a[1] + t * dy)
    return math.hypot(point[0] - q[0], point[1] - q[1])


def _segments_collinear_overlap(a: Coord, b: Coord, c: Coord, d: Coord) -> bool:
    if (
        _point_on_segment_distance(c, a, b) > _SEGMENT_MATCH_TOL
        or _point_on_segment_distance(d, a, b) > _SEGMENT_MATCH_TOL
    ):
        return False
    return _segment_length(c, d) >= _WALL_RECT_MIN_LENGTH_IN * 0.5


def _interior_wall_segments_for_room(
    room: dict[str, Any],
    walls: list[dict[str, Any]],
) -> list[tuple[Coord, Coord]]:
    ring = _open_ring(room.get("polygon") or [])
    if len(ring) < 3:
        return []
    segments: list[tuple[Coord, Coord]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for index in range(len(ring)):
        edge_a, edge_b = ring[index], ring[(index + 1) % len(ring)]
        for wall in walls:
            if wall.get("is_exterior"):
                continue
            start, end = _coord(wall["start"]), _coord(wall["end"])
            if not _segments_collinear_overlap(edge_a, edge_b, start, end):
                continue
            key = (
                round(start[0], 1),
                round(start[1], 1),
                round(end[0], 1),
                round(end[1], 1),
            )
            reverse = (key[2], key[3], key[0], key[1])
            if key in seen or reverse in seen:
                continue
            seen.add(key)
            segments.append((start, end))
    return segments


def _inward_normal(a: Coord, b: Coord, room_poly: ShapelyPolygon) -> Coord | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    for nx, ny in ((-dy / length, dx / length), (dy / length, -dx / length)):
        if room_poly.contains(Point(mid[0] + nx * 2.0, mid[1] + ny * 2.0)):
            return (nx, ny)
    return None


def _wall_rect_polygon(
    a: Coord,
    b: Coord,
    room_poly: ShapelyPolygon,
    rng: random.Random,
) -> list[dict[str, float]] | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    wall_len = math.hypot(dx, dy)
    if wall_len < _WALL_RECT_MIN_WALL_LEN_IN:
        return None

    inward = _inward_normal(a, b, room_poly)
    if inward is None:
        return None

    ux, uy = dx / wall_len, dy / wall_len
    nx, ny = inward
    usable = wall_len - 2.0 * _WALL_RECT_EDGE_MARGIN_IN
    if usable < _WALL_RECT_MIN_LENGTH_IN:
        return None

    rect_len = min(
        rng.uniform(
            usable * _WALL_RECT_LENGTH_FRAC_MIN,
            usable * _WALL_RECT_LENGTH_FRAC_MAX,
        ),
        usable,
    )
    start_t = rng.uniform(
        _WALL_RECT_EDGE_MARGIN_IN,
        wall_len - _WALL_RECT_EDGE_MARGIN_IN - rect_len,
    )
    depth = rect_len * rng.uniform(_WALL_RECT_WIDTH_FRAC_MIN, _WALL_RECT_WIDTH_FRAC_MAX)

    p0 = (a[0] + ux * start_t, a[1] + uy * start_t)
    p1 = (a[0] + ux * (start_t + rect_len), a[1] + uy * (start_t + rect_len))
    p2 = (p1[0] + nx * depth, p1[1] + ny * depth)
    p3 = (p0[0] + nx * depth, p0[1] + ny * depth)

    rect_poly = ShapelyPolygon([p0, p1, p2, p3])
    if rect_poly.is_empty or not rect_poly.is_valid:
        return None
    diff = rect_poly.difference(room_poly)
    if not diff.is_empty and diff.area > 1e-6:
        return None

    return [{"x": x, "y": y} for x, y in (p0, p1, p2, p3)]


def sample_wall_rects(extracted: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rooms = list(extracted.get("rooms") or [])
    openings = extracted.get("openings") or {}
    walls = extracted.get("walls")
    if walls is None:
        from extraction.walls import extract_walls

        walls = extract_walls(
            rooms,
            doors=list(openings.get("doors") or []),
            windows=list(openings.get("windows") or []),
        )

    wall_rects: list[dict[str, Any]] = []
    for room in rooms:
        if rng.random() >= _WALL_RECT_ROOM_PROBABILITY:
            continue
        ring = _open_ring(room.get("polygon") or [])
        if len(ring) < 3:
            continue
        room_poly = ShapelyPolygon(ring)
        if room_poly.is_empty or not room_poly.is_valid:
            continue

        candidates = _interior_wall_segments_for_room(room, walls)
        if not candidates:
            continue
        rng.shuffle(candidates)
        added = 0
        for start, end in candidates:
            if added >= _WALL_RECT_MAX_PER_ROOM:
                break
            polygon = _wall_rect_polygon(start, end, room_poly, rng)
            if polygon is None:
                continue
            wall_rects.append(
                {
                    "polygon": polygon,
                    "room": str(room.get("name") or ""),
                }
            )
            added += 1
    return wall_rects


def rotate_wall_rect(raw: dict[str, Any], cos_t: float, sin_t: float) -> dict[str, Any]:
    updated = dict(raw)
    updated["polygon"] = [
        {
            "x": float(point["x"]) * cos_t - float(point["y"]) * sin_t,
            "y": float(point["x"]) * sin_t + float(point["y"]) * cos_t,
        }
        for point in raw.get("polygon") or []
    ]
    return updated


def transform_wall_rect(
    raw: dict[str, Any],
    mapper: Callable[[float, float], Coord],
    round_point: Callable[[float, float], dict[str, int]],
) -> dict[str, Any]:
    updated = dict(raw)
    updated["polygon"] = [
        round_point(*mapper(float(point["x"]), float(point["y"])))
        for point in raw.get("polygon") or []
    ]
    return updated
