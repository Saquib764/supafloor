"""Scale, center, and offset extracted floorplan coordinates onto a canvas."""

from __future__ import annotations

import math
import random
from typing import Any

Coord = tuple[float, float]

_CANVAS_SIZE = 1024
_SCALE_MIN = 600.0
_SCALE_MAX = 900.0
_INCHES_PER_FOOT = 12.0
_DOOR_WALL_END_MARGIN_IN = 3.0
_DOOR_EDGE_MATCH_TOL = 1.0
_WINDOW_LENGTH_IN = 36.0
_WINDOW_MIN_LENGTH_IN = 12.0
_WINDOW_END_MARGIN_IN = 3.0
_WIDE_WINDOW_PROBABILITY = 0.2
_WIDE_WINDOW_MARGIN_MIN_IN = 3.0
_WIDE_WINDOW_MARGIN_MAX_IN = 10.0
_DOOR_ARC_DISABLE_PROBABILITY = 0.1
_CORNER_MOVE_PROBABILITY = 0.05
_CORNER_MOVE_RANGE = 50.0
_CORNER_MATCH_TOL = 0.5
_SYMBOL_WALL_THICKNESS = 4.0
_LABEL_FONT_BASE = 16.0
_LABEL_FONT_SCALE_MIN = 0.82
_LABEL_FONT_SCALE_MAX = 1.18
_LABEL_DISABLE_PROBABILITY = 0.2
_PILLAR_PROBABILITY = 0.01
_PILLAR_SIDE_MIN_IN = 10.0
_PILLAR_SIDE_MAX_IN = 36.0
_PILLAR_ROTATION_MIN_DEG = 1.0
_PILLAR_ROTATION_MAX_DEG = 10.0

TRANSFORM_STATS_KEY = "transform_stats"


def _open_polygon_ring(polygon: list[dict[str, Any]]) -> list[Coord]:
    ring = [(float(point["x"]), float(point["y"])) for point in polygon]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _signed_area(ring: list[Coord]) -> float:
    area = 0.0
    for i, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(i + 1) % len(ring)]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def rotate_room_polygon_clockwise_from_min(polygon: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Rotate a room ring to start at min (x, y) and enforce clockwise winding."""
    ring = _open_polygon_ring(polygon)
    if len(ring) < 3:
        return [{"x": float(point["x"]), "y": float(point["y"])} for point in polygon]

    start = min(range(len(ring)), key=lambda index: (ring[index][0], ring[index][1]))
    rotated = ring[start:] + ring[:start]
    if _signed_area(rotated) > 0:
        rotated = [rotated[0], *reversed(rotated[1:])]

    return [{"x": x, "y": y} for x, y in rotated]


def _room_first_point_origin_distance(room: dict[str, Any]) -> float:
    polygon = room.get("polygon") or []
    if not polygon:
        return float("inf")
    x = float(polygon[0]["x"])
    y = float(polygon[0]["y"])
    return x * x + y * y


def sort_rooms_by_first_point_to_origin(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rooms by squared distance of each polygon's first vertex to the origin."""
    return sorted(rooms, key=_room_first_point_origin_distance)


def _scale_point(point: dict[str, Any], factor: float) -> dict[str, float]:
    return {"x": float(point["x"]) * factor, "y": float(point["y"]) * factor}


def _scale_segment(raw: dict[str, Any], factor: float) -> dict[str, Any]:
    updated = dict(raw)
    updated["start"] = _scale_point(raw["start"], factor)
    updated["end"] = _scale_point(raw["end"], factor)
    return updated


def _convert_extraction_to_inches(extracted: dict[str, Any]) -> dict[str, Any]:
    """Convert all length coordinates from feet to inches (first transform)."""
    openings = extracted.get("openings") or {}
    return {
        **extracted,
        "rooms": [
            {
                **room,
                "polygon": [
                    _scale_point(point, _INCHES_PER_FOOT)
                    for point in room.get("polygon") or []
                ],
            }
            for room in extracted.get("rooms") or []
        ],
        "walls": [
            _scale_segment(wall, _INCHES_PER_FOOT)
            for wall in extracted.get("walls") or []
        ],
        "openings": {
            "doors": [
                _scale_segment(door, _INCHES_PER_FOOT)
                for door in openings.get("doors") or []
            ],
            "windows": [
                _scale_segment(window, _INCHES_PER_FOOT)
                for window in openings.get("windows") or []
            ],
        },
    }


def _lerp(a: Coord, b: Coord, t: float) -> Coord:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _segment_length(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _project_t(p: Coord, a: Coord, b: Coord) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return 0.0
    return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2


def _dist_to_segment(p: Coord, a: Coord, b: Coord) -> float:
    t = max(0.0, min(1.0, _project_t(p, a, b)))
    qx, qy = _lerp(a, b, t)
    return math.hypot(p[0] - qx, p[1] - qy)


def _coord(point: dict[str, Any]) -> Coord:
    return (float(point["x"]), float(point["y"]))


def _room_edges(extracted: dict[str, Any]) -> list[tuple[Coord, Coord]]:
    edges: list[tuple[Coord, Coord]] = []
    for room in extracted.get("rooms") or []:
        ring = _open_polygon_ring(room.get("polygon") or [])
        for i in range(len(ring)):
            edges.append((ring[i], ring[(i + 1) % len(ring)]))
    return edges


def _host_edge_for_opening(
    start: Coord,
    end: Coord,
    edges: list[tuple[Coord, Coord]],
) -> tuple[Coord, Coord] | None:
    best: tuple[Coord, Coord] | None = None
    best_span = 0.0
    for a, b in edges:
        if _dist_to_segment(start, a, b) > _DOOR_EDGE_MATCH_TOL:
            continue
        if _dist_to_segment(end, a, b) > _DOOR_EDGE_MATCH_TOL:
            continue
        t0, t1 = _project_t(start, a, b), _project_t(end, a, b)
        lo, hi = sorted((t0, t1))
        if hi < -0.02 or lo > 1.02:
            continue
        span = hi - lo
        if span > best_span:
            best_span = span
            best = (a, b)
    return best


def _opening_on_edge(start: Coord, end: Coord, a: Coord, b: Coord) -> bool:
    if _dist_to_segment(start, a, b) > _DOOR_EDGE_MATCH_TOL:
        return False
    if _dist_to_segment(end, a, b) > _DOOR_EDGE_MATCH_TOL:
        return False
    t0, t1 = _project_t(start, a, b), _project_t(end, a, b)
    lo, hi = sorted((t0, t1))
    return not (hi < -0.02 or lo > 1.02)


def _rebuild_walls(extracted: dict[str, Any]) -> dict[str, Any]:
    from dataset.extraction.walls import _extract_walls

    room_polygons = [
        [(float(point["x"]), float(point["y"])) for point in room.get("polygon") or []]
        for room in extracted.get("rooms") or []
    ]
    openings = extracted.get("openings") or {"doors": [], "windows": []}
    return {**extracted, "walls": _extract_walls(room_polygons, openings)}


def _enlarge_random_door(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Pick one door and stretch it to leave 3\" clearance at each wall end."""
    openings = extracted.get("openings") or {}
    doors = list(openings.get("doors") or [])
    if not doors:
        return extracted

    edges = _room_edges(extracted)
    other_openings = [
        (_coord(item["start"]), _coord(item["end"]))
        for group in ("doors", "windows")
        for item in openings.get(group) or []
    ]

    candidates: list[tuple[int, tuple[Coord, Coord], float]] = []
    for index, door in enumerate(doors):
        start, end = _coord(door["start"]), _coord(door["end"])
        host = _host_edge_for_opening(start, end, edges)
        if host is None:
            continue
        a, b = host
        length = _segment_length(a, b)
        if length <= 2.0 * _DOOR_WALL_END_MARGIN_IN + 1e-6:
            continue
        others = 0
        for oi, (os, oe) in enumerate(other_openings):
            # skip this door itself: doors come first in other_openings
            if oi == index:
                continue
            if _opening_on_edge(os, oe, a, b):
                others += 1
        if others:
            continue
        candidates.append((index, host, length))

    if not candidates:
        return extracted

    index, (a, b), length = rng.choice(candidates)
    door = doors[index]
    start, end = _coord(door["start"]), _coord(door["end"])
    margin_t = _DOOR_WALL_END_MARGIN_IN / length
    near = _lerp(a, b, margin_t)
    far = _lerp(a, b, 1.0 - margin_t)
    if _project_t(start, a, b) <= _project_t(end, a, b):
        new_start, new_end = near, far
    else:
        new_start, new_end = far, near

    updated = dict(door)
    updated["start"] = {"x": new_start[0], "y": new_start[1]}
    updated["end"] = {"x": new_end[0], "y": new_end[1]}
    doors[index] = updated

    extracted = {
        **extracted,
        "openings": {
            **openings,
            "doors": doors,
            "windows": list(openings.get("windows") or []),
        },
    }
    return _rebuild_walls(extracted)


def _same_point(a: Coord, b: Coord, *, tol: float = _DOOR_EDGE_MATCH_TOL) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def _segment_hits_other_wall_endpoint(
    p0: Coord,
    p1: Coord,
    walls: list[dict[str, Any]],
    host: tuple[Coord, Coord],
) -> bool:
    host_a, host_b = host
    for wall in walls:
        for endpoint in (_coord(wall["start"]), _coord(wall["end"])):
            if _same_point(endpoint, host_a) or _same_point(endpoint, host_b):
                continue
            if _dist_to_segment(endpoint, p0, p1) <= _DOOR_EDGE_MATCH_TOL:
                return True
    return False


def _wall_candidates(
    walls: list[dict[str, Any]],
    *,
    exterior_only: bool,
) -> list[tuple[Coord, Coord, float]]:
    candidates: list[tuple[Coord, Coord, float]] = []
    for wall in walls:
        if exterior_only and not wall.get("is_exterior"):
            continue
        a, b = _coord(wall["start"]), _coord(wall["end"])
        length = _segment_length(a, b)
        if length < _WINDOW_MIN_LENGTH_IN + 2.0 * _WINDOW_END_MARGIN_IN:
            continue
        candidates.append((a, b, length))
    return candidates


def _try_place_window_on_walls(
    walls: list[dict[str, Any]],
    rng: random.Random,
    *,
    exterior_only: bool,
) -> dict[str, Any] | None:
    candidates = _wall_candidates(walls, exterior_only=exterior_only)
    if not candidates:
        return None
    rng.shuffle(candidates)
    for a, b, length in candidates:
        usable = length - 2.0 * _WINDOW_END_MARGIN_IN
        win_len = min(_WINDOW_LENGTH_IN, usable)
        max_start = length - _WINDOW_END_MARGIN_IN - win_len
        start_dist = rng.uniform(_WINDOW_END_MARGIN_IN, max_start)
        p0 = _lerp(a, b, start_dist / length)
        p1 = _lerp(a, b, (start_dist + win_len) / length)
        if _segment_hits_other_wall_endpoint(p0, p1, walls, (a, b)):
            continue
        return {
            "start": {"x": p0[0], "y": p0[1]},
            "end": {"x": p1[0], "y": p1[1]},
            "room_names": [],
        }
    return None


def _add_random_windows(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Add w random windows on wall segments (no door overlap); first on an exterior wall."""
    rooms = extracted.get("rooms") or []
    room_count = len(rooms)
    if room_count == 0:
        return extracted

    openings = extracted.get("openings") or {}
    doors = list(openings.get("doors") or [])
    windows: list[dict[str, Any]] = []
    target = rng.randint(1, 2 + room_count)

    working = {
        **extracted,
        "openings": {"doors": doors, "windows": windows},
    }

    for index in range(target):
        walls = list(working.get("walls") or [])
        exterior_only = index == 0
        window = _try_place_window_on_walls(walls, rng, exterior_only=exterior_only)
        if window is None and exterior_only:
            window = _try_place_window_on_walls(walls, rng, exterior_only=False)
        if window is None:
            break
        windows.append(window)
        working = _rebuild_walls(
            {
                **working,
                "openings": {"doors": doors, "windows": list(windows)},
            }
        )

    return working


def _gap_overlaps(lo0: float, hi0: float, lo1: float, hi1: float) -> bool:
    return not (hi0 <= lo1 + 1e-9 or hi1 <= lo0 + 1e-9)


def _widen_random_windows(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """With p=0.2 per window, stretch it to independent 3–10\" margins from wall ends."""
    openings = extracted.get("openings") or {}
    windows = list(openings.get("windows") or [])
    doors = list(openings.get("doors") or [])
    if not windows:
        return extracted

    edges = _room_edges(extracted)
    walls = list(extracted.get("walls") or [])
    changed = False
    for index, window in enumerate(windows):
        if rng.random() >= _WIDE_WINDOW_PROBABILITY:
            continue
        start, end = _coord(window["start"]), _coord(window["end"])
        host = _host_edge_for_opening(start, end, edges)
        if host is None:
            continue
        a, b = host
        length = _segment_length(a, b)
        margin_a = rng.uniform(_WIDE_WINDOW_MARGIN_MIN_IN, _WIDE_WINDOW_MARGIN_MAX_IN)
        margin_b = rng.uniform(_WIDE_WINDOW_MARGIN_MIN_IN, _WIDE_WINDOW_MARGIN_MAX_IN)
        if length <= margin_a + margin_b + _WINDOW_MIN_LENGTH_IN:
            continue

        lo_t = margin_a / length
        hi_t = 1.0 - margin_b / length
        overlaps_opening = False
        for other_index, other in enumerate(doors + windows):
            if other_index >= len(doors) and other_index - len(doors) == index:
                continue
            os, oe = _coord(other["start"]), _coord(other["end"])
            if not _opening_on_edge(os, oe, a, b):
                continue
            o_lo, o_hi = sorted((_project_t(os, a, b), _project_t(oe, a, b)))
            if _gap_overlaps(lo_t, hi_t, o_lo, o_hi):
                overlaps_opening = True
                break
        if overlaps_opening:
            continue

        near = _lerp(a, b, lo_t)
        far = _lerp(a, b, hi_t)
        if _project_t(start, a, b) <= _project_t(end, a, b):
            new_start, new_end = near, far
        else:
            new_start, new_end = far, near

        if _segment_hits_other_wall_endpoint(new_start, new_end, walls, (a, b)):
            continue

        updated = dict(window)
        updated["start"] = {"x": new_start[0], "y": new_start[1]}
        updated["end"] = {"x": new_end[0], "y": new_end[1]}
        windows[index] = updated
        changed = True

    if not changed:
        return extracted

    return _rebuild_walls(
        {
            **extracted,
            "openings": {
                **openings,
                "doors": doors,
                "windows": windows,
            },
        }
    )


def _disable_random_door_arcs(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """With p=0.1 per door, set is_arc_disabled so the renderer skips the swing arc."""
    openings = extracted.get("openings") or {}
    doors = list(openings.get("doors") or [])
    if not doors:
        return extracted

    changed = False
    updated_doors: list[dict[str, Any]] = []
    for door in doors:
        updated = dict(door)
        if rng.random() < _DOOR_ARC_DISABLE_PROBABILITY:
            updated["is_arc_disabled"] = True
            changed = True
        updated_doors.append(updated)

    if not changed:
        return extracted

    return {
        **extracted,
        "openings": {
            **openings,
            "doors": updated_doors,
            "windows": list(openings.get("windows") or []),
        },
    }


def _coords_close(a: Coord, b: Coord) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= _CORNER_MATCH_TOL


def _apply_corner_move(ring: list[Coord], old: Coord, new: Coord) -> list[Coord]:
    if len(ring) < 3:
        return ring
    updated: list[Coord] = []
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        updated.append(new if _coords_close(a, old) else a)
        if _coords_close(a, old) or _coords_close(b, old):
            continue
        if _dist_to_segment(old, a, b) > _CORNER_MATCH_TOL:
            continue
        t = _project_t(old, a, b)
        if 0.0 < t < 1.0:
            updated.append(new)
    return updated


def _jitter_room_corners(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """With p=0.05 per unique vertex, offset it by up to 50\" and insert on mid-edge walls."""
    rooms = list(extracted.get("rooms") or [])
    rings = [_open_polygon_ring(room.get("polygon") or []) for room in rooms]
    if not any(len(ring) >= 3 for ring in rings):
        return extracted

    seen: set[tuple[float, float]] = set()
    corners: list[Coord] = []
    for ring in rings:
        for point in ring:
            key = (round(point[0], 3), round(point[1], 3))
            if key in seen:
                continue
            seen.add(key)
            corners.append(point)

    selected = [point for point in corners if rng.random() < _CORNER_MOVE_PROBABILITY]
    if not selected:
        return extracted

    for old in selected:
        dx = rng.uniform(-_CORNER_MOVE_RANGE, _CORNER_MOVE_RANGE)
        dy = rng.uniform(-_CORNER_MOVE_RANGE, _CORNER_MOVE_RANGE)
        new = (old[0] + dx, old[1] + dy)
        rings = [_apply_corner_move(ring, old, new) for ring in rings]

    updated_rooms = [
        {**room, "polygon": [{"x": x, "y": y} for x, y in ring]}
        for room, ring in zip(rooms, rings)
    ]
    return _rebuild_walls({**extracted, "rooms": updated_rooms})


def _rotate_point(point: dict[str, Any], cos_t: float, sin_t: float) -> dict[str, float]:
    x, y = float(point["x"]), float(point["y"])
    return {"x": x * cos_t - y * sin_t, "y": x * sin_t + y * cos_t}


def _rotate_segment(raw: dict[str, Any], cos_t: float, sin_t: float) -> dict[str, Any]:
    updated = dict(raw)
    updated["start"] = _rotate_point(raw["start"], cos_t, sin_t)
    updated["end"] = _rotate_point(raw["end"], cos_t, sin_t)
    return updated


def _rotate_symbol(raw: dict[str, Any], cos_t: float, sin_t: float, rotation_deg: float) -> dict[str, Any]:
    x, y = float(raw["x"]), float(raw["y"])
    updated = dict(raw)
    updated["x"] = x * cos_t - y * sin_t
    updated["y"] = x * sin_t + y * cos_t
    updated["rotation_deg"] = rotation_deg
    return updated


def _rotate_pillar(raw: dict[str, Any], cos_t: float, sin_t: float, rotation_deg: float) -> dict[str, Any]:
    x, y = float(raw["x"]), float(raw["y"])
    updated = dict(raw)
    updated["x"] = x * cos_t - y * sin_t
    updated["y"] = x * sin_t + y * cos_t
    updated["rotation_deg"] = float(raw.get("rotation_deg") or 0.0) + rotation_deg
    return updated


def _rotate_extraction(extracted: dict[str, Any], theta: float) -> dict[str, Any]:
    """Rotate rooms, walls, openings, and symbols around the origin by theta radians."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation_deg = math.degrees(theta)
    openings = extracted.get("openings") or {}
    return {
        **extracted,
        "rooms": [
            {
                **room,
                "polygon": [
                    _rotate_point(point, cos_t, sin_t)
                    for point in room.get("polygon") or []
                ],
            }
            for room in extracted.get("rooms") or []
        ],
        "walls": [
            _rotate_segment(wall, cos_t, sin_t)
            for wall in extracted.get("walls") or []
        ],
        "openings": {
            "doors": [
                _rotate_segment(door, cos_t, sin_t)
                for door in openings.get("doors") or []
            ],
            "windows": [
                _rotate_segment(window, cos_t, sin_t)
                for window in openings.get("windows") or []
            ],
        },
        "symbols": [
            _rotate_symbol(symbol, cos_t, sin_t, rotation_deg)
            for symbol in extracted.get("symbols") or []
        ],
        "pillars": [
            _rotate_pillar(pillar, cos_t, sin_t, rotation_deg)
            for pillar in extracted.get("pillars") or []
        ],
    }


def _normalize_room_polygons(extracted: dict[str, Any]) -> dict[str, Any]:
    rooms = [
        {
            **room,
            "polygon": rotate_room_polygon_clockwise_from_min(room.get("polygon") or []),
        }
        for room in extracted.get("rooms") or []
    ]
    return {
        **extracted,
        "rooms": sort_rooms_by_first_point_to_origin(rooms),
    }


def _bbox(coords: list[Coord]) -> tuple[float, float, float, float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _iter_coords(extracted: dict[str, Any]) -> list[Coord]:
    coords: list[Coord] = []
    for room in extracted.get("rooms") or []:
        for point in room.get("polygon") or []:
            coords.append((float(point["x"]), float(point["y"])))
    for wall in extracted.get("walls") or []:
        for key in ("start", "end"):
            point = wall[key]
            coords.append((float(point["x"]), float(point["y"])))
    openings = extracted.get("openings") or {}
    for group in ("doors", "windows"):
        for item in openings.get(group) or []:
            for key in ("start", "end"):
                point = item[key]
                coords.append((float(point["x"]), float(point["y"])))
    return coords


def _map_point(x: float, y: float, *, minx: float, miny: float, scale: float, offset_x: float, offset_y: float) -> Coord:
    return ((x - minx) * scale + offset_x, (y - miny) * scale + offset_y)


def _round_point(x: float, y: float) -> dict[str, int]:
    return {"x": math.ceil(float(x)), "y": math.ceil(float(y))}


def _polygon_centroid(polygon: list[dict[str, Any]]) -> Coord:
    ring = _open_polygon_ring(polygon)
    if not ring:
        return (0.0, 0.0)
    return (sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring))


def _room_label(
    polygon: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    cx, cy = _polygon_centroid(polygon)
    font_size = _LABEL_FONT_BASE * rng.uniform(_LABEL_FONT_SCALE_MIN, _LABEL_FONT_SCALE_MAX)
    label: dict[str, Any] = {
        "x": math.ceil(cx),
        "y": math.ceil(cy),
        "font_size": round(font_size, 2),
    }
    if rng.random() < _LABEL_DISABLE_PROBABILITY:
        label["is_label_disabled"] = True
    return label


def _transform_segment(raw: dict[str, Any], mapper) -> dict[str, Any]:
    start = raw["start"]
    end = raw["end"]
    sx, sy = mapper(float(start["x"]), float(start["y"]))
    ex, ey = mapper(float(end["x"]), float(end["y"]))
    updated = dict(raw)
    updated["start"] = _round_point(sx, sy)
    updated["end"] = _round_point(ex, ey)
    return updated


def _transform_symbol(raw: dict[str, Any], mapper, scale: float) -> dict[str, Any]:
    x, y = mapper(float(raw["x"]), float(raw["y"]))
    updated = dict(raw)
    updated["x"] = math.ceil(x)
    updated["y"] = math.ceil(y)
    updated["width"] = round(float(raw["width"]) * scale, 2)
    updated["height"] = round(float(raw["height"]) * scale, 2)
    return updated


def _sample_symbols(
    rooms: list[dict[str, Any]],
    rng: random.Random,
    *,
    symbol_probability: float,
) -> list[dict[str, Any]]:
    """Pick furniture/symbols per room and return a separate symbols array."""
    if symbol_probability <= 0.0:
        return []

    from export.symbols import plan_room_symbol

    symbols: list[dict[str, Any]] = []
    for room in rooms:
        if rng.random() > symbol_probability:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in room.get("polygon") or []]
        placement = plan_room_symbol(
            polygon,
            str(room.get("name") or ""),
            wall_thickness=_SYMBOL_WALL_THICKNESS,
            rng=rng,
        )
        if placement is not None:
            symbols.append(placement)
    return symbols


def _unique_wall_ends(extracted: dict[str, Any]) -> list[Coord]:
    seen: set[tuple[float, float]] = set()
    ends: list[Coord] = []
    for wall in extracted.get("walls") or []:
        for key in ("start", "end"):
            point = _coord(wall[key])
            key_xy = (round(point[0], 3), round(point[1], 3))
            if key_xy in seen:
                continue
            seen.add(key_xy)
            ends.append(point)
    return ends


def _sample_pillars(extracted: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    pillars: list[dict[str, Any]] = []
    for x, y in _unique_wall_ends(extracted):
        if rng.random() >= _PILLAR_PROBABILITY:
            continue
        pillars.append(
            {
                "x": x,
                "y": y,
                "width": round(rng.uniform(_PILLAR_SIDE_MIN_IN, _PILLAR_SIDE_MAX_IN), 2),
                "height": round(rng.uniform(_PILLAR_SIDE_MIN_IN, _PILLAR_SIDE_MAX_IN), 2),
                "rotation_deg": round(
                    rng.uniform(_PILLAR_ROTATION_MIN_DEG, _PILLAR_ROTATION_MAX_DEG), 2
                ),
            }
        )
    return pillars


def transform_extraction(
    extracted: dict[str, Any],
    *,
    rng: random.Random | None = None,
    symbol_probability: float = 0.0,
) -> dict[str, Any]:
    """Convert feet→inches, enlarge one door, add windows, jitter corners, rotate, then scale/center/offset."""
    if not 0.0 <= symbol_probability <= 1.0:
        raise ValueError("symbol_probability must be between 0 and 1")

    source_rng = rng or random.Random(str(extracted.get("id") or ""))
    extracted = _convert_extraction_to_inches(extracted)
    # extracted = _enlarge_random_door(extracted, source_rng)
    extracted = _add_random_windows(extracted, source_rng)
    extracted = _widen_random_windows(extracted, source_rng)
    extracted = _disable_random_door_arcs(extracted, source_rng)
    # extracted = _jitter_room_corners(extracted, source_rng)
    extracted = {
        **extracted,
        "symbols": _sample_symbols(
            extracted.get("rooms") or [],
            source_rng,
            symbol_probability=symbol_probability,
        ),
        "pillars": _sample_pillars(extracted, source_rng),
    }
    theta = source_rng.uniform(0.0, 2.0 * math.pi)
    is_rotated = abs(theta) > 1e-9
    extracted = _rotate_extraction(extracted, theta)
    extracted = _normalize_room_polygons(extracted)
    coords = _iter_coords(extracted)
    if not coords:
        return _with_transform_stats(extracted, is_rotated=is_rotated)

    minx, miny, maxx, maxy = _bbox(coords)
    width = maxx - minx
    height = maxy - miny
    max_dim = max(width, height)
    if max_dim < 1e-9:
        return _with_transform_stats(extracted, is_rotated=is_rotated)

    scale_target = source_rng.uniform(_SCALE_MIN, _SCALE_MAX)
    scale = scale_target / max_dim
    scaled_w = width * scale
    scaled_h = height * scale
    margin = (_CANVAS_SIZE - scale_target - 100) / 2.0
    offset_x = (_CANVAS_SIZE - scaled_w) / 2.0 + source_rng.uniform(-margin, margin)
    offset_y = (_CANVAS_SIZE - scaled_h) / 2.0 + source_rng.uniform(-margin, margin)

    def mapper(x: float, y: float) -> Coord:
        return _map_point(
            x,
            y,
            minx=minx,
            miny=miny,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    rooms: list[dict[str, Any]] = []
    for room in extracted.get("rooms") or []:
        polygon = [
            _round_point(*mapper(float(point["x"]), float(point["y"])))
            for point in room.get("polygon") or []
        ]
        rooms.append(
            {
                "name": room["name"],
                "polygon": polygon,
                "label": _room_label(polygon, source_rng),
            }
        )

    openings = extracted.get("openings") or {}
    return _with_transform_stats(
        {
            "id": extracted["id"],
            "rooms": rooms,
            "walls": [
                _transform_segment(wall, mapper)
                for wall in extracted.get("walls") or []
            ],
            "openings": {
                "doors": [
                    _transform_segment(door, mapper)
                    for door in openings.get("doors") or []
                ],
                "windows": [
                    _transform_segment(window, mapper)
                    for window in openings.get("windows") or []
                ],
            },
            "symbols": [
                _transform_symbol(symbol, mapper, scale)
                for symbol in extracted.get("symbols") or []
            ],
            "pillars": [
                _transform_symbol(pillar, mapper, scale)
                for pillar in extracted.get("pillars") or []
            ],
        },
        is_rotated=is_rotated,
    )


def transform_stats(extracted: dict[str, Any], *, is_rotated: bool) -> dict[str, Any]:
    rooms = extracted.get("rooms") or []
    openings = extracted.get("openings") or {}
    doors = openings.get("doors") or []
    windows = openings.get("windows") or []
    return {
        "id": extracted.get("id"),
        "rooms": len(rooms),
        "doors": len(doors),
        "windows": len(windows),
        "is_rotated": bool(is_rotated),
        "missing_label_count": sum(
            1 for room in rooms if (room.get("label") or {}).get("is_label_disabled")
        ),
        "missing_door_symbol": sum(
            1 for door in doors if door.get("is_arc_disabled")
        ),
        "pillars": len(extracted.get("pillars") or []),
    }


def _with_transform_stats(extracted: dict[str, Any], *, is_rotated: bool) -> dict[str, Any]:
    return {**extracted, TRANSFORM_STATS_KEY: transform_stats(extracted, is_rotated=is_rotated)}


def pop_transform_stats(extracted: dict[str, Any]) -> dict[str, Any]:
    stats = extracted.pop(TRANSFORM_STATS_KEY, None)
    if stats is not None:
        return stats
    return transform_stats(extracted, is_rotated=False)


def _to_florence_loc(value: float) -> int:
    return max(0, min(999, round(float(value) * 999 / _CANVAS_SIZE)))


def _florence_point(x: float, y: float) -> str:
    return f"<loc_{_to_florence_loc(x)}><loc_{_to_florence_loc(y)}>"


def format_extraction_florence(extracted: dict[str, Any]) -> str:
    """Serialize rooms as Florence loc tokens on a 0–999 grid."""
    chunks: list[str] = []
    for room in extracted.get("rooms") or []:
        polygon = room.get("polygon") or []
        if not polygon:
            continue
        cx, cy = _polygon_centroid(polygon)
        vertex_lines = [
            _florence_point(float(point["x"]), float(point["y"]))
            for point in polygon
        ]
        chunks.append(
            "\n".join(
                [
                    str(room.get("name") or ""),
                    _florence_point(cx, cy),
                    "<poly>",
                    *vertex_lines,
                    "</poly>",
                    "<sep>",
                ]
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")
