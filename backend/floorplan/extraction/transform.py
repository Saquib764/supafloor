"""Scale, center, and offset extracted floorplan coordinates onto a canvas."""

from __future__ import annotations

import math
import random
from typing import Any

from extraction.dimensions import sample_dimensions
from extraction.margin_noise import sample_margin_noise
from extraction.wall_rects import rotate_wall_rect, sample_wall_rects, transform_wall_rect

Coord = tuple[float, float]

_CANVAS_SIZE = 1024
_SCALE_MIN = 600.0
_SCALE_MAX = 900.0
_INCHES_PER_FOOT = 12.0
_DOOR_EDGE_MATCH_TOL = 1.0
_WINDOW_LENGTH_IN = 36.0
_WINDOW_MIN_LENGTH_IN = 12.0
_WINDOW_END_MARGIN_IN = 3.0
_WIDE_WINDOW_PROBABILITY = 0.2
_WIDE_WINDOW_MARGIN_MIN_IN = 3.0
_WIDE_WINDOW_MARGIN_MAX_IN = 10.0
_DOOR_ARC_DISABLE_PROBABILITY = 0.1
_DOUBLE_DOOR_PROBABILITY = 0.25
_SINGLE_DOOR_ARC_ANGLES = (90.0, 45.0, 30.0, 20.0)
_DOUBLE_DOOR_ARC_MIN_DEG = 10.0
_DOUBLE_DOOR_ARC_MAX_DEG = 30.0
_SYMBOL_WALL_THICKNESS = 4.0
_LABEL_FONT_BASE = 16.0
_LABEL_FONT_SCALE_MIN = 0.82
_LABEL_FONT_SCALE_MAX = 1.18
_LABEL_DISABLE_PROBABILITY = 0.2
_PILLAR_PROBABILITY = 0.01
_PILLAR_SIDE_MIN_IN = 24.0
_PILLAR_SIDE_MAX_IN = 60.0
_PILLAR_ROTATION_MIN_DEG = 1.0
_PILLAR_ROTATION_MAX_DEG = 10.0
_WALL_CENTERLINE_PROBABILITY = 0.5
_WALL_TEXTURE_CUTOFF_MIN = 0.5
_WALL_TEXTURE_CUTOFF_MAX = 1.0
_WALL_HATCHES: tuple[str | None, ...] = (None, ".", "..", "...", "//", "||", "\\")
_IMAGE_GRAY_PROBABILITY = 0.1
_IMAGE_GRAIN_PROBABILITY = 0.1
_ROOM_INTERIOR_TEXTURE_PROBABILITY = 0.15
_ROOM_INTERIOR_HATCHES = ("...", "/", "..", "|", "\\", "++", "+")
_DIMENSION_PLAN_PROBABILITY = 0.1
_DIMENSION_EXTERNAL_PROBABILITY = 0.4
_DIMENSION_INTERNAL_PROBABILITY = 0.4
_MARGIN_NOISE_PROBABILITY = 0.65
_MARGIN_BAND_FRACTION = 0.10
_MARGIN_NOISE_MIN_LEN = 4
_MARGIN_NOISE_MAX_LEN = 200
_MARGIN_NOISE_FONT_MIN = 8.0
_MARGIN_NOISE_FONT_MAX = 14.0
_MARGIN_NOISE_ROTATION_MIN_DEG = -2.0
_MARGIN_NOISE_ROTATION_MAX_DEG = 2.0

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


def canonicalize_room_polygon(polygon: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rotate a room ring to start at the vertex closest to the origin; enforce clockwise winding."""
    n = len(polygon)
    if n < 3:
        return list(polygon)

    ring = [_coord(point) for point in polygon]
    start = min(
        range(n),
        key=lambda index: (ring[index][0] ** 2 + ring[index][1] ** 2, ring[index][0], ring[index][1]),
    )
    order = list(range(start, n)) + list(range(0, start))
    ordered = [polygon[index] for index in order]
    if _signed_area([_coord(point) for point in ordered]) > 0:
        ordered = [ordered[0], *reversed(ordered[1:])]
    return ordered


def export_room_polygon(polygon: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Ceil-round canvas coords and canonicalize ring order (synthetic extraction export)."""
    rounded = [_round_point(float(point["x"]), float(point["y"])) for point in polygon]
    return canonicalize_room_polygon(rounded)  # type: ignore[return-value]


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
    from extraction.walls import _extract_walls

    room_polygons = [
        [(float(point["x"]), float(point["y"])) for point in room.get("polygon") or []]
        for room in extracted.get("rooms") or []
    ]
    openings = extracted.get("openings") or {"doors": [], "windows": []}
    return {**extracted, "walls": _extract_walls(room_polygons, openings)}


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


def _assign_door_type_and_angle(extracted: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Assign door_type (single/double), swing side, and arc angle per door."""
    openings = extracted.get("openings") or {}
    doors = list(openings.get("doors") or [])
    if not doors:
        return extracted

    updated_doors: list[dict[str, Any]] = []
    for door in doors:
        updated = dict(door)
        is_double = rng.random() < _DOUBLE_DOOR_PROBABILITY
        if is_double:
            updated["door_type"] = "double"
            updated["arc_angle_deg"] = round(
                rng.uniform(_DOUBLE_DOOR_ARC_MIN_DEG, _DOUBLE_DOOR_ARC_MAX_DEG), 1
            )
        else:
            updated["door_type"] = "single"
            updated["arc_angle_deg"] = rng.choice(_SINGLE_DOOR_ARC_ANGLES)
        updated["to_the_right"] = rng.choice((True, False))
        updated_doors.append(updated)

    return {
        **extracted,
        "openings": {
            **openings,
            "doors": updated_doors,
            "windows": list(openings.get("windows") or []),
        },
    }


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
        "wall_rects": [
            rotate_wall_rect(wall_rect, cos_t, sin_t)
            for wall_rect in extracted.get("wall_rects") or []
        ],
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
    for wall_rect in extracted.get("wall_rects") or []:
        for point in wall_rect.get("polygon") or []:
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


def _room_interior_texture(rng: random.Random) -> dict[str, Any]:
    if rng.random() >= _ROOM_INTERIOR_TEXTURE_PROBABILITY:
        return {"is_interior_textured": False}
    return {
        "is_interior_textured": True,
        "interior_hatch": rng.choice(_ROOM_INTERIOR_HATCHES),
    }


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


def _texture_flags(n: int, mode: int, rng: random.Random) -> tuple[list[bool], float]:
    if mode == 1:
        return [True] * n, 1.0
    if mode == 3:
        return [False] * n, 0.0
    cutoff = rng.uniform(_WALL_TEXTURE_CUTOFF_MIN, _WALL_TEXTURE_CUTOFF_MAX)
    k = round(n * cutoff)
    flags = [True] * k + [False] * (n - k)
    rng.shuffle(flags)
    return flags, cutoff


def _apply_texture_flags(
    items: list[dict[str, Any]],
    flags: list[bool],
) -> tuple[list[dict[str, Any]], int]:
    updated: list[dict[str, Any]] = []
    textured_count = 0
    for item, is_textured in zip(items, flags):
        copy = dict(item)
        copy["is_textured"] = is_textured
        updated.append(copy)
        if is_textured:
            textured_count += 1
    return updated, textured_count


def assign_wall_rendering(
    walls: list[dict[str, Any]],
    rng: random.Random,
    *,
    pillars: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # Mode 1: texture on all; 2: mixed; 3: texture on none. Outline is always drawn at render time.
    mode = rng.choice([1, 2, 3])
    wall_hatch = rng.choice(_WALL_HATCHES)
    wall_centerline = rng.random() < _WALL_CENTERLINE_PROBABILITY

    wall_flags, cutoff = _texture_flags(len(walls), mode, rng)
    updated_walls, textured_wall_count = _apply_texture_flags(walls, wall_flags)

    pillar_list = list(pillars or [])
    pillar_flags, _ = _texture_flags(len(pillar_list), mode, rng)
    updated_pillars, textured_pillar_count = _apply_texture_flags(pillar_list, pillar_flags)

    return updated_walls, {
        "wall_render_mode": mode,
        "wall_hatch": wall_hatch,
        "wall_texture_cutoff": round(cutoff, 4),
        "wall_centerline": wall_centerline,
        "textured_wall_count": textured_wall_count,
        "untextured_wall_count": len(walls) - textured_wall_count,
        "textured_pillar_count": textured_pillar_count,
        "untextured_pillar_count": len(pillar_list) - textured_pillar_count,
    }, updated_pillars


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
    image_gray = source_rng.random() < _IMAGE_GRAY_PROBABILITY
    image_grain = source_rng.random() < _IMAGE_GRAIN_PROBABILITY
    image_flags = {
        "image_gray": image_gray,
        "image_grain": image_grain,
    }
    extracted = _convert_extraction_to_inches(extracted)
    extracted = _add_random_windows(extracted, source_rng)
    extracted = _widen_random_windows(extracted, source_rng)
    extracted = _assign_door_type_and_angle(extracted, source_rng)
    extracted = _disable_random_door_arcs(extracted, source_rng)
    extracted = {
        **extracted,
        "symbols": _sample_symbols(
            extracted.get("rooms") or [],
            source_rng,
            symbol_probability=symbol_probability,
        ),
        "pillars": _sample_pillars(extracted, source_rng),
        "wall_rects": sample_wall_rects(extracted, source_rng),
    }
    theta = source_rng.uniform(0.0, 2.0 * math.pi)
    is_rotated = abs(theta) > 1e-9
    extracted = _rotate_extraction(extracted, theta)
    coords = _iter_coords(extracted)
    if not coords:
        return _with_transform_stats(
            {**extracted, **image_flags},
            is_rotated=is_rotated,
        )

    minx, miny, maxx, maxy = _bbox(coords)
    width = maxx - minx
    height = maxy - miny
    max_dim = max(width, height)
    if max_dim < 1e-9:
        return _with_transform_stats(
            {**extracted, **image_flags},
            is_rotated=is_rotated,
        )

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
        polygon = export_room_polygon(
            [
                _round_point(*mapper(float(point["x"]), float(point["y"])))
                for point in room.get("polygon") or []
            ]
        )
        rooms.append(
            {
                "name": room["name"],
                "polygon": polygon,
                "label": _room_label(polygon, source_rng),
                **_room_interior_texture(source_rng),
            }
        )

    openings = extracted.get("openings") or {}
    transformed_pillars = [
        _transform_symbol(pillar, mapper, scale)
        for pillar in extracted.get("pillars") or []
    ]
    walls, wall_render, pillars = assign_wall_rendering(
        [_transform_segment(wall, mapper) for wall in extracted.get("walls") or []],
        source_rng,
        pillars=transformed_pillars,
    )
    transformed_openings = {
        "doors": [
            _transform_segment(door, mapper)
            for door in openings.get("doors") or []
        ],
        "windows": [
            _transform_segment(window, mapper)
            for window in openings.get("windows") or []
        ],
    }
    dimensions, dimension_flags = sample_dimensions(
        rooms,
        walls,
        transformed_openings,
        source_rng,
        plan_probability=_DIMENSION_PLAN_PROBABILITY,
        external_probability=_DIMENSION_EXTERNAL_PROBABILITY,
        internal_probability=_DIMENSION_INTERNAL_PROBABILITY,
    )
    transformed_symbols = [
        _transform_symbol(symbol, mapper, scale)
        for symbol in extracted.get("symbols") or []
    ]
    transformed_wall_rects = [
        transform_wall_rect(wall_rect, mapper, _round_point)
        for wall_rect in extracted.get("wall_rects") or []
    ]
    margin_noise, margin_noise_flags = sample_margin_noise(
        rooms=rooms,
        walls=walls,
        openings=transformed_openings,
        symbols=transformed_symbols,
        pillars=pillars,
        wall_rects=transformed_wall_rects,
        dimensions=dimensions,
        rng=source_rng,
        canvas_size=_CANVAS_SIZE,
        probability=_MARGIN_NOISE_PROBABILITY,
        band_fraction=_MARGIN_BAND_FRACTION,
        min_len=_MARGIN_NOISE_MIN_LEN,
        max_len=_MARGIN_NOISE_MAX_LEN,
        font_min=_MARGIN_NOISE_FONT_MIN,
        font_max=_MARGIN_NOISE_FONT_MAX,
        rotation_min_deg=_MARGIN_NOISE_ROTATION_MIN_DEG,
        rotation_max_deg=_MARGIN_NOISE_ROTATION_MAX_DEG,
    )
    return _with_transform_stats(
        {
            "id": extracted["id"],
            "rooms": rooms,
            "walls": walls,
            "openings": transformed_openings,
            "symbols": transformed_symbols,
            "pillars": pillars,
            "wall_rects": transformed_wall_rects,
            "dimensions": dimensions,
            "dimension_flags": dimension_flags,
            "margin_noise": margin_noise,
            "margin_noise_flags": margin_noise_flags,
            **image_flags,
            **wall_render,
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
        "single_doors": sum(1 for door in doors if door.get("door_type", "single") == "single"),
        "double_doors": sum(1 for door in doors if door.get("door_type") == "double"),
        "pillars": len(extracted.get("pillars") or []),
        "wall_rects": len(extracted.get("wall_rects") or []),
        "wall_render_mode": extracted.get("wall_render_mode"),
        "wall_hatch": extracted.get("wall_hatch"),
        "wall_texture_cutoff": extracted.get("wall_texture_cutoff"),
        "wall_centerline": bool(extracted.get("wall_centerline")),
        "textured_wall_count": int(extracted.get("textured_wall_count") or 0),
        "untextured_wall_count": int(extracted.get("untextured_wall_count") or 0),
        "textured_pillar_count": int(extracted.get("textured_pillar_count") or 0),
        "untextured_pillar_count": int(extracted.get("untextured_pillar_count") or 0),
        "image_gray": bool(extracted.get("image_gray")),
        "image_grain": bool(extracted.get("image_grain")),
        "textured_room_count": sum(
            1 for room in rooms if room.get("is_interior_textured")
        ),
        "untextured_room_count": sum(
            1 for room in rooms if not room.get("is_interior_textured")
        ),
        "has_dimensions": bool(extracted.get("dimensions")),
        "dimension_count": len(extracted.get("dimensions") or []),
        "dimension_plan_enabled": bool((extracted.get("dimension_flags") or {}).get("enabled")),
        "dimension_external": bool((extracted.get("dimension_flags") or {}).get("external")),
        "dimension_internal": bool((extracted.get("dimension_flags") or {}).get("internal")),
        "external_dimension_count": sum(
            1 for item in (extracted.get("dimensions") or []) if item.get("kind") == "external"
        ),
        "internal_dimension_count": sum(
            1 for item in (extracted.get("dimensions") or []) if item.get("kind") == "internal"
        ),
        "has_margin_noise": bool(extracted.get("margin_noise")),
        "margin_noise_enabled": bool((extracted.get("margin_noise_flags") or {}).get("enabled")),
        "margin_noise_edge": (extracted.get("margin_noise_flags") or {}).get("edge"),
    }


def _with_transform_stats(extracted: dict[str, Any], *, is_rotated: bool) -> dict[str, Any]:
    return {**extracted, TRANSFORM_STATS_KEY: transform_stats(extracted, is_rotated=is_rotated)}


def pop_transform_stats(extracted: dict[str, Any]) -> dict[str, Any]:
    stats = extracted.pop(TRANSFORM_STATS_KEY, None)
    if stats is not None:
        return stats
    return transform_stats(extracted, is_rotated=False)
