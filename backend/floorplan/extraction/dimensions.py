"""Sample architectural dimension lines for transformed floor plans."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

Coord = tuple[float, float]

_WITNESS_GAP = 8.0
_WITNESS_OVERSHOOT = 4.0
_OFFSET_INNER = 28.0
_OFFSET_OUTER = 52.0
_ENDPOINT_TOL = 2.0
_PARALLEL_TOL = 0.05
_INTERNAL_MIN_SPAN = 60.0
_INTERNAL_INSET = 12.0
_MIN_WALL_SPAN = 8.0
_LABEL_OFFSET = 6.0


def _coord(point: dict[str, Any]) -> Coord:
    return (float(point["x"]), float(point["y"]))


def _pt(x: float, y: float) -> dict[str, int]:
    return {"x": math.ceil(x), "y": math.ceil(y)}


def _format_mm(value: int) -> str:
    return f"{max(value, 1):,}"


def _segment_unit(a: Coord, b: Coord) -> tuple[float, float, float] | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return length, dx / length, dy / length


def _lerp(a: Coord, b: Coord, t: float) -> Coord:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def _scale(v: Coord, factor: float) -> Coord:
    return (v[0] * factor, v[1] * factor)


def _dot(a: Coord, b: Coord) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _polygon_centroid(polygon: list[dict[str, Any]]) -> Coord:
    ring = [(float(p["x"]), float(p["y"])) for p in polygon]
    if not ring:
        return (0.0, 0.0)
    return (sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring))


def _building_centroid(rooms: list[dict[str, Any]]) -> Coord:
    points: list[Coord] = []
    for room in rooms:
        for point in room.get("polygon") or []:
            points.append(_coord(point))
    if not points:
        return (0.0, 0.0)
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def _perpendicular(normal_toward: Coord, ux: float, uy: float) -> Coord:
    left = (-uy, ux)
    right = (uy, -ux)
    if _dot(left, normal_toward) >= _dot(right, normal_toward):
        return left
    return right


def _away_from(point: Coord, reference: Coord) -> Coord:
    dx, dy = point[0] - reference[0], point[1] - reference[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, -1.0)
    return (dx / length, dy / length)


def _label_rotation(ux: float, uy: float) -> float:
    angle = math.degrees(math.atan2(uy, ux))
    if angle > 90.0:
        return angle - 180.0
    if angle < -90.0:
        return angle + 180.0
    return angle


def _mm_label(span: float, reference_span: float, total_mm: int) -> str:
    if reference_span < 1e-6:
        return _format_mm(total_mm)
    return _format_mm(max(1, round(span / reference_span * total_mm)))


def _dimension_along_wall(
    wall_a: Coord,
    wall_b: Coord,
    normal: Coord,
    offset: float,
    *,
    label: str,
    kind: str,
    tier: str | None = None,
) -> dict[str, Any] | None:
    unit = _segment_unit(wall_a, wall_b)
    if unit is None:
        return None
    length, ux, uy = unit
    if length < _MIN_WALL_SPAN:
        return None

    nx, ny = normal
    dim_a = _add(wall_a, _scale(normal, offset))
    dim_b = _add(wall_b, _scale(normal, offset))
    witness_a = _add(wall_a, _scale(normal, _WITNESS_GAP))
    witness_b = _add(wall_a, _scale(normal, offset + _WITNESS_OVERSHOOT))
    witness_c = _add(wall_b, _scale(normal, _WITNESS_GAP))
    witness_d = _add(wall_b, _scale(normal, offset + _WITNESS_OVERSHOOT))

    mid = _lerp(dim_a, dim_b, 0.5)
    label_point = _add(mid, _scale(normal, _LABEL_OFFSET))
    spec: dict[str, Any] = {
        "kind": kind,
        "witness_a": _pt(*witness_a),
        "witness_b": _pt(*witness_b),
        "witness_c": _pt(*witness_c),
        "witness_d": _pt(*witness_d),
        "dim_a": _pt(*dim_a),
        "dim_b": _pt(*dim_b),
        "label": label,
        "label_x": math.ceil(label_point[0]),
        "label_y": math.ceil(label_point[1]),
        "label_rotation": round(_label_rotation(ux, uy), 2),
    }
    if tier is not None:
        spec["tier"] = tier
    return spec


def _endpoint_key(point: Coord) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _points_near(a: Coord, b: Coord, tol: float = _ENDPOINT_TOL) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def _parallel_key(ux: float, uy: float) -> tuple[float, float]:
    if ux < 0.0 or (abs(ux) < 1e-6 and uy < 0.0):
        return (-ux, -uy)
    return (round(ux, 2), round(uy, 2))


def _chain_endpoints(walls: list[tuple[Coord, Coord]]) -> tuple[Coord, Coord]:
    if len(walls) == 1:
        return walls[0]
    degree: dict[tuple[int, int], int] = defaultdict(int)
    for start, end in walls:
        degree[_endpoint_key(start)] += 1
        degree[_endpoint_key(end)] += 1
    endpoints: list[Coord] = []
    for start, end in walls:
        for point in (start, end):
            if degree[_endpoint_key(point)] == 1:
                endpoints.append(point)
    if len(endpoints) >= 2:
        return endpoints[0], endpoints[-1]
    return walls[0][0], walls[-1][1]


def _merge_collinear_chains(
    walls: list[tuple[Coord, Coord, float, float, float]],
) -> list[list[tuple[Coord, Coord]]]:
    remaining = list(walls)
    chains: list[list[tuple[Coord, Coord]]] = []
    while remaining:
        seed = remaining.pop(0)
        chain = [(seed[0], seed[1])]
        seed_u = (seed[2], seed[3])
        changed = True
        while changed:
            changed = False
            for index in range(len(remaining) - 1, -1, -1):
                item = remaining[index]
                if abs(_dot(seed_u, (item[2], item[3])) - 1.0) > _PARALLEL_TOL:
                    continue
                seg_a, seg_b = item[0], item[1]
                head_a, head_b = chain[0][0], chain[0][1]
                tail_a, tail_b = chain[-1][0], chain[-1][1]
                if _points_near(seg_a, tail_b) or _points_near(seg_b, tail_b):
                    chain.append((seg_a, seg_b) if _points_near(seg_a, tail_b) else (seg_b, seg_a))
                    remaining.pop(index)
                    changed = True
                elif _points_near(seg_a, tail_a) or _points_near(seg_b, tail_a):
                    chain.append((seg_b, seg_a) if _points_near(seg_a, tail_a) else (seg_a, seg_b))
                    remaining.pop(index)
                    changed = True
                elif _points_near(seg_a, head_a) or _points_near(seg_b, head_a):
                    chain.insert(0, (seg_b, seg_a) if _points_near(seg_a, head_a) else (seg_a, seg_b))
                    remaining.pop(index)
                    changed = True
                elif _points_near(seg_a, head_b) or _points_near(seg_b, head_b):
                    chain.insert(0, (seg_a, seg_b) if _points_near(seg_a, head_b) else (seg_b, seg_a))
                    remaining.pop(index)
                    changed = True
        chains.append(chain)
    return chains


def _sample_external_dimensions(
    rooms: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    centroid = _building_centroid(rooms)
    exterior: list[tuple[Coord, Coord, float, float, float]] = []
    for wall in walls:
        if not wall.get("is_exterior"):
            continue
        start, end = _coord(wall["start"]), _coord(wall["end"])
        unit = _segment_unit(start, end)
        if unit is None:
            continue
        length, ux, uy = unit
        exterior.append((start, end, ux, uy, length))

    if not exterior:
        return []

    total_mm = rng.randint(12000, 18000)
    max_length = max(item[4] for item in exterior)
    dimensions: list[dict[str, Any]] = []

    for start, end, ux, uy, length in exterior:
        mid = _lerp(start, end, 0.5)
        normal = _perpendicular(_away_from(mid, centroid), ux, uy)
        label = _mm_label(length, max_length, total_mm)
        spec = _dimension_along_wall(
            start,
            end,
            normal,
            _OFFSET_INNER,
            label=label,
            kind="external",
            tier="inner",
        )
        if spec is not None:
            dimensions.append(spec)

    grouped: dict[tuple[float, float], list[tuple[Coord, Coord, float, float, float]]] = defaultdict(list)
    for item in exterior:
        grouped[_parallel_key(item[2], item[3])].append(item)

    for items in grouped.values():
        chains = _merge_collinear_chains(items)
        for chain in chains:
            chain_start, chain_end = _chain_endpoints(chain)
            unit = _segment_unit(chain_start, chain_end)
            if unit is None:
                continue
            length, ux, uy = unit
            if length < _MIN_WALL_SPAN:
                continue
            mid = _lerp(chain_start, chain_end, 0.5)
            normal = _perpendicular(_away_from(mid, centroid), ux, uy)
            label = _mm_label(length, max_length, total_mm)
            spec = _dimension_along_wall(
                chain_start,
                chain_end,
                normal,
                _OFFSET_OUTER,
                label=label,
                kind="external",
                tier="outer",
            )
            if spec is not None:
                dimensions.append(spec)

    return dimensions


def _room_edges(polygon: list[dict[str, Any]]) -> list[tuple[Coord, Coord, float]]:
    ring = [(float(p["x"]), float(p["y"])) for p in polygon]
    if len(ring) < 3:
        return []
    edges: list[tuple[Coord, Coord, float]] = []
    for index in range(len(ring)):
        start = ring[index]
        end = ring[(index + 1) % len(ring)]
        unit = _segment_unit(start, end)
        if unit is None:
            continue
        length, _, _ = unit
        edges.append((start, end, length))
    return edges


def _edges_not_parallel(
    edges: list[tuple[Coord, Coord, float, float, float]],
    reference: tuple[float, float],
) -> bool:
    for _, _, _, ux, uy in edges:
        if abs(_dot(reference, (ux, uy))) > 1.0 - _PARALLEL_TOL:
            return False
    return True


def _sample_internal_dimensions(
    rooms: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    for room in rooms:
        polygon = room.get("polygon") or []
        centroid = _polygon_centroid(polygon)
        edges = _room_edges(polygon)
        if not edges:
            continue

        ranked: list[tuple[Coord, Coord, float, float, float]] = []
        for start, end, length in edges:
            unit = _segment_unit(start, end)
            if unit is None:
                continue
            _, ux, uy = unit
            ranked.append((start, end, length, ux, uy))
        ranked.sort(key=lambda item: item[2], reverse=True)

        chosen: list[tuple[Coord, Coord, float, float, float]] = []
        for candidate in ranked:
            if candidate[2] < _INTERNAL_MIN_SPAN:
                break
            direction = (candidate[3], candidate[4])
            if chosen and not _edges_not_parallel(chosen, direction):
                continue
            chosen.append(candidate)
            if len(chosen) == 2:
                break

        for start, end, length, ux, uy in chosen:
            mid = _lerp(start, end, 0.5)
            toward_room = (centroid[0] - mid[0], centroid[1] - mid[1])
            normal = _perpendicular(toward_room, ux, uy)
            label = _format_mm(rng.randint(2000, 6000))
            spec = _dimension_along_wall(
                start,
                end,
                normal,
                _INTERNAL_INSET,
                label=label,
                kind="internal",
            )
            if spec is not None:
                dimensions.append(spec)

    return dimensions


def sample_dimensions(
    rooms: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    openings: dict[str, Any],
    rng: random.Random,
    *,
    plan_probability: float,
    external_probability: float,
    internal_probability: float,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    flags = {"enabled": False, "external": False, "internal": False}
    if rng.random() >= plan_probability:
        return [], flags

    flags["enabled"] = True
    dimensions: list[dict[str, Any]] = []

    if rng.random() < external_probability:
        flags["external"] = True
        dimensions.extend(_sample_external_dimensions(rooms, walls, rng))

    if rng.random() < internal_probability:
        flags["internal"] = True
        dimensions.extend(_sample_internal_dimensions(rooms, rng))

    return dimensions, flags
