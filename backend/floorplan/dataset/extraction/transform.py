"""Scale, center, and offset extracted floorplan coordinates onto a canvas."""

from __future__ import annotations

import random
from typing import Any

Coord = tuple[float, float]

_CANVAS_SIZE = 1024
_SCALE_MIN = 500.0
_SCALE_MAX = 900.0


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


def _round_point(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 3), "y": round(float(y), 3)}


def _transform_segment(raw: dict[str, Any], mapper) -> dict[str, Any]:
    start = raw["start"]
    end = raw["end"]
    sx, sy = mapper(float(start["x"]), float(start["y"]))
    ex, ey = mapper(float(end["x"]), float(end["y"]))
    updated = dict(raw)
    updated["start"] = _round_point(sx, sy)
    updated["end"] = _round_point(ex, ey)
    return updated


def transform_extraction(
    extracted: dict[str, Any],
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Scale to [0, X], center on canvas, then apply random offset in [0, (canvas-X)/2]."""
    coords = _iter_coords(extracted)
    if not coords:
        return extracted

    source_rng = rng or random.Random(str(extracted.get("id") or ""))
    minx, miny, maxx, maxy = _bbox(coords)
    width = maxx - minx
    height = maxy - miny
    max_dim = max(width, height)
    if max_dim < 1e-9:
        return extracted

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

    openings = extracted.get("openings") or {}
    transformed = {
        "id": extracted["id"],
        "rooms": [
            {
                "name": room["name"],
                "polygon": [
                    _round_point(*mapper(float(point["x"]), float(point["y"])))
                    for point in room.get("polygon") or []
                ],
            }
            for room in extracted.get("rooms") or []
        ],
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
    }
    return transformed
