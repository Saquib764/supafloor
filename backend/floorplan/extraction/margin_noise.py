"""Sample random margin noise text for empty canvas bands."""

from __future__ import annotations

import math
import random
import string
from typing import Any

Coord = tuple[float, float]

_BAND_TOL = 2.0
_EDGE_MARGIN = 20.0
_CHAR_WIDTH_FACTOR = 0.6
_NOISE_CHARS = string.ascii_letters + string.digits + " " + "!@#$%&*+-=/?"
_DIMENSION_POINT_KEYS = (
    "witness_a",
    "witness_b",
    "witness_c",
    "witness_d",
    "dim_a",
    "dim_b",
)


def _coord(point: dict[str, Any]) -> Coord:
    return (float(point["x"]), float(point["y"]))


def _bbox(coords: list[Coord]) -> tuple[float, float, float, float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _extend_rect(
    coords: list[Coord],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    half_w = width / 2.0
    half_h = height / 2.0
    coords.extend(
        [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ]
    )


def _content_coords(
    *,
    rooms: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    openings: dict[str, Any],
    symbols: list[dict[str, Any]],
    pillars: list[dict[str, Any]],
    wall_rects: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
) -> list[Coord]:
    coords: list[Coord] = []
    for room in rooms:
        for point in room.get("polygon") or []:
            coords.append(_coord(point))
    for wall in walls:
        for key in ("start", "end"):
            coords.append(_coord(wall[key]))
    for group in ("doors", "windows"):
        for item in openings.get(group) or []:
            for key in ("start", "end"):
                coords.append(_coord(item[key]))
    for wall_rect in wall_rects:
        for point in wall_rect.get("polygon") or []:
            coords.append(_coord(point))
    for symbol in symbols:
        try:
            _extend_rect(
                coords,
                float(symbol["x"]),
                float(symbol["y"]),
                float(symbol["width"]),
                float(symbol["height"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    for pillar in pillars:
        try:
            _extend_rect(
                coords,
                float(pillar["x"]),
                float(pillar["y"]),
                float(pillar["width"]),
                float(pillar["height"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    for dimension in dimensions:
        for key in _DIMENSION_POINT_KEYS:
            point = dimension.get(key)
            if isinstance(point, dict):
                coords.append(_coord(point))
        try:
            coords.append((float(dimension["label_x"]), float(dimension["label_y"])))
        except (KeyError, TypeError, ValueError):
            pass
    return coords


def _empty_edges(
    bbox: tuple[float, float, float, float],
    *,
    canvas_size: int,
    band_fraction: float,
) -> list[str]:
    band = canvas_size * band_fraction
    minx, miny, maxx, maxy = bbox
    empty: list[str] = []
    if minx >= band - _BAND_TOL:
        empty.append("left")
    if maxx <= canvas_size - band + _BAND_TOL:
        empty.append("right")
    if miny >= band - _BAND_TOL:
        empty.append("top")
    if maxy <= canvas_size - band + _BAND_TOL:
        empty.append("bottom")
    return empty


def _random_noise_text(
    rng: random.Random,
    *,
    min_len: int,
    max_len: int,
) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(_NOISE_CHARS) for _ in range(length)).strip()


def _position_in_band(
    edge: str,
    rng: random.Random,
    *,
    canvas_size: int,
    band_fraction: float,
) -> Coord:
    band = canvas_size * band_fraction
    inner_lo = band * 0.2
    inner_hi = band * 0.8
    span_lo = band
    span_hi = canvas_size - band
    if edge == "left":
        return (rng.uniform(inner_lo, inner_hi), rng.uniform(span_lo, span_hi))
    if edge == "right":
        return (rng.uniform(canvas_size - inner_hi, canvas_size - inner_lo), rng.uniform(span_lo, span_hi))
    if edge == "top":
        return (rng.uniform(span_lo, span_hi), rng.uniform(inner_lo, inner_hi))
    return (rng.uniform(span_lo, span_hi), rng.uniform(canvas_size - inner_hi, canvas_size - inner_lo))


def _max_line_width(
    x: float,
    edge: str,
    *,
    canvas_size: int,
    band: float,
) -> float:
    if edge in ("top", "bottom"):
        return 2.0 * min(x - _EDGE_MARGIN, canvas_size - _EDGE_MARGIN - x)
    if edge == "left":
        return 2.0 * min(x - _EDGE_MARGIN, band - _EDGE_MARGIN - x)
    inner = canvas_size - band
    return 2.0 * min(x - inner - _EDGE_MARGIN, canvas_size - _EDGE_MARGIN - x)


def _wrap_margin_text(
    text: str,
    *,
    font_size: float,
    x: float,
    edge: str,
    canvas_size: int,
    band_fraction: float,
) -> str:
    band = canvas_size * band_fraction
    char_width = font_size * _CHAR_WIDTH_FACTOR
    if char_width <= 0:
        return text

    max_width = _max_line_width(x, edge, canvas_size=canvas_size, band=band)
    if max_width <= 0 or len(text) * char_width <= max_width:
        return text

    max_chars = max(1, int(max_width / char_width))
    lines = [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    return "\n".join(lines)


def sample_margin_noise(
    *,
    rooms: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    openings: dict[str, Any],
    symbols: list[dict[str, Any]],
    pillars: list[dict[str, Any]],
    wall_rects: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    rng: random.Random,
    canvas_size: int,
    probability: float,
    band_fraction: float,
    min_len: int,
    max_len: int,
    font_min: float,
    font_max: float,
    rotation_min_deg: float,
    rotation_max_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flags: dict[str, Any] = {"enabled": False}
    coords = _content_coords(
        rooms=rooms,
        walls=walls,
        openings=openings,
        symbols=symbols,
        pillars=pillars,
        wall_rects=wall_rects,
        dimensions=dimensions,
    )
    if not coords:
        return [], flags

    empty_edges = _empty_edges(
        _bbox(coords),
        canvas_size=canvas_size,
        band_fraction=band_fraction,
    )
    if not empty_edges or rng.random() >= probability:
        return [], flags

    edge = rng.choice(empty_edges)
    text = _random_noise_text(rng, min_len=min_len, max_len=max_len)
    if not text:
        return [], flags

    font_size = round(rng.uniform(font_min, font_max), 2)
    x, y = _position_in_band(edge, rng, canvas_size=canvas_size, band_fraction=band_fraction)
    text = _wrap_margin_text(
        text,
        font_size=font_size,
        x=x,
        edge=edge,
        canvas_size=canvas_size,
        band_fraction=band_fraction,
    )
    flags = {"enabled": True, "edge": edge}
    return [
        {
            "text": text,
            "x": math.ceil(x),
            "y": math.ceil(y),
            "font_size": font_size,
            "rotation": round(rng.uniform(rotation_min_deg, rotation_max_deg), 2),
            "edge": edge,
        }
    ], flags
