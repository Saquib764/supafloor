"""Random notched Manhattan boundary generation (axis-aligned integer edges only)."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from shapely.geometry import Polygon, box
from shapely.validation import make_valid

from floorplan.geometry.ops import to_points
from floorplan.schemas.models import Boundary

_CORNERS = ("sw", "se", "ne", "nw")


def _largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        poly = max(geom.geoms, key=lambda g: g.area)
        return poly if poly.area > 0 else None
    fixed = make_valid(geom)
    if fixed is geom:
        return None
    return _largest_polygon(fixed)


def _corner_xy(bounds: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    if corner == "sw":
        return minx, miny
    if corner == "se":
        return maxx, miny
    if corner == "ne":
        return maxx, maxy
    return minx, maxy


def _dedupe_closed(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not coords:
        return coords
    out: list[tuple[int, int]] = [coords[0]]
    for p in coords[1:]:
        if p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def force_manhattan(coords: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Snap to integers and insert elbows so every edge is horizontal or vertical."""
    if not coords:
        return []
    pts = [(int(round(x)), int(round(y))) for x, y in coords]
    pts = _dedupe_closed(pts)
    if len(pts) < 2:
        return pts
    forced: list[tuple[int, int]] = [pts[0]]
    for x, y in pts[1:]:
        px, py = forced[-1]
        if x != px and y != py:
            forced.append((x, py))
        if (x, y) != forced[-1]:
            forced.append((x, y))
    fx, fy = forced[0]
    lx, ly = forced[-1]
    if lx != fx and ly != fy:
        forced.append((fx, ly))
    return _dedupe_closed(forced)


def is_manhattan(coords: list[tuple[float, float]]) -> bool:
    if len(coords) < 3:
        return False
    ring = list(coords) + [coords[0]]
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if abs(x1 - x0) > 1e-9 and abs(y1 - y0) > 1e-9:
            return False
    return True


def boundary_from_params(start_rect: dict[str, Any], notches: list[dict[str, Any]]) -> tuple[Boundary, dict[str, Any]]:
    """Deterministic boundary from start rectangle + ordered corner notches."""
    w = int(start_rect["w"])
    h = int(start_rect["h"])
    layout = box(0, 0, w, h)
    for notch in notches:
        minx, miny, maxx, maxy = (int(v) for v in layout.bounds)
        corner = notch["corner"]
        cw = max(1, int(notch["w"]))
        ch = max(1, int(notch["h"]))
        cx, cy = _corner_xy((minx, miny, maxx, maxy), corner)
        cut = box(cx - cw // 2, cy - ch // 2, cx - cw // 2 + cw, cy - ch // 2 + ch)
        layout = _largest_polygon(layout.difference(cut))
        if layout is None:
            layout = box(0, 0, w, h)
            break

    coords = force_manhattan(list(layout.exterior.coords[:-1]))
    poly = _largest_polygon(Polygon(coords))
    if poly is None:
        coords = [(0, 0), (w, 0), (w, h), (0, h)]
        poly = Polygon(coords)
    final = force_manhattan(list(poly.exterior.coords[:-1]))
    ox = min(x for x, _ in final)
    oy = min(y for _, y in final)
    final = [(x - ox, y - oy) for x, y in final]
    maxx = max(x for x, _ in final)
    maxy = max(y for _, y in final)
    boundary = Boundary(exterior=to_points(final))
    return boundary, {
        "polygon": [[float(x), float(y)] for x, y in final],
        "bbox_area": float(maxx * maxy),
        "polygon_area": float(Polygon(final).area),
    }


def sample_boundary_params(
    rng: random.Random,
    *,
    target_area: Optional[float] = None,
    area_min: float = 45.0,
    area_max: float = 110.0,
) -> dict[str, Any]:
    """Sample parameter-side choices for a notched boundary (no geometry output)."""
    if target_area is None:
        target_area = float(rng.randint(int(area_min), int(area_max)))
    target_area = max(float(target_area), 16.0)
    aspect = rng.uniform(0.75, 1.4)
    w = max(6, int(round(math.sqrt(target_area * aspect))))
    h = max(6, int(round(target_area / w)))
    n = rng.randint(0, 4)
    notches: list[dict[str, Any]] = []
    # Sample notch sizes against the original rect; corners chosen sequentially at apply time
    for _ in range(n):
        notches.append(
            {
                "corner": rng.choice(_CORNERS),
                "w": max(1, int(round(rng.uniform(0.1, 0.3) * w))),
                "h": max(1, int(round(rng.uniform(0.1, 0.3) * h))),
            }
        )
    return {
        "target_area": target_area,
        "aspect": aspect,
        "start_rect": {"x": 0, "y": 0, "w": w, "h": h},
        "notches": notches,
    }


def generate_notched_boundary(
    rng: random.Random,
    *,
    target_area: Optional[float] = None,
    area_min: float = 45.0,
    area_max: float = 110.0,
    max_attempts: int = 24,
) -> tuple[Boundary, dict[str, Any]]:
    """Sample params + apply. Meta includes both parameter fields and generated polygon."""
    del max_attempts
    params = sample_boundary_params(rng, target_area=target_area, area_min=area_min, area_max=area_max)
    boundary, generated = boundary_from_params(params["start_rect"], params["notches"])
    return boundary, {
        **params,
        "n_notches": len(params["notches"]),
        "boundary_polygon": [(int(p[0]), int(p[1])) for p in generated["polygon"]],
        "bbox_area": generated["bbox_area"],
        "polygon_area": generated["polygon_area"],
    }
