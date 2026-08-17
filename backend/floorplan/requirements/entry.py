"""Sample an entry point on the unit boundary."""

from __future__ import annotations

import random
from typing import Any

from shapely.geometry import LineString, Point, Polygon

from floorplan.units import MIN_OPENING_EDGE_FT


def _boundary_edges(boundary_poly: Polygon) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    coords = list(boundary_poly.exterior.coords)
    edges: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for i in range(len(coords) - 1):
        a = (float(coords[i][0]), float(coords[i][1]))
        b = (float(coords[i + 1][0]), float(coords[i + 1][1]))
        length = LineString([a, b]).length
        if length >= MIN_OPENING_EDGE_FT:
            edges.append((a, b, length))
    return edges


def sample_entry_point(boundary_poly: Polygon, rng: random.Random) -> dict[str, Any]:
    """Return a point on the boundary and the host edge segment."""
    edges = _boundary_edges(boundary_poly)
    if not edges:
        minx, miny, maxx, maxy = boundary_poly.bounds
        mid = (minx + maxx) / 2.0, miny
        return {
            "entry_point": [mid[0], mid[1]],
            "entry_edge": [[minx, miny], [maxx, miny]],
        }

    weights = [length for _, _, length in edges]
    total = sum(weights)
    pick = rng.uniform(0, total)
    acc = 0.0
    chosen = edges[0]
    for edge in edges:
        acc += edge[2]
        if pick <= acc:
            chosen = edge
            break

    a, b, length = chosen
    t = rng.uniform(0.15, 0.85)
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * t
    pt = Point(x, y)
    if boundary_poly.boundary.distance(pt) > 0.05:
        nearest = boundary_poly.boundary.interpolate(boundary_poly.boundary.project(pt))
        x, y = float(nearest.x), float(nearest.y)

    return {
        "entry_point": [round(x, 3), round(y, 3)],
        "entry_edge": [list(a), list(b)],
    }
