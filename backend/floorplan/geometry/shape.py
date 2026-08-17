"""Room shape quality metrics."""

from __future__ import annotations

from shapely.geometry import Point

from geometry.ops import min_width, points_to_polygon, rectangularity

Coord = tuple[float, float]


def _open_ring(coords: list[Coord]) -> list[Coord]:
    pts = list(coords)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        return pts[:-1]
    return pts


def reflex_corner_count(coords: list[Coord]) -> int:
    """Count concave (reflex) vertices on an orthogonal ring."""
    ring = _open_ring(list(coords))
    n = len(ring)
    if n < 4:
        return 0
    count = 0
    for i in range(n):
        x0, y0 = ring[(i - 1) % n]
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            continue
        if cross < 0:
            count += 1
    return count


def compactness(coords: list[Coord]) -> float:
    """4π·A/P² in (0, 1]; circle = 1."""
    poly = points_to_polygon(coords)
    if poly.is_empty or poly.length <= 0:
        return 0.0
    return float(min(1.0, 4.0 * 3.141592653589793 * poly.area / (poly.length * poly.length)))


def max_inscribed_rect_ratio(coords: list[Coord]) -> float:
    """Largest axis-aligned rectangle area / polygon area."""
    poly = points_to_polygon(coords)
    a = poly.area
    if a <= 0:
        return 0.0
    ring = _open_ring(list(coords))
    xs = sorted({x for x, _ in ring})
    ys = sorted({y for _, y in ring})
    best = 0.0
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            for k in range(len(ys) - 1):
                for m in range(k + 1, len(ys)):
                    corners = [(xs[i], ys[k]), (xs[j], ys[k]), (xs[j], ys[m]), (xs[i], ys[m])]
                    if all(poly.contains(Point(x, y)) for x, y in corners):
                        rect_a = (xs[j] - xs[i]) * (ys[m] - ys[k])
                        if rect_a > best:
                            best = rect_a
    if best <= 0:
        minx, miny, maxx, maxy = poly.bounds
        best = (maxx - minx) * (maxy - miny)
    return float(min(1.0, best / a))


def appendage_ratio(coords: list[Coord]) -> float:
    """Fraction of area lost to narrow protrusions (buffer-erode diff)."""
    poly = points_to_polygon(coords)
    if poly.is_empty or poly.area <= 0:
        return 0.0
    mw = min_width(coords)
    r = max(0.15, mw * 0.35)
    core = poly.buffer(-r)
    if core.is_empty:
        return 0.0
    if core.geom_type == "MultiPolygon":
        core = max(core.geoms, key=lambda g: g.area)
    if core.geom_type != "Polygon":
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - core.area / poly.area)))


def shape_subscores(coords: list[Coord]) -> dict[str, float]:
    """Return 0–10 sub-scores for shape components."""
    rect = rectangularity(coords)
    comp = compactness(coords)
    mir = max_inscribed_rect_ratio(coords)
    reflex = reflex_corner_count(coords)
    append = appendage_ratio(coords)
    mw = min_width(coords)

    return {
        "rectangularity": rect * 10.0,
        "compactness": min(10.0, comp * 12.0),
        "max_rect": mir * 10.0,
        "reflex": max(0.0, 10.0 - max(0, reflex - 2) * 2.0),
        "appendage": max(0.0, 10.0 - append * 40.0),
        "min_width": min(10.0, mw * 1.5),
    }
