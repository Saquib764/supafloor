"""Geometry operations backed by Shapely.

All functions accept/return plain coordinate lists or Shapely geometries.
Domain models stay immutable; callers reconstruct models after transforms.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from thefluxtrain.floorplan.schemas.models import Boundary, Point2D, Room


Coord = tuple[float, float]
CoordSeq = Sequence[Coord]


def points_to_polygon(coords: CoordSeq, holes: Optional[Sequence[CoordSeq]] = None) -> Polygon:
    poly = Polygon(coords, holes or [])
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        elif poly.geom_type != "Polygon":
            poly = poly.convex_hull
    return poly


def polygon_from_room(room: Room) -> Polygon:
    return points_to_polygon(room.coords())


def polygon_from_boundary(boundary: Boundary) -> Polygon:
    holes = [[p.as_tuple() for p in hole] for hole in boundary.holes]
    return points_to_polygon(boundary.coords(), holes)


def to_points(coords: Iterable[Coord]) -> list[Point2D]:
    return [Point2D(x=float(x), y=float(y)) for x, y in coords]


def area(coords: CoordSeq) -> float:
    return float(points_to_polygon(coords).area)


def centroid(coords: CoordSeq) -> Coord:
    c = points_to_polygon(coords).centroid
    return (float(c.x), float(c.y))


def bounding_box(coords: CoordSeq) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy)."""
    return points_to_polygon(coords).bounds


def union(a: CoordSeq, b: CoordSeq) -> BaseGeometry:
    return points_to_polygon(a).union(points_to_polygon(b))


def intersection(a: CoordSeq, b: CoordSeq) -> BaseGeometry:
    return points_to_polygon(a).intersection(points_to_polygon(b))


def difference(a: CoordSeq, b: CoordSeq) -> BaseGeometry:
    return points_to_polygon(a).difference(points_to_polygon(b))


def shared_edge(a: CoordSeq, b: CoordSeq, tol: float = 0.05) -> Optional[LineString]:
    """Return shared boundary segment between two polygons, if any."""
    ga = points_to_polygon(a).buffer(0)
    gb = points_to_polygon(b).buffer(0)
    inter = ga.boundary.intersection(gb.boundary)
    if inter.is_empty:
        # Near-touching: expand slightly
        inter = ga.buffer(tol).intersection(gb.buffer(tol)).boundary
    if inter.is_empty:
        return None
    if inter.geom_type == "LineString":
        return inter if inter.length >= tol else None
    if inter.geom_type == "MultiLineString":
        longest = max(inter.geoms, key=lambda g: g.length)
        return longest if longest.length >= tol else None
    if inter.geom_type == "GeometryCollection":
        lines = [g for g in inter.geoms if g.geom_type in ("LineString", "MultiLineString")]
        if not lines:
            return None
        merged = unary_union(lines)
        if merged.geom_type == "LineString":
            return merged if merged.length >= tol else None
        if merged.geom_type == "MultiLineString":
            longest = max(merged.geoms, key=lambda g: g.length)
            return longest if longest.length >= tol else None
    return None


def contains(outer: CoordSeq, inner: CoordSeq) -> bool:
    return points_to_polygon(outer).covers(points_to_polygon(inner))


def touches(a: CoordSeq, b: CoordSeq, tol: float = 0.05) -> bool:
    ga = points_to_polygon(a)
    gb = points_to_polygon(b)
    if ga.touches(gb) or ga.intersects(gb):
        edge = shared_edge(a, b, tol=tol)
        return edge is not None
    return ga.buffer(tol).intersects(gb.buffer(tol)) and not ga.buffer(-tol).intersects(gb.buffer(-tol))


def distance(a: CoordSeq, b: CoordSeq) -> float:
    return float(points_to_polygon(a).distance(points_to_polygon(b)))


def offset(coords: CoordSeq, distance_m: float) -> list[Coord]:
    poly = points_to_polygon(coords).buffer(distance_m, join_style=2)
    if poly.is_empty:
        return list(coords)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return list(poly.exterior.coords[:-1])


def simplify(coords: CoordSeq, tolerance: float = 0.05) -> list[Coord]:
    poly = points_to_polygon(coords).simplify(tolerance, preserve_topology=True)
    if poly.is_empty or poly.geom_type != "Polygon":
        return list(coords)
    return list(poly.exterior.coords[:-1])


def rotate(coords: CoordSeq, angle_deg: float, origin: str | Coord = "centroid") -> list[Coord]:
    poly = affinity.rotate(points_to_polygon(coords), angle_deg, origin=origin)
    return list(poly.exterior.coords[:-1])


def translate(coords: CoordSeq, dx: float, dy: float) -> list[Coord]:
    poly = affinity.translate(points_to_polygon(coords), xoff=dx, yoff=dy)
    return list(poly.exterior.coords[:-1])


def scale(coords: CoordSeq, xfact: float, yfact: float | None = None, origin: str | Coord = "centroid") -> list[Coord]:
    if yfact is None:
        yfact = xfact
    poly = affinity.scale(points_to_polygon(coords), xfact=xfact, yfact=yfact, origin=origin)
    return list(poly.exterior.coords[:-1])


def aspect_ratio(coords: CoordSeq) -> float:
    minx, miny, maxx, maxy = bounding_box(coords)
    w = max(maxx - minx, 1e-9)
    h = max(maxy - miny, 1e-9)
    return max(w / h, h / w)


def perimeter(coords: CoordSeq) -> float:
    return float(points_to_polygon(coords).length)


def min_width(coords: CoordSeq, samples: int = 16) -> float:
    """Approximate minimum width via bounding-box shorter side and inscribed circle radius * 2."""
    poly = points_to_polygon(coords)
    if poly.is_empty:
        return 0.0
    minx, miny, maxx, maxy = poly.bounds
    bbox_min = min(maxx - minx, maxy - miny)
    # Maximum inscribed circle approximation via distance to boundary from interior points
    cx, cy = poly.centroid.x, poly.centroid.y
    if not poly.contains(Point(cx, cy)):
        return float(bbox_min)
    radius = poly.exterior.distance(Point(cx, cy))
    return float(min(bbox_min, radius * 2))


def rectangularity(coords: CoordSeq) -> float:
    """Area / minimum rotated rectangle area, in [0, 1]."""
    poly = points_to_polygon(coords)
    if poly.area <= 0:
        return 0.0
    mrr = poly.minimum_rotated_rectangle
    if mrr.area <= 0:
        return 0.0
    return float(min(1.0, poly.area / mrr.area))


def fit_polygon_in_boundary(inner: CoordSeq, boundary: CoordSeq) -> list[Coord]:
    """Scale and translate inner polygon so its bbox fits inside boundary bbox."""
    b = points_to_polygon(boundary)
    i = points_to_polygon(inner)
    if i.is_empty or b.is_empty:
        return list(inner)
    bx0, by0, bx1, by1 = b.bounds
    ix0, iy0, ix1, iy1 = i.bounds
    bw, bh = max(bx1 - bx0, 1e-9), max(by1 - by0, 1e-9)
    iw, ih = max(ix1 - ix0, 1e-9), max(iy1 - iy0, 1e-9)
    factor = min(bw / iw, bh / ih) * 0.98
    scaled = affinity.scale(i, xfact=factor, yfact=factor, origin="centroid")
    sx0, sy0, sx1, sy1 = scaled.bounds
    dx = (bx0 + bx1) / 2 - (sx0 + sx1) / 2
    dy = (by0 + by1) / 2 - (sy0 + sy1) / 2
    moved = affinity.translate(scaled, xoff=dx, yoff=dy)
    return list(moved.exterior.coords[:-1])


def rect(x: float, y: float, w: float, h: float) -> list[Coord]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def bbox_polygon(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
    return box(minx, miny, maxx, maxy)
