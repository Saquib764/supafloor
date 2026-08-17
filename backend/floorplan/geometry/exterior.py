"""Exterior wall and window placement analysis."""

from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from geometry.ops import polygon_from_boundary, polygon_from_room
from schemas.models import FloorPlan, Room, Window
from schemas.units import MIN_OPENING_EDGE_FT

EDGE_TOL = 0.08
MIN_LINE_LEN = 0.05


def longest_line(geom) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length)
    if geom.geom_type == "GeometryCollection":
        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
        if not lines:
            return None
        return longest_line(unary_union(lines))
    return None


def _ring_edges(coords) -> list[LineString]:
    edges: list[LineString] = []
    pts = list(coords)
    for i in range(len(pts) - 1):
        if pts[i] == pts[i + 1]:
            continue
        edge = LineString([pts[i], pts[i + 1]])
        if edge.length >= MIN_LINE_LEN:
            edges.append(edge)
    return edges


def boundary_overlap(room: Room, boundary_poly: Polygon):
    poly = polygon_from_room(room)
    inter = poly.boundary.intersection(boundary_poly.boundary)
    if inter.is_empty:
        inter = poly.buffer(EDGE_TOL).intersection(boundary_poly.buffer(EDGE_TOL)).boundary
    return inter


def merged_boundary_run(inter) -> LineString | None:
    """Merge collinear boundary overlap segments into the longest wall run."""
    if inter is None or inter.is_empty:
        return None
    lines: list[LineString] = []
    if inter.geom_type == "LineString":
        lines = [inter]
    elif inter.geom_type == "MultiLineString":
        lines = list(inter.geoms)
    else:
        return longest_line(inter)

    h_groups: dict[float, list[LineString]] = {}
    v_groups: dict[float, list[LineString]] = {}
    for ls in lines:
        x0, y0 = ls.coords[0]
        x1, y1 = ls.coords[-1]
        if abs(y0 - y1) <= 1e-6:
            h_groups.setdefault(round(y0, 3), []).append(ls)
        elif abs(x0 - x1) <= 1e-6:
            v_groups.setdefault(round(x0, 3), []).append(ls)

    best: LineString | None = None
    best_len = 0.0
    for y, segs in h_groups.items():
        xs = [c[0] for s in segs for c in s.coords]
        run = LineString([(min(xs), y), (max(xs), y)])
        if run.length > best_len:
            best_len = run.length
            best = run
    for x, segs in v_groups.items():
        ys = [c[1] for s in segs for c in s.coords]
        run = LineString([(x, min(ys)), (x, max(ys))])
        if run.length > best_len:
            best_len = run.length
            best = run
    return best


def room_exterior_edge(room: Room, boundary_poly: Polygon) -> LineString | None:
    """Longest room wall segment that lies on the unit boundary."""
    line = merged_boundary_run(boundary_overlap(room, boundary_poly))
    if line is not None and line.length >= MIN_OPENING_EDGE_FT:
        return line
    poly = polygon_from_room(room)
    best: LineString | None = None
    for edge in _ring_edges(poly.exterior.coords):
        overlap = edge.intersection(boundary_poly.boundary)
        seg = merged_boundary_run(overlap) or longest_line(overlap)
        if seg is None or seg.length < MIN_OPENING_EDGE_FT:
            continue
        if best is None or seg.length > best.length:
            best = seg
    return best


def exterior_edge_length(room: Room, boundary_poly: Polygon) -> float:
    edge = room_exterior_edge(room, boundary_poly)
    return float(edge.length) if edge is not None else 0.0


def window_on_exterior(window: Window, boundary_poly: Polygon, tol: float = EDGE_TOL) -> bool:
    line = LineString([(window.start.x, window.start.y), (window.end.x, window.end.y)])
    return line.distance(boundary_poly.boundary) <= tol


def window_length_on_exterior(window: Window, boundary_poly: Polygon, tol: float = EDGE_TOL) -> float:
    if window_on_exterior(window, boundary_poly, tol=tol):
        return window.length()
    return 0.0


def _outward_normal(edge: LineString, boundary_poly: Polygon) -> tuple[float, float]:
    x0, y0 = edge.coords[0]
    x1, y1 = edge.coords[-1]
    dx, dy = x1 - x0, y1 - y0
    nx, ny = -dy, dx
    ln = (nx * nx + ny * ny) ** 0.5 or 1.0
    nx, ny = nx / ln, ny / ln
    mid = edge.interpolate(0.5, normalized=True)
    test = Point(mid.x + nx * 0.5, mid.y + ny * 0.5)
    if boundary_poly.contains(test):
        nx, ny = -nx, -ny
    return nx, ny


def facade_blocker(
    room: Room,
    boundary_poly: Polygon,
    rooms_by_id: dict[str, Room],
    *,
    balcony_ids: set[str],
    tol: float = EDGE_TOL,
) -> str:
    """Return 'clear', 'balcony', 'blocked', or 'none' for exterior exposure."""
    edge = room_exterior_edge(room, boundary_poly)
    if edge is None:
        return "none"
    mid = edge.interpolate(0.5, normalized=True)
    nx, ny = _outward_normal(edge, boundary_poly)
    ray_end = Point(mid.x + nx * 30.0, mid.y + ny * 30.0)
    ray = LineString([(mid.x, mid.y), (ray_end.x, ray_end.y)])
    room_poly = polygon_from_room(room)

    hit_balcony = False
    hit_block = False
    for other in rooms_by_id.values():
        if other.id == room.id:
            continue
        op = polygon_from_room(other)
        if not ray.intersects(op):
            continue
        inter_pt = ray.intersection(op.boundary)
        if inter_pt.is_empty:
            continue
        if op.contains(Point(mid.x + nx * 0.2, mid.y + ny * 0.2)):
            continue
        dist = room_poly.distance(op)
        if dist > tol:
            continue
        if other.id in balcony_ids:
            hit_balcony = True
        else:
            hit_block = True

    if hit_block:
        return "blocked"
    if hit_balcony:
        return "balcony"
    return "clear"


def room_exterior_segment(room: Room, boundary_poly: Polygon) -> LineString | None:
    """Any boundary-coincident segment (low length threshold — window placement)."""
    line = merged_boundary_run(boundary_overlap(room, boundary_poly))
    return line if line is not None and line.length >= MIN_LINE_LEN else None


def boundary_polygon(plan: FloorPlan) -> Polygon:
    return polygon_from_boundary(plan.ensure_boundary())
