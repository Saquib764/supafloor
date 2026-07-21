"""Boundary adaptation — fit a template layout into an arbitrary boundary."""

from __future__ import annotations

from floorplan.geometry.ops import points_to_polygon, to_points
from floorplan.schemas.models import Boundary, FloorPlan, Point2D
from shapely import affinity


def adapt_to_boundary(plan: FloorPlan, boundary: Boundary) -> FloorPlan:
    """Stretch/compress plan geometry to fit target boundary bbox (topology preserved)."""
    src_boundary = plan.ensure_boundary()
    src = points_to_polygon(src_boundary.coords())
    dst = points_to_polygon(boundary.coords())
    sx0, sy0, sx1, sy1 = src.bounds
    dx0, dy0, dx1, dy1 = dst.bounds
    sw, sh = max(sx1 - sx0, 1e-9), max(sy1 - sy0, 1e-9)
    dw, dh = max(dx1 - dx0, 1e-9), max(dy1 - dy0, 1e-9)
    xfact, yfact = dw / sw, dh / sh

    def xform_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        poly = points_to_polygon(coords)
        moved = affinity.translate(poly, xoff=-sx0, yoff=-sy0)
        scaled = affinity.scale(moved, xfact=xfact, yfact=yfact, origin=(0, 0))
        final = affinity.translate(scaled, xoff=dx0, yoff=dy0)
        return list(final.exterior.coords[:-1])

    def xform_point(p: Point2D) -> Point2D:
        x = (p.x - sx0) * xfact + dx0
        y = (p.y - sy0) * yfact + dy0
        return Point2D(x=x, y=y)

    rooms = [r.model_copy(update={"polygon": to_points(xform_coords(r.coords()))}) for r in plan.rooms]
    doors = [
        d.model_copy(update={"start": xform_point(d.start), "end": xform_point(d.end)})
        for d in plan.doors
    ]
    windows = [
        w.model_copy(update={"start": xform_point(w.start), "end": xform_point(w.end)})
        for w in plan.windows
    ]
    walls = [
        wall.model_copy(update={"start": xform_point(wall.start), "end": xform_point(wall.end)})
        for wall in plan.walls
    ]
    return FloorPlan(
        name=plan.name,
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        walls=walls,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "adapted": True},
    )


def generate_boundary(width: float = 10.0, depth: float = 8.0, shape: str = "rect") -> Boundary:
    if shape == "l":
        coords = [
            (0, 0),
            (width, 0),
            (width, depth * 0.55),
            (width * 0.55, depth * 0.55),
            (width * 0.55, depth),
            (0, depth),
        ]
        return Boundary(exterior=to_points(coords))
    return Boundary(exterior=to_points([(0, 0), (width, 0), (width, depth), (0, depth)]))
