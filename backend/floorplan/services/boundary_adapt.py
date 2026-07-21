"""Boundary adaptation — fit a template layout into an arbitrary boundary."""

from __future__ import annotations

from floorplan.geometry.ops import points_to_polygon, to_points
from floorplan.schemas.models import Boundary, FloorPlan, Point2D


def adapt_to_boundary(plan: FloorPlan, boundary: Boundary) -> FloorPlan:
    """Stretch/compress plan geometry to fit target boundary bbox (topology preserved)."""
    src = points_to_polygon(plan.ensure_boundary().coords())
    dst = points_to_polygon(boundary.coords())
    sx0, sy0, sx1, sy1 = src.bounds
    dx0, dy0, dx1, dy1 = dst.bounds
    sw, sh = max(sx1 - sx0, 1e-9), max(sy1 - sy0, 1e-9)
    dw, dh = max(dx1 - dx0, 1e-9), max(dy1 - dy0, 1e-9)
    xfact, yfact = dw / sw, dh / sh

    def xform_point(p: Point2D) -> Point2D:
        return Point2D(x=(p.x - sx0) * xfact + dx0, y=(p.y - sy0) * yfact + dy0)

    def xform_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [((x - sx0) * xfact + dx0, (y - sy0) * yfact + dy0) for x, y in coords]

    rooms = [r.model_copy(update={"polygon": to_points(xform_coords(r.coords()))}) for r in plan.rooms]
    doors = [d.model_copy(update={"start": xform_point(d.start), "end": xform_point(d.end)}) for d in plan.doors]
    windows = [w.model_copy(update={"start": xform_point(w.start), "end": xform_point(w.end)}) for w in plan.windows]
    return FloorPlan(
        name=plan.name,
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
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
