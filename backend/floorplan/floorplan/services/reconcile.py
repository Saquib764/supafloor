"""Reconcile floor plans: clip to boundary, fix overlaps, fix openings.

Order:
1. Clip each room to boundary ∩ room; drop zero-area rooms
2. Clip doors/windows to boundary; drop empty segments
3. When two rooms overlap, carve intersection out of the larger room
4. Remove hanging doors/windows (not attached to remaining rooms)
5. Ensure every remaining room has at least one door
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from floorplan.geometry.authors import make_door_line
from floorplan.geometry.ops import polygon_from_boundary, polygon_from_room, shared_edge, to_points
from floorplan.schemas.models import Door, FloorPlan, Point2D, Room, Window, _uid

MIN_ROOM_AREA = 0.05  # ft²
MIN_LINE_LEN = 0.05  # ft
MIN_OVERLAP = 0.01  # ft²
DOOR_LEN = 0.9
EDGE_TOL = 0.08


def _longest_line(geom):
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
        return _longest_line(unary_union(lines))
    return None


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        poly = max(geom.geoms, key=lambda g: g.area)
        return poly if poly.area > 0 else None
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return _largest_polygon(unary_union(polys))
    return None


def _poly_to_points(poly: Polygon) -> list[Point2D]:
    return to_points(list(poly.exterior.coords[:-1]))


def _clip_line_to_polygon(start: Point2D, end: Point2D, poly: Polygon) -> tuple[Point2D, Point2D] | None:
    line = LineString([(start.x, start.y), (end.x, end.y)])
    inter = line.intersection(poly)
    if inter.is_empty:
        inter = line.intersection(poly.buffer(EDGE_TOL))
    seg = _longest_line(inter)
    if seg is None or seg.length < MIN_LINE_LEN:
        return None
    c0, c1 = seg.coords[0], seg.coords[-1]
    if c0 == c1:
        return None
    return Point2D(x=float(c0[0]), y=float(c0[1])), Point2D(x=float(c1[0]), y=float(c1[1]))


def _line_on_room_edge(start: Point2D, end: Point2D, room: Room) -> bool:
    line = LineString([(start.x, start.y), (end.x, end.y)])
    return line.distance(polygon_from_room(room).boundary) <= EDGE_TOL


def _segment_on_linestring(ls: LineString, length: float = DOOR_LEN) -> tuple[Point2D, Point2D] | None:
    if ls.length < MIN_LINE_LEN:
        return None
    use = min(length, ls.length * 0.8)
    mid = ls.length / 2.0
    a = max(0.0, mid - use / 2.0)
    b = min(ls.length, a + use)
    p0 = ls.interpolate(a)
    p1 = ls.interpolate(b)
    if p0.coords[0] == p1.coords[0]:
        return None
    return Point2D(x=float(p0.x), y=float(p0.y)), Point2D(x=float(p1.x), y=float(p1.y))


def _clip_rooms(rooms: list[Room], boundary_poly: Polygon) -> list[Room]:
    clipped: list[Room] = []
    for room in rooms:
        inter = _largest_polygon(polygon_from_room(room).intersection(boundary_poly))
        if inter is None or inter.area < MIN_ROOM_AREA:
            continue
        clipped.append(room.model_copy(update={"polygon": _poly_to_points(inter)}))
    return clipped


def _clip_doors(doors: list[Door], boundary_poly: Polygon) -> list[Door]:
    out: list[Door] = []
    for door in doors:
        ends = _clip_line_to_polygon(door.start, door.end, boundary_poly)
        if ends is None:
            continue
        out.append(door.model_copy(update={"start": ends[0], "end": ends[1]}))
    return out


def _clip_windows(windows: list[Window], boundary_poly: Polygon) -> list[Window]:
    out: list[Window] = []
    for win in windows:
        ends = _clip_line_to_polygon(win.start, win.end, boundary_poly)
        if ends is None:
            continue
        out.append(win.model_copy(update={"start": ends[0], "end": ends[1]}))
    return out


def _resolve_overlaps(rooms: list[Room]) -> list[Room]:
    """Carve pairwise overlaps out of the larger room."""
    if len(rooms) < 2:
        return rooms
    polys = [polygon_from_room(r) for r in rooms]
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if polys[i] is None or polys[j] is None:
                continue
            if polys[i].is_empty or polys[j].is_empty:
                continue
            inter = polys[i].intersection(polys[j])
            if inter.is_empty or inter.area < MIN_OVERLAP:
                continue
            if polys[i].area >= polys[j].area:
                polys[i] = _largest_polygon(polys[i].difference(inter)) or Polygon()
            else:
                polys[j] = _largest_polygon(polys[j].difference(inter)) or Polygon()

    out: list[Room] = []
    for room, poly in zip(rooms, polys):
        if poly is None or poly.is_empty or poly.area < MIN_ROOM_AREA:
            continue
        out.append(room.model_copy(update={"polygon": _poly_to_points(poly)}))
    return out


def _drop_hanging_doors(doors: list[Door], rooms: list[Room]) -> list[Door]:
    room_ids = {r.id for r in rooms}
    kept: list[Door] = []
    for door in doors:
        connects = [c for c in door.connects if c in room_ids]
        if not any(_line_on_room_edge(door.start, door.end, r) for r in rooms):
            continue
        kept.append(door.model_copy(update={"connects": connects}))
    return kept


def _drop_hanging_windows(windows: list[Window], rooms: list[Room]) -> list[Window]:
    room_ids = {r.id for r in rooms}
    kept: list[Window] = []
    for win in windows:
        ids = [i for i in win.room_ids if i in room_ids]
        if not any(_line_on_room_edge(win.start, win.end, r) for r in rooms):
            continue
        kept.append(win.model_copy(update={"room_ids": ids}))
    return kept


def _ensure_doors(rooms: list[Room], doors: list[Door]) -> list[Door]:
    covered: set[str] = set()
    for door in doors:
        covered.update(door.connects)

    result = list(doors)
    for room in rooms:
        if room.id in covered:
            continue
        # Prefer a door on a shared edge with a neighbor
        placed = False
        for other in rooms:
            if other.id == room.id:
                continue
            edge = shared_edge(room.coords(), other.coords(), tol=EDGE_TOL)
            if edge is None:
                continue
            ends = _segment_on_linestring(edge)
            if ends is None:
                continue
            result.append(
                make_door_line(
                    (ends[0].x, ends[0].y),
                    (ends[1].x, ends[1].y),
                    connects=[room.id, other.id],
                )
            )
            covered.add(room.id)
            covered.add(other.id)
            placed = True
            break
        if placed:
            continue
        # Fallback: door on the room's own boundary
        boundary = polygon_from_room(room).exterior
        ends = _segment_on_linestring(LineString(boundary.coords))
        if ends is None:
            continue
        result.append(
            make_door_line(
                (ends[0].x, ends[0].y),
                (ends[1].x, ends[1].y),
                connects=[room.id],
                is_exterior=True,
            )
        )
        covered.add(room.id)
    return result


def reconcile_plan(plan: FloorPlan) -> FloorPlan:
    """Apply boundary clip, overlap carve, opening cleanup, and door guarantee."""
    boundary = plan.ensure_boundary()
    boundary_poly = polygon_from_boundary(boundary)

    rooms = _clip_rooms(plan.rooms, boundary_poly)
    doors = _clip_doors(plan.doors, boundary_poly)
    windows = _clip_windows(plan.windows, boundary_poly)

    rooms = _resolve_overlaps(rooms)

    doors = _drop_hanging_doors(doors, rooms)
    windows = _drop_hanging_windows(windows, rooms)
    doors = _ensure_doors(rooms, doors)

    return FloorPlan(
        id=_uid(),
        name=plan.name,
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        zones=plan.zones,
        constraint_graph=plan.constraint_graph,
        metadata={**plan.metadata, "reconciled": True},
    )
