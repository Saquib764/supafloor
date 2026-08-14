"""Geometry generation and opening placement."""

from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Point, Polygon, box

from floorplan.geometry.authors import make_door_line
from floorplan.geometry.ops import polygon_from_boundary, polygon_from_room, shared_edge
from floorplan.geometry.exterior import (
    EDGE_TOL as _EDGE_TOL,
    MIN_LINE_LEN as _MIN_LINE_LEN,
    room_exterior_edge as _boundary_edge,
)
from floorplan.geometry.boundary import boundary_from_params, force_manhattan, is_manhattan
from floorplan.requirements.areas import name_root
from floorplan.requirements.program import (
    bedroom_parent,
    closet_parent,
    is_bedroom,
    is_toilet,
    toilet_parent,
)
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, Door, Room
from floorplan.units import (
    EXTERIOR_DOOR_FT,
    INTERIOR_DOOR_FT,
    MIN_OPENING_EDGE_FT,
)

_MIN_SPACE_AREA = 0.5

_NAME_TO_TYPE: dict[str, RoomType] = {
    "Living/Sleeping": RoomType.LIVING,
    "Living": RoomType.LIVING,
    "Room": RoomType.BEDROOM,
    "Toilet": RoomType.POWDER,
    "Bathroom": RoomType.BATHROOM,
    "Kitchen": RoomType.KITCHEN,
    "Dining": RoomType.DINING,
    "Hallway": RoomType.HALLWAY,
    "Closet": RoomType.CLOSET,
    "Balcony": RoomType.BALCONY,
    "Laundry": RoomType.LAUNDRY,
    "Garage": RoomType.STORAGE,
}

_NAME_TO_ZONE: dict[str, ZoneType] = {
    "Living/Sleeping": ZoneType.PUBLIC,
    "Living": ZoneType.PUBLIC,
    "Room": ZoneType.PRIVATE,
    "Toilet": ZoneType.WET,
    "Bathroom": ZoneType.WET,
    "Kitchen": ZoneType.SERVICE,
    "Dining": ZoneType.PUBLIC,
    "Hallway": ZoneType.CIRCULATION,
    "Closet": ZoneType.PRIVATE,
    "Balcony": ZoneType.OUTDOOR,
    "Laundry": ZoneType.SERVICE,
    "Garage": ZoneType.SERVICE,
}


def _room_type_for_name(name: str) -> RoomType:
    if name == "Room 1":
        return RoomType.MASTER_BEDROOM
    if name in _NAME_TO_TYPE:
        return _NAME_TO_TYPE[name]
    root = name_root(name)
    if name.endswith(" Bath"):
        return RoomType.BATHROOM
    if name.endswith(" Toilet"):
        return RoomType.POWDER
    if name.endswith(" Closet"):
        return RoomType.CLOSET
    return _NAME_TO_TYPE.get(root, RoomType.OTHER)


def _zone_for_name(name: str) -> ZoneType:
    if name in _NAME_TO_ZONE:
        return _NAME_TO_ZONE[name]
    rtype = _room_type_for_name(name)
    if rtype in (RoomType.BEDROOM, RoomType.MASTER_BEDROOM, RoomType.CLOSET):
        return ZoneType.PRIVATE
    if rtype in (RoomType.BATHROOM, RoomType.POWDER):
        return ZoneType.WET
    if rtype == RoomType.HALLWAY:
        return ZoneType.CIRCULATION
    if rtype == RoomType.BALCONY:
        return ZoneType.OUTDOOR
    if rtype in (RoomType.KITCHEN, RoomType.LAUNDRY, RoomType.STORAGE):
        return ZoneType.SERVICE
    return ZoneType.PUBLIC


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        poly = max(geom.geoms, key=lambda g: g.area)
        return poly if poly.area > 0 else None
    return None


def _poly_points(poly: Polygon) -> list[list[float]]:
    coords = force_manhattan(list(poly.exterior.coords[:-1]))
    return [[float(x), float(y)] for x, y in coords]


def _finalize_space_geom(geom: Polygon | None, name: str) -> dict[str, Any] | None:
    if geom is None or geom.area < _MIN_SPACE_AREA:
        return None
    if not is_manhattan(list(geom.exterior.coords[:-1])):
        geom = _largest_polygon(Polygon(force_manhattan(list(geom.exterior.coords[:-1]))))
    if geom is None or geom.area < _MIN_SPACE_AREA:
        return None
    return {"name": name, "polygon": _poly_points(geom), "area": round(float(geom.area), 3)}


def run_generation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Clip grown rectangles into the unit boundary."""
    boundary, boundary_gen = boundary_from_params(parameters["start_rect"], parameters.get("notches") or [])
    boundary_poly = Polygon([(p.x, p.y) for p in boundary.exterior])

    rectangles = parameters.get("rectangles") or []
    boundary_room = parameters.get("boundary_room") or "Living"
    if "Living/Sleeping" in {r.get("name") for r in rectangles}:
        boundary_room = "Living/Sleeping"

    name_order = [r.get("name") for r in rectangles if r.get("name")]
    if boundary_room in name_order:
        name_order = [boundary_room] + [n for n in name_order if n != boundary_room]
    rect_by_name = {r["name"]: r for r in rectangles if r.get("name")}

    geoms: dict[str, Polygon] = {}
    dropped: dict[str, str] = {}

    for name in name_order:
        rect = rect_by_name.get(name)
        if not rect:
            continue
        footprint = rect.get("footprint")
        if footprint:
            geom = _largest_polygon(Polygon(footprint).intersection(boundary_poly))
        else:
            geom = _largest_polygon(
                box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]).intersection(boundary_poly)
            )
        if geom is None or geom.area < _MIN_SPACE_AREA:
            dropped[name] = "outside_boundary"
            continue
        geoms[name] = geom

    spaces: list[dict[str, Any]] = []
    for name in name_order:
        if name in dropped:
            continue
        geom = geoms.get(name)
        if geom is None:
            continue
        finalized = _finalize_space_geom(geom, name)
        if finalized is not None:
            spaces.append(finalized)

    return {"boundary": boundary_gen, "spaces": spaces}



def _segment_on_linestring(
    ls: LineString,
    length: float = INTERIOR_DOOR_FT,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if ls.length < _MIN_LINE_LEN:
        return None
    use = min(length, ls.length * 0.8)
    mid = ls.length / 2.0
    a = max(0.0, mid - use / 2.0)
    b = min(ls.length, a + use)
    p0 = ls.interpolate(a)
    p1 = ls.interpolate(b)
    if p0.coords[0] == p1.coords[0]:
        return None
    return (float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))


def _living_room(rooms: list[Room]) -> Room | None:
    for name in ("Living", "Living/Sleeping"):
        for room in rooms:
            if room.name == name:
                return room
    return None


def _room_by_name(rooms: list[Room], name: str) -> Room | None:
    return next((room for room in rooms if room.name == name), None)


def _door_on_edge(
    edge: LineString,
    connects: list[str],
    *,
    is_exterior: bool = False,
    target: Point | None = None,
) -> Door | None:
    door_len = EXTERIOR_DOOR_FT if is_exterior else INTERIOR_DOOR_FT
    if edge.length < door_len:
        return None
    if target is not None:
        t = edge.project(target)
        use = min(door_len, edge.length * 0.8)
        a = max(0.0, min(t - use / 2.0, edge.length - use))
        b = a + use
        p0 = edge.interpolate(a)
        p1 = edge.interpolate(b)
        if p0.coords[0] != p1.coords[0]:
            return make_door_line(
                (float(p0.x), float(p0.y)),
                (float(p1.x), float(p1.y)),
                connects=connects,
                is_exterior=is_exterior,
            )
    ends = _segment_on_linestring(edge)
    if ends is None:
        return None
    return make_door_line(ends[0], ends[1], connects=connects, is_exterior=is_exterior)


def _toilet_host_ids(rooms: list[Room], doors: list[Door]) -> set[str]:
    """Bedroom room ids that already have a toilet door connected."""
    by_id = {r.id: r for r in rooms}
    hosts: set[str] = set()
    for door in doors:
        if len(door.connects) < 2:
            continue
        a = by_id.get(door.connects[0])
        b = by_id.get(door.connects[1])
        if a and b and is_toilet(a.name) and is_bedroom(b.name):
            hosts.add(b.id)
        if a and b and is_toilet(b.name) and is_bedroom(a.name):
            hosts.add(a.id)
    return hosts


def _toilet_touches_parent(toilet: Room, rooms: list[Room]) -> bool:
    parent_name = toilet_parent(toilet.name)
    if not parent_name:
        return False
    parent = _room_by_name(rooms, parent_name)
    if parent is None:
        return False
    return shared_edge(toilet.coords(), parent.coords(), tol=_EDGE_TOL) is not None


def _toilet_door_allowed(a: Room, b: Room, toilet_hosts: set[str], rooms: list[Room]) -> bool:
    """Toilet doors prefer the parent bedroom; orphans may use any free adjacent host."""
    for toilet, host in ((a, b), (b, a)):
        if not is_toilet(toilet.name):
            continue
        parent = toilet_parent(toilet.name)
        if parent is None:
            continue
        if parent != host.name and _toilet_touches_parent(toilet, rooms):
            return False
        # One toilet per bedroom host; non-bedroom hosts may serve orphan toilets.
        if host.id in toilet_hosts and (parent == host.name or is_bedroom(host.name)):
            return False
    return True


def _kitchen_bedroom_door_allowed(a: Room, b: Room) -> bool:
    """Kitchen must not connect directly to a bedroom."""
    kitchen_a = a.name == "Kitchen"
    kitchen_b = b.name == "Kitchen"
    if kitchen_a and is_bedroom(b.name):
        return False
    if kitchen_b and is_bedroom(a.name):
        return False
    return True


def _closet_door_allowed(a: Room, b: Room) -> bool:
    """Private closet doors must connect only to their bedroom host."""
    for closet, host in ((a, b), (b, a)):
        parent = closet_parent(closet.name)
        if parent is None:
            continue
        if host.name != parent:
            return False
    return True


def _add_door(
    doors: list[Door],
    covered: set[str],
    toilet_hosts: set[str],
    rooms: list[Room],
    a: Room,
    b: Room,
    edge: LineString,
    *,
    is_exterior: bool = False,
) -> bool:
    if not _toilet_door_allowed(a, b, toilet_hosts, rooms):
        return False
    if not _kitchen_bedroom_door_allowed(a, b):
        return False
    if not _closet_door_allowed(a, b):
        return False
    door = _door_on_edge(edge, [a.id, b.id], is_exterior=is_exterior)
    if door is None:
        return False
    doors.append(door)
    covered.add(a.id)
    covered.add(b.id)
    if is_toilet(a.name) and is_bedroom(b.name):
        toilet_hosts.add(b.id)
    if is_toilet(b.name) and is_bedroom(a.name):
        toilet_hosts.add(a.id)
    return True


def _door_on_boundary(door: Door, boundary_poly: Polygon) -> bool:
    line = LineString([(door.start.x, door.start.y), (door.end.x, door.end.y)])
    return line.distance(boundary_poly.boundary) <= _EDGE_TOL


def _has_boundary_door(doors: list[Door], boundary_poly: Polygon) -> bool:
    return any(_door_on_boundary(d, boundary_poly) for d in doors)


def _entry_host_rooms(rooms: list[Room]) -> list[Room]:
    """Preferred hosts for the unit entry door: Living, then Hall."""
    hosts: list[Room] = []
    living = _living_room(rooms)
    if living is not None:
        hosts.append(living)
    hall = _room_by_name(rooms, "Hallway")
    if hall is not None:
        hosts.append(hall)
    return hosts


def _ensure_boundary_door(
    doors: list[Door],
    rooms: list[Room],
    boundary_poly: Polygon,
    entry_point: list[float] | None,
) -> None:
    """Guarantee one door segment on the unit boundary."""
    if _has_boundary_door(doors, boundary_poly):
        return
    target = (
        Point(float(entry_point[0]), float(entry_point[1]))
        if entry_point is not None and len(entry_point) >= 2
        else None
    )
    hosts = _entry_host_rooms(rooms)
    if target is not None and hosts:
        hosts = sorted(hosts, key=lambda r: (_boundary_edge(r, boundary_poly) or LineString()).distance(target))
    candidates = list(hosts)
    seen = {r.id for r in candidates}
    for room in rooms:
        if room.id in seen:
            continue
        if _boundary_edge(room, boundary_poly) is not None:
            candidates.append(room)
            seen.add(room.id)
    for host in candidates:
        edge = _boundary_edge(host, boundary_poly)
        if edge is None:
            continue
        door = _door_on_edge(edge, [host.id], is_exterior=True, target=target)
        if door is not None:
            doors.append(door)
            return


def _place_doors(
    rooms: list[Room],
    *,
    boundary: Boundary | None = None,
    entry_point: list[float] | None = None,
    connection_graph: dict[str, Any] | None = None,
) -> list[Door]:
    doors: list[Door] = []
    covered: set[str] = set()
    toilet_hosts = _toilet_host_ids(rooms, doors)
    by_name = {r.name: r for r in rooms}
    boundary_poly = polygon_from_boundary(boundary) if boundary is not None else None

    if connection_graph:
        for a_name, b_name in connection_graph.get("tree", []):
            a = by_name.get(a_name)
            b = by_name.get(b_name)
            if not a or not b:
                continue
            if a.id in covered and b.id in covered:
                continue
            edge = shared_edge(a.coords(), b.coords(), tol=_EDGE_TOL)
            if edge is None:
                continue
            _add_door(doors, covered, toilet_hosts, rooms, a, b, edge)

        for a_name, b_name in connection_graph.get("edges", []):
            a = by_name.get(a_name)
            b = by_name.get(b_name)
            if not a or not b:
                continue
            if a.id in covered and b.id in covered:
                continue
            edge = shared_edge(a.coords(), b.coords(), tol=_EDGE_TOL)
            if edge is None:
                continue
            _add_door(doors, covered, toilet_hosts, rooms, a, b, edge)

    living = _living_room(rooms)
    hall = _room_by_name(rooms, "Hallway")
    if living and hall and living.id not in covered and hall.id not in covered:
        edge = shared_edge(living.coords(), hall.coords(), tol=_EDGE_TOL)
        if edge is not None:
            door = _door_on_edge(edge, [living.id, hall.id])
            if door is not None:
                doors.append(door)
                covered.update(door.connects)

    for room in rooms:
        if room.id in covered:
            continue
        placed = False
        preferred_names: set[str] = set()
        if is_toilet(room.name):
            parent = toilet_parent(room.name)
            if parent:
                preferred_names.add(parent)
        else:
            parent = closet_parent(room.name)
            if parent:
                preferred_names.add(parent)
        if not preferred_names and connection_graph:
            for a_name, b_name in connection_graph.get("edges", []):
                if a_name == room.name:
                    preferred_names.add(b_name)
                elif b_name == room.name:
                    preferred_names.add(a_name)

        others = [o for o in rooms if o.id != room.id]

        def _neighbor_rank(o: Room) -> tuple[int, int, str]:
            if o.name in preferred_names:
                return (0, 0, o.name)
            # Never prefer kitchen↔bedroom; rank those last so other hosts win.
            if (room.name == "Kitchen" and is_bedroom(o.name)) or (
                is_bedroom(room.name) and o.name == "Kitchen"
            ):
                return (3, 0, o.name)
            if o.name in {"Hallway", "Living", "Living/Sleeping"}:
                return (1, 0, o.name)
            if o.name in {"Dining", "Kitchen"}:
                return (1, 1, o.name)
            if o.name in {"Balcony", "Garage"}:
                return (1, 3, o.name)
            return (1, 2, o.name)

        others.sort(key=_neighbor_rank)
        for other in others:
            edge = shared_edge(room.coords(), other.coords(), tol=_EDGE_TOL)
            if edge is None:
                continue
            if _add_door(doors, covered, toilet_hosts, rooms, room, other, edge):
                placed = True
                break
        if placed:
            continue
        # No ad-hoc exterior doors here — entry is placed on Living/Hall only.

    # Ensure all rooms join the door network via geometric adjacency when needed.
    adj = adjacency_via_doors_from(list(rooms), doors)
    for _ in range(len(rooms)):
        seen = _reachable_ids(adj, rooms)
        progress = False
        for room in rooms:
            if room.id in seen:
                continue
            for other in rooms:
                if other.id == room.id or other.id not in seen:
                    continue
                edge = shared_edge(room.coords(), other.coords(), tol=_EDGE_TOL)
                if edge is None:
                    continue
                if not _toilet_door_allowed(room, other, toilet_hosts, rooms):
                    continue
                if not _kitchen_bedroom_door_allowed(room, other):
                    continue
                if not _closet_door_allowed(room, other):
                    continue
                door = _door_on_edge(edge, [room.id, other.id])
                if door is None:
                    continue
                doors.append(door)
                adj[room.id].append(other.id)
                adj[other.id].append(room.id)
                if is_toilet(room.name) and is_bedroom(other.name):
                    toilet_hosts.add(other.id)
                if is_toilet(other.name) and is_bedroom(room.name):
                    toilet_hosts.add(room.id)
                progress = True
                break
            if progress:
                break
        if not progress:
            break

    if boundary_poly is not None:
        _ensure_boundary_door(doors, rooms, boundary_poly, entry_point)

    return doors


def adjacency_via_doors_from(rooms: list[Room], doors: list[Door]) -> dict[str, list[str]]:
    adj: dict[str, set[str]] = {r.id: set() for r in rooms}
    for door in doors:
        if len(door.connects) >= 2:
            a, b = door.connects[0], door.connects[1]
            adj[a].add(b)
            adj[b].add(a)
    return {k: sorted(v) for k, v in adj.items()}


def _reachable_ids(adj: dict[str, list[str]], rooms: list[Room]) -> set[str]:
    if not rooms:
        return set()
    living = _living_room(rooms)
    hall = _room_by_name(rooms, "Hallway")
    start = living.id if living else (hall.id if hall else rooms[0].id)
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen

