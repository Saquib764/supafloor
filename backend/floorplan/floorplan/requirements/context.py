"""In-memory generation pipeline: parameters → generated geometry → FloorPlan."""

from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from floorplan.geometry.authors import make_door_line, make_room, make_window_line
from floorplan.geometry.ops import polygon_from_boundary, polygon_from_room, shared_edge
from floorplan.geometry.boundary import (
    boundary_from_params,
    force_manhattan,
    is_manhattan,
    sample_boundary_params,
)
from floorplan.requirements.areas import area_options_for, name_root
from floorplan.requirements.centers import sample_centers
from floorplan.requirements.entry import sample_entry_point
from floorplan.requirements.graph import build_connection_graph
from floorplan.requirements.grow import grow_rectangles
from floorplan.requirements.program import (
    bedroom_parent,
    closet_parent,
    is_bedroom,
    is_bedroom_name,
    is_toilet,
    pick_selected_areas,
    scale_selected_areas,
    space_names_for,
    toilet_parent,
)
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, Door, FloorPlan, Point2D, Room, Window

_DOOR_LEN = 0.9
_WINDOW_LEN = 1.0
_EDGE_TOL = 0.08
_MIN_LINE_LEN = 0.05

_MIN_SPACE_AREA = 0.5

_NAME_TO_TYPE: dict[str, RoomType] = {
    "Living/Sleeping": RoomType.LIVING,
    "Living": RoomType.LIVING,
    "Room": RoomType.BEDROOM,
    "Toilet": RoomType.POWDER,
    "Bathroom": RoomType.BATHROOM,
    "Kitchen": RoomType.KITCHEN,
    "Dining": RoomType.DINING,
    "Hall": RoomType.HALLWAY,
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
    "Hall": ZoneType.CIRCULATION,
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


def _rect_for_area(area: float, rng: random.Random) -> tuple[int, int]:
    aspect = rng.uniform(0.6, 1.7)
    w = max(1, int(round(math.sqrt(area * aspect))))
    h = max(1, int(round(area / w)))
    if w * h <= 0:
        w, h = 1, max(1, int(round(area)))
    return w, h


def _size_options(area: float, rng: random.Random) -> list[tuple[int, int]]:
    w, h = _rect_for_area(area, rng)
    if w == h:
        return [(w, h)]
    return [(w, h), (h, w)]


def _rect_touches_boundary(rect: Polygon, boundary_poly: Polygon, tol: float = 1e-6) -> bool:
    inter = rect.boundary.intersection(boundary_poly.boundary)
    return inter.length > tol


def _inward_probe(x0: float, y0: float, x1: float, y1: float, boundary_poly: Polygon) -> tuple[float, float]:
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    nx, ny = -dy, dx
    norm = math.hypot(nx, ny) or 1.0
    nx, ny = nx / norm, ny / norm
    if boundary_poly.contains(Point(mx + nx * 0.5, my + ny * 0.5)):
        return nx, ny
    return -nx, -ny


def _sample_room_at_boundary(
    area: float,
    rng: random.Random,
    boundary_poly: Polygon,
) -> dict[str, Any]:
    """Place a room flush against the unit boundary."""
    coords = list(boundary_poly.exterior.coords)
    candidates: list[dict[str, Any]] = []

    for span, depth in _size_options(area, rng):
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            ix, iy = _inward_probe(x0, y0, x1, y1, boundary_poly)

            if abs(y1 - y0) < 1e-9:
                y_edge = y0
                x_lo, x_hi = int(min(x0, x1)), int(max(x0, x1))
                if x_hi - x_lo < span:
                    continue
                y0_in = y_edge if iy > 0 else y_edge - depth
                y1_in = y_edge + depth if iy > 0 else y_edge
                for x in range(x_lo, x_hi - span + 1):
                    rect = box(x, y0_in, x + span, y1_in)
                    if _rect_touches_boundary(rect, boundary_poly):
                        candidates.append(
                            {"x": x, "y": int(y0_in), "w": span, "h": depth, "area": float(span * depth)}
                        )
            elif abs(x1 - x0) < 1e-9:
                x_edge = x0
                y_lo, y_hi = int(min(y0, y1)), int(max(y0, y1))
                if y_hi - y_lo < span:
                    continue
                x0_in = x_edge if ix > 0 else x_edge - depth
                x1_in = x_edge + depth if ix > 0 else x_edge
                for y in range(y_lo, y_hi - span + 1):
                    rect = box(x0_in, y, x1_in, y + span)
                    if _rect_touches_boundary(rect, boundary_poly):
                        candidates.append(
                            {"x": int(x0_in), "y": y, "w": depth, "h": span, "area": float(span * depth)}
                        )

    if candidates:
        return rng.choice(candidates)
    w, h = _rect_for_area(area, rng)
    return {"x": 0, "y": 0, "w": w, "h": h, "area": float(w * h)}


def _finalize_space_geom(geom: Polygon | None, name: str) -> dict[str, Any] | None:
    if geom is None or geom.area < _MIN_SPACE_AREA:
        return None
    if not is_manhattan(list(geom.exterior.coords[:-1])):
        geom = _largest_polygon(Polygon(force_manhattan(list(geom.exterior.coords[:-1]))))
    if geom is None or geom.area < _MIN_SPACE_AREA:
        return None
    return {"name": name, "polygon": _poly_points(geom), "area": round(float(geom.area), 3)}


def _place_rect_geom(
    rect: dict[str, Any],
    boundary_poly: Polygon,
    occupied: list[Polygon],
) -> tuple[Polygon | None, str | None]:
    geom = box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])
    geom = _largest_polygon(geom.intersection(boundary_poly))
    if geom is None:
        return None, "outside_boundary"
    if occupied:
        geom = _largest_polygon(geom.difference(unary_union(occupied)))
    if geom is None or geom.area < _MIN_SPACE_AREA:
        return None, "overlap"
    return geom, None


def _rect_dict(x: int, y: int, w: int, h: int) -> dict[str, Any]:
    return {"x": x, "y": y, "w": w, "h": h, "area": float(w * h)}


def _box_from_rect(rect: dict[str, Any]) -> Polygon:
    return box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])


def _placement_order(names: list[str], boundary_room: str) -> list[str]:
    def sort_key(name: str) -> tuple[int, int]:
        if name == boundary_room:
            return (0, 0)
        root = name_root(name)
        fixed = {
            "Living": 2,
            "Living/Sleeping": 2,
            "Kitchen": 3,
            "Dining": 4,
            "Hall": 5,
            "Bathroom": 8,
            "Toilet": 9,
            "Closet": 10,
            "Balcony": 11,
            "Laundry": 12,
            "Garage": 13,
        }
        if root == "Room":
            parts = name.rsplit(" ", 1)
            n = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
            return (6, n)
        return (fixed.get(root, 7), 0)

    return sorted(names, key=sort_key)


def _preferred_neighbors(name: str, boundary_room: str) -> list[str]:
    root = name_root(name)
    prefs: dict[str, list[str]] = {
        "Kitchen": ["Living", "Living/Sleeping", "Dining", "Hall"],
        "Dining": ["Living", "Living/Sleeping", "Kitchen"],
        "Hall": ["Living", "Living/Sleeping", boundary_room],
        "Bathroom": ["Hall", "Kitchen", boundary_room],
        "Toilet": ["Bathroom", "Hall"],
        "Room": ["Hall", "Living", "Living/Sleeping", boundary_room],
        "Closet": ["Room", "Hall"],
        "Balcony": ["Living", "Living/Sleeping", "Kitchen"],
        "Laundry": ["Kitchen", "Bathroom"],
        "Garage": [boundary_room, "Hall"],
    }
    parent = bedroom_parent(name)
    if parent:
        return [parent]
    return prefs.get(root, prefs.get(name, []))


def _needs_exterior(name: str) -> bool:
    root = name_root(name)
    return is_bedroom_name(name) or root in ("Living", "Living/Sleeping", "Balcony")


def _free_polygon(boundary_poly: Polygon, occupied_rects: list[dict[str, Any]]) -> Polygon | None:
    if not occupied_rects:
        return boundary_poly
    occupied = unary_union([_box_from_rect(r) for r in occupied_rects])
    return _largest_polygon(boundary_poly.difference(occupied))


def _enumerate_flush_in_free(
    free_poly: Polygon,
    span: int,
    depth: int,
    *,
    boundary_poly: Polygon | None = None,
    require_boundary: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    coords = list(free_poly.exterior.coords)

    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        ix, iy = _inward_probe(x0, y0, x1, y1, free_poly)

        if abs(y1 - y0) < 1e-9:
            y_edge = y0
            x_lo, x_hi = int(min(x0, x1)), int(max(x0, x1))
            if x_hi - x_lo < span:
                continue
            y0_in = y_edge if iy > 0 else y_edge - depth
            for x in range(x_lo, x_hi - span + 1):
                rect = box(x, y0_in, x + span, y0_in + depth)
                if not free_poly.covers(rect):
                    continue
                if require_boundary and boundary_poly is not None and not _rect_touches_boundary(rect, boundary_poly):
                    continue
                candidates.append(_rect_dict(x, int(y0_in), span, depth))
        elif abs(x1 - x0) < 1e-9:
            x_edge = x0
            y_lo, y_hi = int(min(y0, y1)), int(max(y0, y1))
            if y_hi - y_lo < span:
                continue
            x0_in = x_edge if ix > 0 else x_edge - depth
            for y in range(y_lo, y_hi - span + 1):
                rect = box(x0_in, y, x0_in + depth, y + span)
                if not free_poly.covers(rect):
                    continue
                if require_boundary and boundary_poly is not None and not _rect_touches_boundary(rect, boundary_poly):
                    continue
                candidates.append(_rect_dict(int(x0_in), y, depth, span))

    return candidates


def _enumerate_against_rects(free_poly: Polygon, w: int, h: int, rects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add(x: int, y: int) -> None:
        key = (x, y, w, h)
        if key in seen:
            return
        rect = box(x, y, x + w, y + h)
        if free_poly.covers(rect):
            seen.add(key)
            candidates.append(_rect_dict(x, y, w, h))

    for occ in rects:
        ox, oy, ow, oh = occ["x"], occ["y"], occ["w"], occ["h"]
        x_right = ox + ow
        for y in range(oy - h + 1, oy + oh):
            add(x_right, y)
        x_left = ox - w
        for y in range(oy - h + 1, oy + oh):
            add(x_left, y)
        y_below = oy + oh
        for x in range(ox - w + 1, ox + ow):
            add(x, y_below)
        y_above = oy - h
        for x in range(ox - w + 1, ox + ow):
            add(x, y_above)

    return candidates


def _score_rect_candidate(
    rect: dict[str, Any],
    name: str,
    target_area: float,
    boundary_poly: Polygon,
    placed_polys: dict[str, Polygon],
    preferred: list[str],
) -> float:
    geom = _box_from_rect(rect)
    score = 0.0

    area_ratio = geom.area / max(target_area, 1.0)
    if 0.85 <= area_ratio <= 1.25:
        score += 2.0
    elif area_ratio < 0.85:
        score += area_ratio * 1.5
    else:
        score += 1.0

    if _needs_exterior(name):
        if _rect_touches_boundary(geom, boundary_poly):
            score += 3.0
        else:
            score -= 1.5

    if name_root(name) == "Balcony" and not _rect_touches_boundary(geom, boundary_poly):
        score -= 5.0

    for pref in preferred:
        pref_geom = placed_polys.get(pref)
        if pref_geom is None:
            continue
        if geom.boundary.intersection(pref_geom.boundary).length > 0.5:
            score += 4.0

    return score


def _fallback_rect(area: float, rng: random.Random, free_poly: Polygon) -> dict[str, Any]:
    w, h = _rect_for_area(area, rng)
    minx, miny, maxx, maxy = (int(v) for v in free_poly.bounds)
    w = min(w, max(1, maxx - minx))
    h = min(h, max(1, maxy - miny))
    x_hi = max(minx, maxx - w)
    y_hi = max(miny, maxy - h)
    for _ in range(24):
        x = rng.randint(minx, x_hi) if x_hi >= minx else minx
        y = rng.randint(miny, y_hi) if y_hi >= miny else miny
        if free_poly.covers(box(x, y, x + w, y + h)):
            return _rect_dict(x, y, w, h)
    return _rect_dict(minx, miny, w, h)


def _best_rect_for_space(
    name: str,
    area: float,
    free_poly: Polygon,
    boundary_poly: Polygon,
    placed_polys: dict[str, Polygon],
    occupied_rects: list[dict[str, Any]],
    boundary_room: str,
    rng: random.Random,
) -> dict[str, Any]:
    preferred = _preferred_neighbors(name, boundary_room)
    require_boundary = name_root(name) == "Balcony"
    seen: set[tuple[int, int, int, int]] = set()
    scored: list[tuple[float, dict[str, Any]]] = []

    for w, h in _size_options(area, rng):
        candidates = _enumerate_flush_in_free(
            free_poly,
            w,
            h,
            boundary_poly=boundary_poly,
            require_boundary=require_boundary,
        )
        candidates.extend(_enumerate_against_rects(free_poly, w, h, occupied_rects))

        for rect in candidates:
            key = (rect["x"], rect["y"], rect["w"], rect["h"])
            if key in seen:
                continue
            seen.add(key)
            s = _score_rect_candidate(rect, name, area, boundary_poly, placed_polys, preferred)
            scored.append((s, rect))

    if not scored:
        return _fallback_rect(area, rng, free_poly)

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0]
    top = [rect for s, rect in scored if s >= best - 0.5][:5]
    return rng.choice(top)


def run_generation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Pure algorithm under test: parameters → generated geometry."""
    boundary, boundary_gen = boundary_from_params(parameters["start_rect"], parameters.get("notches") or [])
    boundary_poly = Polygon([(p.x, p.y) for p in boundary.exterior])

    rectangles = parameters.get("rectangles") or []
    use_grown = bool(parameters.get("centers"))
    boundary_room = parameters.get("boundary_room") or parameters.get("entry_host") or "Living"
    if "Living/Sleeping" in {r.get("name") for r in rectangles}:
        boundary_room = "Living/Sleeping"

    name_order = [r.get("name") for r in rectangles if r.get("name")]
    if boundary_room in name_order:
        name_order = [boundary_room] + [n for n in name_order if n != boundary_room]
    rect_by_name = {r["name"]: r for r in rectangles if r.get("name")}

    geoms: dict[str, Polygon] = {}
    dropped: dict[str, str] = {}

    if use_grown:
        for name in name_order:
            rect = rect_by_name.get(name)
            if not rect:
                continue
            footprint = rect.get("footprint")
            if footprint:
                geom = _largest_polygon(Polygon(footprint).intersection(boundary_poly))
            else:
                geom = _largest_polygon(
                    box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]).intersection(
                        boundary_poly
                    )
                )
            if geom is None or geom.area < _MIN_SPACE_AREA:
                dropped[name] = "outside_boundary"
                continue
            geoms[name] = geom
    else:
        occupied: list[Polygon] = []

        def _place_room(rect: dict[str, Any]) -> None:
            name = rect["name"]
            geom, reason = _place_rect_geom(rect, boundary_poly, occupied)
            if geom is None:
                dropped[name] = reason or "invalid"
                return
            geoms[name] = geom
            occupied.append(geom)

        for name in name_order:
            rect = rect_by_name.get(name)
            if not rect:
                continue
            if name == boundary_room and name not in geoms:
                geom, reason = _place_rect_geom(rect, boundary_poly, occupied)
                if geom is None or not _rect_touches_boundary(geom, boundary_poly):
                    dropped[name] = reason or "invalid"
                else:
                    geoms[name] = geom
                    occupied.append(geom)
                continue
            if name in geoms or name in dropped:
                continue
            _place_room(rect)

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


def sample_parameters(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample program, entry, graph, centers, then grow seed rectangles."""
    rng = random.Random(seed)
    boundary_params = sample_boundary_params(rng, target_area=target_area)

    names = space_names_for(total_rooms, rng)
    boundary_room = "Living" if "Living" in names else "Living/Sleeping"

    boundary, _ = boundary_from_params(
        boundary_params["start_rect"], boundary_params.get("notches") or []
    )
    boundary_poly = Polygon([(p.x, p.y) for p in boundary.exterior])

    selected_areas = scale_selected_areas(
        pick_selected_areas(names, rng),
        boundary_poly.area,
    )

    entry = sample_entry_point(boundary_poly, rng)
    connection_graph = build_connection_graph(names)
    centers = sample_centers(
        names,
        selected_areas,
        boundary_poly,
        entry["entry_point"],
        connection_graph,
        rng,
    )
    rectangles = grow_rectangles(
        names,
        centers,
        selected_areas,
        boundary_poly,
        connection_graph,
        rng,
    )

    return {
        "seed": seed,
        "total_rooms": max(1, int(total_rooms)),
        "target_area": boundary_params["target_area"],
        "aspect": boundary_params["aspect"],
        "start_rect": boundary_params["start_rect"],
        "notches": boundary_params["notches"],
        "space_names": names,
        "boundary_room": boundary_room,
        "selected_areas": selected_areas,
        "entry_point": entry["entry_point"],
        "entry_edge": entry["entry_edge"],
        "connection_graph": connection_graph,
        "centers": centers,
        "rectangles": rectangles,
    }


def build_document(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
    plan_id: str | None = None,
) -> dict[str, Any]:
    parameters = sample_parameters(total_rooms, target_area=target_area, seed=seed)
    generated = run_generation(parameters)
    return {
        "id": plan_id or uuid4().hex[:12],
        "label": f"{parameters['total_rooms']}-room",
        "parameters": parameters,
        "generated": generated,
    }


def _longest_line(geom) -> LineString | None:
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


def _segment_on_linestring(
    ls: LineString,
    length: float = _DOOR_LEN,
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


def _ring_edges(coords) -> list[LineString]:
    edges: list[LineString] = []
    pts = list(coords)
    for i in range(len(pts) - 1):
        if pts[i] == pts[i + 1]:
            continue
        edge = LineString([pts[i], pts[i + 1]])
        if edge.length >= _MIN_LINE_LEN:
            edges.append(edge)
    return edges


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
    if edge.length < _DOOR_LEN:
        return None
    if target is not None:
        t = edge.project(target)
        use = min(_DOOR_LEN, edge.length * 0.8)
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


def _boundary_edge(room: Room, boundary_poly: Polygon) -> LineString | None:
    """Longest room wall segment that lies on the unit boundary."""
    poly = polygon_from_room(room)
    best: LineString | None = None
    for edge in _ring_edges(poly.exterior.coords):
        overlap = edge.intersection(boundary_poly.boundary)
        line = _longest_line(overlap)
        if line is None or line.length < _DOOR_LEN:
            continue
        if best is None or line.length > best.length:
            best = line
    if best is not None:
        return best
    inter = poly.boundary.intersection(boundary_poly.boundary)
    line = _longest_line(inter)
    if line is not None and line.length >= _DOOR_LEN:
        return line
    return None


def _exterior_edge(room: Room, boundary_poly: Polygon) -> LineString | None:
    poly = polygon_from_room(room)
    inter = poly.boundary.intersection(boundary_poly.boundary)
    if inter.is_empty:
        inter = poly.buffer(_EDGE_TOL).intersection(boundary_poly.buffer(_EDGE_TOL)).boundary
    line = _longest_line(inter)
    return line if line is not None and line.length >= _MIN_LINE_LEN else None


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
    hall = _room_by_name(rooms, "Hall")
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
    hall = _room_by_name(rooms, "Hall")
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
            if o.name in {"Hall", "Living", "Living/Sleeping"}:
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
    hall = _room_by_name(rooms, "Hall")
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


def _window_segments(
    edge: LineString,
    count: int,
    length: float = _WINDOW_LEN,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if edge.length < _MIN_LINE_LEN or count <= 0:
        return []
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    use = min(length, edge.length * 0.35)
    for i in range(count):
        mid = edge.length * (i + 1) / (count + 1)
        a = max(0.0, mid - use / 2.0)
        b = min(edge.length, a + use)
        p0 = edge.interpolate(a)
        p1 = edge.interpolate(b)
        if p0.coords[0] == p1.coords[0]:
            continue
        out.append(((float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))))
    return out


def _window_priority(room: Room) -> tuple[int, int]:
    root = name_root(room.name)
    order = {"Living": 0, "Living/Sleeping": 0, "Room": 1}
    n = 0
    if root == "Room":
        parts = room.name.rsplit(" ", 1)
        n = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
    return (order.get(root, order.get(room.name, 3)), n)


def _place_windows(rooms: list[Room], boundary: Boundary, count: int = 2) -> list[Window]:
    boundary_poly = polygon_from_boundary(boundary)
    candidates = [
        (room, edge)
        for room in rooms
        if (edge := _exterior_edge(room, boundary_poly)) is not None
    ]
    candidates.sort(key=lambda pair: _window_priority(pair[0]))

    windows: list[Window] = []
    for room, edge in candidates:
        need = count - len(windows)
        if need <= 0:
            break
        for start, end in _window_segments(edge, need):
            windows.append(make_window_line(start, end, room_ids=[room.id]))
            if len(windows) >= count:
                break
    return windows


def document_to_plan(document: dict[str, Any]) -> FloorPlan:
    """Build FloorPlan from document.generated, including doors and windows."""
    generated = document["generated"]
    parameters = document["parameters"]
    poly = generated["boundary"]["polygon"]
    boundary = Boundary(exterior=[Point2D(x=float(p[0]), y=float(p[1])) for p in poly])

    rooms = []
    for space in generated.get("spaces") or []:
        pts = space.get("polygon") or []
        if len(pts) < 3:
            continue
        rooms.append(
            make_room(
                space["name"],
                [(p[0], p[1]) for p in pts],
                room_type=_room_type_for_name(space["name"]),
                zone=_zone_for_name(space["name"]),
            )
        )
    if not rooms:
        rooms = [
            make_room(
                "Living",
                boundary.coords(),
                room_type=RoomType.LIVING,
                zone=ZoneType.PUBLIC,
            )
        ]
    doors = _place_doors(
        rooms,
        boundary=boundary,
        entry_point=parameters.get("entry_point"),
        connection_graph=parameters.get("connection_graph"),
    )
    windows = _place_windows(rooms, boundary)
    return FloorPlan(
        name=document.get("label") or f"{parameters.get('total_rooms', '?')}-room",
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"plan_id": document.get("id")},
    )
