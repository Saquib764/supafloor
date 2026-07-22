"""Canonical plan JSON load/save (LLM-shaped)."""

from __future__ import annotations

from typing import Any

from floorplan.geometry.ops import area as polygon_area
from floorplan.requirements.context import _room_type_for_name, _zone_for_name, document_to_plan
from floorplan.requirements.internals import GenerationInternals
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, Door, FloorPlan, Point2D, Room, Window
from floorplan.topology import build_adjacency

SCHEMA_VERSION = "1.0"


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 3), "y": round(float(y), 3)}


def _bbox(coords: list[tuple[float, float]]) -> dict[str, float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return {
        "x": round(minx, 3),
        "y": round(miny, 3),
        "w": round(maxx - minx, 3),
        "h": round(maxy - miny, 3),
    }


def _center(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(bbox["x"] + bbox["w"] / 2.0, 3),
        "y": round(bbox["y"] + bbox["h"] / 2.0, 3),
    }


def _rooms_requested(parameters: dict[str, Any], generated_names: set[str]) -> list[dict[str, Any]]:
    selected = parameters.get("selected_areas") or {}
    rows: list[dict[str, Any]] = []
    for name in parameters.get("space_names") or []:
        target = selected.get(name)
        rows.append(
            {
                "name": name,
                "target_area_sqft": round(float(target), 3) if target is not None else None,
                "room_type": _room_type_for_name(name).value,
                "zone": _zone_for_name(name).value,
                "generated": name in generated_names,
            }
        )
    return rows


def _topology(parameters: dict[str, Any], adjacency: dict[str, list[str]]) -> dict[str, Any]:
    graph = parameters.get("connection_graph") or {}
    connections = [{"from": a, "to": b} for a, b in graph.get("edges", [])]
    return {
        "root": graph.get("root"),
        "connections": connections,
        "adjacency": adjacency,
    }


def _room_rows(plan: FloorPlan) -> tuple[list[dict[str, Any]], float]:
    id_to_name = {room.id: room.name for room in plan.rooms}
    adj_by_id = build_adjacency(plan)
    rows: list[dict[str, Any]] = []
    total_area = 0.0

    for room in plan.rooms:
        coords = room.coords()
        bbox = _bbox(coords)
        room_area = round(float(polygon_area(coords)), 3)
        total_area += room_area
        adjacent = sorted(id_to_name[nid] for nid in adj_by_id.get(room.id, []) if nid in id_to_name)
        rows.append(
            {
                "id": room.id,
                "name": room.name,
                "room_type": room.room_type.value,
                "zone": room.zone.value,
                "area_sqft": room_area,
                "bbox": bbox,
                "center": _center(bbox),
                "adjacent_to": adjacent,
                "polygon": [_point(x, y) for x, y in coords],
            }
        )

    return rows, round(total_area, 3)


def _openings(plan: FloorPlan) -> dict[str, list[dict[str, Any]]]:
    id_to_name = {room.id: room.name for room in plan.rooms}

    def _names(ids: list[str]) -> list[str]:
        return [id_to_name.get(room_id, room_id) for room_id in ids]

    doors = [
        {
            "id": door.id,
            "connects": _names(door.connects),
            "start": _point(door.start.x, door.start.y),
            "end": _point(door.end.x, door.end.y),
            "is_exterior": door.is_exterior,
        }
        for door in plan.doors
    ]
    windows = [
        {
            "id": window.id,
            "room_names": _names(window.room_ids),
            "start": _point(window.start.x, window.start.y),
            "end": _point(window.end.x, window.end.y),
        }
        for window in plan.windows
    ]
    return {"doors": doors, "windows": windows}


def plan_document_from_internals(
    internals: GenerationInternals,
    plan: FloorPlan | None = None,
) -> dict[str, Any]:
    """Convert generation internals to canonical on-disk plan JSON."""
    plan = plan or document_to_plan(internals)
    parameters = internals.parameters
    generated = internals.generated
    generated_names = {space.get("name") for space in generated.get("spaces") or [] if space.get("name")}

    rooms, total_area = _room_rows(plan)
    adjacency = {room["name"]: room["adjacent_to"] for room in rooms}

    boundary = plan.boundary or plan.ensure_boundary()
    exterior = [_point(point.x, point.y) for point in boundary.exterior[:-1]]

    entry = None
    entry_point = parameters.get("entry_point")
    if entry_point and len(entry_point) >= 2:
        entry = {"point": _point(entry_point[0], entry_point[1])}

    return {
        "schema_version": SCHEMA_VERSION,
        "units": {"length": "ft", "area": "ft²"},
        "summary": {
            "id": internals.plan_id,
            "name": f"{parameters.get('total_rooms', '?')}-room",
            "bedroom_count": parameters.get("total_rooms"),
            "room_count": len(rooms),
            "total_area_sqft": total_area,
            "boundary_area_sqft": generated.get("boundary", {}).get("polygon_area"),
            "target_area_sqft": parameters.get("target_area"),
            "entry": entry,
        },
        "requirements": {
            "target_area_sqft": parameters.get("target_area"),
            "bedroom_count": parameters.get("total_rooms"),
            "boundary_room": parameters.get("boundary_room"),
            "rooms_requested": _rooms_requested(parameters, generated_names),
        },
        "topology": _topology(parameters, adjacency),
        "rooms": rooms,
        "openings": _openings(plan),
        "boundary": {"exterior": exterior},
    }


def plan_document_id(document: dict[str, Any]) -> str:
    summary = document.get("summary") or {}
    return str(summary.get("id") or "")


def llm_to_plan(document: dict[str, Any]) -> FloorPlan:
    """Load FloorPlan from canonical plan JSON."""
    summary = document.get("summary") or {}
    name_to_id: dict[str, str] = {}

    rooms: list[Room] = []
    for row in document.get("rooms") or []:
        name_to_id[row["name"]] = row["id"]
        polygon = [Point2D(x=float(p["x"]), y=float(p["y"])) for p in row.get("polygon") or []]
        rooms.append(
            Room(
                id=row["id"],
                name=row["name"],
                polygon=polygon,
                room_type=RoomType(row["room_type"]),
                zone=ZoneType(row["zone"]),
            )
        )

    def _ids(names: list[str]) -> list[str]:
        return [name_to_id.get(name, name) for name in names]

    openings = document.get("openings") or {}
    doors = [
        Door(
            id=door["id"],
            start=Point2D(x=float(door["start"]["x"]), y=float(door["start"]["y"])),
            end=Point2D(x=float(door["end"]["x"]), y=float(door["end"]["y"])),
            connects=_ids(door.get("connects") or []),
            is_exterior=bool(door.get("is_exterior", False)),
        )
        for door in openings.get("doors") or []
    ]
    windows = [
        Window(
            id=window["id"],
            start=Point2D(x=float(window["start"]["x"]), y=float(window["start"]["y"])),
            end=Point2D(x=float(window["end"]["x"]), y=float(window["end"]["y"])),
            room_ids=_ids(window.get("room_names") or []),
        )
        for window in openings.get("windows") or []
    ]

    boundary_data = document.get("boundary") or {}
    exterior = [Point2D(x=float(p["x"]), y=float(p["y"])) for p in boundary_data.get("exterior") or []]
    boundary = Boundary(exterior=exterior) if exterior else None

    plan_id = str(summary.get("id") or "unknown")
    return FloorPlan(
        id=plan_id,
        name=str(summary.get("name") or "untitled"),
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"plan_id": plan_id},
    )
