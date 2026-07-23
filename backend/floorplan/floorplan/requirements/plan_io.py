"""Canonical plan JSON — seed, sample, generate, load."""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from shapely.geometry import Polygon

from floorplan.geometry.authors import make_room
from floorplan.geometry.boundary import boundary_from_params, sample_boundary_params
from floorplan.geometry.ops import area as polygon_area
from floorplan.requirements.context import (
    _place_doors,
    _place_windows,
    _room_type_for_name,
    _zone_for_name,
    run_generation,
)
from floorplan.requirements.entry import sample_entry_point
from floorplan.requirements.grow import grow_rectangles
from floorplan.config.pipeline import pipeline_settings_snapshot, substep_provider
from floorplan.llm.centers import resolve_centers
from floorplan.llm.connection_graph import resolve_connection_graph
from floorplan.llm.target_areas import resolve_selected_areas
from floorplan.requirements.program import space_names_for
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, Door, FloorPlan, Point2D, Room, Window
from floorplan.topology import build_adjacency

SCHEMA_VERSION = "1.1"

_SAMPLING_KEYS = (
    "aspect",
    "start_rect",
    "notches",
    "space_names",
    "selected_areas",
    "entry_point",
    "entry_edge",
    "centers",
    "rectangles",
    "connection_graph",
)


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


def _room_row(room: Room) -> dict[str, Any]:
    coords = room.coords()
    bbox = _bbox(coords)
    return {
        "id": room.id,
        "name": room.name,
        "room_type": room.room_type.value,
        "zone": room.zone.value,
        "area_sqft": round(float(polygon_area(coords)), 3),
        "bbox": bbox,
        "center": _center(bbox),
        "adjacent_to": [],
        "polygon": [_point(x, y) for x, y in coords],
    }


def _generation_params(doc: dict[str, Any]) -> dict[str, Any]:
    sampling = doc["sampling"]
    req = doc["requirements"]
    return {
        "start_rect": sampling["start_rect"],
        "notches": sampling.get("notches") or [],
        "rectangles": sampling.get("rectangles") or [],
        "centers": sampling.get("centers") or {},
        "boundary_room": req.get("boundary_room"),
        "entry_point": sampling.get("entry_point"),
        "connection_graph": sampling.get("connection_graph") or {},
    }


def seed_document(
    bedroom_count: int,
    *,
    target_area: float | None = None,
    seed: int = 0,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Create an empty plan document from user inputs."""
    n = max(1, int(bedroom_count))
    pid = plan_id or uuid4().hex[:12]
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {"length": "ft", "area": "ft²"},
        "summary": {
            "id": pid,
            "name": f"{n}-room",
            "bedroom_count": n,
            "room_count": 0,
            "total_area_sqft": 0.0,
            "boundary_area_sqft": None,
            "target_area_sqft": target_area,
            "entry": None,
        },
        "requirements": {
            "target_area_sqft": target_area,
            "bedroom_count": n,
            "boundary_room": None,
            "rooms_requested": [],
        },
        "sampling": {
            "seed": seed,
            "aspect": None,
            "start_rect": None,
            "notches": [],
            "space_names": [],
            "selected_areas": {},
            "entry_point": None,
            "entry_edge": None,
            "centers": {},
            "rectangles": [],
            "connection_graph": {"root": None, "edges": [], "tree": []},
        },
        "topology": {"root": None, "connections": [], "adjacency": {}},
        "rooms": [],
        "openings": {"doors": [], "windows": []},
        "boundary": {"exterior": []},
        "pipeline": {
            "configured": pipeline_settings_snapshot(),
            "executed": {},
        },
    }


def _rooms_requested(
    space_names: list[str],
    selected_areas: dict[str, float],
    generated_names: set[str],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "target_area_sqft": int(selected_areas[name]) if name in selected_areas else None,
            "room_type": _room_type_for_name(name).value,
            "zone": _zone_for_name(name).value,
            "generated": name in generated_names,
        }
        for name in space_names
    ]


def sample_into_document(doc: dict[str, Any]) -> None:
    """Sample program, entry, graph, centers, and seed rectangles into the document."""
    req = doc["requirements"]
    sampling = doc["sampling"]
    summary = doc["summary"]
    seed = sampling.get("seed", 0)
    rng = random.Random(seed)

    boundary_params = sample_boundary_params(rng, target_area=req.get("target_area_sqft"))
    names = space_names_for(req["bedroom_count"], rng)
    boundary_room = "Living" if "Living" in names else "Living/Sleeping"

    boundary, _ = boundary_from_params(boundary_params["start_rect"], boundary_params.get("notches") or [])
    boundary_poly = Polygon([(p.x, p.y) for p in boundary.exterior])
    selected_areas, target_areas_exec = resolve_selected_areas(
        names,
        boundary_poly.area,
        doc,
        rng,
        boundary_params=boundary_params,
    )

    entry = sample_entry_point(boundary_poly, rng)
    connection_graph, graph_exec = resolve_connection_graph(
        names,
        doc,
        selected_areas=selected_areas,
        layout_total_area_sqft=boundary_poly.area,
    )
    centers, centers_exec = resolve_centers(
        names,
        selected_areas,
        boundary_poly,
        entry["entry_point"],
        connection_graph,
        doc,
        rng,
        entry_edge=entry.get("entry_edge"),
        layout_total_area_sqft=boundary_poly.area,
    )
    rectangles = grow_rectangles(
        names,
        centers,
        selected_areas,
        boundary_poly,
        connection_graph,
        rng,
    )

    sampled = {
        "aspect": boundary_params["aspect"],
        "start_rect": boundary_params["start_rect"],
        "notches": boundary_params["notches"],
        "space_names": names,
        "selected_areas": selected_areas,
        "entry_point": entry["entry_point"],
        "entry_edge": entry["entry_edge"],
        "centers": centers,
        "rectangles": rectangles,
        "connection_graph": connection_graph,
    }
    for key in _SAMPLING_KEYS:
        sampling[key] = sampled[key]

    doc.setdefault("pipeline", {})
    doc["pipeline"]["configured"] = pipeline_settings_snapshot()
    doc["pipeline"]["executed"] = {
        "sample_into_document": {
            "target_areas": {
                "provider": substep_provider("sample_into_document", "target_areas"),
                **target_areas_exec,
            },
            "connection_graph": {
                "provider": substep_provider("sample_into_document", "connection_graph"),
                **graph_exec,
            },
            "centers": {
                "provider": substep_provider("sample_into_document", "centers"),
                **centers_exec,
            },
        },
    }

    req["boundary_room"] = boundary_room
    req["rooms_requested"] = _rooms_requested(names, selected_areas, set())

    doc["topology"] = {
        "root": connection_graph.get("root"),
        "connections": [{"from": a, "to": b} for a, b in connection_graph.get("edges", [])],
        "adjacency": {},
    }

    entry_point = entry["entry_point"]
    summary["entry"] = (
        {"point": _point(entry_point[0], entry_point[1])}
        if entry_point and len(entry_point) >= 2
        else None
    )


def generate_geometry(doc: dict[str, Any]) -> None:
    """Run placement/clipping and write boundary + room polygons into the document."""
    generated = run_generation(_generation_params(doc))

    poly = generated["boundary"]["polygon"]
    doc["boundary"]["exterior"] = [_point(p[0], p[1]) for p in poly]
    doc["summary"]["boundary_area_sqft"] = generated["boundary"].get("polygon_area")

    generated_names: set[str] = set()
    rooms: list[dict[str, Any]] = []
    for space in generated.get("spaces") or []:
        pts = space.get("polygon") or []
        if len(pts) < 3:
            continue
        room = make_room(
            space["name"],
            [(p[0], p[1]) for p in pts],
            room_type=_room_type_for_name(space["name"]),
            zone=_zone_for_name(space["name"]),
        )
        rooms.append(_room_row(room))
        generated_names.add(space["name"])

    if not rooms:
        fallback = make_room(
            "Living",
            [(p[0], p[1]) for p in poly],
            room_type=RoomType.LIVING,
            zone=ZoneType.PUBLIC,
        )
        rooms = [_room_row(fallback)]
        generated_names.add("Living")

    doc["rooms"] = rooms
    for row in doc["requirements"]["rooms_requested"]:
        row["generated"] = row["name"] in generated_names


def place_openings(doc: dict[str, Any]) -> None:
    """Place doors and windows and write them into the document."""
    plan = document_to_plan(doc)
    params = _generation_params(doc)
    doors = _place_doors(
        plan.rooms,
        boundary=plan.boundary,
        entry_point=params.get("entry_point"),
        connection_graph=params.get("connection_graph"),
    )
    windows = _place_windows(plan.rooms, plan.boundary or plan.ensure_boundary())
    id_to_name = {room.id: room.name for room in plan.rooms}
    doc["openings"] = {
        "doors": [
            {
                "id": door.id,
                "connects": [id_to_name.get(rid, rid) for rid in door.connects],
                "start": _point(door.start.x, door.start.y),
                "end": _point(door.end.x, door.end.y),
                "is_exterior": door.is_exterior,
            }
            for door in doors
        ],
        "windows": [
            {
                "id": window.id,
                "room_names": [id_to_name.get(rid, rid) for rid in window.room_ids],
                "start": _point(window.start.x, window.start.y),
                "end": _point(window.end.x, window.end.y),
            }
            for window in windows
        ],
    }


def finalize_derived(doc: dict[str, Any]) -> None:
    """Fill adjacency, room adjacency lists, and summary counts."""
    plan = document_to_plan(doc)
    adj_by_id = build_adjacency(plan)
    id_to_name = {room.id: room.name for room in plan.rooms}

    for room_row in doc["rooms"]:
        room_row["adjacent_to"] = sorted(
            id_to_name[nid] for nid in adj_by_id.get(room_row["id"], []) if nid in id_to_name
        )

    doc["topology"]["adjacency"] = {room["name"]: room["adjacent_to"] for room in doc["rooms"]}
    doc["summary"]["room_count"] = len(doc["rooms"])
    doc["summary"]["total_area_sqft"] = round(sum(float(r["area_sqft"]) for r in doc["rooms"]), 3)


def generate_plan_document(
    bedroom_count: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Generate canonical plan JSON end-to-end."""
    doc = seed_document(bedroom_count, target_area=target_area, seed=seed, plan_id=plan_id)
    sample_into_document(doc)
    generate_geometry(doc)
    place_openings(doc)
    finalize_derived(doc)
    return doc


def plan_document_id(document: dict[str, Any]) -> str:
    summary = document.get("summary") or {}
    return str(summary.get("id") or "")


def document_to_plan(document: dict[str, Any]) -> FloorPlan:
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

    selected_areas: dict[str, float] = {}
    for row in (document.get("requirements") or {}).get("rooms_requested") or []:
        name = row.get("name")
        area = row.get("target_area_sqft")
        if name and area is not None:
            selected_areas[str(name)] = float(area)
    for name, area in ((document.get("sampling") or {}).get("selected_areas") or {}).items():
        selected_areas.setdefault(str(name), float(area))

    plan_id = str(summary.get("id") or "unknown")
    return FloorPlan(
        id=plan_id,
        name=str(summary.get("name") or "untitled"),
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"plan_id": plan_id, "selected_areas": selected_areas},
    )
