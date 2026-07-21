"""Generation document: parameters (inputs) + generated (algorithm output)."""

from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from floorplan.geometry.authors import make_room
from floorplan.geometry.boundary import (
    boundary_from_params,
    force_manhattan,
    is_manhattan,
    sample_boundary_params,
)
from floorplan.requirements.areas import AREA_OPTIONS, area_options_for, name_root
from floorplan.requirements.program import space_names_for
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, Point2D

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
    "Entry": RoomType.ENTRY,
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
    "Entry": ZoneType.CIRCULATION,
    "Closet": ZoneType.PRIVATE,
    "Balcony": ZoneType.OUTDOOR,
    "Laundry": ZoneType.SERVICE,
    "Garage": ZoneType.SERVICE,
}


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


def _sample_rect(
    area: float,
    rng: random.Random,
    bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    w, h = _rect_for_area(area, rng)
    minx, miny, maxx, maxy = bounds
    x_lo = int(math.floor(minx - w * 0.25))
    x_hi = int(math.ceil(maxx - w * 0.75))
    y_lo = int(math.floor(miny - h * 0.25))
    y_hi = int(math.ceil(maxy - h * 0.75))
    if x_hi < x_lo:
        x_lo, x_hi = int(minx), int(minx)
    if y_hi < y_lo:
        y_lo, y_hi = int(miny), int(miny)
    return {
        "x": rng.randint(x_lo, x_hi),
        "y": rng.randint(y_lo, y_hi),
        "w": w,
        "h": h,
        "area": float(w * h),
    }


def run_generation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Pure algorithm under test: parameters → generated geometry."""
    boundary, boundary_gen = boundary_from_params(parameters["start_rect"], parameters.get("notches") or [])
    boundary_poly = Polygon([(p.x, p.y) for p in boundary.exterior])

    spaces: list[dict[str, Any]] = []
    occupied: list[Polygon] = []
    for rect in parameters.get("rectangles") or []:
        name = rect["name"]
        geom = box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])
        geom = _largest_polygon(geom.intersection(boundary_poly))
        if geom is None:
            spaces.append({"name": name, "polygon": [], "area": 0.0, "dropped": "outside_boundary"})
            continue
        if occupied:
            geom = _largest_polygon(geom.difference(unary_union(occupied)))
        if geom is None or geom.area < _MIN_SPACE_AREA:
            spaces.append({"name": name, "polygon": [], "area": 0.0, "dropped": "overlap"})
            continue
        if not is_manhattan(list(geom.exterior.coords[:-1])):
            geom = _largest_polygon(Polygon(force_manhattan(list(geom.exterior.coords[:-1]))))
        if geom is None or geom.area < _MIN_SPACE_AREA:
            spaces.append({"name": name, "polygon": [], "area": 0.0, "dropped": "invalid"})
            continue
        spaces.append({"name": name, "polygon": _poly_points(geom), "area": round(float(geom.area), 3)})
        occupied.append(geom)

    return {"boundary": boundary_gen, "spaces": spaces}


def sample_parameters(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample all RNG choices needed to recreate geometry later."""
    rng = random.Random(seed)
    boundary_params = sample_boundary_params(rng, target_area=target_area)
    # Bounds for placement use the start rect bbox (deterministic before notches)
    sr = boundary_params["start_rect"]
    bounds = (0.0, 0.0, float(sr["w"]), float(sr["h"]))

    names = space_names_for(total_rooms, rng)
    selected_areas: dict[str, float] = {}
    rectangles: list[dict[str, Any]] = []
    for name in names:
        chosen = float(rng.choice(area_options_for(name)))
        selected_areas[name] = chosen
        rect = _sample_rect(chosen, rng, bounds)
        rect["name"] = name
        rectangles.append(rect)

    return {
        "seed": seed,
        "total_rooms": max(1, int(total_rooms)),
        "target_area": boundary_params["target_area"],
        "aspect": boundary_params["aspect"],
        "start_rect": boundary_params["start_rect"],
        "notches": boundary_params["notches"],
        "space_names": names,
        "selected_areas": selected_areas,
        "rectangles": rectangles,
        "area_library": {root: list(opts) for root, opts in AREA_OPTIONS.items()},
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


def recreate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Re-run algorithm from saved parameters; refresh generated section."""
    parameters = document["parameters"]
    generated = run_generation(parameters)
    return {
        **document,
        "generated": generated,
        "label": document.get("label") or f"{parameters.get('total_rooms', '?')}-room",
    }


def document_to_plan(document: dict[str, Any]) -> FloorPlan:
    """Build FloorPlan from document.generated (no doors/windows)."""
    generated = document["generated"]
    parameters = document["parameters"]
    poly = generated["boundary"]["polygon"]
    boundary = Boundary(exterior=[Point2D(x=float(p[0]), y=float(p[1])) for p in poly])

    rooms = []
    for space in generated.get("spaces") or []:
        pts = space.get("polygon") or []
        if len(pts) < 3:
            continue
        root = name_root(space["name"])
        rooms.append(
            make_room(
                space["name"],
                [(p[0], p[1]) for p in pts],
                room_type=_NAME_TO_TYPE.get(root, RoomType.OTHER),
                zone=_NAME_TO_ZONE.get(root, ZoneType.PUBLIC),
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
    return FloorPlan(
        name=document.get("label") or f"{parameters.get('total_rooms', '?')}-room",
        rooms=rooms,
        doors=[],
        windows=[],
        boundary=boundary,
        metadata={"plan_id": document.get("id"), "seed": parameters.get("seed")},
    )


# --- Compat wrappers used by older call sites / tests ---


def build_context(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    doc = build_document(total_rooms, target_area=target_area, seed=seed)
    boundary = Boundary(
        exterior=[Point2D(x=float(p[0]), y=float(p[1])) for p in doc["generated"]["boundary"]["polygon"]]
    )
    return {
        **doc["parameters"],
        "boundary": doc["generated"]["boundary"],
        "spaces": doc["generated"]["spaces"],
        "random": {
            "aspect": doc["parameters"]["aspect"],
            "n_notches": len(doc["parameters"]["notches"]),
            "notches": doc["parameters"]["notches"],
            "selected_areas": doc["parameters"]["selected_areas"],
        },
        "_boundary": boundary,
        "_document": doc,
    }


def context_to_plan(context: dict[str, Any]) -> FloorPlan:
    if "_document" in context:
        return document_to_plan(context["_document"])
    return document_to_plan(
        {
            "id": "ephemeral",
            "label": f"{context.get('total_rooms', '?')}-room",
            "parameters": context,
            "generated": {
                "boundary": context["boundary"],
                "spaces": context["spaces"],
            },
        }
    )


def public_context(context: dict[str, Any]) -> dict[str, Any]:
    if "_document" in context:
        doc = context["_document"]
        return {"parameters": doc["parameters"], "generated": doc["generated"], "id": doc["id"], "label": doc["label"]}
    return {k: v for k, v in context.items() if not k.startswith("_")}
