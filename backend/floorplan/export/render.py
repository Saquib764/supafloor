"""Convert FloorPlan JSON to a realistic PNG."""

from __future__ import annotations

import base64
import io
import random
import re
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from shapely.geometry import Polygon
from shapely.ops import unary_union

from export.draw import draw_door, draw_polygon, draw_window
from floorplan.schemas.models import FloorPlan

Coord = tuple[float, float]

DEFAULT_ROOM_OUTLINE_THICK = 4.0
DEFAULT_BOUNDARY_THICK = 8.0
DEFAULT_OPENING_THICK = 12.0


@dataclass
class _PixelGeom:
    size: int
    rooms: list[tuple[str, list[Coord], str]] = field(default_factory=list)
    doors: list[tuple[Coord, Coord]] = field(default_factory=list)
    windows: list[tuple[Coord, Coord]] = field(default_factory=list)
    plan_id: str = "export"


def _open_ring(polygon: list[Coord]) -> list[Coord]:
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        return list(polygon[:-1])
    return list(polygon)


def _slug_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"room_{index + 1}"


def _geom_from_rooms(
    rooms: list[dict[str, Any]],
    size: int,
    *,
    doors: list[tuple[Coord, Coord]] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
) -> _PixelGeom:
    geom = _PixelGeom(size=size)
    for i, raw in enumerate(rooms):
        name = str(raw.get("name") or f"room_{i + 1}")
        polygon = [(float(p["x"]), float(p["y"])) for p in raw.get("polygon") or []]
        if len(polygon) < 3:
            continue
        geom.rooms.append((_slug_name(name, i), polygon, name))
    geom.doors = list(doors or [])
    geom.windows = list(windows or [])
    return geom


def _geom_from_plan(plan: FloorPlan, size: int) -> _PixelGeom:
    geom = _PixelGeom(size=size, plan_id=plan.id)
    for room in plan.rooms:
        label = room.metadata.get("label")
        display = label.strip() if isinstance(label, str) and label.strip() else room.name.replace("_", " ")
        geom.rooms.append((room.name, list(room.coords()), display))
    for door in plan.doors:
        geom.doors.append(((door.start.x, door.start.y), (door.end.x, door.end.y)))
    for window in plan.windows:
        geom.windows.append(((window.start.x, window.start.y), (window.end.x, window.end.y)))
    return geom


def _resolve_geom(
    plan: FloorPlan | None,
    rooms: list[dict[str, Any]] | None,
    size: int,
    *,
    doors: list[tuple[Coord, Coord]] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
) -> _PixelGeom:
    if rooms:
        geom = _geom_from_rooms(rooms, size, doors=doors, windows=windows)
        geom.plan_id = "rooms"
        return geom
    if plan is not None:
        return _geom_from_plan(plan, size)
    raise ValueError("Provide either plan or rooms")


def _create_canvas(size: int) -> tuple[Any, Axes]:
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def _room_polygons(geom: _PixelGeom) -> list[Polygon]:
    polys: list[Polygon] = []
    for _, polygon, _ in geom.rooms:
        ring = _open_ring(polygon)
        if len(ring) >= 3:
            polys.append(Polygon(ring))
    return polys


def _union_boundary_rings(geom: _PixelGeom) -> list[list[Coord]]:
    polys = _room_polygons(geom)
    if not polys:
        return []
    merged = unary_union(polys)
    if merged.is_empty:
        return []

    if merged.geom_type == "Polygon":
        geoms = [merged]
    elif merged.geom_type == "MultiPolygon":
        geoms = list(merged.geoms)
    else:
        return []

    rings: list[list[Coord]] = []
    for poly in geoms:
        if poly.is_empty:
            continue
        rings.append([(float(x), float(y)) for x, y in poly.exterior.coords])
    return rings


def _draw_room_polygons(ax: Axes, geom: _PixelGeom, *, room_outline_thick: float) -> None:
    for _, polygon, _ in geom.rooms:
        draw_polygon(
            ax,
            polygon,
            facecolor="none",
            edgecolor="black",
            linewidth=room_outline_thick,
        )


def _draw_union_boundary(ax: Axes, geom: _PixelGeom, *, boundary_thick: float) -> int:
    count = 0
    for ring in _union_boundary_rings(geom):
        draw_polygon(
            ax,
            ring,
            facecolor="none",
            edgecolor="black",
            linewidth=boundary_thick,
        )
        count += 1
    return count


def _polygon_centroid(polygon: list[Coord]) -> Coord:
    ring = _open_ring(polygon)
    if not ring:
        return (0.0, 0.0)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _draw_room_labels(ax: Axes, geom: _PixelGeom, *, rng: random.Random | None = None) -> int:
    source_rng = rng or random.Random()
    count = 0
    for _, polygon, label in geom.rooms:
        font_size = 16 * source_rng.uniform(0.82, 1.18)
        cx, cy = _polygon_centroid(polygon)
        ax.text(cx, cy, label, ha="center", va="center", color="black", fontsize=font_size)
        count += 1
    return count


def _draw_doors(ax: Axes, geom: _PixelGeom, *, opening_thick: float) -> int:
    count = 0
    for start, end in geom.doors:
        draw_door(ax, start, end, wall_thickness=opening_thick, to_the_right=count % 2 == 0)
        count += 1
    return count


def _draw_windows(ax: Axes, geom: _PixelGeom, *, opening_thick: float) -> int:
    count = 0
    for start, end in geom.windows:
        draw_window(ax, start, end, wall_thickness=opening_thick)
        count += 1
    return count


def _render_geom(
    geom: _PixelGeom,
    *,
    room_outline_thick: float,
    boundary_thick: float,
    opening_thick: float,
    rng: random.Random | None = None,
) -> tuple[bytes, int, int, int, int, int]:
    if not geom.rooms:
        raise ValueError("Floor plan has no rooms")

    fig, ax = _create_canvas(geom.size)
    _draw_room_polygons(ax, geom, room_outline_thick=room_outline_thick)
    boundary_count = _draw_union_boundary(ax, geom, boundary_thick=boundary_thick)
    door_count = _draw_doors(ax, geom, opening_thick=opening_thick)
    window_count = _draw_windows(ax, geom, opening_thick=boundary_thick)
    label_count = _draw_room_labels(ax, geom, rng=rng)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, facecolor="white")
    plt.close(fig)
    return buf.getvalue(), boundary_count, len(geom.rooms), label_count, door_count, window_count


def export_realistic(
    *,
    plan: FloorPlan | None = None,
    rooms: list[dict[str, Any]] | None = None,
    doors: list[tuple[Coord, Coord]] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
    canvas_size: int = 1024,
    size: int = 1024,
    room_outline_thick: float = DEFAULT_ROOM_OUTLINE_THICK,
    boundary_thick: float = DEFAULT_BOUNDARY_THICK,
    opening_thick: float = DEFAULT_OPENING_THICK,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Render room outlines, boundary, doors, windows, and labels; return base64 + metadata."""
    _ = canvas_size
    geom = _resolve_geom(plan, rooms, size, doors=doors, windows=windows)
    png, boundary_count, room_count, label_count, door_count, window_count = _render_geom(
        geom,
        room_outline_thick=room_outline_thick,
        boundary_thick=boundary_thick,
        opening_thick=opening_thick,
        rng=rng,
    )
    source = "rooms" if rooms else "floor_plan"
    return {
        "png_b64": base64.b64encode(png).decode("ascii"),
        "size": size,
        "metadata": {
            "boundary_count": boundary_count,
            "room_count": room_count,
            "label_count": label_count,
            "door_count": door_count,
            "window_count": window_count,
            "coord_source": source,
            "canvas_size": size,
            "room_outline_thick": room_outline_thick,
            "boundary_thick": boundary_thick,
            "opening_thick": opening_thick,
        },
    }
