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
from export.symbols import draw_planned_symbol
from floorplan.schemas.models import FloorPlan

Coord = tuple[float, float]

DEFAULT_ROOM_OUTLINE_THICK = 4.0
DEFAULT_BOUNDARY_THICK = 8.0
DEFAULT_OPENING_THICK = 12.0
DEFAULT_LABEL_FONT_SIZE = 16.0
BLACKNESS_MIN = 0.7
BLACKNESS_MAX = 1.0


def _ink_color(blackness: float) -> tuple[float, float, float]:
    gray = 1.0 - blackness
    return (gray, gray, gray)


@dataclass
class _RoomPixel:
    slug: str
    polygon: list[Coord]
    display: str
    label_x: float | None = None
    label_y: float | None = None
    font_size: float | None = None
    is_label_disabled: bool = False


@dataclass
class _DoorPixel:
    start: Coord
    end: Coord
    is_arc_disabled: bool = False


@dataclass
class _PixelGeom:
    size: int
    rooms: list[_RoomPixel] = field(default_factory=list)
    doors: list[_DoorPixel] = field(default_factory=list)
    windows: list[tuple[Coord, Coord]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    plan_id: str = "export"


def _open_ring(polygon: list[Coord]) -> list[Coord]:
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        return list(polygon[:-1])
    return list(polygon)


def _slug_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"room_{index + 1}"


def _label_placement(
    raw: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, bool]:
    if not isinstance(raw, dict):
        return None, None, None, False
    try:
        x = float(raw["x"]) if "x" in raw else None
        y = float(raw["y"]) if "y" in raw else None
        font_size = float(raw["font_size"]) if "font_size" in raw else None
    except (TypeError, ValueError):
        return None, None, None, False
    return x, y, font_size, bool(raw.get("is_label_disabled"))


def _parse_door_item(item: Any) -> _DoorPixel | None:
    if isinstance(item, _DoorPixel):
        return item
    if isinstance(item, dict):
        start = item.get("start") or {}
        end = item.get("end") or {}
        try:
            return _DoorPixel(
                start=(float(start["x"]), float(start["y"])),
                end=(float(end["x"]), float(end["y"])),
                is_arc_disabled=bool(item.get("is_arc_disabled")),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        start, end = item[0], item[1]
        is_arc_disabled = bool(item[2]) if len(item) >= 3 else False
        return _DoorPixel(start=start, end=end, is_arc_disabled=is_arc_disabled)
    return None


def _geom_from_rooms(
    rooms: list[dict[str, Any]],
    size: int,
    *,
    doors: list[Any] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
    symbols: list[dict[str, Any]] | None = None,
) -> _PixelGeom:
    geom = _PixelGeom(size=size)
    for i, raw in enumerate(rooms):
        name = str(raw.get("name") or f"room_{i + 1}")
        polygon = [(float(p["x"]), float(p["y"])) for p in raw.get("polygon") or []]
        if len(polygon) < 3:
            continue
        label_x, label_y, font_size, is_label_disabled = _label_placement(raw.get("label"))
        geom.rooms.append(
            _RoomPixel(
                slug=_slug_name(name, i),
                polygon=polygon,
                display=name,
                label_x=label_x,
                label_y=label_y,
                font_size=font_size,
                is_label_disabled=is_label_disabled,
            )
        )
    for item in doors or []:
        door = _parse_door_item(item)
        if door is not None:
            geom.doors.append(door)
    geom.windows = list(windows or [])
    geom.symbols = list(symbols or [])
    return geom


def _geom_from_plan(plan: FloorPlan, size: int) -> _PixelGeom:
    geom = _PixelGeom(size=size, plan_id=plan.id)
    for room in plan.rooms:
        label = room.metadata.get("label")
        display = label.strip() if isinstance(label, str) and label.strip() else room.name.replace("_", " ")
        geom.rooms.append(_RoomPixel(slug=room.name, polygon=list(room.coords()), display=display))
    for door in plan.doors:
        geom.doors.append(_DoorPixel(start=(door.start.x, door.start.y), end=(door.end.x, door.end.y)))
    for window in plan.windows:
        geom.windows.append(((window.start.x, window.start.y), (window.end.x, window.end.y)))
    return geom


def _resolve_geom(
    plan: FloorPlan | None,
    rooms: list[dict[str, Any]] | None,
    size: int,
    *,
    doors: list[Any] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
    symbols: list[dict[str, Any]] | None = None,
) -> _PixelGeom:
    if rooms:
        geom = _geom_from_rooms(rooms, size, doors=doors, windows=windows, symbols=symbols)
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
    for room in geom.rooms:
        ring = _open_ring(room.polygon)
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


def _draw_room_polygons(
    ax: Axes,
    geom: _PixelGeom,
    *,
    room_outline_thick: float,
    ink: tuple[float, float, float],
) -> None:
    for room in geom.rooms:
        draw_polygon(
            ax,
            room.polygon,
            facecolor="none",
            edgecolor=ink,
            linewidth=room_outline_thick,
        )


def _draw_union_boundary(
    ax: Axes,
    geom: _PixelGeom,
    *,
    boundary_thick: float,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for ring in _union_boundary_rings(geom):
        draw_polygon(
            ax,
            ring,
            facecolor="none",
            edgecolor=ink,
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


def _draw_room_symbols(
    ax: Axes,
    geom: _PixelGeom,
    *,
    blackness: float,
) -> int:
    count = 0
    for placement in geom.symbols:
        if draw_planned_symbol(ax, placement, blackness=blackness):
            count += 1
    return count


def _draw_room_labels(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for room in geom.rooms:
        if room.is_label_disabled:
            continue
        if room.label_x is not None and room.label_y is not None:
            cx, cy = room.label_x, room.label_y
        else:
            cx, cy = _polygon_centroid(room.polygon)
        font_size = room.font_size if room.font_size is not None else DEFAULT_LABEL_FONT_SIZE
        ax.text(cx, cy, room.display, ha="center", va="center", color=ink, fontsize=font_size)
        count += 1
    return count


def _draw_doors(
    ax: Axes,
    geom: _PixelGeom,
    *,
    opening_thick: float,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for door in geom.doors:
        draw_door(
            ax,
            door.start,
            door.end,
            wall_thickness=opening_thick,
            to_the_right=count % 2 == 0,
            color=ink,
            draw_arc=not door.is_arc_disabled,
        )
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
) -> tuple[bytes, int, int, int, int, int, int, float]:
    if not geom.rooms:
        raise ValueError("Floor plan has no rooms")

    source_rng = rng or random.Random()
    blackness = source_rng.uniform(BLACKNESS_MIN, BLACKNESS_MAX)
    ink = _ink_color(blackness)

    fig, ax = _create_canvas(geom.size)
    _draw_room_polygons(ax, geom, room_outline_thick=room_outline_thick, ink=ink)
    boundary_count = _draw_union_boundary(ax, geom, boundary_thick=boundary_thick, ink=ink)
    door_count = _draw_doors(ax, geom, opening_thick=opening_thick, ink=ink)
    window_count = _draw_windows(ax, geom, opening_thick=boundary_thick)
    symbol_count = _draw_room_symbols(ax, geom, blackness=blackness)
    label_count = _draw_room_labels(ax, geom, ink=ink)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, facecolor="white")
    plt.close(fig)
    return (
        buf.getvalue(),
        boundary_count,
        len(geom.rooms),
        label_count,
        door_count,
        window_count,
        symbol_count,
        blackness,
    )


def export_realistic(
    *,
    plan: FloorPlan | None = None,
    rooms: list[dict[str, Any]] | None = None,
    doors: list[Any] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
    symbols: list[dict[str, Any]] | None = None,
    canvas_size: int = 1024,
    size: int = 1024,
    room_outline_thick: float = DEFAULT_ROOM_OUTLINE_THICK,
    boundary_thick: float = DEFAULT_BOUNDARY_THICK,
    opening_thick: float = DEFAULT_OPENING_THICK,
    symbol_probability: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Render room outlines, boundary, doors, windows, labels, and planned symbols; return base64 + metadata."""
    _ = canvas_size
    _ = symbol_probability
    geom = _resolve_geom(plan, rooms, size, doors=doors, windows=windows, symbols=symbols)
    (
        png,
        boundary_count,
        room_count,
        label_count,
        door_count,
        window_count,
        symbol_count,
        blackness,
    ) = _render_geom(
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
            "symbol_count": symbol_count,
            "blackness": round(blackness, 4),
        },
    }
