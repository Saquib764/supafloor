"""Convert FloorPlan JSON to a realistic PNG."""

from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes

from export.draw import (
    draw_dimension,
    draw_door,
    draw_pillar_outline,
    draw_pillar_textured,
    draw_polygon,
    draw_room_interior_textured,
    draw_wall_centerline,
    draw_wall_rect_outline,
    draw_wall_strip_outline,
    draw_wall_textured,
    extend_wall_endpoints,
    draw_window,
)
from export.symbols import draw_planned_symbol
from schemas.models import FloorPlan

Coord = tuple[float, float]

_WALL_JUNCTION_TOL = 0.5

DEFAULT_ROOM_OUTLINE_THICK = 4.0
DEFAULT_BOUNDARY_THICK = 8.0
DEFAULT_OPENING_THICK = 12.0
DEFAULT_LABEL_FONT_SIZE = 16.0
DEFAULT_DIMENSION_FONT_SIZE = 10.0
DEFAULT_WALL_CENTERLINE_THICK = 1.0
BLACKNESS_MIN = 0.7
BLACKNESS_MAX = 1.0
GRAY_BLACKNESS_MIN = 0.55
GRAY_BLACKNESS_MAX = 0.78
GRAIN_NOISE_STD = 16.0


def _ink_color(blackness: float) -> tuple[float, float, float]:
    gray = 1.0 - blackness
    return (gray, gray, gray)


def _apply_image_grain(png_bytes: bytes, rng: random.Random) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    arr = np.clip(arr + np_rng.normal(0.0, GRAIN_NOISE_STD, arr.shape), 0.0, 255.0)
    out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class _RoomPixel:
    polygon: list[Coord]
    display: str
    label_x: float | None = None
    label_y: float | None = None
    font_size: float | None = None
    is_label_disabled: bool = False
    is_interior_textured: bool = False
    interior_hatch: str | None = None


@dataclass
class _DoorPixel:
    start: Coord
    end: Coord
    is_arc_disabled: bool = False
    door_type: str = "single"
    arc_angle_deg: float = 90.0
    to_the_right: bool = True


@dataclass
class _PixelGeom:
    size: int
    rooms: list[_RoomPixel] = field(default_factory=list)
    doors: list[_DoorPixel] = field(default_factory=list)
    windows: list[tuple[Coord, Coord]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    pillars: list[dict[str, Any]] = field(default_factory=list)
    wall_rects: list[dict[str, Any]] = field(default_factory=list)
    walls: list[dict[str, Any]] = field(default_factory=list)
    wall_render_mode: int | None = None
    wall_hatch: str | None = None
    wall_centerline: bool = False
    dimensions: list[dict[str, Any]] = field(default_factory=list)
    margin_noise: list[dict[str, Any]] = field(default_factory=list)


def _open_ring(polygon: list[Coord]) -> list[Coord]:
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        return list(polygon[:-1])
    return list(polygon)


def _point_dict(x: float, y: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y)}


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
                door_type=str(item.get("door_type") or "single"),
                arc_angle_deg=float(item.get("arc_angle_deg") or 90.0),
                to_the_right=bool(item.get("to_the_right", True)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        start, end = item[0], item[1]
        is_arc_disabled = bool(item[2]) if len(item) >= 3 else False
        return _DoorPixel(start=start, end=end, is_arc_disabled=is_arc_disabled)
    return None


def _rooms_openings_from_plan(
    plan: FloorPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rooms = [
        {
            "name": room.name,
            "polygon": [_point_dict(x, y) for x, y in room.coords()],
        }
        for room in plan.rooms
    ]
    doors = [
        {
            "start": _point_dict(door.start.x, door.start.y),
            "end": _point_dict(door.end.x, door.end.y),
        }
        for door in plan.doors
    ]
    windows = [
        {
            "start": _point_dict(window.start.x, window.start.y),
            "end": _point_dict(window.end.x, window.end.y),
        }
        for window in plan.windows
    ]
    return rooms, doors, windows


def _geom_from_rooms(
    rooms: list[dict[str, Any]],
    size: int,
    *,
    doors: list[Any] | None = None,
    windows: list[Any] | None = None,
    symbols: list[dict[str, Any]] | None = None,
    pillars: list[dict[str, Any]] | None = None,
    wall_rects: list[dict[str, Any]] | None = None,
    walls: list[dict[str, Any]] | None = None,
    wall_render_mode: int | None = None,
    wall_hatch: str | None = None,
    wall_centerline: bool = False,
    dimensions: list[dict[str, Any]] | None = None,
    margin_noise: list[dict[str, Any]] | None = None,
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
                polygon=polygon,
                display=name,
                label_x=label_x,
                label_y=label_y,
                font_size=font_size,
                is_label_disabled=is_label_disabled,
                is_interior_textured=bool(raw.get("is_interior_textured")),
                interior_hatch=raw.get("interior_hatch"),
            )
        )
    for item in doors or []:
        door = _parse_door_item(item)
        if door is not None:
            geom.doors.append(door)
    parsed_windows: list[tuple[Coord, Coord]] = []
    for item in windows or []:
        if isinstance(item, dict):
            start = item.get("start") or {}
            end = item.get("end") or {}
            try:
                parsed_windows.append(
                    ((float(start["x"]), float(start["y"])), (float(end["x"]), float(end["y"])))
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            parsed_windows.append((item[0], item[1]))
    geom.windows = parsed_windows
    geom.symbols = list(symbols or [])
    geom.pillars = list(pillars or [])
    geom.wall_rects = list(wall_rects or [])
    geom.walls = list(walls or [])
    geom.wall_render_mode = wall_render_mode
    geom.wall_hatch = wall_hatch
    geom.wall_centerline = wall_centerline
    geom.dimensions = list(dimensions or [])
    geom.margin_noise = list(margin_noise or [])
    return geom


def _ensure_walls(
    geom: _PixelGeom,
    rooms: list[dict[str, Any]],
    doors: list[Any] | None,
    windows: list[Any] | None,
    rng: random.Random | None,
) -> None:
    from extraction.transform import assign_wall_rendering
    from extraction.walls import extract_walls

    door_dicts = [item for item in (doors or []) if isinstance(item, dict)]
    window_dicts = [item for item in (windows or []) if isinstance(item, dict)]
    if not geom.walls:
        geom.walls = extract_walls(rooms, doors=door_dicts, windows=window_dicts)
    if geom.wall_render_mode is None:
        source_rng = rng or random.Random()
        geom.walls, render_meta, geom.pillars = assign_wall_rendering(
            geom.walls,
            source_rng,
            pillars=geom.pillars,
        )
        geom.wall_render_mode = render_meta["wall_render_mode"]
        geom.wall_hatch = render_meta["wall_hatch"]
        geom.wall_centerline = bool(render_meta["wall_centerline"])


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


def _polygon_centroid(polygon: list[Coord]) -> Coord:
    ring = _open_ring(polygon)
    if not ring:
        return (0.0, 0.0)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _wall_endpoint(wall: dict[str, Any], key: str) -> Coord | None:
    point = wall.get(key) or {}
    try:
        return (float(point["x"]), float(point["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _shared_wall_endpoints(walls: list[dict[str, Any]]) -> set[tuple[int, str]]:
    refs: list[tuple[int, str, Coord]] = []
    for index, wall in enumerate(walls):
        for key in ("start", "end"):
            point = _wall_endpoint(wall, key)
            if point is not None:
                refs.append((index, key, point))

    tol2 = _WALL_JUNCTION_TOL * _WALL_JUNCTION_TOL
    shared: set[tuple[int, str]] = set()
    for a in range(len(refs)):
        index_a, key_a, (ax, ay) = refs[a]
        for b in range(a + 1, len(refs)):
            index_b, key_b, (bx, by) = refs[b]
            dx, dy = ax - bx, ay - by
            if dx * dx + dy * dy <= tol2:
                shared.add((index_a, key_a))
                shared.add((index_b, key_b))
    return shared


def _draw_room_interiors(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for room in geom.rooms:
        if not room.is_interior_textured or not room.interior_hatch:
            continue
        draw_room_interior_textured(
            ax,
            room.polygon,
            ink=ink,
            hatch=room.interior_hatch,
        )
        count += 1
    return count


def _draw_walls(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
    room_outline_thick: float,
    boundary_thick: float,
    filled_black_walls: bool = False,
) -> int:
    shared = _shared_wall_endpoints(geom.walls)
    count = 0
    for index, wall in enumerate(geom.walls):
        start = _wall_endpoint(wall, "start")
        end = _wall_endpoint(wall, "end")
        if start is None or end is None:
            continue
        thickness = boundary_thick if wall.get("is_exterior") else room_outline_thick
        draw_start, draw_end = extend_wall_endpoints(
            start,
            end,
            thickness=thickness,
            extend_start=(index, "start") in shared,
            extend_end=(index, "end") in shared,
        )
        is_textured = bool(wall.get("is_textured"))
        if is_textured or filled_black_walls:
            draw_wall_textured(
                ax,
                draw_start,
                draw_end,
                thickness=thickness,
                ink=ink,
                wall_hatch=geom.wall_hatch if is_textured else None,
                fill_black=filled_black_walls,
            )
        draw_wall_strip_outline(ax, draw_start, draw_end, thickness=thickness, ink=ink)
        if geom.wall_centerline:
            draw_wall_centerline(
                ax,
                start,
                end,
                ink=ink,
                linewidth=DEFAULT_WALL_CENTERLINE_THICK,
            )
        count += 1
    return count


def _draw_wall_rects(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for raw in geom.wall_rects:
        polygon = raw.get("polygon") or []
        if len(polygon) < 3:
            continue
        try:
            ring = [(float(point["x"]), float(point["y"])) for point in polygon]
        except (KeyError, TypeError, ValueError):
            continue
        draw_wall_rect_outline(ax, ring, ink=ink)
        count += 1
    return count


def _draw_pillars(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
    filled_black_walls: bool = False,
) -> None:
    for raw in geom.pillars:
        try:
            cx = float(raw["x"])
            cy = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
            rotation_deg = float(raw.get("rotation_deg") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        is_textured = bool(raw.get("is_textured"))
        if is_textured or filled_black_walls:
            draw_pillar_textured(
                ax,
                cx,
                cy,
                width,
                height,
                rotation_deg,
                ink=ink,
                wall_hatch=geom.wall_hatch if is_textured else None,
                fill_black=filled_black_walls,
            )
        draw_pillar_outline(ax, cx, cy, width, height, rotation_deg, ink=ink)


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


def _draw_dimensions(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for spec in geom.dimensions:
        draw_dimension(ax, spec, ink=ink, font_size=DEFAULT_DIMENSION_FONT_SIZE)
        count += 1
    return count


def _draw_margin_noise(
    ax: Axes,
    geom: _PixelGeom,
    *,
    ink: tuple[float, float, float],
) -> int:
    count = 0
    for spec in geom.margin_noise:
        try:
            text = str(spec["text"])
            x = float(spec["x"])
            y = float(spec["y"])
            font_size = float(spec.get("font_size") or DEFAULT_DIMENSION_FONT_SIZE)
            rotation = float(spec.get("rotation") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            color=ink,
            fontsize=font_size,
            rotation=rotation,
            family="monospace",
            parse_math=False,
        )
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
            to_the_right=door.to_the_right,
            color=ink,
            draw_arc=not door.is_arc_disabled,
            door_type=door.door_type,
            arc_angle_deg=door.arc_angle_deg,
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
    image_gray: bool = False,
    image_grain: bool = False,
    draw_interiors: bool = True,
    draw_labels: bool = True,
    draw_dimensions: bool = True,
    draw_margin_noise: bool = True,
    draw_symbols: bool = True,
    draw_wall_rects: bool = True,
    filled_black_walls: bool = False,
    blackness_override: float | None = None,
) -> tuple[bytes, int, int, int, int, int, int, int, int, float]:
    if not geom.rooms:
        raise ValueError("Floor plan has no rooms")
    if not geom.walls:
        raise ValueError("Floor plan has no walls")

    source_rng = rng or random.Random()
    if blackness_override is not None:
        blackness = blackness_override
    elif image_gray:
        blackness = source_rng.uniform(GRAY_BLACKNESS_MIN, GRAY_BLACKNESS_MAX)
    else:
        blackness = source_rng.uniform(BLACKNESS_MIN, BLACKNESS_MAX)
    ink = _ink_color(blackness)

    fig, ax = _create_canvas(geom.size)
    if draw_interiors:
        _draw_room_interiors(ax, geom, ink=ink)
    wall_count = _draw_walls(
        ax,
        geom,
        ink=ink,
        room_outline_thick=room_outline_thick,
        boundary_thick=boundary_thick,
        filled_black_walls=filled_black_walls,
    )
    door_count = _draw_doors(ax, geom, opening_thick=opening_thick, ink=ink)
    window_count = _draw_windows(ax, geom, opening_thick=boundary_thick)
    wall_rect_count = _draw_wall_rects(ax, geom, ink=ink) if draw_wall_rects else 0
    _draw_pillars(ax, geom, ink=ink, filled_black_walls=filled_black_walls)
    symbol_count = _draw_room_symbols(ax, geom, blackness=blackness) if draw_symbols else 0
    label_count = _draw_room_labels(ax, geom, ink=ink) if draw_labels else 0
    dimension_count = _draw_dimensions(ax, geom, ink=ink) if draw_dimensions else 0
    if draw_margin_noise:
        _draw_margin_noise(ax, geom, ink=ink)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, facecolor="white")
    plt.close(fig)
    png = buf.getvalue()
    if image_grain:
        png = _apply_image_grain(png, source_rng)
    return (
        png,
        wall_count,
        len(geom.rooms),
        label_count,
        door_count,
        window_count,
        symbol_count,
        wall_rect_count,
        dimension_count,
        blackness,
    )


def export_realistic(
    *,
    plan: FloorPlan | None = None,
    rooms: list[dict[str, Any]] | None = None,
    doors: list[Any] | None = None,
    windows: list[Any] | None = None,
    symbols: list[dict[str, Any]] | None = None,
    pillars: list[dict[str, Any]] | None = None,
    wall_rects: list[dict[str, Any]] | None = None,
    walls: list[dict[str, Any]] | None = None,
    wall_render_mode: int | None = None,
    wall_hatch: str | None = None,
    wall_centerline: bool = False,
    dimensions: list[dict[str, Any]] | None = None,
    margin_noise: list[dict[str, Any]] | None = None,
    size: int = 1024,
    room_outline_thick: float = DEFAULT_ROOM_OUTLINE_THICK,
    boundary_thick: float = DEFAULT_BOUNDARY_THICK,
    opening_thick: float = DEFAULT_OPENING_THICK,
    rng: random.Random | None = None,
    image_gray: bool = False,
    image_grain: bool = False,
) -> dict[str, Any]:
    """Render walls, openings, labels, and planned symbols; return base64 + metadata."""
    if plan is not None:
        rooms_list, doors_list, windows_list = _rooms_openings_from_plan(plan)
    else:
        if not rooms:
            raise ValueError("Provide either plan or rooms")
        rooms_list = list(rooms)
        doors_list = list(doors or [])
        windows_list = list(windows or [])

    geom = _geom_from_rooms(
        rooms_list,
        size,
        doors=doors_list,
        windows=windows_list,
        symbols=symbols,
        pillars=pillars,
        wall_rects=wall_rects,
        walls=walls,
        wall_render_mode=wall_render_mode,
        wall_hatch=wall_hatch,
        wall_centerline=wall_centerline,
        dimensions=dimensions,
        margin_noise=margin_noise,
    )
    _ensure_walls(geom, rooms_list, doors_list, windows_list, rng)

    (
        png,
        wall_count,
        room_count,
        label_count,
        door_count,
        window_count,
        symbol_count,
        wall_rect_count,
        dimension_count,
        blackness,
    ) = _render_geom(
        geom,
        room_outline_thick=room_outline_thick,
        boundary_thick=boundary_thick,
        opening_thick=opening_thick,
        rng=rng,
        image_gray=image_gray,
        image_grain=image_grain,
    )
    return {
        "png_b64": base64.b64encode(png).decode("ascii"),
        "size": size,
        "metadata": {
            "wall_count": wall_count,
            "room_count": room_count,
            "label_count": label_count,
            "door_count": door_count,
            "window_count": window_count,
            "coord_source": "floor_plan" if plan is not None else "rooms",
            "canvas_size": size,
            "room_outline_thick": room_outline_thick,
            "boundary_thick": boundary_thick,
            "opening_thick": opening_thick,
            "symbol_count": symbol_count,
            "wall_rect_count": wall_rect_count,
            "dimension_count": dimension_count,
            "blackness": round(blackness, 4),
            "image_gray": image_gray,
            "image_grain": image_grain,
            "wall_render_mode": geom.wall_render_mode,
            "wall_hatch": geom.wall_hatch,
            "wall_centerline": geom.wall_centerline,
            "textured_room_count": sum(
                1 for room in geom.rooms if room.is_interior_textured
            ),
        },
    }


def export_simple(
    *,
    plan: FloorPlan | None = None,
    rooms: list[dict[str, Any]] | None = None,
    doors: list[Any] | None = None,
    windows: list[Any] | None = None,
    symbols: list[dict[str, Any]] | None = None,
    pillars: list[dict[str, Any]] | None = None,
    wall_rects: list[dict[str, Any]] | None = None,
    walls: list[dict[str, Any]] | None = None,
    wall_render_mode: int | None = None,
    wall_hatch: str | None = None,
    wall_centerline: bool = False,
    size: int = 1024,
    room_outline_thick: float = DEFAULT_ROOM_OUTLINE_THICK,
    boundary_thick: float = DEFAULT_BOUNDARY_THICK,
    opening_thick: float = DEFAULT_OPENING_THICK,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Render like export_realistic but with solid walls and no text, dimensions, room fills, or noise."""
    if plan is not None:
        rooms_list, doors_list, windows_list = _rooms_openings_from_plan(plan)
    else:
        if not rooms:
            raise ValueError("Provide either plan or rooms")
        rooms_list = list(rooms)
        doors_list = list(doors or [])
        windows_list = list(windows or [])

    geom = _geom_from_rooms(
        rooms_list,
        size,
        doors=doors_list,
        windows=windows_list,
        symbols=symbols,
        pillars=pillars,
        wall_rects=wall_rects,
        walls=walls,
        wall_render_mode=wall_render_mode,
        wall_hatch=wall_hatch,
        wall_centerline=wall_centerline,
    )
    _ensure_walls(geom, rooms_list, doors_list, windows_list, rng)

    (
        png,
        wall_count,
        room_count,
        _label_count,
        door_count,
        window_count,
        symbol_count,
        wall_rect_count,
        _dimension_count,
        blackness,
    ) = _render_geom(
        geom,
        room_outline_thick=room_outline_thick,
        boundary_thick=boundary_thick,
        opening_thick=opening_thick,
        rng=rng,
        image_grain=False,
        draw_interiors=False,
        draw_labels=False,
        draw_dimensions=False,
        draw_margin_noise=False,
        draw_symbols=False,
        draw_wall_rects=False,
        filled_black_walls=True,
        blackness_override=BLACKNESS_MAX,
    )
    return {
        "png_b64": base64.b64encode(png).decode("ascii"),
        "size": size,
        "metadata": {
            "wall_count": wall_count,
            "room_count": room_count,
            "door_count": door_count,
            "window_count": window_count,
            "symbol_count": symbol_count,
            "wall_rect_count": wall_rect_count,
            "blackness": round(blackness, 4),
        },
    }
