"""Convert FloorPlan JSON to a realistic PNG."""

from __future__ import annotations

import base64
import io
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes

from export.draw import draw_door, draw_wall, draw_window
from floorplan.schemas.models import FloorPlan

Coord = tuple[float, float]
Edge = tuple[Coord, Coord]
Gap = tuple[float, float]

_EXTERIOR_THICK = 8.0
_INTERIOR_THICK = 3.0
_EDGE_KEY_PREC = 1
_MATCH_TOL = 8.0
_MIN_WALL_PIECE = 4.0


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


def _edge_key(a: Coord, b: Coord) -> tuple[float, float, float, float]:
    if a > b:
        a, b = b, a
    return (
        round(a[0], _EDGE_KEY_PREC),
        round(a[1], _EDGE_KEY_PREC),
        round(b[0], _EDGE_KEY_PREC),
        round(b[1], _EDGE_KEY_PREC),
    )


def _edge_from_key(key: tuple[float, float, float, float]) -> Edge:
    return (key[0], key[1]), (key[2], key[3])


def _lerp(a: Coord, b: Coord, t: float) -> Coord:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _project_t(p: Coord, a: Coord, b: Coord) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return 0.0
    return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2


def _dist_to_segment(p: Coord, a: Coord, b: Coord) -> float:
    t = max(0.0, min(1.0, _project_t(p, a, b)))
    q = _lerp(a, b, t)
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _opening_gap(start: Coord, end: Coord, a: Coord, b: Coord) -> Gap | None:
    if _dist_to_segment(start, a, b) > _MATCH_TOL or _dist_to_segment(end, a, b) > _MATCH_TOL:
        return None
    t0, t1 = _project_t(start, a, b), _project_t(end, a, b)
    lo, hi = sorted((t0, t1))
    if hi < -0.02 or lo > 1.02:
        return None
    return max(0.0, lo), min(1.0, hi)


def _merge_gaps(gaps: list[Gap]) -> list[Gap]:
    if not gaps:
        return []
    ordered = sorted(gaps)
    merged: list[Gap] = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 0.5:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _split_edge(a: Coord, b: Coord, gaps: list[Gap]) -> list[Edge]:
    if not gaps:
        return [(a, b)]
    pieces: list[Edge] = []
    cursor = 0.0
    for lo, hi in _merge_gaps(gaps):
        if lo - cursor >= _MIN_WALL_PIECE:
            pieces.append((_lerp(a, b, cursor), _lerp(a, b, lo)))
        cursor = max(cursor, hi)
    if 1.0 - cursor >= _MIN_WALL_PIECE:
        pieces.append((_lerp(a, b, cursor), _lerp(a, b, 1.0)))
    return pieces


def _collect_edges(rooms: list[tuple[str, list[Coord], str]]) -> Counter[tuple[float, float, float, float]]:
    counts: Counter[tuple[float, float, float, float]] = Counter()
    for _, polygon, _ in rooms:
        ring = _open_ring(polygon)
        for i in range(len(ring)):
            counts[_edge_key(ring[i], ring[(i + 1) % len(ring)])] += 1
    return counts


def _match_openings_to_edges(
    edges: Counter[tuple[float, float, float, float]],
    openings: list[tuple[Coord, Coord]],
) -> dict[tuple[float, float, float, float], list[Gap]]:
    gaps: dict[tuple[float, float, float, float], list[Gap]] = defaultdict(list)
    for start, end in openings:
        best_key: tuple[float, float, float, float] | None = None
        best_gap: Gap | None = None
        best_len = 0.0
        for key in edges:
            a, b = _edge_from_key(key)
            gap = _opening_gap(start, end, a, b)
            if gap is None:
                continue
            span = gap[1] - gap[0]
            if span > best_len:
                best_len = span
                best_key = key
                best_gap = gap
        if best_key is not None and best_gap is not None:
            gaps[best_key].append(best_gap)
    return gaps


def _slug_name(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"room_{index + 1}"


def _opening_lines(openings: list[dict[str, Any]]) -> list[tuple[Coord, Coord]]:
    lines: list[tuple[Coord, Coord]] = []
    for item in openings:
        start = item.get("start") or {}
        end = item.get("end") or {}
        lines.append(
            (
                (float(start["x"]), float(start["y"])),
                (float(end["x"]), float(end["y"])),
            )
        )
    return lines


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
    for win in plan.windows:
        geom.windows.append(((win.start.x, win.start.y), (win.end.x, win.end.y)))
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


def _polygon_centroid(polygon: list[Coord]) -> Coord:
    ring = _open_ring(polygon)
    if not ring:
        return (0.0, 0.0)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


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


def _add_door(ax: Axes, start: Coord, end: Coord) -> None:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    draw_door(ax, start, end, door_width=max(length * 0.85, 20.0), thickness=2.0)


def _add_window(ax: Axes, start: Coord, end: Coord) -> None:
    draw_window(
        ax,
        start,
        end,
        overall_thickness=_EXTERIOR_THICK,
        line_thickness=1.5,
    )


def _add_room_labels(ax: Axes, geom: _PixelGeom) -> int:
    count = 0
    for _, polygon, label in geom.rooms:
        rng = random.Random(f"{geom.plan_id}:{label}:{count}")
        font_size = 16 * rng.uniform(0.82, 1.18)
        cx, cy = _polygon_centroid(polygon)
        ax.text(cx, cy, label, ha="center", va="center", color="black", fontsize=font_size)
        count += 1
    return count


def _render_geom(geom: _PixelGeom) -> tuple[bytes, int, int]:
    if not geom.rooms:
        raise ValueError("Floor plan has no rooms")

    fig, ax = _create_canvas(geom.size)
    edge_counts = _collect_edges(geom.rooms)
    openings = geom.doors + geom.windows
    edge_gaps = _match_openings_to_edges(edge_counts, openings)

    wall_count = 0
    for key, count in edge_counts.items():
        a, b = _edge_from_key(key)
        thickness = _INTERIOR_THICK if count >= 2 else _EXTERIOR_THICK
        for p0, p1 in _split_edge(a, b, edge_gaps.get(key, [])):
            draw_wall(ax, p0, p1, thickness)
            wall_count += 1

    for start, end in geom.doors:
        _add_door(ax, start, end)
    for start, end in geom.windows:
        _add_window(ax, start, end)

    label_count = _add_room_labels(ax, geom)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, facecolor="white")
    plt.close(fig)
    return buf.getvalue(), wall_count, label_count


def export_realistic(
    *,
    plan: FloorPlan | None = None,
    rooms: list[dict[str, Any]] | None = None,
    doors: list[tuple[Coord, Coord]] | None = None,
    windows: list[tuple[Coord, Coord]] | None = None,
    canvas_size: int = 1024,
    size: int = 1024,
) -> dict[str, Any]:
    """Render a realistic floor plan PNG and return base64 + metadata."""
    geom = _resolve_geom(plan, rooms, size, doors=doors, windows=windows)
    png, wall_count, label_count = _render_geom(geom)
    source = "rooms" if rooms else "floor_plan"
    return {
        "png_b64": base64.b64encode(png).decode("ascii"),
        "size": size,
        "metadata": {
            "wall_count": wall_count,
            "door_count": len(geom.doors),
            "window_count": len(geom.windows),
            "label_count": label_count,
            "coord_source": source,
            "room_count": len(geom.rooms),
            "canvas_size": size,
        },
    }
