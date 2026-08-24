"""Extract walls, openings, and room names from a floorplan document."""

from __future__ import annotations

import base64
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from export.render import export_realistic

from .transform import _CANVAS_SIZE, pop_transform_stats, transform_extraction

Coord = tuple[float, float]
Edge = tuple[Coord, Coord]
Gap = tuple[float, float]

_EDGE_KEY_PREC = 1
_MATCH_TOL = 0.5
_MIN_WALL_PIECE = 0.01


def output_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "dev" / "output" / "extraction"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 3), "y": round(float(y), 3)}


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
    dx, dy = p[0] - q[0], p[1] - q[1]
    return (dx * dx + dy * dy) ** 0.5


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
        if lo <= prev_hi + 0.01:
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


def _collect_edges(room_polygons: list[list[Coord]]) -> Counter[tuple[float, float, float, float]]:
    counts: Counter[tuple[float, float, float, float]] = Counter()
    for polygon in room_polygons:
        ring = _open_ring(polygon)
        for i in range(len(ring)):
            counts[_edge_key(ring[i], ring[(i + 1) % len(ring)])] += 1
    return counts


def _opening_lines(items: list[dict[str, Any]]) -> list[tuple[Coord, Coord]]:
    lines: list[tuple[Coord, Coord]] = []
    for item in items:
        start = item.get("start") or {}
        end = item.get("end") or {}
        lines.append(
            (
                (float(start["x"]), float(start["y"])),
                (float(end["x"]), float(end["y"])),
            )
        )
    return lines


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


def _wall_segment(start: Coord, end: Coord, *, is_exterior: bool) -> dict[str, Any]:
    return {
        "start": _point(start[0], start[1]),
        "end": _point(end[0], end[1]),
        "is_exterior": is_exterior,
    }


def _extract_rooms(document: dict[str, Any]) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    for raw in document.get("rooms") or []:
        name = str(raw.get("name") or "")
        polygon = [_point(float(p["x"]), float(p["y"])) for p in raw.get("polygon") or []]
        if len(polygon) < 3 or not name:
            continue
        rooms.append({"name": name, "polygon": polygon})
    return rooms


def _extract_openings(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    openings = document.get("openings") or {}
    doors: list[dict[str, Any]] = []
    for door in openings.get("doors") or []:
        start = door.get("start") or {}
        end = door.get("end") or {}
        doors.append(
            {
                "id": door.get("id"),
                "connects": list(door.get("connects") or []),
                "start": _point(float(start["x"]), float(start["y"])),
                "end": _point(float(end["x"]), float(end["y"])),
                "is_exterior": bool(door.get("is_exterior")),
            }
        )

    windows: list[dict[str, Any]] = []
    for window in openings.get("windows") or []:
        start = window.get("start") or {}
        end = window.get("end") or {}
        windows.append(
            {
                "id": window.get("id"),
                "room_names": list(window.get("room_names") or []),
                "start": _point(float(start["x"]), float(start["y"])),
                "end": _point(float(end["x"]), float(end["y"])),
            }
        )
    return {"doors": doors, "windows": windows}


def _extract_walls(
    room_polygons: list[list[Coord]],
    openings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    edge_counts = _collect_edges(room_polygons)
    opening_items = _opening_lines(openings["doors"]) + _opening_lines(openings["windows"])
    edge_gaps = _match_openings_to_edges(edge_counts, opening_items)

    walls: list[dict[str, Any]] = []
    for key, count in edge_counts.items():
        a, b = _edge_from_key(key)
        is_exterior = count < 2
        for p0, p1 in _split_edge(a, b, edge_gaps.get(key, [])):
            walls.append(_wall_segment(p0, p1, is_exterior=is_exterior))
    return walls


def _plan_id(document: dict[str, Any], source: Path | None = None) -> str:
    summary = document.get("summary") or {}
    plan_id = summary.get("id")
    if plan_id:
        return str(plan_id)
    if source is not None:
        if source.name == "floorplan.json" and source.parent.name:
            return source.parent.name
        return source.stem
    raise ValueError("Could not determine plan id from document or path")


def extract_floorplan(document: dict[str, Any], *, plan_id: str | None = None) -> dict[str, Any]:
    """Transform a floorplan document into walls, openings, and room names."""
    rooms = _extract_rooms(document)
    openings = _extract_openings(document)
    room_polygons = [
        [(float(p["x"]), float(p["y"])) for p in room["polygon"]]
        for room in rooms
    ]
    resolved_id = plan_id or _plan_id(document)
    return {
        "id": resolved_id,
        "rooms": rooms,
        "walls": _extract_walls(room_polygons, openings),
        "openings": openings,
    }


def extract_from_path(path: Path) -> dict[str, Any]:
    """Load `<id>/floorplan.json` and extract walls, openings, and room names."""
    document = json.loads(path.read_text())
    return extract_floorplan(document, plan_id=_plan_id(document, path))


def prepare_extraction(
    path: Path,
    *,
    rng: random.Random | None = None,
    symbol_probability: float = 0.0,
) -> dict[str, Any]:
    """Load a floorplan or extraction JSON and return canvas-ready extraction data."""
    document = json.loads(path.read_text())
    if document.get("walls"):
        return document
    extracted = extract_from_path(path) if path.name == "floorplan.json" else extract_floorplan(
        document,
        plan_id=_plan_id(document, path),
    )
    return transform_extraction(
        extracted,
        rng=rng,
        symbol_probability=symbol_probability,
    )


def write_extraction(extracted: dict[str, Any], directory: Path | None = None) -> Path:
    """Write extraction JSON to dev/output/extraction/{id}.json."""
    out_dir = directory or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_id = str(extracted["id"])
    payload = dict(extracted)
    pop_transform_stats(payload)
    path = out_dir / f"{plan_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def render_extraction_png(
    extracted: dict[str, Any],
    path: Path,
    *,
    rng: random.Random | None = None,
) -> Path:
    """Render transformed extraction JSON to a realistic PNG."""
    openings = extracted.get("openings") or {}
    doors = list(openings.get("doors") or [])
    windows = _opening_lines(openings.get("windows") or [])
    result = export_realistic(
        rooms=extracted["rooms"],
        doors=doors,
        windows=windows,
        symbols=list(extracted.get("symbols") or []),
        pillars=list(extracted.get("pillars") or []),
        wall_rects=list(extracted.get("wall_rects") or []),
        image_gray=bool(extracted.get("image_gray")),
        image_grain=bool(extracted.get("image_grain")),
        size=_CANVAS_SIZE,
        rng=rng,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(result["png_b64"]))
    return path


def extract_and_write(
    path: Path,
    *,
    output_directory: Path | None = None,
) -> dict[str, Path]:
    """Load a floorplan JSON path, transform, write JSON, and render PNG."""
    extracted = prepare_extraction(path)
    plan_id = str(extracted["id"])
    out_dir = output_directory or output_dir()
    json_path = write_extraction(extracted, out_dir)
    png_path = render_extraction_png(extracted, out_dir / f"{plan_id}.png")
    return {"json": json_path, "png": png_path}
