"""Encode/decode Florence loc-token text."""

from __future__ import annotations

import re
from typing import Any, Iterator

_LOC_RE = re.compile(r"<loc_(\d+)>")
_POLY_OPEN = "<poly>"
_POLY_CLOSE = "</poly>"
_STRIP_RE = re.compile(r"</?s>|</?pad>", flags=re.IGNORECASE)
_LEADING_NAME_RE = re.compile(r"^\s*([A-Za-z][^<]*?)\s*(?=<loc_)")
_ROOM_NAME_RE = re.compile(r"Room name:\s*(.+?)\s*$", re.MULTILINE)


def pixel_to_loc(value: float, *, canvas_size: int = 1024, bins: int = 1000) -> int:
    return max(0, min(bins - 1, round(float(value) * (bins - 1) / canvas_size)))


def format_loc_point(
    x: float,
    y: float,
    *,
    canvas_size: int = 1024,
    bins: int = 1000,
) -> str:
    return (
        f"<loc_{pixel_to_loc(x, canvas_size=canvas_size, bins=bins)}>"
        f"<loc_{pixel_to_loc(y, canvas_size=canvas_size, bins=bins)}>"
    )


def _loc_to_pixel(loc: int, image_size: int, bins: int) -> int:
    loc = max(0, min(bins - 1, int(loc)))
    value = (loc + 0.5) * image_size / bins
    return int(round(max(0, min(image_size - 1, value))))


def parse_referring_room(
    text: str,
    *,
    image_size: int,
    bins: int = 1000,
) -> dict[str, Any] | None:
    text = _STRIP_RE.sub("", text)
    open_pos = text.find(_POLY_OPEN)
    close_pos = text.find(_POLY_CLOSE)
    if open_pos == -1 or close_pos == -1 or close_pos < open_pos:
        return None

    content = text[open_pos + len(_POLY_OPEN) : close_pos]
    locs = [int(value) for value in _LOC_RE.findall(content)]
    if len(locs) < 6 or len(locs) % 2 != 0:
        return None

    polygon = [
        [
            _loc_to_pixel(locs[i], image_size, bins),
            _loc_to_pixel(locs[i + 1], image_size, bins),
        ]
        for i in range(0, len(locs), 2)
    ]
    name_match = _ROOM_NAME_RE.search(text[close_pos + len(_POLY_CLOSE) :])
    name = name_match.group(1).strip() if name_match else "Room"
    cx = sum(point[0] for point in polygon) // len(polygon)
    cy = sum(point[1] for point in polygon) // len(polygon)
    return {"name": name, "center": [cx, cy], "polygon": polygon}


def _parse_name_poly_blocks(
    text: str,
    *,
    image_size: int,
    bins: int,
) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    cursor = 0
    for open_pos, close_end, content in _iter_closed_polys(text):
        prefix = text[cursor:open_pos].strip()
        name_lines = [
            line.strip()
            for line in prefix.split("\n")
            if line.strip() and not _LOC_RE.search(line)
        ]
        name = name_lines[-1] if name_lines else "Room"
        locs = [int(value) for value in _LOC_RE.findall(content)]
        if len(locs) < 6 or len(locs) % 2 != 0:
            cursor = close_end
            continue
        polygon = [
            [
                _loc_to_pixel(locs[i], image_size, bins),
                _loc_to_pixel(locs[i + 1], image_size, bins),
            ]
            for i in range(0, len(locs), 2)
        ]
        cx = sum(point[0] for point in polygon) // len(polygon)
        cy = sum(point[1] for point in polygon) // len(polygon)
        rooms.append({"name": name, "center": [cx, cy], "polygon": polygon})
        cursor = close_end
    return rooms


def _iter_closed_polys(text: str) -> Iterator[tuple[int, int, str]]:
    stack: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        open_pos = text.find(_POLY_OPEN, pos)
        close_pos = text.find(_POLY_CLOSE, pos)
        if close_pos == -1 and open_pos == -1:
            break
        if open_pos != -1 and (close_pos == -1 or open_pos < close_pos):
            stack.append((open_pos, open_pos + len(_POLY_OPEN)))
            pos = open_pos + len(_POLY_OPEN)
            continue
        if not stack:
            pos = close_pos + len(_POLY_CLOSE)
            continue
        open_pos, content_start = stack.pop()
        content_end = close_pos
        close_end = close_pos + len(_POLY_CLOSE)
        yield open_pos, close_end, text[content_start:content_end]
        pos = close_end


def _name_from_prefix(prefix: str, cx: int, cy: int) -> str:
    prefix = re.sub(rf"{re.escape(_POLY_CLOSE)}\s*", "", prefix.strip())
    center_tag = f"<loc_{cx}><loc_{cy}>"
    idx = prefix.rfind(center_tag)
    if idx == -1:
        return "Room"
    before = _LOC_RE.sub("", prefix[:idx]).strip()
    if not before:
        return "Room"
    lines = [line.strip() for line in before.split("\n") if line.strip()]
    return lines[-1] if lines else "Room"


def _parse_poly_block(
    prefix: str,
    content: str,
) -> tuple[str, tuple[int, int], list[int]] | None:
    locs = [int(value) for value in _LOC_RE.findall(content)]
    leading = _LEADING_NAME_RE.match(content)
    if leading and leading.group(1).strip():
        name = leading.group(1).strip()
        if len(locs) < 4:
            return None
        center = locs[0], locs[1]
        poly_locs = locs[2:]
    else:
        prefix_locs = [int(value) for value in _LOC_RE.findall(prefix)]
        if len(prefix_locs) < 2:
            return None
        center = prefix_locs[-2], prefix_locs[-1]
        name = _name_from_prefix(prefix, center[0], center[1])
        poly_locs = locs

    if len(poly_locs) < 2 or len(poly_locs) % 2 != 0:
        return None
    return name, center, poly_locs


def parse_rooms(
    text: str,
    *,
    image_size: int,
    bins: int = 1000,
) -> list[dict[str, Any]]:
    cleaned = _STRIP_RE.sub("", text)
    plot_rooms = _parse_name_poly_blocks(cleaned, image_size=image_size, bins=bins)
    if plot_rooms:
        return plot_rooms

    referring = parse_referring_room(cleaned, image_size=image_size, bins=bins)
    if referring is not None:
        return [referring]

    text = cleaned.replace("<sep>", "")
    rooms: list[dict[str, Any]] = []
    cursor = 0
    for open_pos, close_end, content in _iter_closed_polys(text):
        parsed = _parse_poly_block(text[cursor:open_pos], content)
        cursor = close_end
        if parsed is None:
            continue
        name, (cx, cy), poly_locs = parsed
        polygon = [
            [
                _loc_to_pixel(poly_locs[i], image_size, bins),
                _loc_to_pixel(poly_locs[i + 1], image_size, bins),
            ]
            for i in range(0, len(poly_locs), 2)
        ]
        rooms.append(
            {
                "name": name,
                "center": [
                    _loc_to_pixel(cx, image_size, bins),
                    _loc_to_pixel(cy, image_size, bins),
                ],
                "polygon": polygon,
            }
        )
    return rooms


def parse_locs(
    text: str,
    *,
    image_size: int,
    bins: int = 1000,
) -> list[list[int]]:
    text = _STRIP_RE.sub("", text)
    locs = [int(value) for value in _LOC_RE.findall(text)]
    return [
        [
            _loc_to_pixel(locs[i], image_size, bins),
            _loc_to_pixel(locs[i + 1], image_size, bins),
        ]
        for i in range(0, len(locs) - len(locs) % 2, 2)
    ]
