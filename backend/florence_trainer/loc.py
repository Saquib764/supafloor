"""Decode Florence loc-token text into room name + center + polygon."""

from __future__ import annotations

import re
from typing import Any

_ROOM_RE = re.compile(
    r"(.+?)"
    r"<loc_(\d+)>"
    r"<loc_(\d+)>"
    r"\s*<poly>"
    r"(.*?)"
    r"</poly>",
    flags=re.DOTALL,
)
_LOC_RE = re.compile(r"<loc_(\d+)>")
_STRIP_RE = re.compile(r"</?s>|</?pad>|<FLOORPLAN>", flags=re.IGNORECASE)


def _loc_to_pixel(loc: int, image_size: int, bins: int) -> int:
    loc = max(0, min(bins - 1, int(loc)))
    value = (loc + 0.5) * image_size / bins
    return int(round(max(0, min(image_size - 1, value))))


def parse_rooms(
    text: str,
    *,
    image_size: int,
    bins: int = 1000,
) -> list[dict[str, Any]]:
    text = _STRIP_RE.sub("", text)
    rooms: list[dict[str, Any]] = []
    for chunk in text.split("<sep>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _ROOM_RE.match(chunk)
        if not match:
            continue
        locs = [int(value) for value in _LOC_RE.findall(match.group(4))]
        if len(locs) % 2 != 0:
            continue
        polygon = [
            [
                _loc_to_pixel(locs[i], image_size, bins),
                _loc_to_pixel(locs[i + 1], image_size, bins),
            ]
            for i in range(0, len(locs), 2)
        ]
        rooms.append(
            {
                "name": match.group(1).strip(),
                "center": [
                    _loc_to_pixel(int(match.group(2)), image_size, bins),
                    _loc_to_pixel(int(match.group(3)), image_size, bins),
                ],
                "polygon": polygon,
            }
        )
    return rooms
