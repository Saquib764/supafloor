"""Draw parsed room polygons onto a floor-plan image."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

_FILL_COLORS = (
    (232, 213, 183, 180),
    (212, 230, 195, 180),
    (201, 214, 223, 180),
    (184, 197, 214, 180),
    (240, 237, 230, 180),
    (221, 213, 200, 180),
    (168, 197, 184, 180),
    (229, 229, 229, 180),
)


def draw_rooms(image: Image.Image, rooms: list[dict[str, Any]]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, room in enumerate(rooms):
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        points = [(int(x), int(y)) for x, y in polygon]
        color = _FILL_COLORS[index % len(_FILL_COLORS)]
        draw.polygon(points, fill=color, outline=(44, 44, 44, 255))
        center = room.get("center")
        name = str(room.get("name") or "").strip()
        if center and name:
            draw.text((int(center[0]), int(center[1])), name, fill=(26, 26, 26))
    return overlay


def draw_points(
    image: Image.Image,
    points: list[list[int]],
    *,
    radius: int = 3,
    color: tuple[int, int, int] = (220, 50, 50),
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    r = max(1, radius)
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(44, 44, 44))
    return overlay
