"""Draw parsed room polygons onto a floor-plan image."""

from __future__ import annotations

from pathlib import Path
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
        if room.get("accepted") is False:
            fill = (220, 50, 50, 180)
            outline = (140, 20, 20, 255)
        else:
            fill = _FILL_COLORS[index % len(_FILL_COLORS)]
            outline = (44, 44, 44, 255)
        draw.polygon(points, fill=fill, outline=outline)
        sample_point = room.get("sample_point")
        if sample_point:
            sx, sy = int(sample_point[0]), int(sample_point[1])
            r = 4
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(220, 50, 50), outline=(44, 44, 44))
        center = room.get("center")
        name = str(room.get("name") or "").strip()
        if center and name:
            draw.text((int(center[0]), int(center[1])), name, fill=(26, 26, 26))
    return overlay


def _room_bbox(
    polygon: list[list[float]],
    size: tuple[int, int],
    *,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    xs = [float(x) for x, _y in polygon]
    ys = [float(y) for _x, y in polygon]
    width, height = size
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(width, int(max(xs)) + padding + 1)
    y1 = min(height, int(max(ys)) + padding + 1)
    return x0, y0, x1, y1


def _offset_room(room: dict[str, Any], dx: int, dy: int) -> dict[str, Any]:
    polygon = [(float(x) - dx, float(y) - dy) for x, y in room.get("polygon") or []]
    center = room.get("center")
    if center:
        center = [float(center[0]) - dx, float(center[1]) - dy]
    sample_point = room.get("sample_point")
    if sample_point:
        sample_point = [float(sample_point[0]) - dx, float(sample_point[1]) - dy]
    return {
        "name": room.get("name"),
        "polygon": polygon,
        "center": center,
        "sample_point": sample_point,
    }


def draw_room_panel(image: Image.Image, room: dict[str, Any], *, padding: int = 8) -> Image.Image | None:
    polygon = room.get("polygon") or []
    if len(polygon) < 3:
        return None
    bbox = _room_bbox(polygon, image.size, padding=padding)
    panel = image.crop(bbox)
    return draw_rooms(panel, [_offset_room(room, bbox[0], bbox[1])])


def draw_rooms_strip(image: Image.Image, rooms: list[dict[str, Any]], *, gap: int = 4, padding: int = 8) -> Image.Image:
    panels: list[Image.Image] = []
    for index, room in enumerate(rooms):
        panel = draw_room_panel(image, room, padding=padding)
        if panel is not None:
            panels.append(panel)
    if not panels:
        return Image.new("RGB", (1, 1), "white")

    max_h = max(panel.height for panel in panels)
    total_w = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for panel in panels:
        strip.paste(panel, (x, (max_h - panel.height) // 2))
        x += panel.width + gap
    return strip


def rooms_overlay_path(json_path: Path) -> Path:
    return json_path.parent / "output" / "rooms-overlay.png"


def rooms_strip_path(json_path: Path) -> Path:
    return json_path.parent / "output" / "rooms-strip.png"


def save_rooms_overlay(image: Image.Image, rooms: list[dict[str, Any]], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_rooms(image, rooms).save(out)
    return out


def save_rooms_strip(image: Image.Image, rooms: list[dict[str, Any]], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_rooms_strip(image, rooms).save(out)
    return out


def save_rooms_renderings(image: Image.Image, rooms: list[dict[str, Any]], json_path: Path) -> tuple[Path, Path]:
    overlay = save_rooms_overlay(image, rooms, rooms_overlay_path(json_path))
    strip = save_rooms_strip(image, rooms, rooms_strip_path(json_path))
    return overlay, strip


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
