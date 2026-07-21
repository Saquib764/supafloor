"""SVG / PNG floor plan renderer with independent layers."""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from floorplan.geometry import area, bounding_box, centroid
from floorplan.schemas.models import FloorPlan
from floorplan.topology import build_adjacency

ROOM_COLORS = {
    "living": "#E8D5B7",
    "dining": "#E8D5B7",
    "kitchen": "#C9D6DF",
    "bedroom": "#D4E6C3",
    "master_bedroom": "#B8D4A8",
    "bathroom": "#B8C5D6",
    "powder": "#B8C5D6",
    "hallway": "#F0EDE6",
    "entry": "#F0EDE6",
    "closet": "#DDD5C8",
    "storage": "#DDD5C8",
    "utility": "#D0CBBF",
    "laundry": "#D0CBBF",
    "balcony": "#A8C5B8",
    "study": "#D4E6C3",
    "other": "#E5E5E5",
}


def _bounds(plan: FloorPlan, pad: float = 0.5) -> tuple[float, float, float, float]:
    boundary = plan.ensure_boundary()
    minx, miny, maxx, maxy = bounding_box(boundary.coords())
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def _transform(x: float, y: float, bounds: tuple[float, float, float, float], size: int, flip_y: bool = True):
    minx, miny, maxx, maxy = bounds
    w = max(maxx - minx, 1e-9)
    h = max(maxy - miny, 1e-9)
    scale = (size * 0.92) / max(w, h)
    ox = (size - w * scale) / 2
    oy = (size - h * scale) / 2
    px = ox + (x - minx) * scale
    py = oy + (y - miny) * scale
    if flip_y:
        py = size - py
    return px, py


def _poly_svg(coords, bounds, size, fill, stroke="#2C2C2C", stroke_width=1.5, opacity=0.85) -> str:
    pts = " ".join(f"{_transform(x, y, bounds, size)[0]:.1f},{_transform(x, y, bounds, size)[1]:.1f}" for x, y in coords)
    return f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def render_svg(
    plan: FloorPlan,
    *,
    size: int = 800,
    layers: Optional[list[str]] = None,
    scores_by_room: Optional[dict[str, float]] = None,
    highlight_room_id: Optional[str] = None,
) -> str:
    """Render floor plan as SVG string with selectable layers."""
    layers = layers or ["boundary", "rooms", "labels", "doors", "windows", "room_ids"]
    bounds = _bounds(plan)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#FAF8F5"/>',
    ]

    if "boundary" in layers:
        parts.append('<g id="layer-boundary">')
        parts.append(
            _poly_svg(
                plan.ensure_boundary().coords(),
                bounds,
                size,
                fill="none",
                stroke="#1A1A1A",
                stroke_width=3,
                opacity=1,
            )
        )
        parts.append("</g>")

    if "rooms" in layers or "rule_heatmap" in layers:
        parts.append('<g id="layer-rooms">')
        for room in plan.rooms:
            if "rule_heatmap" in layers and scores_by_room and room.id in scores_by_room:
                score = scores_by_room[room.id]
                # 0 red -> 10 green
                t = max(0.0, min(1.0, score / 10.0))
                r = int(200 * (1 - t))
                g = int(180 * t)
                fill = f"rgb({r},{g},90)"
            else:
                fill = ROOM_COLORS.get(room.room_type.value, "#E5E5E5")
            stroke = "#C45C26" if room.id == highlight_room_id else "#2C2C2C"
            sw = 3 if room.id == highlight_room_id else 1.2
            parts.append(_poly_svg(room.coords(), bounds, size, fill=fill, stroke=stroke, stroke_width=sw))
        parts.append("</g>")

    if "labels" in layers or "room_ids" in layers:
        parts.append('<g id="layer-labels">')
        for room in plan.rooms:
            cx, cy = centroid(room.coords())
            px, py = _transform(cx, cy, bounds, size)
            label = room.name if "labels" in layers else room.id[:6]
            if "labels" in layers and "room_ids" in layers:
                label = f"{room.name}"
            parts.append(
                f'<text x="{px:.1f}" y="{py:.1f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-family="Georgia, serif" font-size="11" fill="#1A1A1A">{label}</text>'
            )
            if "labels" in layers:
                a = area(room.coords())
                parts.append(
                    f'<text x="{px:.1f}" y="{py + 14:.1f}" text-anchor="middle" '
                    f'font-family="Georgia, serif" font-size="9" fill="#555">{a:.1f} m²</text>'
                )
        parts.append("</g>")

    if "doors" in layers:
        parts.append('<g id="layer-doors">')
        for door in plan.doors:
            x0, y0 = _transform(door.start.x, door.start.y, bounds, size)
            x1, y1 = _transform(door.end.x, door.end.y, bounds, size)
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                f'stroke="#8B4513" stroke-width="4" stroke-linecap="round"/>'
            )
        parts.append("</g>")

    if "windows" in layers:
        parts.append('<g id="layer-windows">')
        for win in plan.windows:
            x0, y0 = _transform(win.start.x, win.start.y, bounds, size)
            x1, y1 = _transform(win.end.x, win.end.y, bounds, size)
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                f'stroke="#6BA3C9" stroke-width="5" stroke-linecap="round"/>'
            )
        parts.append("</g>")

    if "dimensions" in layers:
        parts.append('<g id="layer-dimensions">')
        minx, miny, maxx, maxy = bounding_box(plan.ensure_boundary().coords())
        # Width dimension along bottom
        x0, y0 = _transform(minx, miny - 0.3, bounds, size)
        x1, y1 = _transform(maxx, miny - 0.3, bounds, size)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#666" stroke-width="1"/>')
        parts.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 + 14:.1f}" text-anchor="middle" '
            f'font-family="Georgia, serif" font-size="10" fill="#444">{maxx - minx:.2f} m</text>'
        )
        parts.append("</g>")

    if "topology" in layers:
        parts.append('<g id="layer-topology">')
        adj = build_adjacency(plan)
        centers = {r.id: centroid(r.coords()) for r in plan.rooms}
        drawn = set()
        for a, neighbors in adj.items():
            for b in neighbors:
                key = tuple(sorted((a, b)))
                if key in drawn:
                    continue
                drawn.add(key)
                if a not in centers or b not in centers:
                    continue
                ax, ay = _transform(*centers[a], bounds, size)
                bx, by = _transform(*centers[b], bounds, size)
                parts.append(
                    f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                    f'stroke="#C45C26" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>'
                )
        parts.append("</g>")

    if "circulation" in layers:
        parts.append('<g id="layer-circulation">')
        for room in plan.rooms:
            if room.room_type.value in ("hallway", "entry"):
                parts.append(_poly_svg(room.coords(), bounds, size, fill="#F5E6D3", stroke="#C45C26", stroke_width=2, opacity=0.5))
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def render_png(plan: FloorPlan, *, size: int = 800, layers: Optional[list[str]] = None) -> bytes:
    """Rasterize a simple PNG via Pillow (independent of SVG for reliability)."""
    layers = layers or ["boundary", "rooms", "labels", "doors", "windows"]
    bounds = _bounds(plan)
    img = Image.new("RGB", (size, size), "#FAF8F5")
    draw = ImageDraw.Draw(img, "RGBA")

    if "rooms" in layers:
        for room in plan.rooms:
            color = ROOM_COLORS.get(room.room_type.value, "#E5E5E5")
            rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            pts = [_transform(x, y, bounds, size) for x, y in room.coords()]
            draw.polygon(pts, fill=rgb + (220,), outline=(44, 44, 44, 255))

    if "boundary" in layers:
        pts = [_transform(x, y, bounds, size) for x, y in plan.ensure_boundary().coords()]
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        draw.line(pts, fill=(26, 26, 26), width=3)

    if "doors" in layers:
        for door in plan.doors:
            p0 = _transform(door.start.x, door.start.y, bounds, size)
            p1 = _transform(door.end.x, door.end.y, bounds, size)
            draw.line([p0, p1], fill=(139, 69, 19), width=4)

    if "windows" in layers:
        for win in plan.windows:
            p0 = _transform(win.start.x, win.start.y, bounds, size)
            p1 = _transform(win.end.x, win.end.y, bounds, size)
            draw.line([p0, p1], fill=(107, 163, 201), width=5)

    if "labels" in layers:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for room in plan.rooms:
            cx, cy = centroid(room.coords())
            px, py = _transform(cx, cy, bounds, size)
            draw.text((px, py), room.name, fill=(26, 26, 26), font=font, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_plan(
    plan: FloorPlan,
    *,
    size: int = 800,
    layers: Optional[list[str]] = None,
    scores_by_room: Optional[dict[str, float]] = None,
    highlight_room_id: Optional[str] = None,
) -> dict[str, Any]:
    svg = render_svg(
        plan,
        size=size,
        layers=layers,
        scores_by_room=scores_by_room,
        highlight_room_id=highlight_room_id,
    )
    png = render_png(plan, size=size, layers=layers)
    return {
        "svg": svg,
        "png": base64.b64encode(png).decode("ascii"),
        "metadata": {
            "plan_id": plan.id,
            "name": plan.name,
            "room_count": len(plan.rooms),
            "layers": layers or ["boundary", "rooms", "labels", "doors", "windows", "room_ids"],
            "size": size,
        },
    }
