"""Room mask utilities for the FPN pipeline."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from util.render import draw_masks, floorplan_view, merge_panels, stack_rows, with_label

RoomLabel = dict[str, object]


def _room_binary(room_mask: np.ndarray) -> np.ndarray:
    if room_mask.ndim == 3:
        return room_mask[0] > 0.5
    return room_mask > 0.5


def scale_room_mask(
    room_mask: np.ndarray,
    target_size: int,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Upscale/downscale a room probability mask."""
    plane = room_mask[0].astype(np.float32) if room_mask.ndim == 3 else room_mask.astype(np.float32)
    height, width = plane.shape
    if height == target_size and width == target_size:
        return plane
    return cv2.resize(plane, (target_size, target_size), interpolation=interpolation)


def component_masks_from_binary(binary: np.ndarray) -> list[np.ndarray]:
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return [(labels == label) for label in range(1, count)]


def room_components_from_mask(room_mask: np.ndarray, *, min_area_px: int) -> list[np.ndarray]:
    components = component_masks_from_binary(_room_binary(room_mask).astype(np.uint8))
    return [component for component in components if int(component.sum()) >= min_area_px]


def _largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _component_contour_polygon(
    component: np.ndarray,
    *,
    approx_epsilon: float,
) -> list[tuple[int, int]] | None:
    contour = _largest_external_contour(component)
    if contour is None or len(contour) < 3:
        return None
    approx = cv2.approxPolyDP(contour, approx_epsilon, True)
    if len(approx) < 3:
        approx = contour
    return [(int(x), int(y)) for x, y in approx[:, 0]]


def _avg_in_box(gray: np.ndarray, x: int, y: int, *, radius: int) -> float:
    height, width = gray.shape
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    return float(gray[y0:y1, x0:x1].mean())


def _polygon_has_mostly_bright_vertices(
    polygon: list[tuple[int, int]],
    gray: np.ndarray,
    *,
    box_radius: int,
    bright_threshold: float,
    reject_fraction: float,
) -> bool:
    if len(polygon) < 3:
        return True
    bright_count = sum(
        1
        for x, y in polygon
        if _avg_in_box(gray, x, y, radius=box_radius) > bright_threshold
    )
    return bright_count / len(polygon) >= reject_fraction


def filter_bright_room_components(
    components: list[np.ndarray],
    image: Image.Image,
    *,
    approx_epsilon: float,
    box_radius: int,
    bright_threshold: float,
    reject_fraction: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Drop room components whose contour vertices mostly sit on bright pixels."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    kept: list[np.ndarray] = []
    rejected = np.zeros(gray.shape, dtype=bool)
    for component in components:
        polygon = _component_contour_polygon(component, approx_epsilon=approx_epsilon)
        if polygon is None:
            continue
        if _polygon_has_mostly_bright_vertices(
            polygon,
            gray,
            box_radius=box_radius,
            bright_threshold=bright_threshold,
            reject_fraction=reject_fraction,
        ):
            rejected |= component
            continue
        kept.append(component)
    return kept, rejected


def contour_polygons_from_components(
    components: list[np.ndarray],
    *,
    approx_epsilon: float,
) -> list[list[tuple[int, int]]]:
    """Fit one approxPolyDP polygon per connected mask component."""
    polygons: list[list[tuple[int, int]]] = []
    for component in components:
        polygon = _component_contour_polygon(component, approx_epsilon=approx_epsilon)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def room_polygons_from_components(
    components: list[np.ndarray],
    *,
    approx_epsilon: float,
) -> list[list[tuple[int, int]]]:
    return contour_polygons_from_components(components, approx_epsilon=approx_epsilon)


_OPENING_POLYGON_COLOR: dict[Literal["door", "window"], tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "door": ((80, 220, 80), (40, 180, 40)),
    "window": ((80, 160, 255), (40, 120, 220)),
}
_POLYGON_FILL_ALPHA = 150
_POLYGON_OUTLINE_ALPHA = 255
_BOUNDARY_OUTLINE = (220, 40, 40)


def _draw_unfilled_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: list[tuple[int, int]],
    *,
    outline: tuple[int, int, int],
    width: int = 2,
) -> None:
    if len(polygon) < 3:
        return
    draw.line([*polygon, polygon[0]], fill=(*outline, _POLYGON_OUTLINE_ALPHA), width=width)


def _room_preview_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.92)
    return int(red * 255), int(green * 255), int(blue * 255)


def annotate_room_labels(
    image: Image.Image,
    room_labels: list[RoomLabel] | None,
) -> Image.Image:
    """Draw LLM room name labels at their reported positions."""
    if not room_labels:
        return image
    out = image.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for entry in room_labels:
        name = str(entry.get("room") or "").strip()
        position = entry.get("position")
        if not name or not isinstance(position, dict):
            continue
        x = int(position.get("x", 0))
        y = int(position.get("y", 0))
        text = name
        bbox = draw.textbbox((x, y), text, anchor="mm")
        pad = 2
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(255, 255, 255, 210),
        )
        draw.text((x, y), text, fill=(20, 20, 20, 255), anchor="mm")
    return Image.alpha_composite(out, overlay).convert("RGB")


def render_combined_contour_overlay(
    input_image: Image.Image,
    *,
    room_polygons: list[list[tuple[int, int]]],
    door_polygons: list[list[tuple[int, int]]],
    window_polygons: list[list[tuple[int, int]]],
    boundary_polygon: list[tuple[int, int]] | None = None,
) -> Image.Image:
    base = input_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, polygon in enumerate(room_polygons):
        if len(polygon) < 3:
            continue
        rgb = _room_preview_color(index)
        draw.polygon(
            polygon,
            fill=(*rgb, _POLYGON_FILL_ALPHA),
            outline=(*rgb, _POLYGON_OUTLINE_ALPHA),
        )
    door_fill, door_outline = _OPENING_POLYGON_COLOR["door"]
    for polygon in door_polygons:
        if len(polygon) >= 3:
            draw.polygon(
                polygon,
                fill=(*door_fill, _POLYGON_FILL_ALPHA),
                outline=(*door_outline, _POLYGON_OUTLINE_ALPHA),
            )
    window_fill, window_outline = _OPENING_POLYGON_COLOR["window"]
    for polygon in window_polygons:
        if len(polygon) >= 3:
            draw.polygon(
                polygon,
                fill=(*window_fill, _POLYGON_FILL_ALPHA),
                outline=(*window_outline, _POLYGON_OUTLINE_ALPHA),
            )
    if boundary_polygon:
        _draw_unfilled_polygon(draw, boundary_polygon, outline=_BOUNDARY_OUTLINE, width=3)
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_mask_polygon_comparison(
    floorplan: Image.Image,
    room_components: list[np.ndarray],
    *,
    room_polygons: list[list[tuple[int, int]]],
    door_polygons: list[list[tuple[int, int]]],
    window_polygons: list[list[tuple[int, int]]],
    boundary_polygon: list[tuple[int, int]] | None,
    out_path: Path,
    size: int,
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    room_labels: list[RoomLabel] | None = None,
) -> Path:
    """Side-by-side: all pixel masks on input | all contour polygons on input."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_panel = compose_room_components_overlay(
        floorplan,
        room_components,
        size=size,
        door_mask=door_mask,
        window_mask=window_mask,
    )
    polygon_panel = render_combined_contour_overlay(
        floorplan_view(floorplan, size),
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        boundary_polygon=boundary_polygon,
    )
    if room_labels:
        mask_panel = annotate_room_labels(mask_panel, room_labels)
        polygon_panel = annotate_room_labels(polygon_panel, room_labels)
    merge_panels(
        [
            with_label(mask_panel, "Pixel mask"),
            with_label(polygon_panel, "Contour polygon"),
        ]
    ).save(out_path, format="JPEG", quality=95)
    return out_path


def compose_room_components_overlay(
    floorplan: Image.Image,
    components: list[np.ndarray],
    *,
    size: int,
    door_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
) -> Image.Image:
    """Per-room colored masks on the raw floor plan (background is not region-masked)."""
    view = floorplan_view(floorplan, size)
    overlay = draw_masks(view, components, alpha=120) if components else view.copy()
    if door_mask is not None and window_mask is not None:
        overlay = _overlay_openings(overlay, door_mask, window_mask)
    return overlay


def _overlay_openings(
    base: Image.Image,
    door_mask: np.ndarray,
    window_mask: np.ndarray,
) -> Image.Image:
    size = base.size[0]
    rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
    doors = scale_room_mask(door_mask, size) > 0.5
    windows = scale_room_mask(window_mask, size) > 0.5
    out = rgb.copy()
    out[doors, 0] = np.clip(out[doors, 0] * 0.25 + 255 * 0.75, 0, 255)
    out[doors, 1] *= 0.25
    out[doors, 2] *= 0.25
    out[windows, 1] = np.clip(out[windows, 1] * 0.25 + 255 * 0.75, 0, 255)
    out[windows, 0] *= 0.25
    out[windows, 2] *= 0.25
    return Image.fromarray(out.astype(np.uint8))


def save_all_room_mask_overlays(
    floorplan: Image.Image,
    sources: list[tuple[str, list[np.ndarray]]],
    out_dir: Path,
    *,
    size: int,
    door_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
    room_labels: list[RoomLabel] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    view = floorplan_view(floorplan, size)
    if room_labels:
        view = annotate_room_labels(view, room_labels)
    rows = [
        merge_panels(
            [
                view,
                with_label(
                    annotate_room_labels(
                        compose_room_components_overlay(
                            floorplan,
                            components,
                            size=size,
                            door_mask=door_mask,
                            window_mask=window_mask,
                        ),
                        room_labels,
                    ),
                    label,
                ),
                with_label(
                    draw_masks(
                        Image.new("RGB", (size, size), (255, 255, 255)),
                        components,
                        alpha=255,
                    )
                    if components
                    else Image.new("RGB", (size, size), (255, 255, 255)),
                    label,
                ),
            ]
        )
        for label, components in sources
    ]
    path = out_dir / "room-mask-overlays.jpg"
    stack_rows(rows).save(path, format="JPEG", quality=95)
    return path


def write_room_counts_markdown(
    counts: list[tuple[str, int]],
    path: Path,
    *,
    component_size: int,
    min_area_px: int,
) -> Path:
    lines = [
        f"# Room counts ({component_size}px)",
        "",
        f"Minimum component area: {min_area_px} px",
        "",
        "| Source | Rooms |",
        "| --- | ---: |",
    ]
    for source, room_count in counts:
        lines.append(f"| {source} | {room_count} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
