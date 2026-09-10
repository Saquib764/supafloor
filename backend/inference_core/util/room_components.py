"""Room mask utilities for the FPN pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from util.render import draw_masks, floorplan_view, merge_panels, stack_rows, with_label


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
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    view = floorplan_view(floorplan, size)
    rows = [
        merge_panels(
            [
                view,
                with_label(
                    _overlay_openings(
                        draw_masks(view, components, alpha=120) if components else view.copy(),
                        door_mask,
                        window_mask,
                    )
                    if door_mask is not None and window_mask is not None
                    else (draw_masks(view, components, alpha=120) if components else view.copy()),
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
