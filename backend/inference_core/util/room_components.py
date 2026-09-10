"""Room mask utilities for the FPN pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from util.render import draw_mask, draw_masks, floorplan_view, merge_panels, stack_rows, with_label


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


def compute_area_masks(
    roi_mask: np.ndarray,
    combined_rooms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Open area = ROI ∩ rooms; wall area = ROI − rooms."""
    roi = roi_mask > 0.5
    rooms = combined_rooms > 0.5 if combined_rooms.dtype != bool else combined_rooms
    open_area = roi & rooms
    wall_area = roi & ~rooms
    return open_area, wall_area


def _mask_to_rgb(mask: np.ndarray, *, channel: int) -> Image.Image:
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., channel] = (np.clip(mask.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_area_masks(
    open_area: np.ndarray,
    wall_area: np.ndarray,
    out_dir: Path,
    *,
    floorplan: Image.Image,
    size: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    open_path = out_dir / "open-area-mask.jpg"
    wall_path = out_dir / "wall-area-mask.jpg"
    overlay_path = out_dir / "open-area-overlay.jpg"
    _mask_to_rgb(open_area, channel=1).save(open_path, format="JPEG", quality=95)
    _mask_to_rgb(wall_area, channel=0).save(wall_path, format="JPEG", quality=95)

    draw_mask(floorplan_view(floorplan, size), open_area, alpha=120, color=(0, 180, 60)).save(
        overlay_path,
        format="JPEG",
        quality=95,
    )
    return {"open_area": open_path, "wall_area": wall_path, "open_area_overlay": overlay_path}


def save_all_room_mask_overlays(
    floorplan: Image.Image,
    sources: list[tuple[str, list[np.ndarray]]],
    out_dir: Path,
    *,
    size: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    view = floorplan_view(floorplan, size)
    rows = [
        merge_panels(
            [
                view,
                with_label(
                    draw_masks(view, components, alpha=120) if components else view.copy(),
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
