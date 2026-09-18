"""Discover room masks as connected white regions in a grayscale floorplan."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import Config
from util.canvas import load_canvas_image
from util.render import draw_mask, draw_masks


def _white_connected_masks(
    gray: np.ndarray,
    *,
    white_threshold: int,
    min_area_px: int,
    exclude_largest: bool,
) -> list[np.ndarray]:
    white = (gray >= white_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    components = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px
    ]
    if exclude_largest and len(components) > 1:
        largest_label = max(components, key=lambda item: item[0])[1]
        components = [item for item in components if item[1] != largest_label]

    masks: list[np.ndarray] = []
    for _, label in sorted(components, key=lambda item: item[0], reverse=True):
        masks.append((labels == label).astype(bool))
    return masks


def run_mask_discovery_loop(
    cfg: Config,
    image_path: Path | None = None,
    *,
    source_image: Image.Image | None = None,
    output_dir: Path | None = None,
) -> list[np.ndarray]:
    if source_image is None and image_path is None:
        raise ValueError("provide image_path or source_image")

    run_dir = output_dir or cfg.output
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if source_image is not None:
        image = source_image.convert("RGB")
    else:
        image = load_canvas_image(image_path, canvas_size=cfg.canvas_size)

    gray = np.array(image.convert("L"), dtype=np.uint8)
    gray_path = masks_dir / "gray.png"
    Image.fromarray(gray, mode="L").save(gray_path)
    print(f"wrote {gray_path}")

    room_masks = cfg.room_masks
    discovered = _white_connected_masks(
        gray,
        white_threshold=room_masks.white_threshold,
        min_area_px=room_masks.min_area_px,
        exclude_largest=room_masks.exclude_largest,
    )

    for index, mask in enumerate(discovered, start=1):
        step_path = masks_dir / f"step-{index:03d}.png"
        draw_mask(image, mask).save(step_path)
        area = int(mask.sum())
        print(f"wrote {step_path} (area={area})")

    if discovered:
        all_path = masks_dir / "all-masks.png"
        draw_masks(image, discovered).save(all_path)
        print(f"wrote {all_path} ({len(discovered)} mask(s))")
    else:
        print("no white connected regions found")

    print(f"done: {len(discovered)} mask(s)")
    return discovered
