"""Multi-resolution fpn_five structure mask inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import Config
from util.canvas import fit_to_canvas_rgb
from util.floorplan_region import (
    apply_region_mask,
    compute_floorplan_region,
    save_floorplan_region,
    save_masked_canvas,
)
from util.fpn_five_runtime import FpnFiveRuntime, MaskChannels, prepare_bw
from util.mask_clean import clean_binary_mask
from util.render import merge_panels, stack_rows
from util.room_components import (
    room_components_from_mask,
    save_all_room_mask_overlays,
    scale_room_mask,
    write_room_counts_markdown,
)
from util.structure_polygons import opening_centerlines_from_masks
from util.wall_centerlines import save_wall_centerline_outputs, wall_skeletons_from_masks
from util.wall_room_masks import render_wall_room_masks


def _mask_channel_view(
    masks: np.ndarray,
    channel: int,
    *,
    background_channel: int,
) -> Image.Image:
    if channel == background_channel:
        gray = np.full(masks.shape[-2:], 255, dtype=np.uint8)
        gray[masks[channel] <= 0.5] = 0
        return Image.fromarray(gray, mode="L").convert("RGB")
    active = (masks[channel] > 0.5).astype(np.uint8) * 255
    return Image.fromarray(active, mode="L").convert("RGB")


def _structure_overlay(
    prepared: Image.Image,
    masks: np.ndarray,
    channels: MaskChannels,
) -> Image.Image:
    """Door on red channel, window on green channel, composited on wall input."""
    gray = np.asarray(prepared.convert("L"), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    doors = masks[channels.doors] > 0.5
    windows = masks[channels.windows] > 0.5
    rgb[doors, 0] = 255
    rgb[doors, 1] = (rgb[doors, 1].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[doors, 2] = (rgb[doors, 2].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[windows, 1] = 255
    rgb[windows, 0] = (rgb[windows, 0].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[windows, 2] = (rgb[windows, 2].astype(np.float32) * 0.25).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _five_mask_row(
    prepared: Image.Image,
    masks: np.ndarray,
    channels: MaskChannels,
) -> Image.Image:
    panels = [prepared.convert("RGB")]
    for channel in (
        channels.background,
        channels.rooms,
        channels.walls,
        channels.doors,
        channels.windows,
    ):
        panels.append(
            _mask_channel_view(
                masks,
                channel,
                background_channel=channels.background,
            )
        )
    panels.append(_structure_overlay(prepared, masks, channels))
    return merge_panels(panels)


def _scaled_mask_clean(size: int, cfg: Config) -> tuple[int, int, int]:
    scale = size / 768
    clean = cfg.fpn.mask_clean
    open_px = max(0, round(clean.open_px * scale))
    close_px = max(0, round(clean.close_px * scale))
    min_area_px = max(0, round(clean.min_area_px * scale * scale))
    return open_px, close_px, min_area_px


def _clean_mask(mask: np.ndarray, *, size: int, cfg: Config) -> np.ndarray:
    open_px, close_px, min_area_px = _scaled_mask_clean(size, cfg)
    if mask.ndim == 2:
        cleaned = clean_binary_mask(
            mask > 0.5,
            open_px=open_px,
            close_px=close_px,
            min_area_px=min_area_px,
        )
        return cleaned.astype(np.float32)
    cleaned = np.zeros_like(mask, dtype=np.float32)
    for channel in range(mask.shape[0]):
        cleaned[channel] = clean_binary_mask(
            mask[channel] > 0.5,
            open_px=open_px,
            close_px=close_px,
            min_area_px=min_area_px,
        ).astype(np.float32)
    return cleaned


def _save_step(
    out_dir: Path,
    step: str,
    *,
    mask_rows: list[Image.Image],
) -> Path:
    step_dir = out_dir / step
    step_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("room_masks.jpg", "structure_masks.jpg"):
        stale_path = step_dir / stale
        if stale_path.is_file():
            stale_path.unlink()
    path = step_dir / "masks.jpg"
    stack_rows(mask_rows).save(path, format="JPEG", quality=95)
    return path


def _scaled_floorplan_region_params(size: int, cfg: Config) -> tuple[int, float]:
    scale = size / 768
    region = cfg.fpn.floorplan_region
    close_px = max(0, round(region.close_px * scale))
    approx_epsilon = max(0.5, region.approx_epsilon * scale)
    return close_px, approx_epsilon


def run_fpn_pipeline(
    cfg: Config,
    image_path: Path,
    *,
    runtime: FpnFiveRuntime | None = None,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    full_canvas = fit_to_canvas_rgb(Image.open(image_path), size=cfg.canvas_size)
    runtime = runtime or FpnFiveRuntime.load(
        checkpoint=checkpoint or cfg.fpn.checkpoint_latest,
    )
    channels = runtime.channels

    out_dir = (output_dir or cfg.fpn.output_dir) / image_path.stem
    step_zero_dir = out_dir / "step-000"
    region_size = cfg.fpn.floorplan_region.size
    room_settings = cfg.fpn.room_components
    component_size = room_settings.component_size

    region_prepared = prepare_bw(full_canvas, region_size)
    region_masks = runtime.infer(region_prepared)
    close_px, approx_epsilon = _scaled_floorplan_region_params(region_size, cfg)
    floorplan_region = compute_floorplan_region(
        region_masks[channels.background],
        close_px=close_px,
        approx_epsilon=approx_epsilon,
    )
    region_output = save_floorplan_region(
        floorplan_region,
        out_dir,
        size=region_size,
    )

    image = apply_region_mask(full_canvas, floorplan_region.mask)
    masked_canvas_path = save_masked_canvas(image, out_dir)

    raw_mask_rows: list[Image.Image] = []
    cleaned_mask_rows: list[Image.Image] = []
    room_masks_by_size: dict[int, np.ndarray] = {}
    wall_masks_by_size: dict[int, np.ndarray] = {}
    door_masks_by_size: dict[int, np.ndarray] = {}
    window_masks_by_size: dict[int, np.ndarray] = {}

    for size in cfg.fpn.sizes:
        prepared = prepare_bw(image, size)
        masks = runtime.infer(prepared)
        room_masks_by_size[size] = masks[channels.rooms]
        wall_masks_by_size[size] = masks[channels.walls]
        door_masks_by_size[size] = masks[channels.doors]
        window_masks_by_size[size] = masks[channels.windows]
        raw_mask_rows.append(_five_mask_row(prepared, masks, channels))

        cleaned = _clean_mask(masks, size=size, cfg=cfg)
        cleaned_mask_rows.append(_five_mask_row(prepared, cleaned, channels))

    outputs: dict[str, object] = {
        "floorplan_region": {**region_output, "masked_canvas": masked_canvas_path},
        "step-000": _save_step(out_dir, "step-000", mask_rows=raw_mask_rows),
        "step-001": _save_step(out_dir, "step-001", mask_rows=cleaned_mask_rows),
    }

    min_area_px = max(
        1,
        round(room_settings.min_area_px * (component_size / 768) ** 2),
    )
    roi_mask = floorplan_region.mask.astype(np.float32)
    if roi_mask.shape != (component_size, component_size):
        roi_mask = scale_room_mask(
            roi_mask,
            component_size,
            interpolation=cv2.INTER_NEAREST,
        )

    opening_size = 512
    if opening_size not in room_masks_by_size:
        fpn512_masks = runtime.infer(prepare_bw(image, opening_size))
        fpn512_room = fpn512_masks[channels.rooms]
        fpn512_door = fpn512_masks[channels.doors]
        fpn512_window = fpn512_masks[channels.windows]
        wall_masks_by_size[opening_size] = fpn512_masks[channels.walls]
    else:
        fpn512_room = room_masks_by_size[opening_size]
        fpn512_door = door_masks_by_size[opening_size]
        fpn512_window = window_masks_by_size[opening_size]

    wall_or_sizes = (512, 768, 1024)
    wall_masks_for_subtract: list[np.ndarray] = []
    for wall_size in wall_or_sizes:
        if wall_size not in wall_masks_by_size:
            wall_masks_by_size[wall_size] = runtime.infer(
                prepare_bw(image, wall_size)
            )[channels.walls]
        wall_masks_for_subtract.append(wall_masks_by_size[wall_size])

    wall_output = render_wall_room_masks(
        wall_masks_for_subtract,
        fpn512_door,
        fpn512_window,
        fpn512_room,
        region_mask=roi_mask,
        size=component_size,
        step_dir=step_zero_dir,
    )
    wall_components = [
        component
        for component in wall_output["wall_subtracted_components"]
        if int(component.sum()) >= min_area_px
    ]

    room_sources: list[tuple[str, list[np.ndarray]]] = [
        (f"Wall-subtract ({component_size}px)", wall_components),
    ]
    for size in cfg.fpn.sizes:
        scaled = scale_room_mask(room_masks_by_size[size], component_size)
        room_sources.append(
            (
                f"FPN Five {size}px → {component_size}px",
                room_components_from_mask(scaled, min_area_px=min_area_px),
            )
        )
    rooms_md_path = write_room_counts_markdown(
        [(label, len(components)) for label, components in room_sources],
        out_dir / "rooms.md",
        component_size=component_size,
        min_area_px=min_area_px,
    )
    room_overlays_path = save_all_room_mask_overlays(
        image,
        room_sources,
        out_dir,
        size=component_size,
        door_mask=fpn512_door,
        window_mask=fpn512_window,
    )

    structure_dir = out_dir / "structure-polygons"

    opening_centerlines = opening_centerlines_from_masks(
        fpn512_door,
        fpn512_window,
        size=component_size,
        min_area_px=min_area_px,
        polyline_epsilon=3.0,
        polyline_min_length=6.0,
        polyline_max_spur_len=4,
    )

    wall_skeletons = wall_skeletons_from_masks(
        wall_masks_for_subtract,
        size=component_size,
    )
    wall_centerline_outputs = save_wall_centerline_outputs(
        image,
        wall_skeletons,
        structure_dir,
        size=component_size,
        opening_centerlines=opening_centerlines,
    )

    outputs["rooms"] = {
        "markdown": rooms_md_path,
        "overlays": room_overlays_path,
        "input": wall_output["input"],
    }
    outputs["structure_polygons"] = {
        "dir": structure_dir,
        **wall_centerline_outputs,
    }

    return outputs
