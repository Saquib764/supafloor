"""Multi-resolution fpn_five structure mask inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import Config
from util.canvas import fit_to_canvas_rgb
from util.floorplan_region import (
    apply_region_mask,
    compute_floorplan_region,
    crop_region_bbox,
    region_mask_for_image,
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
from util.final_mask_polygons import save_final_mask_polygons
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


def _resize_panel(
    image: Image.Image,
    panel_size: int,
    *,
    nearest: bool = False,
) -> Image.Image:
    if image.size == (panel_size, panel_size):
        return image
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize((panel_size, panel_size), resample)


def _five_mask_row(
    prepared: Image.Image,
    masks: np.ndarray,
    channels: MaskChannels,
    *,
    panel_size: int,
) -> Image.Image:
    panels = [_resize_panel(prepared.convert("RGB"), panel_size)]
    for channel in (
        channels.background,
        channels.rooms,
        channels.walls,
        channels.doors,
        channels.windows,
    ):
        panels.append(
            _resize_panel(
                _mask_channel_view(
                    masks,
                    channel,
                    background_channel=channels.background,
                ),
                panel_size,
                nearest=True,
            )
        )
    panels.append(
        _resize_panel(_structure_overlay(prepared, masks, channels), panel_size)
    )
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
    step_dir: Path,
    *,
    mask_rows: list[Image.Image],
    filename: str = "masks.jpg",
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("room_masks.jpg", "structure_masks.jpg"):
        stale_path = step_dir / stale
        if stale_path.is_file():
            stale_path.unlink()
    path = step_dir / filename
    stack_rows(mask_rows).save(path, format="JPEG", quality=95)
    return path


def _scaled_floorplan_region_params(size: int, cfg: Config) -> tuple[int, float]:
    scale = size / 768
    region = cfg.fpn.floorplan_region
    close_px = max(0, round(region.close_px * scale))
    approx_epsilon = max(0.5, region.approx_epsilon * scale)
    return close_px, approx_epsilon


@dataclass
class _SizeInference:
    room_masks_by_size: dict[int, np.ndarray]
    wall_masks_by_size: dict[int, np.ndarray]
    door_masks_by_size: dict[int, np.ndarray]
    window_masks_by_size: dict[int, np.ndarray]
    raw_mask_rows: list[Image.Image]
    cleaned_mask_rows: list[Image.Image]
    final_cleaned_masks: np.ndarray
    final_size: int


_FINAL_MASK_LAYERS: tuple[tuple[str, str], ...] = (
    ("background", "background"),
    ("room", "rooms"),
    ("wall", "walls"),
    ("door", "doors"),
    ("window", "windows"),
)


def _mask_plane_at_size(
    masks: np.ndarray,
    channel: int,
    *,
    target_size: int,
) -> np.ndarray:
    plane = masks[channel] > 0.5
    if plane.shape != (target_size, target_size):
        plane = scale_room_mask(
            plane.astype(np.float32),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
    return plane


def _save_final_masks(
    step_dir: Path,
    masks: np.ndarray,
    channels: MaskChannels,
    *,
    target_size: int,
    min_area_px: int,
) -> dict[str, Path | dict[str, Path]]:
    """Save cleaned five-class masks as PNGs and structure polygons as JSON."""
    out_dir = step_dir / "final-masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_by_layer = {
        "background": channels.background,
        "rooms": channels.rooms,
        "walls": channels.walls,
        "doors": channels.doors,
        "windows": channels.windows,
    }
    masks_by_layer: dict[str, np.ndarray] = {}
    paths: dict[str, Path | dict[str, Path]] = {}
    for filename, channel_key in _FINAL_MASK_LAYERS:
        plane = _mask_plane_at_size(masks, channel_by_layer[channel_key], target_size=target_size)
        path = out_dir / f"{filename}.png"
        Image.fromarray(plane.astype(np.uint8) * 255, mode="L").save(path)
        paths[filename] = path
        if filename != "background":
            masks_by_layer[filename] = plane

    paths["polygons"] = save_final_mask_polygons(
        out_dir,
        masks_by_layer,
        size=target_size,
        min_area_px=min_area_px,
    )
    return paths


def _infer_at_sizes(
    image: Image.Image,
    *,
    runtime: FpnFiveRuntime,
    channels: MaskChannels,
    sizes: tuple[int, ...],
    cfg: Config,
) -> _SizeInference:
    room_masks_by_size: dict[int, np.ndarray] = {}
    wall_masks_by_size: dict[int, np.ndarray] = {}
    door_masks_by_size: dict[int, np.ndarray] = {}
    window_masks_by_size: dict[int, np.ndarray] = {}
    raw_mask_rows: list[Image.Image] = []
    cleaned_mask_rows: list[Image.Image] = []
    panel_size = cfg.fpn.room_components.component_size
    final_size = max(sizes)
    final_cleaned_masks: np.ndarray | None = None

    for size in sizes:
        fitted = fit_to_canvas_rgb(image, size=size)
        prepared = prepare_bw(fitted, size)
        masks = runtime.infer(prepared)
        room_masks_by_size[size] = masks[channels.rooms]
        wall_masks_by_size[size] = masks[channels.walls]
        door_masks_by_size[size] = masks[channels.doors]
        window_masks_by_size[size] = masks[channels.windows]
        raw_mask_rows.append(_five_mask_row(prepared, masks, channels, panel_size=panel_size))

        cleaned = _clean_mask(masks, size=size, cfg=cfg)
        cleaned_mask_rows.append(
            _five_mask_row(prepared, cleaned, channels, panel_size=panel_size)
        )
        if size == final_size:
            final_cleaned_masks = cleaned

    if final_cleaned_masks is None:
        raise RuntimeError(f"Missing inference at final size {final_size}px")

    return _SizeInference(
        room_masks_by_size=room_masks_by_size,
        wall_masks_by_size=wall_masks_by_size,
        door_masks_by_size=door_masks_by_size,
        window_masks_by_size=window_masks_by_size,
        raw_mask_rows=raw_mask_rows,
        cleaned_mask_rows=cleaned_mask_rows,
        final_cleaned_masks=final_cleaned_masks,
        final_size=final_size,
    )


def _run_step_analysis(
    *,
    overlay_floorplan: Image.Image,
    inference: _SizeInference,
    runtime: FpnFiveRuntime,
    channels: MaskChannels,
    cfg: Config,
    step_dir: Path,
    region_mask: np.ndarray,
) -> dict[str, object]:
    room_settings = cfg.fpn.room_components
    component_size = room_settings.component_size
    min_area_px = max(
        1,
        round(room_settings.min_area_px * (component_size / 768) ** 2),
    )

    opening_size = 512
    room_masks_by_size = inference.room_masks_by_size
    wall_masks_by_size = inference.wall_masks_by_size
    door_masks_by_size = inference.door_masks_by_size
    window_masks_by_size = inference.window_masks_by_size

    if opening_size not in room_masks_by_size:
        fpn512_masks = runtime.infer(
            prepare_bw(fit_to_canvas_rgb(overlay_floorplan, size=opening_size), opening_size)
        )
        fpn512_room = fpn512_masks[channels.rooms]
        fpn512_door = fpn512_masks[channels.doors]
        fpn512_window = fpn512_masks[channels.windows]
        wall_masks_by_size[opening_size] = fpn512_masks[channels.walls]
    else:
        fpn512_room = room_masks_by_size[opening_size]
        fpn512_door = door_masks_by_size[opening_size]
        fpn512_window = window_masks_by_size[opening_size]

    wall_or_sizes = tuple(cfg.fpn.sizes)
    wall_masks_for_subtract: list[np.ndarray] = []
    for wall_size in wall_or_sizes:
        if wall_size not in wall_masks_by_size:
            wall_masks_by_size[wall_size] = runtime.infer(
                prepare_bw(fit_to_canvas_rgb(overlay_floorplan, size=wall_size), wall_size)
            )[channels.walls]
        wall_masks_for_subtract.append(wall_masks_by_size[wall_size])

    roi_mask = region_mask.astype(np.float32)
    if roi_mask.shape != (component_size, component_size):
        roi_mask = scale_room_mask(
            roi_mask,
            component_size,
            interpolation=cv2.INTER_NEAREST,
        )

    wall_output = render_wall_room_masks(
        wall_masks_for_subtract,
        fpn512_door,
        fpn512_window,
        fpn512_room,
        region_mask=roi_mask,
        size=component_size,
        step_dir=step_dir,
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
        step_dir / "rooms.md",
        component_size=component_size,
        min_area_px=min_area_px,
    )
    room_overlays_path = save_all_room_mask_overlays(
        overlay_floorplan,
        room_sources,
        step_dir,
        size=component_size,
        door_mask=fpn512_door,
        window_mask=fpn512_window,
    )

    structure_dir = step_dir / "structure-polygons"
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
        overlay_floorplan,
        wall_skeletons,
        structure_dir,
        size=component_size,
        opening_centerlines=opening_centerlines,
    )

    final_masks = _save_final_masks(
        step_dir,
        inference.final_cleaned_masks,
        channels,
        target_size=component_size,
        min_area_px=min_area_px,
    )

    return {
        "dir": step_dir,
        "masks": _save_step(step_dir, mask_rows=inference.raw_mask_rows),
        "masks_clean": _save_step(
            step_dir,
            mask_rows=inference.cleaned_mask_rows,
            filename="masks-clean.jpg",
        ),
        "final_masks": final_masks,
        "rooms": {
            "markdown": rooms_md_path,
            "overlays": room_overlays_path,
            "input": wall_output["input"],
        },
        "structure_polygons": {
            "dir": structure_dir,
            **wall_centerline_outputs,
        },
    }


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
    step_one_dir = out_dir / "step-001"
    step_two_dir = out_dir / "step-002"
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
        step_one_dir,
        size=region_size,
    )

    region_mask_canvas = region_mask_for_image(floorplan_region.mask, full_canvas.size)
    masked_canvas = apply_region_mask(full_canvas, region_mask_canvas)
    masked_canvas_path = save_masked_canvas(masked_canvas, step_one_dir)

    step_one_canvas = fit_to_canvas_rgb(full_canvas, size=component_size)
    step_one_inference = _infer_at_sizes(
        masked_canvas,
        runtime=runtime,
        channels=channels,
        sizes=tuple(cfg.fpn.sizes),
        cfg=cfg,
    )
    step_one_outputs = _run_step_analysis(
        overlay_floorplan=step_one_canvas,
        inference=step_one_inference,
        runtime=runtime,
        channels=channels,
        cfg=cfg,
        step_dir=step_one_dir,
        region_mask=floorplan_region.mask.astype(np.float32),
    )

    cropped = crop_region_bbox(masked_canvas, region_mask_canvas, padding_px=100)
    step_two_dir.mkdir(parents=True, exist_ok=True)
    cropped_path = step_two_dir / "input-cropped.jpg"
    cropped.save(cropped_path, format="JPEG", quality=95)

    cropped_canvas = fit_to_canvas_rgb(cropped, size=component_size)
    cropped_canvas_path = save_masked_canvas(cropped_canvas, step_two_dir)

    cropped_inference = _infer_at_sizes(
        cropped,
        runtime=runtime,
        channels=channels,
        sizes=tuple(cfg.fpn.sizes),
        cfg=cfg,
    )
    full_roi_mask = np.ones((component_size, component_size), dtype=np.float32)
    step_two_outputs = _run_step_analysis(
        overlay_floorplan=cropped_canvas,
        inference=cropped_inference,
        runtime=runtime,
        channels=channels,
        cfg=cfg,
        step_dir=step_two_dir,
        region_mask=full_roi_mask,
    )

    outputs: dict[str, object] = {
        "step-001": {
            **region_output,
            **step_one_outputs,
            "input_masked": masked_canvas_path,
        },
        "step-002": {
            **step_two_outputs,
            "input_cropped": cropped_path,
            "input_masked": cropped_canvas_path,
        },
    }
    return outputs
