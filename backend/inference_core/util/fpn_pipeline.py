"""Multi-resolution FPN room and structure mask inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from config import Config
from util.canvas import fit_to_canvas_rgb
from util.floorplan_region import (
    apply_region_mask,
    compute_floorplan_region,
    save_floorplan_region,
    save_masked_canvas,
)
from util.fpn_runtime import FpnRuntime, prepare_bw
from util.mask_clean import clean_binary_mask
from util.render import merge_panels, stack_rows
from util.room_components import (
    compute_area_masks,
    room_components_from_mask,
    save_all_room_mask_overlays,
    save_area_masks,
    scale_room_mask,
    write_room_counts_markdown,
)
from util.wall_room_masks import render_wall_room_masks


def _probability_to_rgb(plane: np.ndarray, *, channel: int) -> Image.Image:
    height, width = plane.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., channel] = (np.clip(plane.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _openings_to_rgb(masks: np.ndarray) -> Image.Image:
    """Door (green) + window (blue) channels combined."""
    height, width = masks.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    if masks.shape[0] > 1:
        rgb[..., 1] = (np.clip(masks[1].astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    if masks.shape[0] > 2:
        rgb[..., 2] = (np.clip(masks[2].astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _room_row(prepared: Image.Image, room_mask: np.ndarray, structure_mask: np.ndarray) -> Image.Image:
    input_view = prepared.convert("RGB")
    if room_mask.ndim == 3:
        room_mask = room_mask[0]
    room_output = _probability_to_rgb(room_mask, channel=1)
    wall_output = _probability_to_rgb(structure_mask[0], channel=0)
    return merge_panels([input_view, room_output, wall_output])


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


def _structure_row(prepared: Image.Image, structure_mask: np.ndarray) -> Image.Image:
    input_view = prepared.convert("RGB")
    wall = _probability_to_rgb(structure_mask[0], channel=0)
    openings = _openings_to_rgb(structure_mask)
    return merge_panels([input_view, wall, openings])


def _save_step(
    out_dir: Path,
    step: str,
    *,
    room_rows: list[Image.Image],
    structure_rows: list[Image.Image],
) -> tuple[Path, Path]:
    step_dir = out_dir / step
    step_dir.mkdir(parents=True, exist_ok=True)
    room_path = step_dir / "room_masks.jpg"
    structure_path = step_dir / "structure_masks.jpg"
    stack_rows(room_rows).save(room_path, format="JPEG", quality=95)
    stack_rows(structure_rows).save(structure_path, format="JPEG", quality=95)
    return room_path, structure_path


def _scaled_floorplan_region_params(size: int, cfg: Config) -> tuple[int, int, float]:
    scale = size / 768
    region = cfg.fpn.floorplan_region
    wall_dilate_px = max(0, round(region.wall_dilate_px * scale))
    close_px = max(0, round(region.close_px * scale))
    approx_epsilon = max(0.5, region.approx_epsilon * scale)
    return wall_dilate_px, close_px, approx_epsilon


def run_fpn_pipeline(
    cfg: Config,
    image_path: Path,
    *,
    runtime: FpnRuntime | None = None,
) -> dict[str, object]:
    full_canvas = fit_to_canvas_rgb(
        Image.open(image_path),
        size=cfg.canvas_size,
        margin=cfg.margin,
    )
    runtime = runtime or FpnRuntime.load(
        structure_checkpoint=cfg.fpn.structure_checkpoint,
        room_checkpoint=cfg.fpn.room_checkpoint,
    )

    out_dir = cfg.fpn.output_dir / image_path.stem
    step_zero_dir = out_dir / "step-000"
    region_size = cfg.fpn.floorplan_region.size
    room_settings = cfg.fpn.room_components
    infer_size = room_settings.infer_size
    component_size = room_settings.component_size

    region_prepared = prepare_bw(full_canvas, region_size)
    region_room_mask = runtime.infer_room(region_prepared)
    region_structure_mask = runtime.infer_structure(region_prepared)
    wall_dilate_px, close_px, approx_epsilon = _scaled_floorplan_region_params(region_size, cfg)
    floorplan_region = compute_floorplan_region(
        region_room_mask,
        region_structure_mask,
        wall_dilate_px=wall_dilate_px,
        close_px=close_px,
        approx_epsilon=approx_epsilon,
    )
    region_output = save_floorplan_region(
        region_prepared,
        floorplan_region,
        out_dir,
        size=region_size,
    )

    image = apply_region_mask(full_canvas, floorplan_region.mask)
    masked_canvas_path = save_masked_canvas(image, out_dir)

    original_room_rows: list[Image.Image] = []
    original_structure_rows: list[Image.Image] = []
    improved_room_rows: list[Image.Image] = []
    improved_structure_rows: list[Image.Image] = []
    infer_structure_mask: np.ndarray | None = None
    fpn_room_masks: dict[int, np.ndarray] = {}

    for size in cfg.fpn.sizes:
        prepared = prepare_bw(image, size)
        room_mask = runtime.infer_room(prepared)
        structure_mask = runtime.infer_structure(prepared)
        original_room_rows.append(_room_row(prepared, room_mask, structure_mask))
        original_structure_rows.append(_structure_row(prepared, structure_mask))
        fpn_room_masks[size] = room_mask

        if size == infer_size:
            infer_structure_mask = structure_mask

        cleaned_room = _clean_mask(room_mask, size=size, cfg=cfg)
        cleaned_structure = _clean_mask(structure_mask, size=size, cfg=cfg)
        improved_room_rows.append(_room_row(prepared, cleaned_room, cleaned_structure))
        improved_structure_rows.append(_structure_row(prepared, cleaned_structure))

    outputs: dict[str, object] = {
        "floorplan_region": {**region_output, "masked_canvas": masked_canvas_path},
        "step-000": _save_step(
            out_dir,
            "step-000",
            room_rows=original_room_rows,
            structure_rows=original_structure_rows,
        ),
        "step-001": _save_step(
            out_dir,
            "step-001",
            room_rows=improved_room_rows,
            structure_rows=improved_structure_rows,
        ),
    }

    if infer_structure_mask is None:
        infer_structure_mask = runtime.infer_structure(prepare_bw(image, infer_size))

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
    fpn_open_source = scale_room_mask(fpn_room_masks[infer_size], component_size)
    open_area, wall_area = compute_area_masks(roi_mask, fpn_open_source)
    area_mask_paths = save_area_masks(
        open_area,
        wall_area,
        out_dir,
        floorplan=image,
        size=component_size,
    )
    wall_output = render_wall_room_masks(
        infer_structure_mask,
        fpn_room_masks[512],
        floorplan=image,
        region_mask=roi_mask,
        size=component_size,
        out_dir=out_dir,
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
        scaled = scale_room_mask(fpn_room_masks[size], component_size)
        room_sources.append(
            (
                f"FPN {size}px → {component_size}px",
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
    )
    outputs["rooms"] = {
        "markdown": rooms_md_path,
        "overlays": room_overlays_path,
        "input": wall_output["input"],
        "wall_based_room_mask_final": wall_output["wall_based_room_mask_final"],
        "wall_based_room_mask_final_root": wall_output["wall_based_room_mask_final_root"],
    }
    outputs["area_masks"] = area_mask_paths

    return outputs
