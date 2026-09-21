"""FPN Five structure mask inference at 768px."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from config import Config
from util.canvas import fit_to_canvas_rgb
from util.floorplan_region import (
    apply_region_mask,
    compute_floorplan_region,
    crop_region_bbox,
    region_mask_for_image,
)
from util.fpn_five_runtime import FpnFiveRuntime, MaskChannels, prepare_bw
from util.mask_clean import clean_binary_mask, close_binary_mask
from util.room_components import (
    RoomLabel,
    contour_polygons_from_components,
    filter_bright_room_components,
    room_components_from_mask,
    save_all_room_mask_overlays,
    save_mask_polygon_comparison,
    write_room_counts_markdown,
)
from util.room_names_llm import detect_room_names
from util.final_mask_polygons import (
    load_layer_polygons,
    polygons_to_mask,
    save_final_mask_contour_polygons,
)
from util.render import merge_panels
from util.room_polygon_llm import (
    refine_room_polygons,
    render_room_polygons,
    save_final_mask_polygon_preview,
    save_room_polygon_outputs,
    scale_image,
)
from util.wall_room_masks import render_wall_room_masks

StepLogFn = Callable[[str], None]


def _format_timestamp(when: datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _step_run(step_name: str):
    debug: list[str] = []
    started = datetime.now()
    print(f"{step_name} started → {_format_timestamp(started)}")

    def log(message: str) -> None:
        debug.append(message)

    try:
        yield log
    finally:
        ended = datetime.now()
        elapsed = (ended - started).total_seconds()
        print(f"{step_name} finished → {_format_timestamp(ended)} ({elapsed:.1f}s)")
        print(f"{step_name} debug log:")
        if debug:
            for line in debug:
                print(f"  {line}")
        else:
            print("  (empty)")


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


def _clean_masks(masks: np.ndarray, cfg: Config) -> np.ndarray:
    clean = cfg.fpn.mask_clean
    cleaned = np.zeros_like(masks, dtype=np.float32)
    for channel in range(masks.shape[0]):
        cleaned[channel] = clean_binary_mask(
            masks[channel] > 0.5,
            open_px=clean.open_px,
            close_px=clean.close_px,
            min_area_px=clean.min_area_px,
        ).astype(np.float32)
    return cleaned


def _save_final_masks(
    step_dir: Path,
    *,
    size: int,
    min_area_px: int,
    room_polygons: list[list[tuple[int, int]]],
    door_polygons: list[list[tuple[int, int]]],
    window_polygons: list[list[tuple[int, int]]],
    boundary_polygon: list[tuple[int, int]],
    wall_approx_epsilon: float,
    input_image: Image.Image | None = None,
) -> dict[str, Path | dict[str, Path]]:
    out_dir = step_dir / "final-masks"
    polygon_paths = save_final_mask_contour_polygons(
        out_dir,
        size=size,
        min_area_px=min_area_px,
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        boundary_polygon=boundary_polygon,
        wall_approx_epsilon=wall_approx_epsilon,
    )
    paths: dict[str, Path | dict[str, Path]] = {
        "background": out_dir / "background.png",
        "room": out_dir / "room.png",
        "wall": out_dir / "wall.png",
        "door": out_dir / "door.png",
        "window": out_dir / "window.png",
        "polygons": polygon_paths,
    }
    if input_image is not None:
        _, overlay_path = save_final_mask_polygon_preview(
            input_image,
            room_polygons=load_layer_polygons(polygon_paths["room"]),
            wall_polygons=load_layer_polygons(polygon_paths["wall"]),
            door_polygons=load_layer_polygons(polygon_paths["door"]),
            window_polygons=load_layer_polygons(polygon_paths["window"]),
            out_dir=out_dir,
            save_input=False,
        )
        paths["polygon_overlay"] = overlay_path
    return paths


def _infer_on_canvas(
    canvas: Image.Image,
    *,
    size: int,
    runtime: FpnFiveRuntime,
    cfg: Config,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    prepared = prepare_bw(canvas, size)
    masks = runtime.infer(prepared)
    return prepared, masks, _clean_masks(masks, cfg)


def _write_step_artifacts(
    step_dir: Path,
    *,
    prepared: Image.Image,
    masks: np.ndarray,
    cleaned: np.ndarray,
    overlay: Image.Image,
    region_mask: np.ndarray,
    channels: MaskChannels,
    cfg: Config,
    size: int,
    room_labels: list[RoomLabel] | None = None,
) -> dict[str, object]:
    room_cfg = cfg.fpn.room_components
    min_area_px = max(1, room_cfg.min_area_px)
    approx_epsilon = room_cfg.approx_epsilon
    opening_approx_epsilon = room_cfg.opening_approx_epsilon

    room_mask = cleaned[channels.rooms]
    wall_mask = cleaned[channels.walls]
    door_mask = cleaned[channels.doors]
    window_mask = cleaned[channels.windows]
    background_mask = cleaned[channels.background]
    region = cfg.fpn.floorplan_region

    wall_output = render_wall_room_masks(
        wall_mask,
        door_mask,
        window_mask,
        room_mask,
        region_mask=region_mask,
        size=size,
        step_dir=step_dir,
    )
    wall_components = [
        component
        for component in wall_output["wall_subtracted_components"]
        if int(component.sum()) >= min_area_px
    ]
    wall_components, rejected_background = filter_bright_room_components(
        wall_components,
        overlay,
        approx_epsilon=approx_epsilon,
        box_radius=room_cfg.bright_vertex_box_radius,
        bright_threshold=room_cfg.bright_vertex_threshold,
        reject_fraction=room_cfg.bright_vertex_reject_fraction,
    )
    background_plane = background_mask > 0.5
    if rejected_background.any():
        background_plane = close_binary_mask(
            background_plane | rejected_background,
            close_px=region.close_px,
        )
    floorplan_footprint = compute_floorplan_region(
        background_plane.astype(np.float32),
        close_px=region.close_px,
        approx_epsilon=region.boundary_approx_epsilon,
        dilate_px=region.boundary_dilate_px,
        min_component_area_ratio=region.min_component_area_ratio,
    )
    boundary_polygon = [
        (int(point[0]), int(point[1]))
        for point in floorplan_footprint.polygon
        if len(point) >= 2
    ]
    room_sources: list[tuple[str, list[np.ndarray]]] = [
        (f"Wall-subtract ({size}px)", wall_components),
        (
            f"FPN Five {size}px",
            room_components_from_mask(room_mask > 0.5, min_area_px=min_area_px),
        ),
    ]

    rooms_md_path = write_room_counts_markdown(
        [(label, len(components)) for label, components in room_sources],
        step_dir / "rooms.md",
        component_size=size,
        min_area_px=min_area_px,
    )
    room_overlays_path = save_all_room_mask_overlays(
        overlay,
        room_sources,
        step_dir,
        size=size,
        door_mask=door_mask,
        window_mask=window_mask,
        room_labels=room_labels,
    )
    room_contour_polygons = contour_polygons_from_components(
        wall_components,
        approx_epsilon=approx_epsilon,
    )
    door_components = room_components_from_mask(door_mask, min_area_px=min_area_px)
    door_contour_polygons = contour_polygons_from_components(
        door_components,
        approx_epsilon=opening_approx_epsilon,
    )
    window_components = room_components_from_mask(window_mask, min_area_px=min_area_px)
    window_contour_polygons = contour_polygons_from_components(
        window_components,
        approx_epsilon=opening_approx_epsilon,
    )
    mask_polygon_comparison_path = save_mask_polygon_comparison(
        overlay,
        wall_components,
        room_polygons=room_contour_polygons,
        door_polygons=door_contour_polygons,
        window_polygons=window_contour_polygons,
        boundary_polygon=boundary_polygon or None,
        out_path=step_dir / "mask-polygon-comparison.jpg",
        size=size,
        door_mask=door_mask,
        window_mask=window_mask,
        room_labels=room_labels,
    )

    final_masks = _save_final_masks(
        step_dir,
        size=size,
        min_area_px=min_area_px,
        room_polygons=room_contour_polygons,
        door_polygons=door_contour_polygons,
        window_polygons=window_contour_polygons,
        boundary_polygon=boundary_polygon,
        wall_approx_epsilon=opening_approx_epsilon,
        input_image=overlay,
    )

    step_dir.mkdir(parents=True, exist_ok=True)
    wall_mask_path = step_dir / "wall-mask.png"
    Image.fromarray((wall_mask > 0.5).astype(np.uint8) * 255, mode="L").save(wall_mask_path)
    _five_mask_row(prepared, masks, channels).save(step_dir / "masks.jpg", format="JPEG", quality=95)
    _five_mask_row(prepared, cleaned, channels).save(
        step_dir / "masks-clean.jpg",
        format="JPEG",
        quality=95,
    )

    return {
        "dir": step_dir,
        "masks": step_dir / "masks.jpg",
        "masks_clean": step_dir / "masks-clean.jpg",
        "wall_mask": wall_mask_path,
        "final_masks": final_masks,
        "mask_polygon_comparison": mask_polygon_comparison_path,
        "rooms": {
            "markdown": rooms_md_path,
            "overlays": room_overlays_path,
            "input": wall_output["input"],
        },
        "room_mask": room_mask,
    }


def _run_step_001(
    overlay: Image.Image,
    *,
    prepared: Image.Image,
    masks: np.ndarray,
    cleaned: np.ndarray,
    floorplan_region_mask: np.ndarray,
    step_dir: Path,
    channels: MaskChannels,
    cfg: Config,
    size: int,
    log: StepLogFn | None = None,
) -> dict[str, object]:
    llm = cfg.fpn.room_polygon_llm
    room_labels: list[RoomLabel] | None = None
    room_names_meta: dict[str, object] | None = None
    if llm.model and llm.openai_api_key:
        if log is not None:
            log(f"room names llm → {llm.model} at {llm.size}px")
        room_names_meta = detect_room_names(
            overlay,
            step_dir,
            llm_size=llm.size,
            pipeline_size=size,
            model=llm.model,
            detail=llm.detail,
            api_key=llm.openai_api_key,
            input_cost_per_million_usd=llm.input_cost_per_million_usd,
            cached_input_cost_per_million_usd=llm.cached_input_cost_per_million_usd,
            cache_write_cost_per_million_usd=llm.cache_write_cost_per_million_usd,
            output_cost_per_million_usd=llm.output_cost_per_million_usd,
            log=log,
        )
        room_labels = room_names_meta["rooms"]  # type: ignore[assignment]
        usage = room_names_meta["usage"]
        if log is not None and isinstance(usage, dict):
            log(
                "room names llm usage → "
                f"input: {usage.get('input_tokens', 0):,}  "
                f"output: {usage.get('output_tokens', 0):,}  "
                f"cost: ${usage.get('cost_usd', 0.0):.4f}"
            )
    elif log is not None:
        log("room names llm → skipped (missing model or OPENAI_API_KEY)")

    outputs = _write_step_artifacts(
        step_dir,
        prepared=prepared,
        masks=masks,
        cleaned=cleaned,
        overlay=overlay,
        region_mask=floorplan_region_mask.astype(np.float32),
        channels=channels,
        cfg=cfg,
        size=size,
        room_labels=room_labels,
    )
    outputs.pop("room_mask")
    if room_names_meta is not None:
        outputs["room_names"] = {
            "json": room_names_meta["json"],
            "input": room_names_meta["input"],
            "usage": room_names_meta["usage"],
        }
    return outputs


def _run_step_002(
    cropped_canvas: Image.Image,
    *,
    step_dir: Path,
    runtime: FpnFiveRuntime,
    channels: MaskChannels,
    cfg: Config,
    size: int,
) -> dict[str, object]:
    full_roi = np.ones((size, size), dtype=np.float32)
    prepared, masks, cleaned = _infer_on_canvas(
        cropped_canvas,
        size=size,
        runtime=runtime,
        cfg=cfg,
    )
    outputs = _write_step_artifacts(
        step_dir,
        prepared=prepared,
        masks=masks,
        cleaned=cleaned,
        overlay=cropped_canvas,
        region_mask=full_roi,
        channels=channels,
        cfg=cfg,
        size=size,
    )
    return outputs


def _run_step_003(
    input_image: Image.Image,
    room_mask: np.ndarray,
    *,
    step_dir: Path,
    cfg: Config,
    log: StepLogFn,
    pipeline_size: int,
) -> dict[str, object]:
    size = pipeline_size
    llm = cfg.fpn.room_polygon_llm
    min_area_px = max(1, cfg.fpn.room_components.min_area_px)
    if not llm.model:
        raise RuntimeError("Step 3 requires fpn.room_polygon_llm.model in config.yaml.")
    if not llm.openai_api_key:
        raise RuntimeError(
            "Step 3 requires OPENAI_API_KEY in backend/inference_core/.env."
        )

    step_dir.mkdir(parents=True, exist_ok=True)
    log(f"llm model → {llm.model}")
    log(f"llm size → {llm.size}px")

    room_polygons, door_polygons, window_polygons, floorplan_boundary, llm_meta = (
        refine_room_polygons(
            input_image,
            room_mask,
            step_dir,
            llm_size=llm.size,
            pipeline_size=size,
            min_area_px=min_area_px,
            model=llm.model,
            detail=llm.detail,
            api_key=llm.openai_api_key,
            input_cost_per_million_usd=llm.input_cost_per_million_usd,
            cached_input_cost_per_million_usd=llm.cached_input_cost_per_million_usd,
            cache_write_cost_per_million_usd=llm.cache_write_cost_per_million_usd,
            output_cost_per_million_usd=llm.output_cost_per_million_usd,
            log=log,
        )
    )

    usage = llm_meta["usage"]
    log(
        "llm usage → "
        f"input: {usage['input_tokens']:,}  "
        f"cached: {usage.get('cached_tokens', 0):,}  "
        f"cache write: {usage.get('cache_write_tokens', 0):,}  "
        f"output: {usage['output_tokens']:,}  "
        f"total: {usage['total_tokens']:,}  "
        f"cost: ${usage['cost_usd']:.4f}"
    )
    log(
        f"refined → {len(room_polygons)} rooms, "
        f"{len(door_polygons)} doors, {len(window_polygons)} windows"
    )

    polygons_llm: list = [entry["polygon"] for entry in llm_meta["rooms"]]
    doors_llm: list = [entry["polygon"] for entry in llm_meta["doors"]]
    windows_llm: list = [entry["polygon"] for entry in llm_meta["windows"]]
    boundary_llm = llm_meta["floorplan_boundary"]
    render_path = step_dir / "room-polygons.jpg"
    svg_path = step_dir / "room-polygons.svg"
    render_room_polygons(
        scale_image(input_image, llm.size),
        polygons_llm,
        size=llm.size,
        png_path=render_path,
        svg_path=svg_path,
        door_polygons=doors_llm,
        window_polygons=windows_llm,
        floorplan_boundary=boundary_llm,
    )

    final_masks = save_room_polygon_outputs(
        step_dir,
        pipeline_size=size,
        min_area_px=min_area_px,
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        floorplan_boundary=floorplan_boundary,
        wall_approx_epsilon=cfg.fpn.room_components.opening_approx_epsilon,
        input_image=input_image,
    )

    room_components = [polygons_to_mask([polygon], size) for polygon in room_polygons]
    room_sources: list[tuple[str, list[np.ndarray]]] = [
        (f"LLM refined ({size}px)", room_components),
        (
            f"Step-002 mask ({size}px)",
            room_components_from_mask(room_mask > 0.5, min_area_px=min_area_px),
        ),
    ]
    rooms_md_path = write_room_counts_markdown(
        [(label, len(components)) for label, components in room_sources],
        step_dir / "rooms.md",
        component_size=size,
        min_area_px=min_area_px,
    )
    room_overlays_path = save_all_room_mask_overlays(
        input_image,
        room_sources,
        step_dir,
        size=size,
    )

    return {
        "dir": step_dir,
        "input_1024": llm_meta["input_1024"],
        "room_mask_1024": llm_meta["room_mask_1024"],
        "floorplan_1024_json": llm_meta["floorplan_1024_json"],
        "room_polygons": render_path,
        "room_polygons_svg": svg_path,
        "final_masks": final_masks,
        "rooms": {
            "markdown": rooms_md_path,
            "overlays": room_overlays_path,
        },
        "llm_usage": llm_meta["usage"],
    }


def run_fpn_pipeline(
    cfg: Config,
    image_path: Path,
    *,
    runtime: FpnFiveRuntime | None = None,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    canvas_size: int | None = None,
    skip_step_2: bool = False,
    skip_step_3: bool = False,
) -> dict[str, object]:
    size = canvas_size or cfg.fpn.size
    min_area_px = max(1, cfg.fpn.room_components.min_area_px)
    full_canvas = fit_to_canvas_rgb(Image.open(image_path), size=size)
    checkpoint_path = checkpoint or cfg.fpn.checkpoint_latest
    runtime = runtime or FpnFiveRuntime.load(checkpoint=checkpoint_path)
    channels = runtime.channels

    out_dir = (output_dir or cfg.fpn.output_dir) / image_path.stem
    step_one_dir = out_dir / "step-001"
    step_two_dir = out_dir / "step-002"
    step_three_dir = out_dir / "step-003"
    region = cfg.fpn.floorplan_region
    step_two_outputs: dict[str, object] | None = None

    with _step_run("step-001") as log:
        log(f"image → {image_path}")
        log(f"output → {step_one_dir}")
        log(f"canvas → {size}×{size}")
        log(f"skip step-002 → {skip_step_2}")
        log(f"skip step-003 → {skip_step_3}")
        log(f"checkpoint → {checkpoint_path}")
        log(f"device → {runtime.device}")

        prepared, masks, cleaned = _infer_on_canvas(
            full_canvas,
            size=size,
            runtime=runtime,
            cfg=cfg,
        )
        floorplan_region = compute_floorplan_region(
            masks[channels.background],
            close_px=region.close_px,
            approx_epsilon=region.approx_epsilon,
            min_component_area_ratio=region.min_component_area_ratio,
        )
        region_pixels = int(floorplan_region.mask.sum())
        log(
            f"floorplan region → {region_pixels:,} px, "
            f"{len(floorplan_region.polygon)} polygon vertices"
        )
        region_mask_canvas = region_mask_for_image(floorplan_region.mask, full_canvas.size)
        masked_canvas = apply_region_mask(full_canvas, region_mask_canvas)

        step_one_outputs = _run_step_001(
            masked_canvas,
            prepared=prepared,
            masks=masks,
            cleaned=cleaned,
            floorplan_region_mask=floorplan_region.mask,
            step_dir=step_one_dir,
            channels=channels,
            cfg=cfg,
            size=size,
            log=log,
        )
        step_one_room_mask = np.asarray(
            Image.open(step_one_outputs["final_masks"]["room"]).convert("L")
        ) > 0
        log(
            f"room components → "
            f"{len(room_components_from_mask(step_one_room_mask, min_area_px=min_area_px))}"
        )
        step_one_final_dir = step_one_outputs["final_masks"]["background"].parent

    if not skip_step_2:
        with _step_run("step-002") as log:
            log(f"output → {step_two_dir}")
            cropped = crop_region_bbox(masked_canvas, region_mask_canvas, padding_px=100)
            log(f"crop → {cropped.width}×{cropped.height}px")
            step_two_dir.mkdir(parents=True, exist_ok=True)
            cropped_path = step_two_dir / "input-cropped.jpg"
            cropped.save(cropped_path, format="JPEG", quality=95)

            cropped_canvas = fit_to_canvas_rgb(cropped, size=size)
            log("fpn inference → cropped canvas")

            step_two_outputs = _run_step_002(
                cropped_canvas,
                step_dir=step_two_dir,
                runtime=runtime,
                channels=channels,
                cfg=cfg,
                size=size,
            )
            room_mask = step_two_outputs.pop("room_mask")
            step_two_room_count = len(
                room_components_from_mask(room_mask > 0.5, min_area_px=min_area_px)
            )
            log(f"room components → {step_two_room_count}")
            step_one_final_dir = step_two_outputs["final_masks"]["background"].parent
            step_three_input = cropped_canvas
    else:
        room_mask = step_one_room_mask
        step_three_input = full_canvas

    step_three_outputs: dict[str, object] | None = None
    if not skip_step_3:
        with _step_run("step-003") as log:
            log(f"output → {step_three_dir}")
            step_three_outputs = _run_step_003(
                step_three_input,
                room_mask,
                step_dir=step_three_dir,
                cfg=cfg,
                log=log,
                pipeline_size=size,
            )

    result: dict[str, object] = {
        "step-001": step_one_outputs,
    }
    if step_two_outputs is not None:
        result["step-002"] = {
            **step_two_outputs,
            "input_cropped": step_two_dir / "input-cropped.jpg",
        }
    if step_three_outputs is not None:
        result["step-003"] = step_three_outputs
    return result
