#!/usr/bin/env python3
"""Interactive pygame UI to refine FPN structure masks by hand.

Source images live in ``real_floorplan/``. Saves into ``floorplan-manual-mask/``:

  input/<id>.jpg           — source image fit to 1024×1024 canvas (unmasked)
  output/<id>/wall.png     — white = active
  output/<id>/door.png
  output/<id>/window.png
  output/<id>/room.png
  output/<id>/background.png — floorplan boundary mask
  output/<id>/polygon-overlay.jpg — translucent polygon preview on input

Polygons are stored under ``output/label_data/<id>/step-003/final-masks/`` (not in the manual dataset).

Wall is mask-only (brush or **Compute wall**); it has no polygon edit. Masks are always
derived from polygons (background = black fill on white; wall = background − rooms).
Polygons live in ``output/label_data/<id>/step-003/final-masks/`` and are edited in place.
**Infer masks** runs the FPN pipeline into ``output/label_data/``. Final mask PNGs are saved
to ``floorplan-manual-mask/output/``.
**T** toggles brush vs polygon tool. Brush strokes refit layer polygons from the painted
mask. **Polygon edit dropdown** (select point / add point / new polygon). **Del** removes
a vertex.

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python label_data.py

Batch processed inference for every source image (no UI):

    uv run python label_data.py --auto

Skip images that already have saved masks/polygons (default). Re-run all with ``--force``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_rgb, fit_to_canvas_placement
from util.final_mask_polygons import (
    PolygonRing,
    StructureLayer,
    fit_polygons_from_mask,
    load_layer_polygons,
    polygons_to_mask,
    save_layer_polygons,
    transform_mask_to_full_canvas,
    transform_polygons_to_full_canvas,
    transform_polygons_from_full_canvas,
)
from util.floorplan_region import region_crop_box
from util.fpn_pipeline import run_fpn_pipeline
from util.room_polygon_llm import save_final_mask_polygon_preview

_PIPELINE_CROP_PADDING_PX = 100

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]
LABEL_INFERENCE_OUTPUT_DIR = _ROOT / "output" / "label_data"
LABEL_CANVAS_SIZE = 1024
TEST_DIR = _ROOT / "real_floorplan"
DATASET_ROOT = _REPO_ROOT / "floorplan-manual-mask"
INPUT_DIR = DATASET_ROOT / "input"
OUTPUT_DIR = DATASET_ROOT / "output"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MaskLayer = Literal["wall", "door", "window", "room", "background"]
BrushMode = Literal["add", "remove"]
EditTool = Literal["brush", "polygon"]
PolygonEditMode = Literal["select_point", "add_point", "new_polygon"]
POLYGON_EDIT_OPTIONS: tuple[tuple[PolygonEditMode, str], ...] = (
    ("select_point", "Select point"),
    ("add_point", "Add point"),
    ("new_polygon", "New polygon"),
)
POLYGON_EDIT_DROPDOWN_ROW_HEIGHT = 26
STRUCTURE_LAYERS: tuple[MaskLayer, ...] = ("wall", "door", "window", "room")
POLYGON_EDIT_LAYERS: tuple[MaskLayer, ...] = ("room", "door", "window", "background")
MASK_LAYERS: tuple[MaskLayer, ...] = (*STRUCTURE_LAYERS, "background")

LIST_WIDTH = 200
PANEL_WIDTH = 280
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
POLY_LIST_ROW_HEIGHT = 20
POLY_LIST_HEADER_HEIGHT = 18
VIEWER_DISPLAY_SIZE = 800  # UI viewer px; canvas/masks stay at LABEL_CANVAS_SIZE
DEFAULT_BRUSH_RADIUS = 10
MIN_BRUSH_RADIUS = 2
MAX_BRUSH_RADIUS = 64
POLYGON_CLOSE_RADIUS = 8
JPEG_QUALITY = 95

SIDE_BG = (28, 28, 28)
LIST_BG = (22, 22, 22)
LIST_HOVER = (45, 45, 45)
LIST_SELECTED = (60, 80, 120)
LIST_TEXT = (220, 220, 220)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_ACTIVE = (70, 90, 70)
BTN_ACTIVE_HOVER = (90, 110, 90)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
STATUS_COLOR = (120, 180, 255)
LAYER_COLORS: dict[MaskLayer, tuple[int, int, int]] = {
    "wall": (255, 80, 80),
    "door": (80, 220, 80),
    "window": (80, 160, 255),
    "room": (255, 200, 80),
    "background": (180, 180, 255),
}
OVERLAY_ALPHA = 110
POLYGON_DRAFT_COLOR = (255, 200, 0)
POLYGON_VERTEX_RADIUS = 4
POLYGON_HIT_RADIUS = 8
SELECTED_VERTEX_COLOR = (255, 60, 220)
SELECTED_POLY_FILL_ALPHA = 70
SELECTED_VERTEX_RADIUS = 3


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def background_polygons_to_mask(components: list[PolygonRing], size: int) -> np.ndarray:
    """Black polygon interior on a white canvas (active = white outside polygon)."""
    mask = np.ones((size, size), dtype=bool)
    for polygon in components:
        if len(polygon) < 3:
            continue
        fill = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(fill, [np.array(polygon, dtype=np.int32)], 1)
        mask[fill.astype(bool)] = False
    return mask


def compute_wall_mask_from_boundary_and_rooms(
    boundary_polygons: list[PolygonRing],
    room_polygons: list[PolygonRing],
    *,
    size: int,
) -> np.ndarray:
    """Wall = white floorplan interior with black room cutouts on a black canvas."""
    mask = np.zeros((size, size), dtype=bool)
    if boundary_polygons:
        boundary_fill = polygons_to_mask(boundary_polygons, size)
        mask[boundary_fill] = True
    if room_polygons:
        room_fill = polygons_to_mask(room_polygons, size)
        mask[room_fill] = False
    return mask


def empty_masks(size: int) -> dict[MaskLayer, np.ndarray]:
    blank = np.zeros((size, size), dtype=bool)
    return {layer: blank.copy() for layer in MASK_LAYERS}


def _label_image_dir(stem: str) -> Path:
    return LABEL_INFERENCE_OUTPUT_DIR / stem


def _pipeline_crop_box(image_path: Path, *, canvas_size: int) -> tuple[int, int, int, int]:
    region_mask_path = _label_image_dir(image_path.stem) / "step-001" / "floorplan-region-mask.jpg"
    if not region_mask_path.is_file():
        raise FileNotFoundError(f"Missing pipeline region mask: {region_mask_path}")
    gray = np.asarray(Image.open(region_mask_path).convert("L"))
    if gray.shape != (canvas_size, canvas_size):
        gray = cv2.resize(gray, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)
    crop_box = region_crop_box(
        gray > 10,
        (canvas_size, canvas_size),
        padding_px=_PIPELINE_CROP_PADDING_PX,
    )
    if crop_box is None:
        raise RuntimeError("Could not derive crop box for pipeline polygons")
    return crop_box


def _step003_final_dir(stem: str) -> Path:
    return _label_image_dir(stem) / "step-003" / "final-masks"


def _step003_source_canvas(stem: str, image_path: Path, *, canvas_size: int) -> Image.Image:
    saved_input = _step003_final_dir(stem) / "input.jpg"
    if saved_input.is_file():
        with Image.open(saved_input) as image:
            canvas = image.convert("RGB")
        if canvas.size != (canvas_size, canvas_size):
            canvas = canvas.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
        return canvas
    return fit_to_canvas_rgb(Image.open(image_path), size=canvas_size)


def _pipeline_step_content_size(stem: str, crop_box: tuple[int, int, int, int]) -> tuple[int, int]:
    cropped_path = _label_image_dir(stem) / "step-002" / "input-cropped.jpg"
    if cropped_path.is_file():
        with Image.open(cropped_path) as cropped:
            return cropped.size
    x0, y0, x1, y1 = crop_box
    return max(1, x1 - x0), max(1, y1 - y0)


def _final_mask_step_size(final_dir: Path, default: int = LABEL_CANVAS_SIZE) -> int:
    for name in ("room.json", "background.json", "door.json", "window.json"):
        json_path = final_dir / name
        if not json_path.is_file():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        size = data.get("size")
        if isinstance(size, int) and size > 0:
            return size
    return default


def _remap_rings_to_label_canvas(
    rings: list[PolygonRing],
    image_path: Path,
    *,
    from_size: int,
) -> list[PolygonRing]:
    if from_size == LABEL_CANVAS_SIZE:
        return rings
    with Image.open(image_path) as image:
        orig_w, orig_h = image.size
    px0, py0, cw0, ch0 = fit_to_canvas_placement(orig_w, orig_h, size=from_size)
    px1, py1, cw1, ch1 = fit_to_canvas_placement(orig_w, orig_h, size=LABEL_CANVAS_SIZE)
    remapped: list[PolygonRing] = []
    for ring in rings:
        remapped.append([
            (
                int(round(px1 + ((x - px0) / cw0 if cw0 else 0.0) * cw1)),
                int(round(py1 + ((y - py0) / ch0 if ch0 else 0.0) * ch1)),
            )
            for x, y in ring
        ])
    return remapped


def load_step003_polygons(
    cfg,
    image_path: Path,
    *,
    full_canvas: Image.Image,
) -> tuple[
    dict[MaskLayer, np.ndarray],
    dict[StructureLayer, list[PolygonRing]],
    list[PolygonRing],
] | None:
    """Load step-003 LLM polygons and masks scaled to the full canvas."""
    final_dir = _step003_final_dir(image_path.stem)
    if not final_dir.is_dir():
        return None

    step_size = _final_mask_step_size(final_dir)
    transform_kwargs = None
    if step_size != LABEL_CANVAS_SIZE:
        try:
            crop_box = _pipeline_crop_box(image_path, canvas_size=step_size)
            content_size = _pipeline_step_content_size(image_path.stem, crop_box)
            transform_kwargs = {
                "crop_box": crop_box,
                "step_size": step_size,
                "content_size": content_size,
            }
        except (FileNotFoundError, RuntimeError):
            pass

    layer_polygons: dict[StructureLayer, list[PolygonRing]] = {
        layer: [] for layer in STRUCTURE_LAYERS
    }
    masks = empty_masks(LABEL_CANVAS_SIZE)

    def _polygons_from_json(json_path: Path) -> list[PolygonRing]:
        if not json_path.is_file():
            return []
        rings = load_layer_polygons(json_path)
        if transform_kwargs is None:
            return rings
        rings = transform_polygons_to_full_canvas(rings, **transform_kwargs)
        return _remap_rings_to_label_canvas(rings, image_path, from_size=step_size)

    for layer in STRUCTURE_LAYERS:
        json_path = final_dir / f"{layer}.json"
        if json_path.is_file():
            layer_polygons[layer] = _polygons_from_json(json_path)
        if layer == "wall":
            continue
        if layer_polygons[layer]:
            masks[layer] = polygons_to_mask(layer_polygons[layer], LABEL_CANVAS_SIZE)
        else:
            png_path = final_dir / f"{layer}.png"
            if png_path.is_file():
                step_mask = np.asarray(Image.open(png_path).convert("L")) > 127
                if transform_kwargs is not None:
                    masks[layer] = transform_mask_to_full_canvas(
                        step_mask,
                        full_size=LABEL_CANVAS_SIZE,
                        **transform_kwargs,
                    )
                elif step_mask.shape != (LABEL_CANVAS_SIZE, LABEL_CANVAS_SIZE):
                    masks[layer] = cv2.resize(
                        step_mask.astype(np.uint8),
                        (LABEL_CANVAS_SIZE, LABEL_CANVAS_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    masks[layer] = step_mask

    background_polygons: list[PolygonRing] = []
    background_json = final_dir / "background.json"
    if background_json.is_file():
        background_polygons = _polygons_from_json(background_json)
    if background_polygons:
        masks["background"] = background_polygons_to_mask(
            background_polygons,
            LABEL_CANVAS_SIZE,
        )
    else:
        background_path = final_dir / "background.png"
        if background_path.is_file():
            step_mask = np.asarray(Image.open(background_path).convert("L")) > 127
            if transform_kwargs is not None:
                masks["background"] = transform_mask_to_full_canvas(
                    step_mask,
                    full_size=LABEL_CANVAS_SIZE,
                    **transform_kwargs,
                )
            elif step_mask.shape != (LABEL_CANVAS_SIZE, LABEL_CANVAS_SIZE):
                masks["background"] = cv2.resize(
                    step_mask.astype(np.uint8),
                    (LABEL_CANVAS_SIZE, LABEL_CANVAS_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                masks["background"] = step_mask
        else:
            masks["background"] = np.ones((LABEL_CANVAS_SIZE, LABEL_CANVAS_SIZE), dtype=bool)

    layer_polygons["wall"] = []
    masks["wall"] = compute_wall_mask_from_boundary_and_rooms(
        background_polygons,
        layer_polygons["room"],
        size=LABEL_CANVAS_SIZE,
    )

    return masks, layer_polygons, background_polygons


def infer_processed_from_pipeline(
    cfg,
    image_path: Path,
) -> tuple[
    Image.Image,
    dict[MaskLayer, np.ndarray],
    dict[StructureLayer, list[PolygonRing]],
    list[PolygonRing],
]:
    """Run the FPN pipeline and load step-003 LLM polygons."""
    full_canvas = _step003_source_canvas(image_path.stem, image_path, canvas_size=LABEL_CANVAS_SIZE)
    run_fpn_pipeline(
        cfg,
        image_path,
        checkpoint=cfg.fpn.checkpoint_latest,
        output_dir=LABEL_INFERENCE_OUTPUT_DIR,
        canvas_size=LABEL_CANVAS_SIZE,
        skip_step_2=True,
    )
    loaded = load_step003_polygons(cfg, image_path, full_canvas=full_canvas)
    if loaded is None:
        raise FileNotFoundError(f"Missing pipeline output: {_step003_final_dir(image_path.stem)}")
    masks, layer_polygons, background_polygons = loaded
    return full_canvas, masks, layer_polygons, background_polygons


def mask_to_gray_image(mask: np.ndarray) -> Image.Image:
    gray = (mask.astype(np.uint8)) * 255
    return Image.fromarray(gray, mode="L")


def _load_mask_png(path: Path, size: int) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"))
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
    return gray > 127


def _min_area_px(cfg) -> int:
    return max(1, cfg.fpn.room_components.min_area_px)


def save_source_polygon_labels(
    cfg,
    image_path: Path,
    *,
    full_canvas: Image.Image,
    layer_polygons: dict[StructureLayer, list[PolygonRing]],
    background_polygons: list[PolygonRing],
    canvas_size: int,
    min_area_px: int,
) -> Path:
    final_dir = _step003_final_dir(image_path.stem)
    final_dir.mkdir(parents=True, exist_ok=True)
    save_size = LABEL_CANVAS_SIZE
    transform_kwargs: dict[str, object] | None = None
    if _final_mask_step_size(final_dir) != LABEL_CANVAS_SIZE:
        try:
            crop_box = _pipeline_crop_box(image_path, canvas_size=LABEL_CANVAS_SIZE)
            content_size = _pipeline_step_content_size(image_path.stem, crop_box)
            transform_kwargs = {
                "crop_box": crop_box,
                "step_size": LABEL_CANVAS_SIZE,
                "content_size": content_size,
            }
        except (FileNotFoundError, RuntimeError):
            pass

    def to_source(rings: list[PolygonRing]) -> list[PolygonRing]:
        if transform_kwargs is None:
            return rings
        return transform_polygons_from_full_canvas(rings, **transform_kwargs)

    for layer in ("room", "door", "window"):
        save_layer_polygons(
            final_dir / f"{layer}.json",
            layer,
            to_source(layer_polygons.get(layer, [])),
            size=save_size,
            min_area_px=min_area_px,
        )
    if background_polygons:
        save_layer_polygons(
            final_dir / "background.json",
            "room",
            to_source(background_polygons),
            size=save_size,
            min_area_px=min_area_px,
        )
    save_final_mask_polygon_preview(
        full_canvas,
        room_polygons=layer_polygons.get("room", []),
        door_polygons=layer_polygons.get("door", []),
        window_polygons=layer_polygons.get("window", []),
        floorplan_boundary=background_polygons[0] if background_polygons else None,
        out_dir=final_dir,
    )
    return final_dir


def load_saved_masks(stem: str, size: int) -> dict[MaskLayer, np.ndarray] | None:
    out_dir = OUTPUT_DIR / stem
    structure_paths = {layer: out_dir / f"{layer}.png" for layer in STRUCTURE_LAYERS}
    if not all(path.is_file() for path in structure_paths.values()):
        return None
    masks = empty_masks(size)
    for layer, path in structure_paths.items():
        masks[layer] = _load_mask_png(path, size)
    background_path = out_dir / "background.png"
    if background_path.is_file():
        masks["background"] = _load_mask_png(background_path, size)
    return masks


def save_manual_masks(
    stem: str,
    canvas: Image.Image,
    masks: dict[MaskLayer, np.ndarray],
    *,
    layer_polygons: dict[StructureLayer, list[PolygonRing]] | None = None,
    background_polygons: list[PolygonRing] | None = None,
) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = INPUT_DIR / f"{stem}.jpg"
    canvas.save(input_path, format="JPEG", quality=JPEG_QUALITY)

    for layer in MASK_LAYERS:
        mask_to_gray_image(masks[layer]).save(out_dir / f"{layer}.png")

    if layer_polygons is not None:
        save_final_mask_polygon_preview(
            canvas,
            room_polygons=layer_polygons.get("room", []),
            door_polygons=layer_polygons.get("door", []),
            window_polygons=layer_polygons.get("window", []),
            floorplan_boundary=background_polygons[0] if background_polygons else None,
            out_dir=out_dir,
            save_input=False,
        )
    return out_dir


def _auto_outputs_exist(stem: str) -> bool:
    """True when batch auto-save artifacts are already on disk."""
    out_dir = OUTPUT_DIR / stem
    if not all((out_dir / f"{layer}.png").is_file() for layer in MASK_LAYERS):
        return False
    return (_step003_final_dir(stem) / "room.json").is_file()


def auto_process_all_images(
    cfg,
    image_paths: list[Path],
    *,
    skip_existing: bool = True,
) -> int:
    """Run processed pipeline inference and save masks/polygons for every image."""
    min_area = _min_area_px(cfg)
    size = LABEL_CANVAS_SIZE
    failed: list[tuple[str, str]] = []
    skipped = 0
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        print(f"[{index}/{total}] {image_path.name}", flush=True)
        if skip_existing and _auto_outputs_exist(stem):
            skipped += 1
            print("  skipped (outputs already exist)", flush=True)
            continue
        try:
            canvas, masks, layer_polygons, background_polygons = infer_processed_from_pipeline(
                cfg,
                image_path,
            )
            save_source_polygon_labels(
                cfg,
                image_path,
                full_canvas=canvas,
                layer_polygons=layer_polygons,
                background_polygons=background_polygons,
                canvas_size=size,
                min_area_px=min_area,
            )
            out_dir = save_manual_masks(
                stem,
                canvas,
                masks,
                layer_polygons=layer_polygons,
                background_polygons=background_polygons,
            )
            print(
                f"  masks -> {out_dir.relative_to(_REPO_ROOT)}  "
                f"polygons -> {_step003_final_dir(stem).relative_to(_ROOT)}",
                flush=True,
            )
        except Exception as exc:
            failed.append((image_path.name, str(exc)))
            print(f"  failed: {exc}", flush=True)

    saved = total - len(failed) - skipped
    print(
        f"Done — saved {saved}/{total} to {DATASET_ROOT.relative_to(_REPO_ROOT)}"
        + (f" ({skipped} skipped)" if skipped else ""),
        flush=True,
    )
    if failed:
        print("Failures:", flush=True)
        for name, message in failed:
            print(f"  {name}: {message}", flush=True)
        return 1
    return 0


def _draw_vertex_handle(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int],
    selected: bool,
) -> None:
    r = SELECTED_VERTEX_RADIUS if selected else POLYGON_VERTEX_RADIUS
    fill = SELECTED_VERTEX_COLOR if selected else color
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(*fill, 255))


def _draw_polygon_ring(
    draw: ImageDraw.ImageDraw,
    ring: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    fill_alpha: int | None = None,
    width: int = 2,
    selected_vertex: int | None = None,
) -> None:
    if len(ring) < 2:
        return
    if fill_alpha is not None and len(ring) >= 3:
        draw.polygon(ring, fill=(*color, fill_alpha))
    draw.line([*ring, ring[0]], fill=(*color, 230), width=width)
    for index, (x, y) in enumerate(ring):
        _draw_vertex_handle(
            draw,
            x,
            y,
            color=color,
            selected=index == selected_vertex,
        )


def _draw_polygon_draft_overlay(
    canvas: Image.Image,
    draft: list[tuple[int, int]],
    *,
    cursor: tuple[int, int] | None,
    color: tuple[int, int, int] = POLYGON_DRAFT_COLOR,
    selected_vertex: int | None = None,
) -> Image.Image:
    if not draft:
        return canvas
    overlay = canvas.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    if len(draft) >= 2:
        draw.line(draft, fill=(*color, 220), width=2)
    if cursor is not None and draft:
        draw.line([draft[-1], cursor], fill=(*color, 160), width=1)
    for index, (x, y) in enumerate(draft):
        _draw_vertex_handle(
            draw,
            x,
            y,
            color=color,
            selected=index == selected_vertex,
        )
    return overlay


def _draw_polygon_components_overlay(
    canvas: Image.Image,
    components: list[PolygonRing],
    *,
    color: tuple[int, int, int],
    selected_index: int | None = None,
    selected_vertex: int | None = None,
) -> Image.Image:
    if not components:
        return canvas
    overlay = canvas.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, ring in enumerate(components):
        is_selected = index == selected_index
        _draw_polygon_ring(
            draw,
            ring,
            color=color,
            fill_alpha=SELECTED_POLY_FILL_ALPHA if is_selected else 40,
            width=3 if is_selected else 2,
            selected_vertex=selected_vertex if is_selected else None,
        )
    return overlay


def compose_overlay(
    canvas: Image.Image,
    masks: dict[MaskLayer, np.ndarray],
    *,
    active_layer: MaskLayer,
    edit_tool: EditTool = "brush",
    layer_polygons: dict[StructureLayer, list[PolygonRing]] | None = None,
    background_polygons: list[PolygonRing] | None = None,
    polygon_draft: list[tuple[int, int]] | None = None,
    polygon_cursor: tuple[int, int] | None = None,
    selected_poly_index: int | None = None,
    selected_vertex_index: int | None = None,
) -> Image.Image:
    rgb = np.asarray(canvas.convert("RGB"))
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    layer_rgb = LAYER_COLORS[active_layer]
    polygon_components: list[PolygonRing] = []
    if layer_polygons is not None and active_layer in ("room", "door", "window"):
        polygon_components = layer_polygons[active_layer]
    elif background_polygons is not None and active_layer == "background":
        polygon_components = background_polygons
    show_polygons = (
        edit_tool == "polygon"
        and active_layer in POLYGON_EDIT_LAYERS
        and bool(polygon_components or polygon_draft)
    )
    if show_polygons and polygon_components:
        rgb = Image.fromarray(rgba[:, :, :3], mode="RGB")
        rgb = _draw_polygon_components_overlay(
            rgb,
            polygon_components,
            color=layer_rgb,
            selected_index=selected_poly_index,
            selected_vertex=selected_vertex_index,
        )
    else:
        active = masks[active_layer]
        color = np.zeros((*active.shape, 4), dtype=np.uint8)
        color[active] = (*layer_rgb, OVERLAY_ALPHA)
        overlay = Image.fromarray(color, mode="RGBA")
        composed = Image.alpha_composite(Image.fromarray(rgba, mode="RGBA"), overlay)
        rgb = composed.convert("RGB")
    if polygon_draft:
        rgb = _draw_polygon_draft_overlay(
            rgb,
            polygon_draft,
            cursor=polygon_cursor,
            color=layer_rgb,
            selected_vertex=selected_vertex_index,
        )
    return rgb


def _point_in_polygon(x: int, y: int, ring: PolygonRing) -> bool:
    if len(ring) < 3:
        return False
    path = np.array(ring, dtype=np.int32)
    return cv2.pointPolygonTest(path, (float(x), float(y)), False) >= 0


def _find_vertex_near(point: tuple[int, int], ring: PolygonRing, *, radius: int) -> int | None:
    px, py = point
    radius_sq = radius * radius
    for index, (x, y) in enumerate(ring):
        if (x - px) ** 2 + (y - py) ** 2 <= radius_sq:
            return index
    return None


def _nearest_edge_insert_index(point: tuple[int, int], ring: PolygonRing) -> int | None:
    if len(ring) < 2:
        return None
    px, py = point
    best_index = 0
    best_dist = float("inf")
    count = len(ring)
    for index in range(count):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % count]
        dx = x1 - x0
        dy = y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_index = index + 1
    if best_dist > POLYGON_HIT_RADIUS ** 2:
        return None
    return best_index


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgb = image.convert("RGB")
    return pygame.image.frombuffer(rgb.tobytes(), rgb.size, "RGB").copy()


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    visible: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False, active: bool = False) -> None:
        if not self.visible:
            return
        if active:
            color = BTN_ACTIVE_HOVER if hover else BTN_ACTIVE
        else:
            color = BTN_HOVER if hover else BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (90, 90, 90), self.rect, 1, border_radius=4)
        text = font.render(self.label, True, BTN_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.visible and self.rect.collidepoint(pos)


@dataclass
class PolygonEditDropdown:
    rect: pygame.Rect
    selected: PolygonEditMode = "select_point"
    open: bool = False
    visible: bool = False

    def label_for(self, mode: PolygonEditMode) -> str:
        for option, label in POLYGON_EDIT_OPTIONS:
            if option == mode:
                return label
        return mode

    @property
    def button_label(self) -> str:
        return f"{self.label_for(self.selected)} ▾"

    def option_rects(self) -> list[tuple[PolygonEditMode, pygame.Rect]]:
        if not self.open:
            return []
        return [
            (
                mode,
                pygame.Rect(
                    self.rect.x,
                    self.rect.bottom + index * POLYGON_EDIT_DROPDOWN_ROW_HEIGHT,
                    self.rect.width,
                    POLYGON_EDIT_DROPDOWN_ROW_HEIGHT - 2,
                ),
            )
            for index, (mode, _label) in enumerate(POLYGON_EDIT_OPTIONS)
        ]

    def menu_rect(self) -> pygame.Rect | None:
        if not self.open:
            return None
        count = len(POLYGON_EDIT_OPTIONS)
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom,
            self.rect.width,
            count * POLYGON_EDIT_DROPDOWN_ROW_HEIGHT,
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False) -> None:
        if not self.visible:
            return
        Button(self.rect, self.button_label).draw(
            surface,
            font,
            hover=hover,
            active=self.open,
        )
        if not self.open:
            return
        menu = self.menu_rect()
        if menu is None:
            return
        pygame.draw.rect(surface, LIST_BG, menu)
        pygame.draw.rect(surface, (90, 90, 90), menu, 1)
        mouse = pygame.mouse.get_pos()
        for mode, rect in self.option_rects():
            if mode == self.selected:
                pygame.draw.rect(surface, LIST_SELECTED, rect)
            elif rect.collidepoint(mouse):
                pygame.draw.rect(surface, LIST_HOVER, rect)
            text = font.render(self.label_for(mode), True, BTN_TEXT)
            surface.blit(text, (rect.x + 8, rect.y + 5))

    def hit_button(self, pos: tuple[int, int]) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    def hit_option(self, pos: tuple[int, int]) -> PolygonEditMode | None:
        for mode, rect in self.option_rects():
            if rect.collidepoint(pos):
                return mode
        return None


class LabelDataUI:
    def __init__(self, cfg, test_images: list[Path]) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.test_images = test_images
        self.canvas_size = LABEL_CANVAS_SIZE
        self.selected_index = 0
        self.list_scroll = 0
        self.active_layer: MaskLayer = "wall"
        self.edit_tool: EditTool = "brush"
        self.brush_mode: BrushMode = "add"
        self.brush_radius = DEFAULT_BRUSH_RADIUS
        self.polygon_draft: list[tuple[int, int]] | None = None
        self.polygon_edit_mode: PolygonEditMode = "select_point"
        self.selected_poly_index: int | None = None
        self.selected_vertex_index: int | None = None
        self.dragging_vertex_index: int | None = None
        self.poly_list_scroll = 0
        self.point_list_scroll = 0
        self._poly_list_hit_rect: pygame.Rect | None = None
        self._point_list_hit_rect: pygame.Rect | None = None
        self.painting = False
        self.running_inference = False
        self.masks = empty_masks(self.canvas_size)
        self.layer_polygons: dict[StructureLayer, list[PolygonRing]] = {
            layer: [] for layer in STRUCTURE_LAYERS
        }
        self.background_polygons: list[PolygonRing] = []
        self.source_canvas = Image.new("RGB", (self.canvas_size, self.canvas_size), (255, 255, 255))
        self.display_surface: pygame.Surface | None = None
        self.display_rect = pygame.Rect(0, 0, 0, 0)
        self.list_w = LIST_WIDTH
        self.panel_w = PANEL_WIDTH
        self.viewer_x = self.list_w
        self.status = ""
        try:
            self.model_address = str(cfg.fpn.checkpoint_latest.relative_to(_ROOT.parent))
        except ValueError:
            self.model_address = str(cfg.fpn.checkpoint_latest)
        self._load_test_image(0, initial=True)
        self.display_w = VIEWER_DISPLAY_SIZE
        self.display_h = VIEWER_DISPLAY_SIZE
        self.panel_x = self.viewer_x + self.display_w
        self.window_w = self.panel_x + self.panel_w
        self.window_h = VIEWER_DISPLAY_SIZE
        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Label masks — floorplan-manual-mask")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 12, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self._layout_buttons()
        self._update_tool_ui()
        self._rebuild_display()

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = self.panel_w - 20
        y = 10
        gap = 36
        self.infer_btn = Button(pygame.Rect(x, y, btn_w, 28), "Infer masks")
        y += gap
        self.save_btn = Button(pygame.Rect(x, y, btn_w, 28), "Save masks")
        y += gap
        self.layer_buttons: dict[MaskLayer, Button] = {}
        for layer in STRUCTURE_LAYERS:
            self.layer_buttons[layer] = Button(
                pygame.Rect(x, y, btn_w, 26),
                f"Layer: {layer}",
            )
            y += 30
        y += 4
        self.compute_wall_btn = Button(pygame.Rect(x, y, btn_w, 28), "Compute wall")
        y += 34
        self.layer_buttons["background"] = Button(
            pygame.Rect(x, y, btn_w, 26),
            "Layer: background",
        )
        y += 30
        y += 6
        self.tool_btn = Button(pygame.Rect(x, y, btn_w, 28), "Tool: brush")
        y += gap
        self.brush_mode_btn = Button(pygame.Rect(x, y, btn_w, 28), "Brush: add")
        y += gap
        self.poly_edit_dropdown = PolygonEditDropdown(
            pygame.Rect(x, y, btn_w, 28),
            visible=False,
        )
        y += gap
        self.brush_size_label_y = y
        y += 22
        self.polygon_list_y = y + 26
        self.status_y = self.polygon_list_y + 180

    def _current_stem(self) -> str:
        return self.test_images[self.selected_index].stem

    def _clear_polygon_selection(self) -> None:
        self.selected_poly_index = None
        self.selected_vertex_index = None
        self.dragging_vertex_index = None

    def _active_structure_layer(self) -> StructureLayer | None:
        layer = self.active_layer
        return layer if layer in STRUCTURE_LAYERS else None

    def _editable_polygon_layer(self) -> MaskLayer | None:
        if self.active_layer in POLYGON_EDIT_LAYERS:
            return self.active_layer
        return None

    def _active_polygon_list(self) -> list[PolygonRing]:
        layer = self._editable_polygon_layer()
        if layer is None:
            return []
        if layer == "background":
            return self.background_polygons
        return self.layer_polygons[layer]

    def _select_polygon(self, index: int | None) -> None:
        polygons = self._active_polygon_list()
        if index is not None and (index < 0 or index >= len(polygons)):
            index = None
        self.selected_poly_index = index
        self.selected_vertex_index = None
        self.dragging_vertex_index = None
        self.point_list_scroll = 0
        if index is not None:
            count = len(polygons[index])
            self.status = f"Selected polygon {index + 1} ({count} points)."
        else:
            self.status = "No polygon selected."

    def _select_vertex(self, poly_index: int, vertex_index: int) -> None:
        polygons = self._active_polygon_list()
        if poly_index < 0 or poly_index >= len(polygons):
            return
        ring = polygons[poly_index]
        if vertex_index < 0 or vertex_index >= len(ring):
            return
        self.selected_poly_index = poly_index
        self.selected_vertex_index = vertex_index
        self._ensure_point_visible(vertex_index)
        x, y = ring[vertex_index]
        self.status = f"Selected P{vertex_index} ({x}, {y})."

    def _polygon_edit_status(self, mode: PolygonEditMode) -> str:
        if mode == "select_point":
            return "Select point — click polygon or vertex to edit."
        if mode == "add_point":
            return "Add point — click near edge of selected polygon to insert."
        return "New polygon — place points, click start point or Enter to finish."

    def _set_polygon_edit_mode(self, mode: PolygonEditMode) -> None:
        self.polygon_edit_mode = mode
        self.poly_edit_dropdown.selected = mode
        self.poly_edit_dropdown.open = False
        if mode != "new_polygon":
            self.polygon_draft = None
        if mode == "new_polygon":
            self._clear_polygon_selection()
        self.status = self._polygon_edit_status(mode)
        self._rebuild_display()

    def _cycle_polygon_edit_mode(self) -> None:
        modes = [mode for mode, _label in POLYGON_EDIT_OPTIONS]
        index = modes.index(self.polygon_edit_mode)
        self._set_polygon_edit_mode(modes[(index + 1) % len(modes)])

    def _handle_polygon_edit_dropdown_click(self, pos: tuple[int, int]) -> bool:
        if not self.poly_edit_dropdown.visible:
            return False
        if self.poly_edit_dropdown.open:
            picked = self.poly_edit_dropdown.hit_option(pos)
            if picked is not None:
                self._set_polygon_edit_mode(picked)
                return True
            if self.poly_edit_dropdown.menu_rect() and self.poly_edit_dropdown.menu_rect().collidepoint(pos):
                return True
            self.poly_edit_dropdown.open = False
            return self.poly_edit_dropdown.hit_button(pos)
        if self.poly_edit_dropdown.hit_button(pos):
            self.poly_edit_dropdown.open = True
            return True
        return False

    def _empty_layer_polygons(self) -> dict[StructureLayer, list[PolygonRing]]:
        return {layer: [] for layer in STRUCTURE_LAYERS}

    def _sync_masks_from_polygons(self) -> None:
        size = self.canvas_size
        for layer in ("room", "door", "window"):
            self.masks[layer] = polygons_to_mask(self.layer_polygons[layer], size)
        if self.background_polygons:
            self.masks["background"] = background_polygons_to_mask(
                self.background_polygons,
                size,
            )
        self.masks["wall"] = compute_wall_mask_from_boundary_and_rooms(
            self.background_polygons,
            self.layer_polygons["room"],
            size=size,
        )

    def _save_polygon_labels(self) -> None:
        image_path = self.test_images[self.selected_index]
        save_source_polygon_labels(
            self.cfg,
            image_path,
            full_canvas=self.source_canvas,
            layer_polygons=self.layer_polygons,
            background_polygons=self.background_polygons,
            canvas_size=self.canvas_size,
            min_area_px=_min_area_px(self.cfg),
        )

    def _persist_labels(self, *, show_status: bool = True) -> None:
        if self.running_inference:
            return
        stem = self._current_stem()
        self._sync_masks_from_polygons()
        self._save_polygon_labels()
        out_dir = save_manual_masks(
            stem,
            self.source_canvas,
            self.masks,
            layer_polygons=self.layer_polygons,
            background_polygons=self.background_polygons,
        )
        if show_status:
            polygon_dir = _step003_final_dir(stem)
            self.status = (
                f"Saved masks + overlay -> {out_dir.relative_to(_REPO_ROOT)}; "
                f"polygons -> {polygon_dir.relative_to(_ROOT)}"
            )

    def _autosave(self) -> None:
        self._persist_labels(show_status=False)

    def _fit_polygons_from_masks(self) -> None:
        min_area = _min_area_px(self.cfg)
        for layer in STRUCTURE_LAYERS:
            if layer == "wall":
                continue
            self.layer_polygons[layer] = fit_polygons_from_mask(
                layer,
                self.masks[layer],
                size=self.canvas_size,
                min_area_px=min_area,
            )

    def _refit_active_layer_from_mask(self) -> None:
        min_area = _min_area_px(self.cfg)
        layer = self._editable_polygon_layer()
        if layer is None:
            return
        if layer == "background":
            self.background_polygons = fit_polygons_from_mask(
                "room",
                self.masks["background"],
                size=self.canvas_size,
                min_area_px=min_area,
            )
        else:
            self.layer_polygons[layer] = fit_polygons_from_mask(
                layer,
                self.masks[layer],
                size=self.canvas_size,
                min_area_px=min_area,
            )
        self._clear_polygon_selection()

    def _update_tool_ui(self) -> None:
        polygon_tool = self.edit_tool == "polygon"
        self.tool_btn.label = f"Tool: {self.edit_tool}"
        self.brush_mode_btn.visible = not polygon_tool
        polygon_edit_layer = self._editable_polygon_layer()
        self.poly_edit_dropdown.visible = polygon_edit_layer is not None and polygon_tool
        if not polygon_tool:
            self.poly_edit_dropdown.open = False

    def _finish_brush_stroke(self) -> None:
        self._refit_active_layer_from_mask()
        self._autosave()
        self._rebuild_display()

    def _load_polygons_for_stem(self, stem: str) -> None:
        image_path = self.test_images[self.selected_index]
        loaded = load_step003_polygons(
            self.cfg,
            image_path,
            full_canvas=self.source_canvas,
        )
        if loaded is not None:
            self.masks, self.layer_polygons, self.background_polygons = loaded
            return
        self._fit_polygons_from_masks()
        self.layer_polygons["wall"] = []
        self._sync_masks_from_polygons()

    def _load_test_image(self, index: int, *, initial: bool = False) -> None:
        path = self.test_images[index]
        self.selected_index = index
        if not initial:
            self._ensure_list_visible(index)
        stem = path.stem
        self.source_canvas = _step003_source_canvas(stem, path, canvas_size=self.canvas_size)
        saved_masks = load_saved_masks(stem, self.canvas_size)
        if saved_masks is not None:
            self.masks = saved_masks
        else:
            self.masks = empty_masks(self.canvas_size)
        self.layer_polygons = self._empty_layer_polygons()
        self.background_polygons = []
        self._load_polygons_for_stem(stem)
        step_dir = _step003_final_dir(stem)
        if step_dir.is_dir():
            self.status = f"Loaded step-003 LLM polygons — {path.name}"
        elif saved_masks is not None:
            self.status = f"Loaded saved labels — {path.name}"
        else:
            self.status = f"Loaded {path.name} — infer or edit polygons."
        self.painting = False
        self.polygon_draft = None
        self._clear_polygon_selection()
        self.poly_list_scroll = 0
        self.point_list_scroll = 0
        if not initial:
            self._rebuild_display()

    def _rebuild_display(self, *, polygon_cursor: tuple[int, int] | None = None) -> None:
        composed = compose_overlay(
            self.source_canvas,
            self.masks,
            active_layer=self.active_layer,
            edit_tool=self.edit_tool,
            layer_polygons=self.layer_polygons,
            background_polygons=self.background_polygons,
            polygon_draft=self.polygon_draft,
            polygon_cursor=polygon_cursor,
            selected_poly_index=self.selected_poly_index,
            selected_vertex_index=self.selected_vertex_index,
        )
        surface = _pil_to_surface(composed)
        self.display_surface = pygame.transform.smoothscale(surface, (self.display_w, self.display_h))
        self.display_rect = pygame.Rect(self.viewer_x, 0, self.display_w, self.display_h)

    def _display_scale(self) -> float:
        return self.display_w / self.canvas_size

    def _canvas_point(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        if not self.display_rect.collidepoint(pos):
            return None
        rel_x = pos[0] - self.display_rect.x
        rel_y = pos[1] - self.display_rect.y
        x = int(rel_x * self.canvas_size / self.display_w)
        y = int(rel_y * self.canvas_size / self.display_h)
        x = max(0, min(self.canvas_size - 1, x))
        y = max(0, min(self.canvas_size - 1, y))
        return x, y

    def _draw_brush_cursor(self, pos: tuple[int, int]) -> None:
        if self.running_inference or not self.display_rect.collidepoint(pos):
            return
        radius_px = max(1, round(self.brush_radius * self._display_scale()))
        color = LAYER_COLORS[self.active_layer]
        if self.brush_mode == "remove":
            color = (255, 90, 90)
        old_clip = self.window.get_clip()
        self.window.set_clip(self.display_rect)
        pygame.draw.circle(self.window, color, pos, radius_px, 2)
        pygame.draw.circle(self.window, color, pos, 2)
        self.window.set_clip(old_clip)

    def _paint_at(self, pos: tuple[int, int]) -> None:
        point = self._canvas_point(pos)
        if point is None:
            return
        cx, cy = point
        layer = self.active_layer
        mask = self.masks[layer]
        radius = self.brush_radius
        y0 = max(0, cy - radius)
        y1 = min(self.canvas_size, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(self.canvas_size, cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        if self.brush_mode == "add":
            mask[y0:y1, x0:x1][disk] = True
        else:
            mask[y0:y1, x0:x1][disk] = False

    def _set_layer(self, layer: MaskLayer) -> None:
        self.active_layer = layer
        self.polygon_draft = None
        self._clear_polygon_selection()
        self.poly_list_scroll = 0
        self.point_list_scroll = 0
        if layer == "wall" and self.edit_tool == "polygon":
            self.edit_tool = "brush"
            self._update_tool_ui()
        editing = (
            "polygons"
            if self.edit_tool == "polygon" and layer in POLYGON_EDIT_LAYERS
            else "mask"
        )
        self.status = f"Editing {layer} {editing}."
        self._rebuild_display()

    def _toggle_brush_mode(self) -> None:
        self.brush_mode = "remove" if self.brush_mode == "add" else "add"
        self.brush_mode_btn.label = f"Brush: {self.brush_mode}"
        self.status = f"Brush mode: {self.brush_mode}."

    def _toggle_edit_tool(self) -> None:
        if self.edit_tool == "brush":
            if self.active_layer == "wall":
                self.status = "Wall layer — brush only (no polygon edit)."
                return
            self.edit_tool = "polygon"
            self.painting = False
            self._set_polygon_edit_mode("select_point")
            self.status = "Polygon tool — click to place vertices, Enter or first point to close."
        else:
            self.edit_tool = "brush"
            self.polygon_draft = None
            self.dragging_vertex_index = None
            self._clear_polygon_selection()
            self.status = "Brush tool — paint on the image."
        self._update_tool_ui()

    def _cancel_polygon_draft(self) -> None:
        if not self.polygon_draft:
            return
        self.polygon_draft = None
        self.dragging_vertex_index = None
        self.status = "Polygon draft cancelled."

    def _commit_polygon_draft(self) -> bool:
        if not self.polygon_draft or len(self.polygon_draft) < 3:
            self.status = "Need at least 3 vertices to close a polygon."
            return False
        layer = self.active_layer
        ring = list(self.polygon_draft)
        vertex_count = len(ring)
        if layer not in POLYGON_EDIT_LAYERS:
            self.status = f"Cannot add polygon on {layer} layer."
            return False
        if layer == "background":
            self.background_polygons.append(ring)
            self._select_polygon(len(self.background_polygons) - 1)
        else:
            self.layer_polygons[layer].append(ring)
            self._select_polygon(len(self.layer_polygons[layer]) - 1)
        self._sync_masks_from_polygons()
        self.status = f"Added {layer} polygon ({vertex_count} vertices)."
        self.polygon_draft = None
        self.dragging_vertex_index = None
        if self.polygon_edit_mode == "new_polygon":
            self.polygon_edit_mode = "select_point"
            self.poly_edit_dropdown.selected = "select_point"
            self.poly_edit_dropdown.open = False
            self.status = f"{self.status} Switched to select point."
        self._autosave()
        self._rebuild_display()
        return True

    def _close_polygon_draft(self) -> bool:
        return self._commit_polygon_draft()

    def _add_polygon_vertex(self, x: int, y: int) -> None:
        if self.polygon_draft is None:
            self.polygon_draft = []
        if len(self.polygon_draft) >= 3:
            first_x, first_y = self.polygon_draft[0]
            if (x - first_x) ** 2 + (y - first_y) ** 2 <= POLYGON_CLOSE_RADIUS ** 2:
                self._close_polygon_draft()
                return
        self.polygon_draft.append((x, y))
        self.status = (
            f"Vertex {len(self.polygon_draft)} — click start point or Enter to close."
        )

    def _pick_polygon_at(self, point: tuple[int, int]) -> int | None:
        if self._editable_polygon_layer() is None:
            return None
        px, py = point
        polygons = self._active_polygon_list()
        for index in range(len(polygons) - 1, -1, -1):
            if _point_in_polygon(px, py, polygons[index]):
                return index
        return None

    def _delete_selected_vertex(self) -> None:
        layer = self._editable_polygon_layer()
        if layer is None or self.selected_poly_index is None:
            return
        polygons = self._active_polygon_list()
        ring = polygons[self.selected_poly_index]
        if self.selected_vertex_index is None:
            del polygons[self.selected_poly_index]
            self._sync_masks_from_polygons()
            self._clear_polygon_selection()
            self.status = f"Deleted polygon from {layer}."
            self._autosave()
            self._rebuild_display()
            return
        if len(ring) <= 3:
            del polygons[self.selected_poly_index]
            self._sync_masks_from_polygons()
            self._clear_polygon_selection()
            self.status = f"Deleted polygon from {layer} (too few points)."
        else:
            del ring[self.selected_vertex_index]
            self.selected_vertex_index = None
            self._sync_masks_from_polygons()
            self.status = f"Deleted vertex from polygon {self.selected_poly_index + 1}."
        self._autosave()
        self._rebuild_display()

    def _insert_vertex_on_selected_polygon(self, point: tuple[int, int]) -> bool:
        if self._editable_polygon_layer() is None or self.selected_poly_index is None:
            return False
        ring = self._active_polygon_list()[self.selected_poly_index]
        insert_at = _nearest_edge_insert_index(point, ring)
        if insert_at is None:
            return False
        ring.insert(insert_at, point)
        self.selected_vertex_index = insert_at
        self._sync_masks_from_polygons()
        self.status = f"Inserted point at P{insert_at}."
        self._autosave()
        self._rebuild_display()
        return True

    def _handle_select_point_click(self, point: tuple[int, int]) -> None:
        if self._editable_polygon_layer() is None:
            return
        if self.selected_poly_index is not None:
            ring = self._active_polygon_list()[self.selected_poly_index]
            vertex_index = _find_vertex_near(point, ring, radius=POLYGON_HIT_RADIUS)
            if vertex_index is not None:
                self._select_vertex(self.selected_poly_index, vertex_index)
                self.dragging_vertex_index = vertex_index
                self._rebuild_display()
                return
        poly_index = self._pick_polygon_at(point)
        if poly_index is not None:
            self._select_polygon(poly_index)
            ring = self._active_polygon_list()[poly_index]
            vertex_index = _find_vertex_near(point, ring, radius=POLYGON_HIT_RADIUS)
            if vertex_index is not None:
                self.selected_vertex_index = vertex_index
                self.dragging_vertex_index = vertex_index
            self._rebuild_display()
            return
        self._select_polygon(None)
        self._rebuild_display()

    def _handle_add_point_click(self, point: tuple[int, int]) -> None:
        if self._editable_polygon_layer() is None:
            return
        if self.selected_poly_index is not None:
            ring = self._active_polygon_list()[self.selected_poly_index]
            if self._insert_vertex_on_selected_polygon(point):
                return
            vertex_index = _find_vertex_near(point, ring, radius=POLYGON_HIT_RADIUS)
            if vertex_index is not None:
                self._select_vertex(self.selected_poly_index, vertex_index)
                self.dragging_vertex_index = vertex_index
                self._rebuild_display()
                return
        poly_index = self._pick_polygon_at(point)
        if poly_index is not None:
            self._select_polygon(poly_index)
            ring = self._active_polygon_list()[poly_index]
            if self._insert_vertex_on_selected_polygon(point):
                return
            vertex_index = _find_vertex_near(point, ring, radius=POLYGON_HIT_RADIUS)
            if vertex_index is not None:
                self.selected_vertex_index = vertex_index
                self.dragging_vertex_index = vertex_index
            else:
                self.status = "Add point — click near a polygon edge to insert."
            self._rebuild_display()
            return
        self.status = "Add point — select a polygon first."

    def _handle_new_polygon_click(self, point: tuple[int, int]) -> None:
        if self.polygon_draft:
            vertex_index = _find_vertex_near(point, self.polygon_draft, radius=POLYGON_HIT_RADIUS)
            if vertex_index is not None:
                self.selected_vertex_index = vertex_index
                self.dragging_vertex_index = vertex_index
                return
            self.selected_vertex_index = None
            self._add_polygon_vertex(*point)
            self._rebuild_display()
            return
        self.selected_vertex_index = None
        self._add_polygon_vertex(*point)
        self._rebuild_display()

    def _handle_polygon_click(self, pos: tuple[int, int]) -> None:
        point = self._canvas_point(pos)
        if point is None:
            return
        if self.polygon_edit_mode == "select_point":
            self._handle_select_point_click(point)
        elif self.polygon_edit_mode == "add_point":
            self._handle_add_point_click(point)
        else:
            self._handle_new_polygon_click(point)

    def _nudge_selected_vertex(self, dx: int, dy: int) -> bool:
        if self.selected_vertex_index is None:
            return False
        index = self.selected_vertex_index
        max_coord = self.canvas_size - 1

        if self.polygon_draft and 0 <= index < len(self.polygon_draft):
            x, y = self.polygon_draft[index]
            self.polygon_draft[index] = (
                max(0, min(max_coord, x + dx)),
                max(0, min(max_coord, y + dy)),
            )
            x, y = self.polygon_draft[index]
            self.status = f"Moved P{index} to ({x}, {y})."
            self._rebuild_display()
            return True

        if self._editable_polygon_layer() is None or self.selected_poly_index is None:
            return False
        ring = self._active_polygon_list()[self.selected_poly_index]
        if index < 0 or index >= len(ring):
            return False
        x, y = ring[index]
        ring[index] = (
            max(0, min(max_coord, x + dx)),
            max(0, min(max_coord, y + dy)),
        )
        x, y = ring[index]
        self._sync_masks_from_polygons()
        self.status = f"Moved P{index} to ({x}, {y})."
        self._autosave()
        self._rebuild_display()
        return True

    def _handle_polygon_drag(self, pos: tuple[int, int]) -> None:
        if self.dragging_vertex_index is None:
            return
        point = self._canvas_point(pos)
        if point is None:
            return
        if self.polygon_draft:
            self.polygon_draft[self.dragging_vertex_index] = point
            self._rebuild_display()
            return
        if self._editable_polygon_layer() is None or self.selected_poly_index is None:
            return
        ring = self._active_polygon_list()[self.selected_poly_index]
        ring[self.dragging_vertex_index] = point
        self.selected_vertex_index = self.dragging_vertex_index
        self._sync_masks_from_polygons()
        self._rebuild_display()

    def _on_compute_wall(self) -> None:
        if self.running_inference:
            return
        if not self.background_polygons:
            self.status = "Need a floorplan boundary on the background layer."
            return
        self.layer_polygons["wall"] = []
        self.masks["wall"] = compute_wall_mask_from_boundary_and_rooms(
            self.background_polygons,
            self.layer_polygons["room"],
            size=self.canvas_size,
        )
        self.status = "Computed wall mask from background − rooms."
        self._autosave()
        self._rebuild_display()

    def _on_infer(self) -> None:
        if self.running_inference:
            return
        path = self.test_images[self.selected_index]
        self.running_inference = True
        self.status = f"Running FPN inference on {path.name}..."
        self._draw()
        pygame.display.flip()
        try:
            _canvas, masks, polygons, background_polygons = infer_processed_from_pipeline(
                self.cfg,
                path,
            )
            self.source_canvas = _canvas
            self.masks = masks
            self.layer_polygons = polygons
            self.background_polygons = background_polygons
            self._clear_polygon_selection()
            self.polygon_draft = None
            self.status = (
                f"Loaded step-003 LLM polygons — edit {self.active_layer}, then save."
            )
            self._autosave()
            self._rebuild_display()
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
        finally:
            self.running_inference = False

    def _on_save(self) -> None:
        if self.running_inference:
            return
        if self.polygon_draft and len(self.polygon_draft) >= 3:
            self._commit_polygon_draft()
            return
        self._persist_labels(show_status=True)

    def _list_visible_count(self) -> int:
        return max(1, (self.window_h - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT)

    def _max_list_scroll(self) -> int:
        return max(0, len(self.test_images) - self._list_visible_count())

    def _ensure_list_visible(self, index: int) -> None:
        if index < self.list_scroll:
            self.list_scroll = index
        elif index >= self.list_scroll + self._list_visible_count():
            self.list_scroll = index - self._list_visible_count() + 1
        self.list_scroll = max(0, min(self._max_list_scroll(), self.list_scroll))

    def _list_index_at(self, pos: tuple[int, int]) -> int | None:
        mx, my = pos
        if mx >= self.list_w or my < LIST_HEADER_HEIGHT:
            return None
        row = (my - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT
        index = self.list_scroll + row
        if index < 0 or index >= len(self.test_images):
            return None
        return index

    def _handle_click(self, pos: tuple[int, int]) -> None:
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_test_image(list_index)
            return
        if self.infer_btn.hit(pos):
            self._on_infer()
            return
        if self.save_btn.hit(pos):
            self._on_save()
            return
        if self.compute_wall_btn.hit(pos):
            self._on_compute_wall()
            return
        if self.tool_btn.hit(pos):
            self._toggle_edit_tool()
            return
        if self.brush_mode_btn.hit(pos):
            self._toggle_brush_mode()
            return
        if self._handle_polygon_edit_dropdown_click(pos):
            return
        if self._handle_polygon_panel_click(pos):
            return
        for layer, button in self.layer_buttons.items():
            if button.hit(pos):
                self._set_layer(layer)
                return
        if self.display_rect.collidepoint(pos) and not self.running_inference:
            if self.edit_tool == "polygon":
                self._handle_polygon_click(pos)
                return
            self.painting = True
            self._paint_at(pos)
            self._rebuild_display()

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, self.list_w, self.window_h))
        header = self.label_font.render("real_floorplan/", True, LABEL_COLOR)
        self.window.blit(header, (8, 4))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.list_w - 1, 0),
            (self.list_w - 1, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        for row in range(self._list_visible_count()):
            index = self.list_scroll + row
            if index >= len(self.test_images):
                break
            y = LIST_HEADER_HEIGHT + row * LIST_ITEM_HEIGHT
            rect = pygame.Rect(0, y, self.list_w, LIST_ITEM_HEIGHT)
            if index == self.selected_index:
                pygame.draw.rect(self.window, LIST_SELECTED, rect)
            elif rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, rect)
            label = self.test_images[index].name
            if self.font.size(label)[0] > self.list_w - 12:
                while label and self.font.size(label + "…")[0] > self.list_w - 12:
                    label = label[:-1]
                label += "…"
            text = self.font.render(label, True, LIST_TEXT)
            self.window.blit(text, (8, y + 3))

    def _draw_viewer(self) -> None:
        if self.edit_tool == "polygon" and self._editable_polygon_layer() is not None:
            if self.active_layer == "background":
                component_count = len(self.background_polygons)
            else:
                component_count = len(self.layer_polygons[self.active_layer])
            draft_count = len(self.polygon_draft or [])
            mode_label = self.poly_edit_dropdown.label_for(self.polygon_edit_mode)
            layer_label = (
                f"{self.active_layer} ({mode_label}, "
                f"{component_count} polys, draft {draft_count} pts)"
            )
        else:
            layer_label = f"{self.active_layer} (brush {self.brush_mode}, r={self.brush_radius})"
        self.window.blit(
            self.label_font.render(layer_label, True, LAYER_COLORS[self.active_layer]),
            (self.viewer_x, 6),
        )
        if self.display_surface is not None:
            self.window.blit(self.display_surface, self.display_rect.topleft)
            pygame.draw.rect(self.window, (70, 70, 70), self.display_rect, 1)
            if self.edit_tool == "brush":
                self._draw_brush_cursor(pygame.mouse.get_pos())

    def _draw_side_panel(self) -> None:
        pygame.draw.rect(self.window, SIDE_BG, (self.panel_x, 0, self.panel_w, self.window_h))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        disabled = self.running_inference
        self.infer_btn.draw(
            self.window,
            self.btn_font,
            hover=self.infer_btn.hit(mouse) and not disabled,
        )
        self.save_btn.draw(
            self.window,
            self.btn_font,
            hover=self.save_btn.hit(mouse) and not disabled,
        )
        self.compute_wall_btn.draw(
            self.window,
            self.btn_font,
            hover=self.compute_wall_btn.hit(mouse) and not disabled,
        )
        for layer, button in self.layer_buttons.items():
            button.draw(
                self.window,
                self.btn_font,
                hover=button.hit(mouse) and not disabled,
                active=layer == self.active_layer,
            )
        self.tool_btn.draw(
            self.window,
            self.btn_font,
            hover=self.tool_btn.hit(mouse) and not disabled,
            active=self.edit_tool == "polygon",
        )
        if self.brush_mode_btn.visible:
            self.brush_mode_btn.draw(
                self.window,
                self.btn_font,
                hover=self.brush_mode_btn.hit(mouse) and not disabled,
                active=self.brush_mode == "add",
            )
        self.poly_edit_dropdown.draw(
            self.window,
            self.btn_font,
            hover=self.poly_edit_dropdown.hit_button(mouse) and not disabled,
        )
        if self.edit_tool == "polygon" and self._editable_polygon_layer() is not None:
            self._draw_polygon_panel_lists()
        brush_text = self.small_font.render(
            f"Brush size: {self.brush_radius}px  ([ ] to resize)",
            True,
            LABEL_COLOR,
        )
        self.window.blit(brush_text, (self.panel_x + 10, self.brush_size_label_y))
        hint = self.small_font.render(
            "P cycle edit · Arrows nudge point · Del · Enter · Esc",
            True,
            LABEL_COLOR,
        )
        self.window.blit(hint, (self.panel_x + 10, self.brush_size_label_y + 18))

        status = self.status
        if self.running_inference:
            status = f"{status} (please wait)"
        wrapped = self._wrap_text(status, self.panel_w - 20)
        y = self.status_y
        for line in wrapped:
            text = self.font.render(line, True, STATUS_COLOR)
            self.window.blit(text, (self.panel_x + 10, y))
            y += 18
        self._draw_model_address()

    def _polygon_panel_rows(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        poly_rows: list[tuple[str, str]] = []
        point_rows: list[tuple[str, str]] = []
        layer = self._editable_polygon_layer()
        if layer is None:
            return poly_rows, point_rows
        for index, ring in enumerate(self._active_polygon_list()):
            poly_rows.append((f"poly-{index}", f"Poly {index + 1} · {len(ring)} pts"))
        if self.selected_poly_index is not None:
            ring = self._active_polygon_list()[self.selected_poly_index]
            for index, (x, y) in enumerate(ring):
                point_rows.append((f"pt-{index}", f"P{index}: {x}, {y}"))
        return poly_rows, point_rows

    def _polygon_list_visible_rows(self) -> int:
        model_block = 50
        available = self.window_h - self.polygon_list_y - 120 - model_block
        return max(3, available // POLY_LIST_ROW_HEIGHT)

    def _polygon_panel_capacities(
        self,
        poly_count: int,
        point_count: int,
    ) -> tuple[int, int]:
        budget = self._polygon_list_visible_rows()
        if point_count <= 0:
            return budget, 0
        poly_cap = max(2, min(poly_count, budget // 3))
        point_cap = max(3, budget - poly_cap)
        return poly_cap, point_cap

    def _ensure_point_visible(self, point_index: int) -> None:
        poly_rows, point_rows = self._polygon_panel_rows()
        _, point_cap = self._polygon_panel_capacities(len(poly_rows), len(point_rows))
        if point_cap <= 0:
            return
        max_scroll = max(0, len(point_rows) - point_cap)
        if point_index < self.point_list_scroll:
            self.point_list_scroll = point_index
        elif point_index >= self.point_list_scroll + point_cap:
            self.point_list_scroll = point_index - point_cap + 1
        self.point_list_scroll = max(0, min(max_scroll, self.point_list_scroll))

    def _draw_polygon_panel_lists(self) -> None:
        layer = self._editable_polygon_layer()
        self._poly_list_hit_rect = None
        self._point_list_hit_rect = None
        if layer is None:
            return
        x = self.panel_x + 10
        width = self.panel_w - 20
        y = self.polygon_list_y
        header = self.label_font.render(f"Polygons ({layer})", True, LABEL_COLOR)
        self.window.blit(header, (x, y))
        y += POLY_LIST_HEADER_HEIGHT
        poly_rows, point_rows = self._polygon_panel_rows()
        poly_cap, point_cap = self._polygon_panel_capacities(len(poly_rows), len(point_rows))
        mouse = pygame.mouse.get_pos()
        max_poly_scroll = max(0, len(poly_rows) - poly_cap)
        self.poly_list_scroll = max(0, min(max_poly_scroll, self.poly_list_scroll))
        poly_list_top = y
        for row in range(poly_cap):
            index = self.poly_list_scroll + row
            if index >= len(poly_rows):
                break
            row_id, label = poly_rows[index]
            rect = pygame.Rect(x, y, width, POLY_LIST_ROW_HEIGHT - 2)
            poly_index = int(row_id.split("-")[1])
            if poly_index == self.selected_poly_index:
                pygame.draw.rect(self.window, LIST_SELECTED, rect)
            elif rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, rect)
            text = self.small_font.render(label, True, LIST_TEXT)
            self.window.blit(text, (x + 4, y + 2))
            y += POLY_LIST_ROW_HEIGHT
        if poly_cap > 0:
            self._poly_list_hit_rect = pygame.Rect(x, poly_list_top, width, y - poly_list_top)
        if point_rows:
            y += 4
            points_header_y = y
            sub_header = self.label_font.render("Points", True, LABEL_COLOR)
            self.window.blit(sub_header, (x, y))
            y += POLY_LIST_HEADER_HEIGHT
            max_point_scroll = max(0, len(point_rows) - point_cap)
            self.point_list_scroll = max(0, min(max_point_scroll, self.point_list_scroll))
            point_list_top = y
            for row in range(point_cap):
                index = self.point_list_scroll + row
                if index >= len(point_rows):
                    break
                row_id, label = point_rows[index]
                rect = pygame.Rect(x, y, width, POLY_LIST_ROW_HEIGHT - 2)
                point_index = int(row_id.split("-")[1])
                if point_index == self.selected_vertex_index:
                    pygame.draw.rect(self.window, LIST_SELECTED, rect)
                elif rect.collidepoint(mouse):
                    pygame.draw.rect(self.window, LIST_HOVER, rect)
                text = self.small_font.render(label, True, LIST_TEXT)
                self.window.blit(text, (x + 4, y + 2))
                y += POLY_LIST_ROW_HEIGHT
            self._point_list_hit_rect = pygame.Rect(
                x,
                points_header_y,
                width,
                y - points_header_y,
            )
        self.status_y = min(y + 8, self.window_h - 120)

    def _handle_polygon_panel_click(self, pos: tuple[int, int]) -> bool:
        if self._editable_polygon_layer() is None:
            return False
        x = self.panel_x + 10
        width = self.panel_w - 20
        y = self.polygon_list_y + POLY_LIST_HEADER_HEIGHT
        if pos[0] < x or pos[0] > x + width or pos[1] < self.polygon_list_y:
            return False
        poly_rows, point_rows = self._polygon_panel_rows()
        poly_cap, point_cap = self._polygon_panel_capacities(len(poly_rows), len(point_rows))
        for row in range(poly_cap):
            index = self.poly_list_scroll + row
            if index >= len(poly_rows):
                break
            rect = pygame.Rect(x, y, width, POLY_LIST_ROW_HEIGHT - 2)
            if rect.collidepoint(pos):
                self._select_polygon(index)
                self._rebuild_display()
                return True
            y += POLY_LIST_ROW_HEIGHT
        if point_rows:
            y += 4 + POLY_LIST_HEADER_HEIGHT
            for row in range(point_cap):
                index = self.point_list_scroll + row
                if index >= len(point_rows):
                    break
                row_id, _label = point_rows[index]
                rect = pygame.Rect(x, y, width, POLY_LIST_ROW_HEIGHT - 2)
                if rect.collidepoint(pos):
                    point_index = int(row_id.split("-")[1])
                    if self.selected_poly_index is not None:
                        self._select_vertex(self.selected_poly_index, point_index)
                        self._rebuild_display()
                    return True
                y += POLY_LIST_ROW_HEIGHT
        return False

    def _draw_model_address(self) -> None:
        path_lines = self._wrap_text(self.model_address, self.panel_w - 20, font=self.small_font)
        line_h = 16
        block_h = 18 + len(path_lines) * line_h
        y = self.window_h - block_h - 8
        header = self.label_font.render("Model", True, LABEL_COLOR)
        self.window.blit(header, (self.panel_x + 10, y))
        y += 18
        for line in path_lines:
            text = self.small_font.render(line, True, LABEL_COLOR)
            self.window.blit(text, (self.panel_x + 10, y))
            y += line_h

    def _wrap_text(
        self,
        text: str,
        max_width: int,
        *,
        font: pygame.font.Font | None = None,
    ) -> list[str]:
        if not text:
            return [""]
        font = font or self.font
        lines: list[str] = []
        current = text
        while current:
            if font.size(current)[0] <= max_width:
                lines.append(current)
                break
            cut = len(current)
            while cut > 1 and font.size(current[:cut] + "…")[0] > max_width:
                cut -= 1
            lines.append(current[:cut] + "…")
            current = current[cut:].lstrip()
        return lines

    def _draw(self) -> None:
        self.window.fill((18, 18, 18))
        self._draw_file_list()
        self._draw_viewer()
        self._draw_side_panel()

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            polygon_cursor: tuple[int, int] | None = None
            mouse = pygame.mouse.get_pos()
            if (
                self.edit_tool == "polygon"
                and self.polygon_draft
                and self.display_rect.collidepoint(mouse)
            ):
                point = self._canvas_point(mouse)
                if point is not None:
                    polygon_cursor = point
                    self._rebuild_display(polygon_cursor=polygon_cursor)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.polygon_draft:
                            self._cancel_polygon_draft()
                            self._rebuild_display()
                        else:
                            running = False
                    elif event.key == pygame.K_RETURN:
                        self._close_polygon_draft()
                    elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        if (
                            self.polygon_edit_mode == "select_point"
                            and self.selected_poly_index is not None
                        ):
                            self._delete_selected_vertex()
                        elif self.polygon_draft:
                            if self.selected_vertex_index is not None and self.polygon_draft:
                                del self.polygon_draft[self.selected_vertex_index]
                                self.selected_vertex_index = None
                                self.dragging_vertex_index = None
                                if not self.polygon_draft:
                                    self.polygon_draft = None
                            else:
                                self.polygon_draft.pop()
                            if not self.polygon_draft:
                                self.polygon_draft = None
                                self.status = "Polygon draft cleared."
                            else:
                                self.status = (
                                    f"Vertex {len(self.polygon_draft)} — keep placing points."
                                )
                            self._rebuild_display()
                    elif event.key == pygame.K_p:
                        if self.edit_tool == "polygon":
                            self._cycle_polygon_edit_mode()
                    elif event.key == pygame.K_t:
                        self._toggle_edit_tool()
                        self._rebuild_display()
                    elif event.key == pygame.K_a:
                        self.brush_mode = "add"
                        self.brush_mode_btn.label = "Brush: add"
                    elif event.key == pygame.K_r:
                        self.brush_mode = "remove"
                        self.brush_mode_btn.label = "Brush: remove"
                    elif event.key == pygame.K_LEFT:
                        self._nudge_selected_vertex(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self._nudge_selected_vertex(1, 0)
                    elif event.key == pygame.K_UP:
                        self._nudge_selected_vertex(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self._nudge_selected_vertex(0, 1)
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.brush_radius = max(MIN_BRUSH_RADIUS, self.brush_radius - 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.brush_radius = min(MAX_BRUSH_RADIUS, self.brush_radius + 2)
                    elif event.key in (
                        pygame.K_1,
                        pygame.K_2,
                        pygame.K_3,
                        pygame.K_4,
                        pygame.K_5,
                    ):
                        layer_index = event.key - pygame.K_1
                        if layer_index < len(MASK_LAYERS):
                            self._set_layer(MASK_LAYERS[layer_index])
                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if mx < self.list_w:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                    elif (
                        self.panel_x <= mx < self.panel_x + self.panel_w
                        and my >= self.polygon_list_y
                    ):
                        poly_rows, point_rows = self._polygon_panel_rows()
                        poly_cap, point_cap = self._polygon_panel_capacities(
                            len(poly_rows),
                            len(point_rows),
                        )
                        if (
                            self._point_list_hit_rect is not None
                            and self._point_list_hit_rect.collidepoint(mx, my)
                        ):
                            max_scroll = max(0, len(point_rows) - point_cap)
                            self.point_list_scroll = max(
                                0,
                                min(max_scroll, self.point_list_scroll - event.y),
                            )
                        else:
                            max_scroll = max(0, len(poly_rows) - poly_cap)
                            self.poly_list_scroll = max(
                                0,
                                min(max_scroll, self.poly_list_scroll - event.y),
                            )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    was_painting = self.painting
                    was_dragging_vertex = self.dragging_vertex_index is not None
                    self.painting = False
                    if was_dragging_vertex:
                        self._autosave()
                    self.dragging_vertex_index = None
                    if was_painting and self.edit_tool == "brush":
                        self._finish_brush_stroke()
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_vertex_index is not None:
                        self._handle_polygon_drag(event.pos)
                    elif self.painting and self.edit_tool == "brush":
                        self._paint_at(event.pos)
                        self._rebuild_display()

            self._draw()
            pygame.display.flip()
            clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual FPN structure mask labeling UI")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run processed inference and save masks/polygons for all source images (no UI)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --auto, re-run inference even when outputs already exist",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    if args.auto:
        raise SystemExit(auto_process_all_images(cfg, test_images, skip_existing=not args.force))

    pygame.init()
    print(f"Ready — {len(test_images)} test image(s), dataset root: {DATASET_ROOT}")
    ui = LabelDataUI(cfg, test_images)
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
