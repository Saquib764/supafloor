#!/usr/bin/env python3
"""Pygame UI to evaluate FPN step-001 polygon inference against ground truth.

Left panel lists test_floorplan/ files; click to run step-001 inference and compare
to GT in output/test_data/<stem>/step-001/final-masks/ (1024 px). Predictions are written to
eval_floorplan/<stem>/step-001/ (FPN canvas_size, default 1000 px). Row 1 shows GT vs FPN
overlays on the FPN canvas; row 2 shows image-only LLM extract (1024 px). Area metrics map
FPN polygons to 1024 px before comparing to GT.

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python infer_ui_eval.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_placement, fit_to_canvas_rgb
from util.final_mask_polygons import PolygonRing, load_layer_polygons
from util.floorplan_image_llm import extract_floorplan_from_image, load_floorplan_extract
from util.room_components import annotate_room_labels, render_combined_contour_overlay

_ROOT = Path(__file__).resolve().parent
TEST_DIR = _ROOT / "test_floorplan"
EVAL_OUTPUT_DIR = _ROOT / "eval_floorplan"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LIST_WIDTH = 200
PANEL_WIDTH = 400
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
VIEWER_GAP = 8
VIEWER_ROW_GAP = 8
VIEWER_LABEL_HEIGHT = 18
VIEWER_SLOT_SIZE = 400
BTN_HEIGHT = 32
BTN_GAP = 6
PANEL_BTN_TOP = 10
ROOM_ROW_HEIGHT = 16
ROOM_SECTION_HEADER_HEIGHT = 22
ROOM_NAME_ROW_HEIGHT = 16
LLM_EXTRACT_SIZE = 1024
GT_CANVAS_SIZE = 1024  # GT polygon storage; FPN pred mapped here for area comparison

SIDE_BG = (28, 28, 28)
LIST_BG = (22, 22, 22)
LIST_HOVER = (45, 45, 45)
LIST_SELECTED = (60, 80, 120)
LIST_TEXT = (220, 220, 220)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
STATUS_COLOR = (120, 180, 255)
PANEL_LABEL_COLOR = (200, 200, 200)
EMPTY_SLOT_BG = (32, 32, 32)
EMPTY_SLOT_BORDER = (70, 70, 70)
MATCH_COLOR = (120, 220, 120)
MISMATCH_COLOR = (255, 120, 120)
STAT_HEADER_COLOR = (180, 180, 180)
MATCH_ERROR_RATIO = 0.05

_BOUNDARY_OUTLINE = (220, 40, 40)
_BOUNDARY_OUTLINE_ALPHA = 255


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _step_dir(base_dir: Path, stem: str) -> Path:
    return base_dir / stem / "step-001"


def _final_masks_dir(base_dir: Path, stem: str) -> Path:
    return _step_dir(base_dir, stem) / "final-masks"


def _llm_extract_dir(stem: str) -> Path:
    return EVAL_OUTPUT_DIR / stem / "llm-extract"


def _inference_timing_path(stem: str) -> Path:
    return _step_dir(EVAL_OUTPUT_DIR, stem) / "timing.json"


def _load_inference_seconds(stem: str) -> float | None:
    path = _inference_timing_path(stem)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    elapsed = payload.get("elapsed_seconds")
    return float(elapsed) if elapsed is not None else None


def _save_inference_seconds(stem: str, elapsed_seconds: float) -> None:
    path = _inference_timing_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"elapsed_seconds": round(elapsed_seconds, 3)}, indent=2),
        encoding="utf-8",
    )


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "(none)"
    return f"{seconds:.1f}s"


def _format_cost_usd(cost_usd: float) -> str:
    return "$0.00" if cost_usd <= 0 else f"${cost_usd:.4f}"


def load_room_names_meta(step_dir: Path) -> dict[str, object] | None:
    path = step_dir / "room-names.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def load_room_names(step_dir: Path) -> list[dict[str, object]]:
    payload = load_room_names_meta(step_dir)
    if payload is None:
        return []
    rooms = payload.get("rooms")
    return rooms if isinstance(rooms, list) else []


def _room_names_is_local(meta: dict[str, object]) -> bool:
    backend = str(meta.get("backend") or "").lower()
    if backend == "paddle":
        return True
    if backend == "llm":
        return False
    return "llm_size" not in meta


def _room_names_label(meta: dict[str, object] | None) -> str:
    if meta is None:
        return "Room names"
    if _room_names_is_local(meta):
        model = str(meta.get("model") or "Paddle")
        return f"Room names ({model})"
    model = str(meta.get("model") or "LLM")
    return f"Room names ({model})"


def _room_names_seconds(meta: dict[str, object] | None) -> float | None:
    if meta is None:
        return None
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return None
    elapsed_ms = usage.get("elapsed_ms")
    if elapsed_ms is None:
        return None
    return float(elapsed_ms) / 1000.0


def _room_names_cost(meta: dict[str, object] | None) -> float:
    if meta is None or _room_names_is_local(meta):
        return 0.0
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    return float(usage.get("cost_usd", 0.0))


def polygon_counts(final_dir: Path) -> dict[str, int]:
    if not final_dir.is_dir():
        return {"rooms": 0, "doors": 0, "windows": 0}
    return {
        "rooms": len(load_layer_polygons(final_dir / "room.json")),
        "doors": len(load_layer_polygons(final_dir / "door.json")),
        "windows": len(load_layer_polygons(final_dir / "window.json")),
    }


def _draw_unfilled_boundary(
    draw: ImageDraw.ImageDraw,
    polygon: list[tuple[int, int]],
    *,
    width: int = 3,
) -> None:
    if len(polygon) < 3:
        return
    draw.line(
        [*polygon, polygon[0]],
        fill=(*_BOUNDARY_OUTLINE, _BOUNDARY_OUTLINE_ALPHA),
        width=width,
    )


def _polygon_canvas_size(final_dir: Path) -> int | None:
    for layer in ("room", "door", "window", "background"):
        layer_json = final_dir / f"{layer}.json"
        if not layer_json.is_file():
            continue
        size = json.loads(layer_json.read_text(encoding="utf-8")).get("size")
        if size is not None:
            return int(size)
    return None


def _shared_input_image(
    gt_dir: Path,
    image_path: Path,
    *,
    canvas_size: int,
) -> Image.Image:
    input_path = gt_dir / "input.jpg"
    if input_path.is_file():
        return Image.open(input_path).convert("RGB")
    return fit_to_canvas_rgb(Image.open(image_path), size=canvas_size)


def _map_polygons_to_canvas(
    components: list[PolygonRing],
    *,
    from_size: int,
    to_size: int,
    content_size: tuple[int, int],
) -> list[PolygonRing]:
    if from_size == to_size:
        return components
    orig_w, orig_h = content_size
    paste_from = fit_to_canvas_placement(orig_w, orig_h, size=from_size)
    paste_to = fit_to_canvas_placement(orig_w, orig_h, size=to_size)
    px_f, py_f, cw_f, ch_f = paste_from
    px_t, py_t, cw_t, ch_t = paste_to
    mapped: list[PolygonRing] = []
    for ring in components:
        if len(ring) < 3:
            continue
        mapped.append(
            [
                (
                    int(round(px_t + ((x - px_f) / cw_f) * cw_t)),
                    int(round(py_t + ((y - py_f) / ch_f) * ch_t)),
                )
                for x, y in ring
            ]
        )
    return mapped


def _map_point_to_canvas(
    x: int,
    y: int,
    *,
    from_size: int,
    to_size: int,
    content_size: tuple[int, int],
) -> tuple[int, int]:
    if from_size == to_size:
        return int(x), int(y)
    orig_w, orig_h = content_size
    paste_from = fit_to_canvas_placement(orig_w, orig_h, size=from_size)
    paste_to = fit_to_canvas_placement(orig_w, orig_h, size=to_size)
    px_f, py_f, cw_f, ch_f = paste_from
    px_t, py_t, cw_t, ch_t = paste_to
    return (
        int(round(px_t + ((x - px_f) / cw_f) * cw_t)),
        int(round(py_t + ((y - py_f) / ch_f) * ch_t)),
    )


def _map_names_to_canvas(
    names: list[dict[str, object]],
    *,
    from_size: int,
    to_size: int,
    content_size: tuple[int, int],
) -> list[dict[str, object]]:
    mapped: list[dict[str, object]] = []
    for entry in names:
        position = entry.get("position")
        if not isinstance(position, dict):
            continue
        mx, my = _map_point_to_canvas(
            int(position.get("x", 0)),
            int(position.get("y", 0)),
            from_size=from_size,
            to_size=to_size,
            content_size=content_size,
        )
        mapped.append({"room": entry.get("room"), "position": {"x": mx, "y": my}})
    return mapped


def render_llm_overlay(
    input_image: Image.Image,
    extract: dict[str, object],
    *,
    content_size: tuple[int, int],
    canvas_size: int,
) -> Image.Image | None:
    llm_size = int(extract.get("size", LLM_EXTRACT_SIZE))
    rooms = extract.get("rooms")
    doors = extract.get("doors")
    windows = extract.get("windows")
    boundary = extract.get("boundary")
    names = extract.get("names")
    if not isinstance(rooms, list) or not rooms:
        return None
    map_kwargs = {
        "from_size": llm_size,
        "to_size": canvas_size,
        "content_size": content_size,
    }
    room_polygons = _map_polygons_to_canvas(rooms, **map_kwargs)
    door_polygons = _map_polygons_to_canvas(
        doors if isinstance(doors, list) else [],
        **map_kwargs,
    )
    window_polygons = _map_polygons_to_canvas(
        windows if isinstance(windows, list) else [],
        **map_kwargs,
    )
    boundary_polygon: PolygonRing | None = None
    if isinstance(boundary, list) and len(boundary) >= 3:
        mapped_boundary = _map_polygons_to_canvas([boundary], **map_kwargs)
        boundary_polygon = mapped_boundary[0] if mapped_boundary else None
    overlay = render_combined_contour_overlay(
        input_image,
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        boundary_polygon=boundary_polygon,
    )
    if isinstance(names, list) and names:
        mapped_names = _map_names_to_canvas(
            names,
            from_size=llm_size,
            to_size=canvas_size,
            content_size=content_size,
        )
        return annotate_room_labels(overlay, mapped_names)
    return overlay


def _load_final_mask_polygons(
    final_dir: Path,
    *,
    target_size: int,
    content_size: tuple[int, int],
) -> tuple[list[PolygonRing], list[PolygonRing], list[PolygonRing], list[PolygonRing]]:
    polygon_size = _polygon_canvas_size(final_dir)
    map_kwargs = {}
    if polygon_size is not None and polygon_size != target_size:
        map_kwargs = {
            "from_size": polygon_size,
            "to_size": target_size,
            "content_size": content_size,
        }

    def _load(layer: str) -> list[PolygonRing]:
        polygons = load_layer_polygons(final_dir / f"{layer}.json")
        if map_kwargs:
            return _map_polygons_to_canvas(polygons, **map_kwargs)
        return polygons

    return (
        _load("room"),
        _load("door"),
        _load("window"),
        _load("background"),
    )


def render_eval_overlay(
    final_dir: Path,
    *,
    input_image: Image.Image,
    content_size: tuple[int, int],
) -> Image.Image | None:
    """Background outline only; rooms, doors, and windows filled."""
    if not final_dir.is_dir():
        return None
    target_size = input_image.size[0]
    room_polygons, door_polygons, window_polygons, background_polygons = _load_final_mask_polygons(
        final_dir,
        target_size=target_size,
        content_size=content_size,
    )
    overlay = render_combined_contour_overlay(
        input_image,
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        boundary_polygon=background_polygons[0] if background_polygons else None,
    )
    if len(background_polygons) <= 1:
        return overlay
    base = overlay.convert("RGBA")
    extra = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(extra, "RGBA")
    for polygon in background_polygons[1:]:
        _draw_unfilled_boundary(draw, polygon)
    return Image.alpha_composite(base, extra).convert("RGB")


@dataclass
class EvalCounts:
    rooms: int
    doors: int
    windows: int


@dataclass
class RoomAreaMatch:
    gt_index: int
    gt_area: int
    pred_index: int | None
    pred_area: int
    error_ratio: float


@dataclass
class FootprintArea:
    gt_area: int
    pred_area: int
    error_ratio: float


@dataclass
class LlmFootprintArea:
    gt_area: int
    llm_area: int
    error_ratio: float


@dataclass
class LlmUsageSummary:
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_seconds: float | None = None


@dataclass
class RunMetricRow:
    label: str
    seconds: float | None
    cost_usd: float


def _llm_usage_from_extract(extract: dict[str, object] | None) -> LlmUsageSummary:
    if extract is None:
        return LlmUsageSummary(0, 0, 0.0)
    usage = extract.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    elapsed = extract.get("elapsed_seconds")
    return LlmUsageSummary(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost_usd=float(usage.get("cost_usd", 0.0)),
        elapsed_seconds=float(elapsed) if elapsed is not None else None,
    )


def _polygon_area_px(polygon: PolygonRing) -> int:
    if len(polygon) < 3:
        return 0
    return int(abs(cv2.contourArea(np.array(polygon, dtype=np.int32))))


def _polygons_total_area_px(polygons: list[PolygonRing]) -> int:
    return sum(_polygon_area_px(polygon) for polygon in polygons)


def _area_error_ratio(gt_area: int, pred_area: int) -> float:
    if gt_area > 0:
        return abs(pred_area - gt_area) / gt_area
    return 0.0 if pred_area == 0 else 1.0


def _average_room_error(matches: list[RoomAreaMatch]) -> float | None:
    if not matches:
        return None
    return sum(match.error_ratio for match in matches) / len(matches)


def _average_room_error_filtered(
    matches: list[RoomAreaMatch],
    *,
    max_error_ratio: float = MATCH_ERROR_RATIO,
) -> float | None:
    filtered = [match for match in matches if match.error_ratio < max_error_ratio]
    if not filtered:
        return None
    return sum(match.error_ratio for match in filtered) / len(filtered)


def compare_footprint_area(
    gt_background: list[PolygonRing],
    pred_background: list[PolygonRing],
) -> FootprintArea:
    gt_area = _polygons_total_area_px(gt_background)
    pred_area = _polygons_total_area_px(pred_background)
    return FootprintArea(
        gt_area=gt_area,
        pred_area=pred_area,
        error_ratio=_area_error_ratio(gt_area, pred_area),
    )


def _intersection_area_px(
    polygon_a: PolygonRing,
    polygon_b: PolygonRing,
    *,
    canvas_size: int,
) -> int:
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        return 0
    mask_a = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    mask_b = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(mask_a, [np.array(polygon_a, dtype=np.int32)], 1)
    cv2.fillPoly(mask_b, [np.array(polygon_b, dtype=np.int32)], 1)
    return int(np.logical_and(mask_a, mask_b).sum())


def match_gt_rooms_to_pred(
    gt_rooms: list[PolygonRing],
    pred_rooms: list[PolygonRing],
    *,
    canvas_size: int,
) -> list[RoomAreaMatch]:
    matches: list[RoomAreaMatch] = []
    for gt_index, gt_room in enumerate(gt_rooms):
        gt_area = _polygon_area_px(gt_room)
        best_pred_index: int | None = None
        best_intersection = 0
        for pred_index, pred_room in enumerate(pred_rooms):
            intersection = _intersection_area_px(
                gt_room,
                pred_room,
                canvas_size=canvas_size,
            )
            if intersection > best_intersection:
                best_intersection = intersection
                best_pred_index = pred_index
        pred_area = (
            _polygon_area_px(pred_rooms[best_pred_index])
            if best_pred_index is not None
            else 0
        )
        matches.append(
            RoomAreaMatch(
                gt_index=gt_index,
                gt_area=gt_area,
                pred_index=best_pred_index,
                pred_area=pred_area,
                error_ratio=_area_error_ratio(gt_area, pred_area),
            )
        )
    return matches


@dataclass
class Button:
    rect: pygame.Rect
    label: str

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False) -> None:
        color = BTN_HOVER if hover else BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (90, 90, 90), self.rect, 1, border_radius=4)
        text = font.render(self.label, True, BTN_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgb = image.convert("RGB")
    return pygame.image.frombuffer(rgb.tobytes(), rgb.size, "RGB").copy()


def _fit_image_to_slot(image: Image.Image, slot_w: int, slot_h: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return Image.new("RGB", (slot_w, slot_h), EMPTY_SLOT_BG)
    scale = min(slot_w / width, slot_h / height)
    fitted_w = max(1, int(width * scale))
    fitted_h = max(1, int(height * scale))
    resized = image.convert("RGB").resize((fitted_w, fitted_h), Image.Resampling.LANCZOS)
    slot = Image.new("RGB", (slot_w, slot_h), EMPTY_SLOT_BG)
    slot.paste(resized, ((slot_w - fitted_w) // 2, (slot_h - fitted_h) // 2))
    return slot


class InferUIEval:
    def __init__(self, cfg, test_images: list[Path], *, config_path: Path) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.config_path = config_path
        self.show_llm_extract = cfg.eval_ui.llm_extract
        self.test_images = test_images
        self.gt_base_dir = cfg.label_data.pipeline_output_dir
        self.selected_index = 0
        self.list_scroll = 0
        self.running_inference = False
        self.running_llm = False
        self.gt_counts = EvalCounts(0, 0, 0)
        self.pred_counts = EvalCounts(0, 0, 0)
        self.llm_counts = EvalCounts(0, 0, 0)
        self.room_matches: list[RoomAreaMatch] = []
        self.llm_room_matches: list[RoomAreaMatch] = []
        self.footprint = FootprintArea(0, 0, 0.0)
        self.llm_footprint = LlmFootprintArea(0, 0, 0.0)
        self.llm_usage = LlmUsageSummary(0, 0, 0.0)
        self.inference_seconds: float | None = None
        self.room_names_meta: dict[str, object] | None = None
        self.panel_scroll = 0
        self.gt_overlay: Image.Image | None = None
        self.pred_overlay: Image.Image | None = None
        self.llm_overlay: Image.Image | None = None
        self.room_names: list[dict[str, object]] = []
        self.llm_names: list[dict[str, object]] = []
        self.content_size = (0, 0)
        self.input_image: Image.Image | None = None
        self.status = f"Loaded {test_images[0].name}"
        self.col_width = VIEWER_SLOT_SIZE
        self.viewer_x = LIST_WIDTH
        self.viewer_width = self.col_width * 2 + VIEWER_GAP
        self.llm_viewer_x = self.viewer_x + (self.viewer_width - self.col_width) // 2
        self.panel_x = self.viewer_x + self.viewer_width + VIEWER_GAP
        self.window_w = self.panel_x + PANEL_WIDTH
        self.window_h = self._viewer_content_height()
        self.window: pygame.Surface | None = None
        pygame.display.set_caption("Infer UI — Eval (step-001 vs GT)")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self._layout_buttons()
        self.gt_surface = self._empty_slot_surface(self.col_width, self.col_width)
        self.pred_surface = self._empty_slot_surface(self.col_width, self.col_width)
        self.llm_surface = self._empty_slot_surface(self.col_width, self.col_width)
        self._update_window_size()
        self._load_image(0, initial=True)

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = PANEL_WIDTH - 20
        self.rerun_btn = Button(
            pygame.Rect(x, PANEL_BTN_TOP, btn_w, BTN_HEIGHT),
            "Re-run inference",
        )
        if self.show_llm_extract:
            self.llm_extract_btn = Button(
                pygame.Rect(x, PANEL_BTN_TOP + BTN_HEIGHT + BTN_GAP, btn_w, BTN_HEIGHT),
                "LLM extract",
            )

    def _empty_slot_surface(self, width: int, height: int) -> pygame.Surface:
        surface = pygame.Surface((width, height))
        surface.fill(EMPTY_SLOT_BG)
        pygame.draw.rect(surface, EMPTY_SLOT_BORDER, surface.get_rect(), 1)
        return surface

    def _slot_surface(self, image: Image.Image | None, slot_w: int, slot_h: int) -> pygame.Surface:
        if image is None:
            return self._empty_slot_surface(slot_w, slot_h)
        return _pil_to_surface(_fit_image_to_slot(image, slot_w, slot_h))

    def _gt_dir(self, image_path: Path) -> Path:
        return _final_masks_dir(self.gt_base_dir, image_path.stem)

    def _pred_dir(self, image_path: Path) -> Path:
        return _final_masks_dir(EVAL_OUTPUT_DIR, image_path.stem)

    def _pred_has_outputs(self, image_path: Path) -> bool:
        final_dir = self._pred_dir(image_path)
        if not (final_dir / "room.json").is_file():
            return False
        return _polygon_canvas_size(final_dir) == self.cfg.canvas_size

    def _llm_extract_path(self, image_path: Path) -> Path:
        return _llm_extract_dir(image_path.stem) / "floorplan.json"

    def _refresh_llm_display(
        self,
        image_path: Path,
        *,
        input_image: Image.Image,
        content_size: tuple[int, int],
        gt_rooms: list[PolygonRing],
        gt_background: list[PolygonRing],
    ) -> None:
        extract = load_floorplan_extract(self._llm_extract_path(image_path))
        if extract is None:
            self.llm_overlay = None
            self.llm_counts = EvalCounts(0, 0, 0)
            self.llm_room_matches = []
            self.llm_footprint = LlmFootprintArea(0, 0, 0.0)
            self.llm_names = []
            self.llm_usage = LlmUsageSummary(0, 0, 0.0)
            self.llm_surface = self._empty_slot_surface(self.col_width, self.col_width)
            return

        self.llm_usage = _llm_usage_from_extract(extract)
        display_size = input_image.size[0]
        llm_size = int(extract.get("size", LLM_EXTRACT_SIZE))
        compare_map_kwargs = {
            "from_size": llm_size,
            "to_size": GT_CANVAS_SIZE,
            "content_size": content_size,
        }
        llm_rooms_raw = extract.get("rooms")
        llm_doors_raw = extract.get("doors")
        llm_windows_raw = extract.get("windows")
        llm_boundary_raw = extract.get("boundary")
        llm_rooms = _map_polygons_to_canvas(
            llm_rooms_raw if isinstance(llm_rooms_raw, list) else [],
            **compare_map_kwargs,
        )
        llm_doors = _map_polygons_to_canvas(
            llm_doors_raw if isinstance(llm_doors_raw, list) else [],
            **compare_map_kwargs,
        )
        llm_windows = _map_polygons_to_canvas(
            llm_windows_raw if isinstance(llm_windows_raw, list) else [],
            **compare_map_kwargs,
        )
        llm_background: list[PolygonRing] = []
        if isinstance(llm_boundary_raw, list) and len(llm_boundary_raw) >= 3:
            llm_background = _map_polygons_to_canvas([llm_boundary_raw], **compare_map_kwargs)

        self.llm_counts = EvalCounts(
            rooms=len(llm_rooms),
            doors=len(llm_doors),
            windows=len(llm_windows),
        )
        self.llm_room_matches = match_gt_rooms_to_pred(
            gt_rooms,
            llm_rooms,
            canvas_size=GT_CANVAS_SIZE,
        )
        gt_area = _polygons_total_area_px(gt_background)
        llm_area = _polygons_total_area_px(llm_background)
        self.llm_footprint = LlmFootprintArea(
            gt_area=gt_area,
            llm_area=llm_area,
            error_ratio=_area_error_ratio(gt_area, llm_area),
        )
        names_raw = extract.get("names")
        self.llm_names = (
            _map_names_to_canvas(
                names_raw if isinstance(names_raw, list) else [],
                from_size=llm_size,
                to_size=display_size,
                content_size=content_size,
            )
            if isinstance(names_raw, list)
            else []
        )
        self.llm_overlay = render_llm_overlay(
            input_image,
            extract,
            content_size=content_size,
            canvas_size=display_size,
        )
        self.llm_surface = self._slot_surface(self.llm_overlay, self.col_width, self.col_width)

    def _refresh_display(self, image_path: Path) -> None:
        gt_dir = self._gt_dir(image_path)
        pred_dir = self._pred_dir(image_path)
        pred_step = _step_dir(EVAL_OUTPUT_DIR, image_path.stem)
        gt_raw = polygon_counts(gt_dir)
        pred_raw = polygon_counts(pred_dir)
        self.gt_counts = EvalCounts(**gt_raw)
        self.pred_counts = EvalCounts(**pred_raw)
        self.inference_seconds = _load_inference_seconds(image_path.stem)
        with Image.open(image_path) as source:
            content_size = source.size
        self.content_size = content_size
        input_image = _shared_input_image(
            gt_dir,
            image_path,
            canvas_size=self.cfg.canvas_size,
        )
        self.input_image = input_image
        self.gt_overlay = render_eval_overlay(
            gt_dir,
            input_image=input_image,
            content_size=content_size,
        )
        self.room_names_meta = load_room_names_meta(pred_step)
        self.room_names = load_room_names(pred_step)
        pred_overlay = render_eval_overlay(
            pred_dir,
            input_image=input_image,
            content_size=content_size,
        )
        self.pred_overlay = (
            annotate_room_labels(pred_overlay, self.room_names)
            if pred_overlay is not None and self.room_names
            else pred_overlay
        )
        gt_rooms, _, _, gt_background = _load_final_mask_polygons(
            gt_dir,
            target_size=GT_CANVAS_SIZE,
            content_size=content_size,
        )
        pred_rooms, _, _, pred_background = _load_final_mask_polygons(
            pred_dir,
            target_size=GT_CANVAS_SIZE,
            content_size=content_size,
        )
        self.room_matches = match_gt_rooms_to_pred(
            gt_rooms,
            pred_rooms,
            canvas_size=GT_CANVAS_SIZE,
        )
        self.footprint = compare_footprint_area(gt_background, pred_background)
        print(
            f"Footprint {image_path.name}: "
            f"GT={self.footprint.gt_area} px  "
            f"Pred={self.footprint.pred_area} px  "
            f"Err={self.footprint.error_ratio:.4f}"
        )
        if self.room_names:
            print(f"Room labels {image_path.name}: {len(self.room_names)}")
        if self.show_llm_extract:
            self._refresh_llm_display(
                image_path,
                input_image=input_image,
                content_size=content_size,
                gt_rooms=gt_rooms,
                gt_background=gt_background,
            )
        slot = self.col_width
        self.gt_surface = self._slot_surface(self.gt_overlay, slot, slot)
        self.pred_surface = self._slot_surface(self.pred_overlay, slot, slot)
        self.panel_scroll = 0
        self._update_window_size()

    def _run_inference(self, image_path: Path) -> None:
        self.running_inference = True
        self.status = f"Running step-001 on {image_path.name}..."
        self._draw()
        if self.window is not None:
            pygame.display.flip()
        try:
            started = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "infer_fpn.py"),
                    "--config",
                    str(self.config_path),
                    "--image",
                    str(image_path),
                    "--checkpoint",
                    str(self.cfg.fpn.checkpoint_latest),
                    "--output-dir",
                    str(EVAL_OUTPUT_DIR),
                    "--canvas-size",
                    str(self.cfg.canvas_size),
                    "--skip-step-2",
                    "--skip-step-3",
                ],
                check=True,
                cwd=str(_ROOT),
            )
            elapsed = time.perf_counter() - started
            _save_inference_seconds(image_path.stem, elapsed)
            self.inference_seconds = elapsed
            print(f"Inference {image_path.name}: time={elapsed:.1f}s")
            self._refresh_display(image_path)
            self.status = f"Inference complete — {elapsed:.1f}s — {image_path.name}"
        except subprocess.CalledProcessError as exc:
            self.status = f"Inference failed (exit {exc.returncode})"
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
        finally:
            self.running_inference = False

    def _run_llm_extract(self, image_path: Path) -> None:
        if not self.show_llm_extract:
            return
        llm = self.cfg.fpn.room_polygon_llm
        if not llm.model or not llm.openai_api_key:
            self.status = "LLM extract requires OPENAI_API_KEY and fpn.room_polygon_llm.model"
            return
        self.running_llm = True
        self.status = f"Running LLM extract on {image_path.name}..."
        self._draw()
        if self.window is not None:
            pygame.display.flip()
        try:
            with Image.open(image_path) as source:
                source_image = source.convert("RGB")
            out_dir = _llm_extract_dir(image_path.stem)
            meta = extract_floorplan_from_image(
                source_image,
                out_dir,
                llm_size=llm.size,
                model=llm.model,
                detail=llm.detail,
                api_key=llm.openai_api_key,
                input_cost_per_million_usd=llm.input_cost_per_million_usd,
                cached_input_cost_per_million_usd=llm.cached_input_cost_per_million_usd,
                cache_write_cost_per_million_usd=llm.cache_write_cost_per_million_usd,
                output_cost_per_million_usd=llm.output_cost_per_million_usd,
                log=print,
            )
            usage = meta.get("usage")
            cost = usage.get("cost_usd", 0.0) if isinstance(usage, dict) else 0.0
            input_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
            output_tokens = int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0
            elapsed = meta.get("elapsed_seconds")
            elapsed_seconds = float(elapsed) if elapsed is not None else None
            print(
                f"LLM extract {image_path.name}: model={llm.model} "
                f"in={input_tokens:,} out={output_tokens:,} cost=${cost:.4f} "
                f"time={_format_seconds(elapsed_seconds)}"
            )
            if self.input_image is not None:
                gt_dir = self._gt_dir(image_path)
                gt_rooms, _, _, gt_background = _load_final_mask_polygons(
                    gt_dir,
                    target_size=GT_CANVAS_SIZE,
                    content_size=self.content_size,
                )
                self._refresh_llm_display(
                    image_path,
                    input_image=self.input_image,
                    content_size=self.content_size,
                    gt_rooms=gt_rooms,
                    gt_background=gt_background,
                )
            time_label = _format_seconds(elapsed_seconds)
            self.status = f"LLM extract complete — ${cost:.4f} — {time_label} — {image_path.name}"
        except Exception as exc:
            self.status = f"LLM extract failed: {exc}"
        finally:
            self.running_llm = False

    def _load_image(self, index: int, *, initial: bool = False, force_infer: bool = False) -> None:
        path = self.test_images[index]
        self.selected_index = index
        if not initial:
            self._ensure_list_visible(index)
        if force_infer or not self._pred_has_outputs(path):
            self._run_inference(path)
        else:
            self._refresh_display(path)
            self.status = f"Loaded cached prediction — {path.name}"
        if not initial:
            self._draw()

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
        if mx >= LIST_WIDTH or my < LIST_HEADER_HEIGHT:
            return None
        row = (my - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT
        index = self.list_scroll + row
        if index < 0 or index >= len(self.test_images):
            return None
        return index

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.running_inference or self.running_llm:
            return
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_image(list_index)
            return
        if self.rerun_btn.hit(pos):
            self._load_image(self.selected_index, force_infer=True)
            return
        if self.show_llm_extract and self.llm_extract_btn.hit(pos):
            self._run_llm_extract(self.test_images[self.selected_index])

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, LIST_WIDTH, self.window_h))
        header = self.label_font.render("test_floorplan/", True, LABEL_COLOR)
        self.window.blit(header, (8, 4))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (LIST_WIDTH - 1, 0),
            (LIST_WIDTH - 1, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        for row in range(self._list_visible_count()):
            index = self.list_scroll + row
            if index >= len(self.test_images):
                break
            y = LIST_HEADER_HEIGHT + row * LIST_ITEM_HEIGHT
            rect = pygame.Rect(0, y, LIST_WIDTH, LIST_ITEM_HEIGHT)
            if index == self.selected_index:
                pygame.draw.rect(self.window, LIST_SELECTED, rect)
            elif rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, rect)
            label = self.test_images[index].name
            if self.font.size(label)[0] > LIST_WIDTH - 12:
                while label and self.font.size(label + "…")[0] > LIST_WIDTH - 12:
                    label = label[:-1]
                label += "…"
            text = self.font.render(label, True, LIST_TEXT)
            self.window.blit(text, (8, y + 3))

    def _draw_viewer(self) -> None:
        y = 0
        self.window.blit(
            self.label_font.render("Ground truth", True, PANEL_LABEL_COLOR),
            (self.viewer_x, y),
        )
        self.window.blit(
            self.label_font.render("Predicted", True, PANEL_LABEL_COLOR),
            (self.viewer_x + self.col_width + VIEWER_GAP, y),
        )
        y += VIEWER_LABEL_HEIGHT
        self.window.blit(self.gt_surface, (self.viewer_x, y))
        self.window.blit(self.pred_surface, (self.viewer_x + self.col_width + VIEWER_GAP, y))

        if not self.show_llm_extract:
            return
        row2_y = y + self.col_width + VIEWER_ROW_GAP
        llm_label = self.label_font.render("LLM extract", True, PANEL_LABEL_COLOR)
        self.window.blit(
            llm_label,
            (self.llm_viewer_x + (self.col_width - llm_label.get_width()) // 2, row2_y),
        )
        row2_y += VIEWER_LABEL_HEIGHT
        self.window.blit(self.llm_surface, (self.llm_viewer_x, row2_y))

    def _count_color(self, gt_value: int, pred_value: int) -> tuple[int, int, int]:
        return MATCH_COLOR if gt_value == pred_value else MISMATCH_COLOR

    def _count_color_3(
        self,
        gt_value: int,
        pred_value: int,
        llm_value: int,
    ) -> tuple[int, int, int]:
        if gt_value == pred_value == llm_value:
            return MATCH_COLOR
        return MISMATCH_COLOR

    def _status_lines(self) -> list[str]:
        status = self.status
        if self.running_inference or self.running_llm:
            status = f"{status} (please wait)"
        return self._wrap_text(status, PANEL_WIDTH - 20)

    def _viewer_content_height(self) -> int:
        row1 = VIEWER_LABEL_HEIGHT + self.col_width
        if not self.show_llm_extract:
            return row1
        row2 = VIEWER_ROW_GAP + VIEWER_LABEL_HEIGHT + self.col_width
        return row1 + row2

    def _panel_scroll_top(self) -> int:
        if self.show_llm_extract:
            return PANEL_BTN_TOP + BTN_HEIGHT + BTN_GAP + BTN_HEIGHT + 8
        return PANEL_BTN_TOP + BTN_HEIGHT + 8

    def _comparison_columns(self) -> tuple[int, int, int, int]:
        col_label = self.panel_x + 10
        col_gt = self.panel_x + 100
        col_pred = self.panel_x + 170
        col_llm = self.panel_x + 240
        return col_label, col_gt, col_pred, col_llm

    def _run_metric_rows(self) -> list[RunMetricRow]:
        room_seconds = _room_names_seconds(self.room_names_meta)
        room_cost = _room_names_cost(self.room_names_meta)
        fpn_seconds = self.inference_seconds
        if fpn_seconds is not None and room_seconds is not None:
            fpn_seconds = max(0.0, fpn_seconds - room_seconds)

        rows = [
            RunMetricRow("FPN", fpn_seconds, 0.0),
            RunMetricRow(_room_names_label(self.room_names_meta), room_seconds, room_cost),
        ]
        if self.show_llm_extract:
            rows.append(
                RunMetricRow(
                    "LLM extract",
                    self.llm_usage.elapsed_seconds,
                    self.llm_usage.cost_usd,
                )
            )
        return rows

    def _run_metrics_table_height(self) -> int:
        row_count = len(self._run_metric_rows())
        return 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_ROW_HEIGHT + row_count * ROOM_ROW_HEIGHT

    def _avg_room_error_row_count(self) -> int:
        return 1 + (1 if self.show_llm_extract else 0)

    def _avg_room_error_summary_height(self) -> int:
        block = 8 + ROOM_SECTION_HEADER_HEIGHT + self._avg_room_error_row_count() * ROOM_ROW_HEIGHT
        return block * 2

    def _draw_avg_room_error_block(
        self,
        y: int,
        *,
        title: str,
        fpn_avg: float | None,
        llm_avg: float | None,
    ) -> int:
        col_label = self.panel_x + 10
        col_err = self.panel_x + 130
        y += 8
        self.window.blit(
            self.label_font.render(title, True, PANEL_LABEL_COLOR),
            (col_label, y),
        )
        y += ROOM_SECTION_HEADER_HEIGHT
        summary_rows = [("FPN", fpn_avg)]
        if self.show_llm_extract:
            summary_rows.append(("LLM", llm_avg))
        for label, avg_err in summary_rows:
            if avg_err is None:
                err_text = "(none)"
                err_color = LABEL_COLOR
            else:
                err_text = f"{avg_err:.4f}"
                err_color = MATCH_COLOR if avg_err < MATCH_ERROR_RATIO else MISMATCH_COLOR
            self.window.blit(self.font.render(f"{label}:", True, LIST_TEXT), (col_label, y))
            self.window.blit(self.font.render(err_text, True, err_color), (col_err, y))
            y += ROOM_ROW_HEIGHT
        return y

    def _draw_avg_room_error_summary(self, y: int) -> int:
        y = self._draw_avg_room_error_block(
            y,
            title="Avg room area err",
            fpn_avg=_average_room_error(self.room_matches),
            llm_avg=_average_room_error(self.llm_room_matches),
        )
        return self._draw_avg_room_error_block(
            y,
            title="Avg (<5% err)",
            fpn_avg=_average_room_error_filtered(self.room_matches),
            llm_avg=_average_room_error_filtered(self.llm_room_matches),
        )

    def _draw_run_metrics_table(self, y: int) -> int:
        col_label = self.panel_x + 10
        col_time = self.panel_x + 210
        col_cost = self.panel_x + 300
        self.window.blit(
            self.label_font.render("Run cost & time", True, PANEL_LABEL_COLOR),
            (col_label, y),
        )
        y += ROOM_SECTION_HEADER_HEIGHT
        self.window.blit(self.font.render("", True, STAT_HEADER_COLOR), (col_label, y))
        self.window.blit(self.font.render("Time", True, STAT_HEADER_COLOR), (col_time, y))
        self.window.blit(self.font.render("Cost", True, STAT_HEADER_COLOR), (col_cost, y))
        y += ROOM_ROW_HEIGHT

        for row in self._run_metric_rows():
            label = row.label
            if self.font.size(label)[0] > 190:
                while label and self.font.size(label + "…")[0] > 190:
                    label = label[:-1]
                label += "…"
            time_color = STATUS_COLOR if row.seconds is not None else LABEL_COLOR
            cost_color = STATUS_COLOR if row.cost_usd > 0 else LIST_TEXT
            self.window.blit(self.font.render(label, True, LIST_TEXT), (col_label, y))
            self.window.blit(
                self.font.render(_format_seconds(row.seconds), True, time_color),
                (col_time, y),
            )
            self.window.blit(
                self.font.render(_format_cost_usd(row.cost_usd), True, cost_color),
                (col_cost, y),
            )
            y += ROOM_ROW_HEIGHT
        return y

    def _panel_scrollable_height(self) -> int:
        y = 0
        y += len(self._status_lines()) * 18
        y += 12 + 22 + 20 + 20 * 3
        y += self._run_metrics_table_height()
        y += 8 + 22 + 20 + 20
        y += self._avg_room_error_summary_height()
        y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_ROW_HEIGHT
        y += len(self.room_matches) * ROOM_ROW_HEIGHT
        y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_NAME_ROW_HEIGHT
        y += max(1, len(self.room_names)) * ROOM_NAME_ROW_HEIGHT
        if self.show_llm_extract:
            y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_ROW_HEIGHT
            y += len(self.llm_room_matches) * ROOM_ROW_HEIGHT
            y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_NAME_ROW_HEIGHT
            y += max(1, len(self.llm_names)) * ROOM_NAME_ROW_HEIGHT
        y += 8
        return y

    def _max_panel_scroll(self) -> int:
        visible_h = max(0, self.window_h - self._panel_scroll_top())
        return max(0, self._panel_scrollable_height() - visible_h)

    def _update_window_size(self) -> None:
        self.window_h = self._viewer_content_height()
        self.window = pygame.display.set_mode((self.window_w, self.window_h))

    def _draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(self.panel_x, 0, PANEL_WIDTH, self.window_h)
        pygame.draw.rect(self.window, SIDE_BG, panel_rect)
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        busy = self.running_inference or self.running_llm
        self.rerun_btn.draw(
            self.window,
            self.btn_font,
            hover=self.rerun_btn.hit(mouse) and not busy,
        )
        if self.show_llm_extract:
            self.llm_extract_btn.draw(
                self.window,
                self.btn_font,
                hover=self.llm_extract_btn.hit(mouse) and not busy,
            )
        scroll_top = self._panel_scroll_top()
        clip = pygame.Rect(self.panel_x, scroll_top, PANEL_WIDTH, self.window_h - scroll_top)
        self.window.set_clip(clip)
        y = scroll_top - self.panel_scroll
        for line in self._status_lines():
            self.window.blit(
                self.font.render(line, True, STATUS_COLOR),
                (self.panel_x + 10, y),
            )
            y += 18
        y += 12
        self.window.blit(
            self.label_font.render("Counts", True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += 22
        col_label, col_gt, col_pred, col_llm = self._comparison_columns()
        self.window.blit(self.font.render("GT", True, STAT_HEADER_COLOR), (col_gt, y))
        self.window.blit(self.font.render("Pred", True, STAT_HEADER_COLOR), (col_pred, y))
        if self.show_llm_extract:
            self.window.blit(self.font.render("LLM", True, STAT_HEADER_COLOR), (col_llm, y))
        y += 20
        for label, gt_val, pred_val, llm_val in (
            ("Rooms", self.gt_counts.rooms, self.pred_counts.rooms, self.llm_counts.rooms),
            ("Doors", self.gt_counts.doors, self.pred_counts.doors, self.llm_counts.doors),
            ("Windows", self.gt_counts.windows, self.pred_counts.windows, self.llm_counts.windows),
        ):
            if self.show_llm_extract:
                color = self._count_color_3(gt_val, pred_val, llm_val)
            else:
                color = self._count_color(gt_val, pred_val)
            self.window.blit(self.font.render(f"{label}:", True, LIST_TEXT), (col_label, y))
            self.window.blit(self.font.render(str(gt_val), True, color), (col_gt, y))
            self.window.blit(self.font.render(str(pred_val), True, color), (col_pred, y))
            if self.show_llm_extract:
                self.window.blit(self.font.render(str(llm_val), True, color), (col_llm, y))
            y += 20

        y = self._draw_run_metrics_table(y)

        y += 8
        self.window.blit(
            self.label_font.render("Footprint", True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += 22
        if self.show_llm_extract:
            col_gt_area = self.panel_x + 100
            col_pred_area = self.panel_x + 170
            col_llm_area = self.panel_x + 240
            col_pred_err = self.panel_x + 300
            col_llm_err = self.panel_x + 350
            self.window.blit(self.font.render("GT px", True, STAT_HEADER_COLOR), (col_gt_area, y))
            self.window.blit(self.font.render("Pred", True, STAT_HEADER_COLOR), (col_pred_area, y))
            self.window.blit(self.font.render("LLM", True, STAT_HEADER_COLOR), (col_llm_area, y))
            self.window.blit(self.font.render("P err", True, STAT_HEADER_COLOR), (col_pred_err, y))
            self.window.blit(self.font.render("L err", True, STAT_HEADER_COLOR), (col_llm_err, y))
        else:
            col_gt_area = self.panel_x + 110
            col_pred_area = self.panel_x + 190
            col_err = self.panel_x + 250
            self.window.blit(self.font.render("GT px", True, STAT_HEADER_COLOR), (col_gt_area, y))
            self.window.blit(self.font.render("Pred px", True, STAT_HEADER_COLOR), (col_pred_area, y))
            self.window.blit(self.font.render("Err", True, STAT_HEADER_COLOR), (col_err, y))
        y += 20
        pred_err_color = (
            MATCH_COLOR if self.footprint.error_ratio < MATCH_ERROR_RATIO else MISMATCH_COLOR
        )
        self.window.blit(
            self.font.render(str(self.footprint.gt_area), True, LIST_TEXT),
            (col_gt_area, y),
        )
        self.window.blit(
            self.font.render(str(self.footprint.pred_area), True, LIST_TEXT),
            (col_pred_area, y),
        )
        if self.show_llm_extract:
            llm_err_color = (
                MATCH_COLOR
                if self.llm_footprint.error_ratio < MATCH_ERROR_RATIO
                else MISMATCH_COLOR
            )
            self.window.blit(
                self.font.render(str(self.llm_footprint.llm_area), True, LIST_TEXT),
                (col_llm_area, y),
            )
            self.window.blit(
                self.font.render(f"{self.footprint.error_ratio:.3f}", True, pred_err_color),
                (col_pred_err, y),
            )
            self.window.blit(
                self.font.render(f"{self.llm_footprint.error_ratio:.3f}", True, llm_err_color),
                (col_llm_err, y),
            )
        else:
            self.window.blit(
                self.font.render(f"{self.footprint.error_ratio:.4f}", True, pred_err_color),
                (col_err, y),
            )
        y += 20

        y = self._draw_avg_room_error_summary(y)

        y += 8
        room_areas_label = "Room areas (FPN)" if self.show_llm_extract else "Room areas"
        self.window.blit(
            self.label_font.render(room_areas_label, True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += ROOM_SECTION_HEADER_HEIGHT
        col_room = self.panel_x + 10
        col_gt_area = self.panel_x + 52
        col_pred_area = self.panel_x + 130
        col_err = self.panel_x + 210
        self.window.blit(self.font.render("Rm", True, STAT_HEADER_COLOR), (col_room, y))
        self.window.blit(self.font.render("GT px", True, STAT_HEADER_COLOR), (col_gt_area, y))
        self.window.blit(self.font.render("Pred px", True, STAT_HEADER_COLOR), (col_pred_area, y))
        self.window.blit(self.font.render("Err", True, STAT_HEADER_COLOR), (col_err, y))
        y += ROOM_ROW_HEIGHT
        for index, match in enumerate(self.room_matches):
            row_y = y + index * ROOM_ROW_HEIGHT
            err_color = MATCH_COLOR if match.error_ratio < MATCH_ERROR_RATIO else MISMATCH_COLOR
            self.window.blit(
                self.font.render(f"{match.gt_index + 1:02d}", True, LIST_TEXT),
                (col_room, row_y),
            )
            self.window.blit(
                self.font.render(str(match.gt_area), True, LIST_TEXT),
                (col_gt_area, row_y),
            )
            self.window.blit(
                self.font.render(str(match.pred_area), True, LIST_TEXT),
                (col_pred_area, row_y),
            )
            self.window.blit(
                self.font.render(f"{match.error_ratio:.4f}", True, err_color),
                (col_err, row_y),
            )
        y += len(self.room_matches) * ROOM_ROW_HEIGHT
        y += 8
        self.window.blit(
            self.label_font.render("Room labels (FPN step-001)", True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += ROOM_SECTION_HEADER_HEIGHT
        col_name = self.panel_x + 10
        col_pos = self.panel_x + 220
        self.window.blit(self.font.render("Name", True, STAT_HEADER_COLOR), (col_name, y))
        self.window.blit(self.font.render("x, y", True, STAT_HEADER_COLOR), (col_pos, y))
        y += ROOM_NAME_ROW_HEIGHT
        if not self.room_names:
            self.window.blit(
                self.font.render("(none)", True, LABEL_COLOR),
                (col_name, y),
            )
            y += ROOM_NAME_ROW_HEIGHT
        else:
            for index, entry in enumerate(self.room_names):
                row_y = y + index * ROOM_NAME_ROW_HEIGHT
                name = str(entry.get("room") or "")
                position = entry.get("position")
                pos_text = ""
                if isinstance(position, dict):
                    pos_text = f"{position.get('x', '?')}, {position.get('y', '?')}"
                label = name
                if self.font.size(label)[0] > 190:
                    while label and self.font.size(label + "…")[0] > 190:
                        label = label[:-1]
                    label += "…"
                self.window.blit(self.font.render(label, True, LIST_TEXT), (col_name, row_y))
                self.window.blit(self.font.render(pos_text, True, LIST_TEXT), (col_pos, row_y))
            y += len(self.room_names) * ROOM_NAME_ROW_HEIGHT

        if self.show_llm_extract:
            y += 8
            self.window.blit(
                self.label_font.render("Room areas (LLM)", True, PANEL_LABEL_COLOR),
                (self.panel_x + 10, y),
            )
            y += ROOM_SECTION_HEADER_HEIGHT
            col_room = self.panel_x + 10
            col_gt_area = self.panel_x + 52
            col_llm_area = self.panel_x + 130
            col_err = self.panel_x + 210
            self.window.blit(self.font.render("Rm", True, STAT_HEADER_COLOR), (col_room, y))
            self.window.blit(self.font.render("GT px", True, STAT_HEADER_COLOR), (col_gt_area, y))
            self.window.blit(self.font.render("LLM px", True, STAT_HEADER_COLOR), (col_llm_area, y))
            self.window.blit(self.font.render("Err", True, STAT_HEADER_COLOR), (col_err, y))
            y += ROOM_ROW_HEIGHT
            if not self.llm_room_matches:
                self.window.blit(
                    self.font.render("(none)", True, LABEL_COLOR),
                    (col_room, y),
                )
                y += ROOM_ROW_HEIGHT
            else:
                for index, match in enumerate(self.llm_room_matches):
                    row_y = y + index * ROOM_ROW_HEIGHT
                    err_color = (
                        MATCH_COLOR if match.error_ratio < MATCH_ERROR_RATIO else MISMATCH_COLOR
                    )
                    self.window.blit(
                        self.font.render(f"{match.gt_index + 1:02d}", True, LIST_TEXT),
                        (col_room, row_y),
                    )
                    self.window.blit(
                        self.font.render(str(match.gt_area), True, LIST_TEXT),
                        (col_gt_area, row_y),
                    )
                    self.window.blit(
                        self.font.render(str(match.pred_area), True, LIST_TEXT),
                        (col_llm_area, row_y),
                    )
                    self.window.blit(
                        self.font.render(f"{match.error_ratio:.4f}", True, err_color),
                        (col_err, row_y),
                    )
                y += len(self.llm_room_matches) * ROOM_ROW_HEIGHT

            y += 8
            self.window.blit(
                self.label_font.render("LLM room names", True, PANEL_LABEL_COLOR),
                (self.panel_x + 10, y),
            )
            y += ROOM_SECTION_HEADER_HEIGHT
            col_name = self.panel_x + 10
            col_pos = self.panel_x + 220
            self.window.blit(self.font.render("Name", True, STAT_HEADER_COLOR), (col_name, y))
            self.window.blit(self.font.render("x, y", True, STAT_HEADER_COLOR), (col_pos, y))
            y += ROOM_NAME_ROW_HEIGHT
            if not self.llm_names:
                self.window.blit(
                    self.font.render("(none)", True, LABEL_COLOR),
                    (col_name, y),
                )
            else:
                for index, entry in enumerate(self.llm_names):
                    row_y = y + index * ROOM_NAME_ROW_HEIGHT
                    name = str(entry.get("room") or "")
                    position = entry.get("position")
                    pos_text = ""
                    if isinstance(position, dict):
                        pos_text = f"{position.get('x', '?')}, {position.get('y', '?')}"
                    label = name
                    if self.font.size(label)[0] > 190:
                        while label and self.font.size(label + "…")[0] > 190:
                            label = label[:-1]
                        label += "…"
                    self.window.blit(self.font.render(label, True, LIST_TEXT), (col_name, row_y))
                    self.window.blit(self.font.render(pos_text, True, LIST_TEXT), (col_pos, row_y))
        self.window.set_clip(None)

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        if not text:
            return [""]
        lines: list[str] = []
        current = text
        while current:
            if self.font.size(current)[0] <= max_width:
                lines.append(current)
                break
            cut = len(current)
            while cut > 1 and self.font.size(current[:cut] + "…")[0] > max_width:
                cut -= 1
            lines.append(current[:cut] + "…")
            current = current[cut:].lstrip()
        return lines

    def _draw(self) -> None:
        if self.window is None:
            return
        self.window.fill((18, 18, 18))
        self._draw_file_list()
        self._draw_viewer()
        self._draw_side_panel()

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEWHEEL:
                    mx, _my = pygame.mouse.get_pos()
                    if mx < LIST_WIDTH:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                    elif mx >= self.panel_x:
                        self.panel_scroll = max(
                            0,
                            min(
                                self._max_panel_scroll(),
                                self.panel_scroll - event.y * ROOM_ROW_HEIGHT,
                            ),
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            self._draw()
            pygame.display.flip()
            clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FPN step-001 polygons against GT")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    size = cfg.canvas_size
    print(f"Inference canvas: {size}×{size}px")
    print(f"Ready — {len(test_images)} floor plan(s), showing {test_images[0].name}.")
    ui = InferUIEval(cfg, test_images, config_path=args.config)
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
