#!/usr/bin/env python3
"""Pygame UI to evaluate FPN step-001 polygon inference against ground truth.

Left panel lists test_floorplan/ files; click to run step-001 inference and compare
to GT in output/test_data/<stem>/step-001/final-masks/. Predictions are written to
eval_floorplan/<stem>/step-001/ at 1024×1024. The viewer shows GT vs predicted polygon
overlays (predicted panel includes LLM room labels). The right panel lists room/door/window counts and per-room area
comparison (GT vs best-matching predicted room by intersection, with error ratio),
and footprint area from background polygons.

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python infer_ui_eval.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_placement, fit_to_canvas_rgb
from util.final_mask_polygons import PolygonRing, load_layer_polygons
from util.room_components import annotate_room_labels, render_combined_contour_overlay

_ROOT = Path(__file__).resolve().parent
TEST_DIR = _ROOT / "test_floorplan"
EVAL_OUTPUT_DIR = _ROOT / "eval_floorplan"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LIST_WIDTH = 200
PANEL_WIDTH = 300
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
VIEWER_GAP = 8
VIEWER_LABEL_HEIGHT = 18
VIEWER_SLOT_SIZE = 400
ROOM_ROW_HEIGHT = 16
ROOM_SECTION_HEADER_HEIGHT = 22
ROOM_NAME_ROW_HEIGHT = 16

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


def load_room_names(step_dir: Path) -> list[dict[str, object]]:
    path = step_dir / "room-names.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rooms = payload.get("rooms")
    return rooms if isinstance(rooms, list) else []


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
        self.test_images = test_images
        self.gt_base_dir = cfg.label_data.pipeline_output_dir
        self.selected_index = 0
        self.list_scroll = 0
        self.running_inference = False
        self.gt_counts = EvalCounts(0, 0, 0)
        self.pred_counts = EvalCounts(0, 0, 0)
        self.room_matches: list[RoomAreaMatch] = []
        self.footprint = FootprintArea(0, 0, 0.0)
        self.panel_scroll = 0
        self.gt_overlay: Image.Image | None = None
        self.pred_overlay: Image.Image | None = None
        self.room_names: list[dict[str, object]] = []
        self.status = f"Loaded {test_images[0].name}"
        self.col_width = VIEWER_SLOT_SIZE
        self.viewer_x = LIST_WIDTH
        self.panel_x = self.viewer_x + self.col_width * 2 + VIEWER_GAP
        self.window_w = self.panel_x + PANEL_WIDTH
        self.window_h = VIEWER_LABEL_HEIGHT + self.col_width
        self.window: pygame.Surface | None = None
        pygame.display.set_caption("Infer UI — Eval (step-001 vs GT)")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self._layout_buttons()
        self.gt_surface = self._empty_slot_surface(self.col_width, self.col_width)
        self.pred_surface = self._empty_slot_surface(self.col_width, self.col_width)
        self._update_window_size()
        self._load_image(0, initial=True)

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = PANEL_WIDTH - 20
        self.rerun_btn = Button(pygame.Rect(x, 10, btn_w, 32), "Re-run inference")

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

    def _refresh_display(self, image_path: Path) -> None:
        gt_dir = self._gt_dir(image_path)
        pred_dir = self._pred_dir(image_path)
        pred_step = _step_dir(EVAL_OUTPUT_DIR, image_path.stem)
        gt_raw = polygon_counts(gt_dir)
        pred_raw = polygon_counts(pred_dir)
        self.gt_counts = EvalCounts(**gt_raw)
        self.pred_counts = EvalCounts(**pred_raw)
        with Image.open(image_path) as source:
            content_size = source.size
        input_image = _shared_input_image(
            gt_dir,
            image_path,
            canvas_size=self.cfg.canvas_size,
        )
        self.gt_overlay = render_eval_overlay(
            gt_dir,
            input_image=input_image,
            content_size=content_size,
        )
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
            target_size=input_image.size[0],
            content_size=content_size,
        )
        pred_rooms, _, _, pred_background = _load_final_mask_polygons(
            pred_dir,
            target_size=input_image.size[0],
            content_size=content_size,
        )
        self.room_matches = match_gt_rooms_to_pred(
            gt_rooms,
            pred_rooms,
            canvas_size=input_image.size[0],
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
            self._refresh_display(image_path)
            self.status = f"Inference complete — {image_path.name}"
        except subprocess.CalledProcessError as exc:
            self.status = f"Inference failed (exit {exc.returncode})"
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
        finally:
            self.running_inference = False

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
        if self.running_inference:
            return
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_image(list_index)
            return
        if self.rerun_btn.hit(pos):
            self._load_image(self.selected_index, force_infer=True)

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

    def _count_color(self, gt_value: int, pred_value: int) -> tuple[int, int, int]:
        return MATCH_COLOR if gt_value == pred_value else MISMATCH_COLOR

    def _status_lines(self) -> list[str]:
        status = self.status
        if self.running_inference:
            status = f"{status} (please wait)"
        return self._wrap_text(status, PANEL_WIDTH - 20)

    def _viewer_content_height(self) -> int:
        return VIEWER_LABEL_HEIGHT + self.col_width

    def _panel_scroll_top(self) -> int:
        return 52

    def _panel_scrollable_height(self) -> int:
        y = 0
        y += len(self._status_lines()) * 18
        y += 12 + 22 + 20 + 20 * 3
        y += 8 + 22 + 20 + 20
        y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_ROW_HEIGHT
        y += len(self.room_matches) * ROOM_ROW_HEIGHT
        y += 8 + ROOM_SECTION_HEADER_HEIGHT + ROOM_NAME_ROW_HEIGHT
        y += len(self.room_names) * ROOM_NAME_ROW_HEIGHT
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
        self.rerun_btn.draw(
            self.window,
            self.btn_font,
            hover=self.rerun_btn.hit(mouse) and not self.running_inference,
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
        col_label = self.panel_x + 10
        col_gt = self.panel_x + 110
        col_pred = self.panel_x + 190
        gt_header = self.font.render("GT", True, STAT_HEADER_COLOR)
        pred_header = self.font.render("Pred", True, STAT_HEADER_COLOR)
        self.window.blit(gt_header, (col_gt, y))
        self.window.blit(pred_header, (col_pred, y))
        y += 20
        for label, gt_val, pred_val in (
            ("Rooms", self.gt_counts.rooms, self.pred_counts.rooms),
            ("Doors", self.gt_counts.doors, self.pred_counts.doors),
            ("Windows", self.gt_counts.windows, self.pred_counts.windows),
        ):
            color = self._count_color(gt_val, pred_val)
            self.window.blit(self.font.render(f"{label}:", True, LIST_TEXT), (col_label, y))
            gt_text = self.font.render(str(gt_val), True, color)
            pred_text = self.font.render(str(pred_val), True, color)
            self.window.blit(gt_text, (col_gt, y))
            self.window.blit(pred_text, (col_pred, y))
            y += 20

        y += 8
        self.window.blit(
            self.label_font.render("Footprint", True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += 22
        col_gt_area = self.panel_x + 110
        col_pred_area = self.panel_x + 190
        col_err = self.panel_x + 250
        self.window.blit(self.font.render("GT px", True, STAT_HEADER_COLOR), (col_gt_area, y))
        self.window.blit(self.font.render("Pred px", True, STAT_HEADER_COLOR), (col_pred_area, y))
        self.window.blit(self.font.render("Err", True, STAT_HEADER_COLOR), (col_err, y))
        y += 20
        footprint_err_color = (
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
        self.window.blit(
            self.font.render(f"{self.footprint.error_ratio:.4f}", True, footprint_err_color),
            (col_err, y),
        )
        y += 20

        y += 8
        self.window.blit(
            self.label_font.render("Room areas", True, PANEL_LABEL_COLOR),
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
        y += 8
        self.window.blit(
            self.label_font.render("Room labels (LLM)", True, PANEL_LABEL_COLOR),
            (self.panel_x + 10, y),
        )
        y += ROOM_SECTION_HEADER_HEIGHT
        col_name = self.panel_x + 10
        col_pos = self.panel_x + 150
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
                if self.font.size(label)[0] > 130:
                    while label and self.font.size(label + "…")[0] > 130:
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
