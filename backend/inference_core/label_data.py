#!/usr/bin/env python3
"""Interactive pygame UI to refine FPN structure masks by hand.

Source images live in ``test_image/``. Saves into ``floorplan-manual-mask/``:

  input/<id>.jpg           — 1024×1024 canvas image
  output/<id>/wall.png     — white = active
  output/<id>/door.png
  output/<id>/window.png
  output/<id>/room.png
  output/<id>/background.png — painted or computed on demand

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python label_data.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pygame
from PIL import Image

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_rgb
from util.floorplan_region import apply_region_mask, compute_floorplan_region
from util.fpn_five_runtime import FpnFiveRuntime, prepare_bw
from util.mask_clean import clean_binary_mask

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]
TEST_DIR = _ROOT / "test_image"
DATASET_ROOT = _REPO_ROOT / "floorplan-manual-mask"
INPUT_DIR = DATASET_ROOT / "input"
OUTPUT_DIR = DATASET_ROOT / "output"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MaskLayer = Literal["wall", "door", "window", "room", "background"]
BrushMode = Literal["add", "remove"]
STRUCTURE_LAYERS: tuple[MaskLayer, ...] = ("wall", "door", "window", "room")
MASK_LAYERS: tuple[MaskLayer, ...] = (*STRUCTURE_LAYERS, "background")

LIST_WIDTH = 200
PANEL_WIDTH = 260
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
DISPLAY_SCALE = 0.72
DEFAULT_BRUSH_RADIUS = 10
MIN_BRUSH_RADIUS = 2
MAX_BRUSH_RADIUS = 64
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


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _scaled_mask_clean(size: int, cfg) -> tuple[int, int, int]:
    scale = size / 768
    clean = cfg.fpn.mask_clean
    open_px = max(0, round(clean.open_px * scale))
    close_px = max(0, round(clean.close_px * scale))
    min_area_px = max(0, round(clean.min_area_px * scale * scale))
    return open_px, close_px, min_area_px


def _clean_channel(mask: np.ndarray, *, size: int, cfg) -> np.ndarray:
    open_px, close_px, min_area_px = _scaled_mask_clean(size, cfg)
    return clean_binary_mask(
        mask > 0.5,
        open_px=open_px,
        close_px=close_px,
        min_area_px=min_area_px,
    )


def compute_background_mask(wall: np.ndarray, room: np.ndarray) -> np.ndarray:
    """Exterior / non-structure pixels from wall + room footprint."""
    occupied = np.clip(wall.astype(np.float32) + room.astype(np.float32), 0.0, 1.0)
    binary = ((occupied > 0.5).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed, kernel, iterations=1)
    return eroded == 0


def empty_masks(size: int) -> dict[MaskLayer, np.ndarray]:
    blank = np.zeros((size, size), dtype=bool)
    return {layer: blank.copy() for layer in MASK_LAYERS}


def infer_structure_masks(cfg, image_path: Path) -> tuple[Image.Image, dict[MaskLayer, np.ndarray]]:
    """Run FPN Five on the canvas-fitted image and return cleaned editable masks."""
    canvas_size = cfg.canvas_size
    full_canvas = fit_to_canvas_rgb(Image.open(image_path), size=canvas_size)
    runtime = FpnFiveRuntime.load(checkpoint=cfg.fpn.checkpoint)
    channels = runtime.channels

    region_size = cfg.fpn.floorplan_region.size
    region_prepared = prepare_bw(full_canvas, region_size)
    region_masks = runtime.infer(region_prepared)
    close_px = max(0, round(cfg.fpn.floorplan_region.close_px * (region_size / 768)))
    approx_epsilon = max(0.5, cfg.fpn.floorplan_region.approx_epsilon * (region_size / 768))
    floorplan_region = compute_floorplan_region(
        region_masks[channels.background],
        close_px=close_px,
        approx_epsilon=approx_epsilon,
    )
    masked_canvas = apply_region_mask(full_canvas, floorplan_region.mask)

    infer_size = cfg.fpn.room_components.component_size
    prepared = prepare_bw(masked_canvas, infer_size)
    raw = runtime.infer(prepared)

    masks: dict[MaskLayer, np.ndarray] = {}
    for layer, channel in (
        ("room", channels.rooms),
        ("wall", channels.walls),
        ("door", channels.doors),
        ("window", channels.windows),
    ):
        cleaned = _clean_channel(raw[channel], size=infer_size, cfg=cfg)
        masks[layer] = cleaned

    if infer_size != canvas_size:
        resized: dict[MaskLayer, np.ndarray] = {}
        for layer, mask in masks.items():
            resized[layer] = cv2.resize(
                mask.astype(np.uint8),
                (canvas_size, canvas_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        masks = resized
        canvas = masked_canvas
    else:
        canvas = masked_canvas

    return canvas, masks


def mask_to_gray_image(mask: np.ndarray) -> Image.Image:
    gray = (mask.astype(np.uint8)) * 255
    return Image.fromarray(gray, mode="L")


def _load_mask_png(path: Path, size: int) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"))
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
    return gray > 127


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


def load_saved_canvas(stem: str, size: int) -> Image.Image | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = INPUT_DIR / f"{stem}{ext}"
        if path.is_file():
            image = fit_to_canvas_rgb(Image.open(path), size=size)
            return image
    return None


def save_manual_masks(
    stem: str,
    canvas: Image.Image,
    masks: dict[MaskLayer, np.ndarray],
) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = INPUT_DIR / f"{stem}.jpg"
    canvas.save(input_path, format="JPEG", quality=JPEG_QUALITY)

    for layer in MASK_LAYERS:
        mask_to_gray_image(masks[layer]).save(out_dir / f"{layer}.png")
    return out_dir


def compose_overlay(
    canvas: Image.Image,
    masks: dict[MaskLayer, np.ndarray],
    *,
    active_layer: MaskLayer,
) -> Image.Image:
    rgb = np.asarray(canvas.convert("RGB"))
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    active = masks[active_layer]
    color = np.zeros((*active.shape, 4), dtype=np.uint8)
    layer_rgb = LAYER_COLORS[active_layer]
    color[active] = (*layer_rgb, OVERLAY_ALPHA)
    overlay = Image.fromarray(color, mode="RGBA")
    composed = Image.alpha_composite(Image.fromarray(rgba, mode="RGBA"), overlay)
    return composed.convert("RGB")


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


class LabelDataUI:
    def __init__(self, cfg, test_images: list[Path]) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.test_images = test_images
        self.canvas_size = cfg.canvas_size
        self.selected_index = 0
        self.list_scroll = 0
        self.active_layer: MaskLayer = "wall"
        self.brush_mode: BrushMode = "add"
        self.brush_radius = DEFAULT_BRUSH_RADIUS
        self.painting = False
        self.running_inference = False
        self.masks = empty_masks(self.canvas_size)
        self.source_canvas = Image.new("RGB", (self.canvas_size, self.canvas_size), (255, 255, 255))
        self.display_surface: pygame.Surface | None = None
        self.display_rect = pygame.Rect(0, 0, 0, 0)
        self.list_w = LIST_WIDTH
        self.panel_w = PANEL_WIDTH
        self.viewer_x = self.list_w
        self.status = ""
        self._load_test_image(0, initial=True)
        self.display_w = max(1, int(self.canvas_size * DISPLAY_SCALE))
        self.display_h = max(1, int(self.canvas_size * DISPLAY_SCALE))
        self.panel_x = self.viewer_x + self.display_w
        self.window_w = self.panel_x + self.panel_w
        self.window_h = max(self.display_h, 620)
        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Label masks — floorplan-manual-mask")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 12, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self._layout_buttons()
        self._rebuild_display()

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = self.panel_w - 20
        y = 10
        gap = 36
        self.infer_btn = Button(pygame.Rect(x, y, btn_w, 28), "Infer masks")
        y += gap
        self.save_btn = Button(pygame.Rect(x, y, btn_w, 28), "Save masks")
        y += gap + 8
        self.layer_buttons: dict[MaskLayer, Button] = {}
        for layer in STRUCTURE_LAYERS:
            self.layer_buttons[layer] = Button(
                pygame.Rect(x, y, btn_w, 26),
                f"Layer: {layer}",
            )
            y += 30
        y += 4
        self.compute_background_btn = Button(pygame.Rect(x, y, btn_w, 28), "Compute background")
        y += 34
        self.layer_buttons["background"] = Button(
            pygame.Rect(x, y, btn_w, 26),
            "Layer: background",
        )
        y += 30
        y += 6
        self.brush_mode_btn = Button(pygame.Rect(x, y, btn_w, 28), "Brush: add")
        y += gap
        self.brush_size_label_y = y
        y += 22
        self.status_y = y + 8

    def _current_stem(self) -> str:
        return self.test_images[self.selected_index].stem

    def _load_test_image(self, index: int, *, initial: bool = False) -> None:
        path = self.test_images[index]
        self.selected_index = index
        if not initial:
            self._ensure_list_visible(index)
        stem = path.stem
        saved_canvas = load_saved_canvas(stem, self.canvas_size)
        saved_masks = load_saved_masks(stem, self.canvas_size)
        if saved_canvas is not None and saved_masks is not None:
            self.source_canvas = saved_canvas
            self.masks = saved_masks
            self.status = f"Loaded saved masks — {path.name}"
        else:
            self.source_canvas = fit_to_canvas_rgb(Image.open(path), size=self.canvas_size)
            self.masks = empty_masks(self.canvas_size)
            self.status = f"Loaded {path.name} — infer or paint masks."
        self.painting = False
        if not initial:
            self._rebuild_display()

    def _rebuild_display(self) -> None:
        composed = compose_overlay(
            self.source_canvas,
            self.masks,
            active_layer=self.active_layer,
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
        self.status = f"Editing {layer} mask."
        self._rebuild_display()

    def _toggle_brush_mode(self) -> None:
        self.brush_mode = "remove" if self.brush_mode == "add" else "add"
        self.brush_mode_btn.label = f"Brush: {self.brush_mode}"
        self.status = f"Brush mode: {self.brush_mode}."

    def _on_compute_background(self) -> None:
        if self.running_inference:
            return
        self.masks["background"] = compute_background_mask(self.masks["wall"], self.masks["room"])
        self.status = "Background computed from wall + room — edit or save."
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
            canvas, masks = infer_structure_masks(self.cfg, path)
            self.source_canvas = canvas
            self.masks = masks
            self.masks["background"] = np.zeros((self.canvas_size, self.canvas_size), dtype=bool)
            self.status = f"Inference loaded — refine {self.active_layer}, then save."
            self._rebuild_display()
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
        finally:
            self.running_inference = False

    def _on_save(self) -> None:
        if self.running_inference:
            return
        stem = self._current_stem()
        out_dir = save_manual_masks(stem, self.source_canvas, self.masks)
        self.status = f"Saved to {out_dir.relative_to(_REPO_ROOT)}"

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
        if self.compute_background_btn.hit(pos):
            self._on_compute_background()
            return
        if self.brush_mode_btn.hit(pos):
            self._toggle_brush_mode()
            return
        for layer, button in self.layer_buttons.items():
            if button.hit(pos):
                self._set_layer(layer)
                return
        if self.display_rect.collidepoint(pos) and not self.running_inference:
            self.painting = True
            self._paint_at(pos)
            self._rebuild_display()

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, self.list_w, self.window_h))
        header = self.label_font.render("test_image/", True, LABEL_COLOR)
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
        layer_label = f"{self.active_layer} ({self.brush_mode}, r={self.brush_radius})"
        self.window.blit(
            self.label_font.render(layer_label, True, LAYER_COLORS[self.active_layer]),
            (self.viewer_x, 6),
        )
        if self.display_surface is not None:
            self.window.blit(self.display_surface, self.display_rect.topleft)
            pygame.draw.rect(self.window, (70, 70, 70), self.display_rect, 1)
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
        self.compute_background_btn.draw(
            self.window,
            self.btn_font,
            hover=self.compute_background_btn.hit(mouse) and not disabled,
        )
        for layer, button in self.layer_buttons.items():
            button.draw(
                self.window,
                self.btn_font,
                hover=button.hit(mouse) and not disabled,
                active=layer == self.active_layer,
            )
        self.brush_mode_btn.draw(
            self.window,
            self.btn_font,
            hover=self.brush_mode_btn.hit(mouse) and not disabled,
            active=self.brush_mode == "add",
        )
        brush_text = self.small_font.render(
            f"Brush size: {self.brush_radius}px  ([ ] to resize)",
            True,
            LABEL_COLOR,
        )
        self.window.blit(brush_text, (self.panel_x + 10, self.brush_size_label_y))
        hint = self.small_font.render("Paint on image · 1-5 layers · A/R brush", True, LABEL_COLOR)
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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_a:
                        self.brush_mode = "add"
                        self.brush_mode_btn.label = "Brush: add"
                    elif event.key == pygame.K_r:
                        self.brush_mode = "remove"
                        self.brush_mode_btn.label = "Brush: remove"
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
                    mx, _my = pygame.mouse.get_pos()
                    if mx < self.list_w:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.painting = False
                elif event.type == pygame.MOUSEMOTION and self.painting:
                    self._paint_at(event.pos)
                    self._rebuild_display()

            self._draw()
            pygame.display.flip()
            clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual FPN structure mask labeling UI")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    print(f"Ready — {len(test_images)} test image(s), dataset root: {DATASET_ROOT}")
    ui = LabelDataUI(cfg, test_images)
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
