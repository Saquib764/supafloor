#!/usr/bin/env python3
"""Interactive pygame UI: click points to infer room polygons or SAM masks.

Left panel lists test/ images (sorted by name); first image loads by default.
Toggle backbone between Florence (room polygons) and SAM 2.1 (segmentation masks).
Single mode infers immediately on each click. Multiple mode places points first,
then runs one inference when Infer is clicked. Florence multiple mode also supports
optional exclusion points. UNet/VAE buttons apply only in Florence mode.

Usage (from backend/florence_trainer):

    uv sync --group dev
    uv run python dev/infer_ui.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
_DEV = Path(__file__).resolve().parent
TEST_DIR = _ROOT / "test"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

import numpy as np
import pygame
import torch
from PIL import Image

from config import CONFIG_PATH, load_config
from data import (
    format_prompt,
    prepare_image,
)
from loc import parse_referring_room, parse_rooms
from model import load_model_and_processor
from plot import draw_rooms
from pre_clean import pre_clean
from fit_test_canvas import fit_to_canvas

from sam_infer import get_sam_runtime
from unet_infer import get_unet_runtime
from vae_infer import get_vae_runtime

Backbone = Literal["florence", "sam"]

def _generate_for_prompts(model, processor, cfg, image, prompts: list[str]) -> list[str]:
    gen_kwargs = {
        "max_new_tokens": cfg.infer.max_new_tokens,
        "num_beams": cfg.infer.num_beams,
        "do_sample": False,
        "use_cache": True,
    }
    if torch.cuda.is_available():
        inputs = processor(
            text=prompts,
            images=[image] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
        inputs = {
            key: (
                value.to(device=device, dtype=dtype)
                if torch.is_floating_point(value)
                else value.to(device)
            )
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        return processor.batch_decode(generated, skip_special_tokens=False)

    texts: list[str] = []
    for prompt in prompts:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
        inputs = {
            key: (
                value.to(device=device, dtype=dtype)
                if torch.is_floating_point(value)
                else value.to(device)
            )
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        texts.extend(processor.batch_decode(generated, skip_special_tokens=False))
    return texts


def infer_point(model, processor, cfg, image, x: float, y: float) -> dict:
    prompt = format_prompt(
        [(x, y)],
        canvas_size=cfg.canvas_size,
        bins=cfg.loc_bins,
    )
    text = _generate_for_prompts(model, processor, cfg, image, [prompt])[0]
    if text.startswith(prompt):
        text = text[len(prompt) :]
    room = parse_referring_room(text, image_size=cfg.image_size, bins=cfg.loc_bins)
    if room is not None and len(room.get("polygon") or []) >= 3:
        point_scale = cfg.image_size / cfg.canvas_size
        room["sample_point"] = [x * point_scale, y * point_scale]
    return {"prompt": prompt, "raw": text, "room": room}


def infer_points(
    model,
    processor,
    cfg,
    image,
    points: list[tuple[float, float]],
    *,
    excluded_points: list[tuple[float, float]] | None = None,
) -> dict:
    if not points:
        return {"prompt": "", "raw": "", "rooms": []}

    prompt = format_prompt(
        points,
        canvas_size=cfg.canvas_size,
        bins=cfg.loc_bins,
        excluded_points=excluded_points,
    )
    text = _generate_for_prompts(model, processor, cfg, image, [prompt])[0]
    if text.startswith(prompt):
        text = text[len(prompt) :]
    parsed = parse_rooms(text, image_size=cfg.image_size, bins=cfg.loc_bins)
    point_scale = cfg.image_size / cfg.canvas_size
    rooms: list[dict | None] = []
    for index, (x, y) in enumerate(points):
        room = parsed[index] if index < len(parsed) else None
        if room is not None and len(room.get("polygon") or []) >= 3:
            room["sample_point"] = [x * point_scale, y * point_scale]
            rooms.append(room)
        else:
            rooms.append(None)
    return {"prompt": prompt, "raw": text, "rooms": rooms}

PANEL_WIDTH_RATIO = 0.6
LIST_WIDTH = 200
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
SIDE_BG = (28, 28, 28)
LIST_BG = (22, 22, 22)
LIST_HOVER = (45, 45, 45)
LIST_SELECTED = (60, 80, 120)
LIST_TEXT = (220, 220, 220)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
TEXT_COLOR = (235, 235, 235)
STATUS_COLOR = (120, 180, 255)
PENDING_COLOR = (255, 200, 0)
EXCLUDED_COLOR = (255, 90, 90)
DISPLAY_SCALE = 0.8


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def load_viewer_image(path: Path, cfg) -> Image.Image:
    return prepare_image(
        fit_to_canvas(pre_clean(Image.open(path)), size=cfg.canvas_size),
        cfg.image_size,
    )


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgb = image.convert("RGB")
    return pygame.image.frombuffer(rgb.tobytes(), rgb.size, "RGB").copy()


def _wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


@dataclass
class MaskEntry:
    mask: np.ndarray
    points: list[tuple[float, float]]
    score: float


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    visible: bool = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False) -> None:
        if not self.visible:
            return
        color = BTN_HOVER if hover else BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (90, 90, 90), self.rect, 1, border_radius=4)
        text = font.render(self.label, True, BTN_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.visible and self.rect.collidepoint(pos)


class InferUI:
    def __init__(
        self,
        cfg,
        model,
        processor,
        test_images: list[Path],
        *,
        sam_config: Path,
        unet_checkpoint: Path,
        vae_checkpoint: Path,
    ) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.sam_config = sam_config
        self.sam_runtime = None
        self.unet_checkpoint = unet_checkpoint
        self.vae_checkpoint = vae_checkpoint
        self.test_images = test_images
        self.selected_index = 0
        self.list_scroll = 0
        self.backbone: Backbone = "florence"
        self.base_image = load_viewer_image(test_images[0], cfg)
        self.rooms: list[dict] = []
        self.sam_entries: list[MaskEntry] = []
        self.selected_points: list[tuple[int, int]] = []
        self.excluded_points: list[tuple[int, int]] = []
        self.placing_exclusion = False
        self.mode: Literal["single", "multiple"] = "single"
        self.last_prompt = ""
        self.last_raw = ""
        self.status = f"Loaded {test_images[0].name} — florence, single mode, click to infer."
        self.layout_w = cfg.image_size
        self.layout_h = cfg.image_size
        self.display_scale = DISPLAY_SCALE
        self.display_w = max(1, int(self.layout_w * self.display_scale))
        self.display_h = max(1, int(self.layout_h * self.display_scale))
        self.list_w = LIST_WIDTH
        self.viewer_x = self.list_w
        self.panel_w = int(self.layout_w * PANEL_WIDTH_RATIO)
        self.panel_x = self.viewer_x + self.display_w
        self.window_w = self.panel_x + self.panel_w
        self.image_w, self.image_h = self.base_image.size
        self.window = pygame.display.set_mode((self.window_w, self.display_h))
        pygame.display.set_caption("Infer UI — florence")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self._layout_buttons()
        self._rebuild_surface()

    def _sync_image_size(self) -> None:
        self.image_w, self.image_h = self.base_image.size

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = self.panel_w - 20
        y = 10
        self.clear_btn = Button(pygame.Rect(x, y, btn_w, 28), "Clear all")
        y += 34
        self.backbone_btn = Button(pygame.Rect(x, y, btn_w, 28), "Backbone: florence")
        y += 34
        self.unet_btn = Button(pygame.Rect(x, y, btn_w, 28), "UNet")
        y += 34
        self.vae_btn = Button(pygame.Rect(x, y, btn_w, 28), "VAE")
        y += 34
        self.mode_btn = Button(pygame.Rect(x, y, btn_w, 28), "Mode: single")
        y += 34
        self.exclude_btn = Button(
            pygame.Rect(x, y, btn_w, 28),
            "Add exclusion points",
            visible=False,
        )
        y += 34
        self.infer_btn = Button(pygame.Rect(x, y, btn_w, 28), "Infer", visible=False)
        self.text_top = y + 42

    def _list_visible_count(self) -> int:
        return max(1, (self.display_h - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT)

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

    def _viewer_image_for_path(self, path: Path) -> Image.Image:
        if self.backbone == "florence":
            return load_viewer_image(path, self.cfg)
        runtime = self._ensure_sam_runtime()
        return runtime.viewer_image(path)

    def _ensure_sam_runtime(self):
        if self.sam_runtime is None:
            self.status = "Loading SAM 2.1..."
            self._draw()
            pygame.display.flip()
            try:
                self.sam_runtime = get_sam_runtime(str(self.sam_config.resolve()))
            except Exception as exc:
                self.status = f"SAM load failed: {exc}"
                raise
        return self.sam_runtime

    def _load_test_image(self, index: int) -> None:
        path = self.test_images[index]
        self.selected_index = index
        self._ensure_list_visible(index)
        self.base_image = self._viewer_image_for_path(path)
        self.rooms.clear()
        self.sam_entries.clear()
        self._reset_point_selection()
        self.last_prompt = ""
        self.last_raw = ""
        if self.mode == "single":
            self.status = f"Loaded {path.name} — {self.backbone}, click to infer."
        else:
            self._update_multi_status()
        self._sync_image_size()
        self._rebuild_surface()

    def _in_image(self, pos: tuple[int, int]) -> bool:
        mx, my = pos
        return self.viewer_x <= mx < self.panel_x and my < self.display_h

    def _sam_masks_and_points(self) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        masks = [entry.mask for entry in self.sam_entries]
        points = [point for entry in self.sam_entries for point in entry.points]
        return masks, points

    def _rebuild_surface(self) -> None:
        if self.backbone == "florence":
            overlay = draw_rooms(self.base_image.copy(), self.rooms)
        else:
            runtime = self._ensure_sam_runtime()
            masks, points = self._sam_masks_and_points()
            overlay = runtime.overlay(self.base_image, masks, points)
        self.image_surface = _pil_to_surface(overlay)
        self.display_surface = pygame.transform.smoothscale(
            self.image_surface,
            (self.display_w, self.display_h),
        )

    def _screen_to_image(self, sx: int, sy: int) -> tuple[int, int]:
        rx = sx - self.viewer_x
        ix = int(rx * self.image_w / self.display_w) if self.display_w else 0
        iy = int(sy * self.image_h / self.display_h) if self.display_h else 0
        return max(0, min(self.image_w - 1, ix)), max(0, min(self.image_h - 1, iy))

    def _image_to_screen(self, ix: int, iy: int) -> tuple[int, int]:
        sx = self.viewer_x + int(ix * self.display_w / self.image_w)
        sy = int(iy * self.display_h / self.image_h)
        return sx, sy

    def _image_to_canvas(self, x: int, y: int) -> tuple[float, float]:
        scale = self.cfg.canvas_size / self.cfg.image_size
        return x * scale, y * scale

    def _florence_image(self) -> Image.Image:
        return self.base_image

    def _reset_point_selection(self) -> None:
        self.selected_points.clear()
        self.excluded_points.clear()
        self.placing_exclusion = False
        self.exclude_btn.label = "Add exclusion points"

    def _apply_viewer_image(self, output: Image.Image, status: str) -> None:
        if self.backbone == "sam":
            runtime = self._ensure_sam_runtime()
            self.base_image = runtime.fit_output(output)
        else:
            self.base_image = prepare_image(output, self.cfg.image_size)
        self.rooms.clear()
        self.sam_entries.clear()
        self._reset_point_selection()
        self.last_prompt = ""
        self.last_raw = ""
        self.status = status
        self._sync_image_size()
        self._rebuild_surface()

    def _on_unet(self) -> None:
        self.status = "Running UNet (taef2)..."
        self._draw()
        pygame.display.flip()
        try:
            runtime = get_unet_runtime(str(self.unet_checkpoint.resolve()))
            self._apply_viewer_image(runtime.infer(self.base_image), "UNet pass complete — viewer image updated.")
        except Exception as exc:
            self.status = f"UNet failed: {exc}"

    def _on_vae(self) -> None:
        self.status = "Running VAE..."
        self._draw()
        pygame.display.flip()
        try:
            runtime = get_vae_runtime(str(self.vae_checkpoint.resolve()))
            self._apply_viewer_image(runtime.infer(self.base_image), "VAE pass complete — viewer image updated.")
        except Exception as exc:
            self.status = f"VAE failed: {exc}"

    def _florence_only_visible(self) -> bool:
        return self.backbone == "florence"

    def _sync_backbone_controls(self) -> None:
        self.exclude_btn.visible = self._florence_only_visible() and self.mode == "multiple"
        self.backbone_btn.label = f"Backbone: {self.backbone}"
        pygame.display.set_caption(f"Infer UI — {self.backbone}")

    def _toggle_backbone(self) -> None:
        next_backbone: Backbone = "sam" if self.backbone == "florence" else "florence"
        if next_backbone == "sam":
            try:
                self._ensure_sam_runtime()
            except Exception:
                self._draw()
                return

        self.backbone = next_backbone
        self.rooms.clear()
        self.sam_entries.clear()
        self._reset_point_selection()
        self.last_prompt = ""
        self.last_raw = ""
        self.base_image = self._viewer_image_for_path(self.test_images[self.selected_index])
        self._sync_backbone_controls()
        if self.mode == "single":
            self.status = f"Switched to {self.backbone} — click the floor plan to infer."
        else:
            self._update_multi_status()
        self._sync_image_size()
        self._rebuild_surface()

    def _set_mode(self, mode: Literal["single", "multiple"]) -> None:
        self.mode = mode
        self.selected_points.clear()
        self.excluded_points.clear()
        self.placing_exclusion = False
        self.infer_btn.visible = mode == "multiple"
        self.exclude_btn.visible = self._florence_only_visible() and mode == "multiple"
        self.mode_btn.label = f"Mode: {mode}"
        if mode == "single":
            self.status = f"Single mode ({self.backbone}) — click the floor plan to infer immediately."
        else:
            self._update_multi_status()

    def _toggle_exclusion_placement(self) -> None:
        self.placing_exclusion = not self.placing_exclusion
        self.exclude_btn.label = (
            "Add positive points" if self.placing_exclusion else "Add exclusion points"
        )
        self._update_multi_status()

    def _update_multi_status(self) -> None:
        positive = len(self.selected_points)
        if self.backbone == "sam":
            self.status = f"Placing points ({positive} set) — press Infer."
            return
        excluded = len(self.excluded_points)
        if self.placing_exclusion:
            self.status = (
                f"Placing exclusion points ({excluded} set, {positive} positive) — press Infer."
            )
        else:
            self.status = (
                f"Placing positive points ({positive} set, {excluded} excluded) — press Infer."
            )

    def _toggle_mode(self) -> None:
        self._set_mode("multiple" if self.mode == "single" else "single")

    def _clear_all(self) -> None:
        self.rooms.clear()
        self.sam_entries.clear()
        self._reset_point_selection()
        self.last_prompt = ""
        self.last_raw = ""
        if self.mode == "multiple":
            self._update_multi_status()
        elif self.backbone == "sam":
            self.status = "Cleared all masks."
        else:
            self.status = "Cleared all polygons."
        self._rebuild_surface()

    def _append_sam_entry(
        self,
        mask: np.ndarray,
        points: list[tuple[float, float]],
        score: float,
    ) -> None:
        self.sam_entries.append(MaskEntry(mask=mask, points=points, score=score))

    def _apply_result(self, result: dict, image_x: int, image_y: int) -> bool:
        self.last_prompt = result["prompt"]
        self.last_raw = result["raw"]
        room = result["room"]
        if room is not None and len(room.get("polygon") or []) >= 3:
            self.rooms.append(room)
            name = room.get("name") or "room"
            self.status = f"Inferred {name} at ({image_x}, {image_y}) — {len(self.rooms)} polygon(s) shown"
            self._rebuild_surface()
            return True
        self.status = f"No valid polygon at ({image_x}, {image_y})"
        return False

    def _on_sam_image_click(self, x: int, y: int) -> None:
        runtime = self._ensure_sam_runtime()
        if self.mode == "single":
            self.status = f"Inferring point ({x}, {y})..."
            self._draw()
            pygame.display.flip()
            mask, score = runtime.predict_point(self.base_image, (float(x), float(y)))
            self._append_sam_entry(mask, [(float(x), float(y))], score)
            self.status = (
                f"Mask {len(self.sam_entries)} at ({x}, {y}) score={score:.3f} — click to add more."
            )
            self._rebuild_surface()
            return

        if (x, y) not in self.selected_points:
            self.selected_points.append((x, y))
        self._update_multi_status()

    def _on_sam_infer_batch(self) -> None:
        if not self.selected_points:
            self.status = "No points selected."
            return

        runtime = self._ensure_sam_runtime()
        count = len(self.selected_points)
        self.status = f"Inferring {count} point(s)..."
        self._draw()
        pygame.display.flip()

        canvas_points = [(float(x), float(y)) for x, y in self.selected_points]
        mask, score = runtime.predict_points(self.base_image, canvas_points)
        self._append_sam_entry(mask, canvas_points, score)
        self._reset_point_selection()
        self.status = (
            f"Inferred mask from {count} point(s) score={score:.3f} — {len(self.sam_entries)} total."
        )
        self._rebuild_surface()

    def _on_image_click(self, x: int, y: int) -> None:
        if self.backbone == "sam":
            self._on_sam_image_click(x, y)
            return

        if self.mode == "single":
            canvas_x, canvas_y = self._image_to_canvas(x, y)
            self.status = f"Inferring point ({x}, {y})..."
            self._draw()
            pygame.display.flip()
            result = infer_point(
                self.model,
                self.processor,
                self.cfg,
                self._florence_image(),
                canvas_x,
                canvas_y,
            )
            self._apply_result(result, x, y)
            return

        target = self.excluded_points if self.placing_exclusion else self.selected_points
        if (x, y) not in target:
            target.append((x, y))
        self._update_multi_status()

    def _on_infer_batch(self) -> None:
        if self.backbone == "sam":
            self._on_sam_infer_batch()
            return

        if not self.selected_points:
            self.status = "No positive points selected."
            return

        canvas_points = [self._image_to_canvas(x, y) for x, y in self.selected_points]
        canvas_excluded = [self._image_to_canvas(x, y) for x, y in self.excluded_points]
        count = len(canvas_points)
        self.status = f"Inferring {count} point(s)..."
        self._draw()
        pygame.display.flip()

        result = infer_points(
            self.model,
            self.processor,
            self.cfg,
            self._florence_image(),
            canvas_points,
            excluded_points=canvas_excluded or None,
        )
        self._reset_point_selection()

        ok = 0
        for room in result["rooms"]:
            if room is not None:
                self.rooms.append(room)
                ok += 1

        self.last_prompt = result["prompt"]
        self.last_raw = result["raw"]
        self.status = f"Inferred {ok}/{count} polygon(s) — {len(self.rooms)} total shown"
        self._rebuild_surface()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_test_image(list_index)
            return
        if self.clear_btn.hit(pos):
            self._clear_all()
            return
        if self.backbone_btn.hit(pos):
            self._toggle_backbone()
            return
        if self.unet_btn.hit(pos):
            self._on_unet()
            return
        if self.vae_btn.hit(pos):
            self._on_vae()
            return
        if self.mode_btn.hit(pos):
            self._toggle_mode()
            return
        if self.exclude_btn.hit(pos):
            self._toggle_exclusion_placement()
            return
        if self.infer_btn.hit(pos):
            self._on_infer_batch()
            return

        mx, my = pos
        if self._in_image(pos):
            self._on_image_click(*self._screen_to_image(mx, my))

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, self.list_w, self.display_h))
        header = self.label_font.render("test/", True, LABEL_COLOR)
        self.window.blit(header, (8, 4))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.list_w - 1, 0),
            (self.list_w - 1, self.display_h),
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

    def _draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_w, self.display_h)
        pygame.draw.rect(self.window, SIDE_BG, panel_rect)
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.display_h),
            1,
        )

        mouse = pygame.mouse.get_pos()
        for btn in (
            self.clear_btn,
            self.backbone_btn,
            self.unet_btn,
            self.vae_btn,
            self.mode_btn,
            self.exclude_btn,
            self.infer_btn,
        ):
            btn.draw(self.window, self.btn_font, hover=btn.hit(mouse))

        x = self.panel_x + 10
        y = self.text_top
        max_width = self.panel_w - 20

        for line in _wrap_text(self.font, self.status, max_width):
            status = self.label_font.render(line, True, STATUS_COLOR)
            self.window.blit(status, (x, y))
            y += 18
        y += 8

        if self.backbone == "sam":
            label = self.label_font.render("MASKS:", True, LABEL_COLOR)
            self.window.blit(label, (x, y))
            y += 18
            if not self.sam_entries:
                line = self.font.render("(none)", True, TEXT_COLOR)
                self.window.blit(line, (x, y))
            else:
                for index, entry in enumerate(self.sam_entries, start=1):
                    if y > self.display_h - 16:
                        break
                    if len(entry.points) == 1:
                        px, py = entry.points[0]
                        text = f"{index}. ({px:.0f}, {py:.0f}) score={entry.score:.3f}"
                    else:
                        text = f"{index}. {len(entry.points)} pts score={entry.score:.3f}"
                    line = self.font.render(text, True, TEXT_COLOR)
                    self.window.blit(line, (x, y))
                    y += 16
            return

        for label, text in (("INPUT", self.last_prompt), ("OUTPUT", self.last_raw)):
            label_surface = self.label_font.render(f"{label}:", True, LABEL_COLOR)
            self.window.blit(label_surface, (x, y))
            y += 18
            for line in _wrap_text(self.font, text, max_width):
                if y > self.display_h - 16:
                    break
                line_surface = self.font.render(line, True, TEXT_COLOR)
                self.window.blit(line_surface, (x, y))
                y += 16
            y += 8

    def _draw(self) -> None:
        self.window.fill((18, 18, 18))
        self._draw_file_list()
        self.window.blit(self.display_surface, (self.viewer_x, 0))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.display_h),
            1,
        )
        for px, py in self.selected_points:
            sx, sy = self._image_to_screen(px, py)
            pygame.draw.circle(self.window, PENDING_COLOR, (sx, sy), 5)
            pygame.draw.circle(self.window, (44, 44, 44), (sx, sy), 5, 1)
        for px, py in self.excluded_points:
            sx, sy = self._image_to_screen(px, py)
            pygame.draw.circle(self.window, EXCLUDED_COLOR, (sx, sy), 5)
            pygame.draw.circle(self.window, (44, 44, 44), (sx, sy), 5, 1)
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
                    if mx < self.list_w:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            self._draw()
            pygame.display.flip()
            clock.tick(60)


_SAM_ROOT = _ROOT.parent / "sam_trainer"
_UNET_ROOT = _ROOT.parent / "unet_trainer"
_VAE_ROOT = _ROOT.parent / "vae_trainer"
_DEFAULT_SAM_CONFIG = _SAM_ROOT / "config.yaml"
_DEFAULT_UNET_CHECKPOINT = _UNET_ROOT / "training_session" / "last.pt"
_DEFAULT_VAE_CHECKPOINT = _VAE_ROOT / "training_session" / "last.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Click floor-plan points to infer room polygons")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Florence config path")
    parser.add_argument(
        "--sam-config",
        type=Path,
        default=_DEFAULT_SAM_CONFIG,
        help="SAM config path (default: sam_trainer/config.yaml)",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=_DEFAULT_UNET_CHECKPOINT,
        help="UNet trainable weights (default: unet_trainer/training_session/last.pt)",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=_DEFAULT_VAE_CHECKPOINT,
        help="VAE weights (default: vae_trainer/training_session/last.pt)",
    )
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    adapter = args.adapter or cfg.train.session_dir

    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    print(f"Loading model from {adapter}...")
    model, processor = load_model_and_processor(cfg, adapter=adapter)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")

    print(f"Ready — {len(test_images)} test image(s), showing {test_images[0].name}.")
    ui = InferUI(
        cfg,
        model,
        processor,
        test_images,
        sam_config=args.sam_config,
        unet_checkpoint=args.unet_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
    )
    ui._sync_backbone_controls()
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
