#!/usr/bin/env python3
"""Interactive pygame UI: click a point to infer a room polygon.

Usage (from backend/florence_trainer):

    uv sync --group dev
    uv run python dev/infer_ui.py --image test/fp1.jpg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

MAX_POINTS_PER_PROMPT = 6


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


def infer_points(model, processor, cfg, image, points: list[tuple[float, float]]) -> list[dict]:
    if not points:
        return []

    batches = [
        points[i : i + MAX_POINTS_PER_PROMPT]
        for i in range(0, len(points), MAX_POINTS_PER_PROMPT)
    ]
    prompts = [
        format_prompt(
            batch,
            canvas_size=cfg.canvas_size,
            bins=cfg.loc_bins,
        )
        for batch in batches
    ]
    texts = _generate_for_prompts(model, processor, cfg, image, prompts)
    point_scale = cfg.image_size / cfg.canvas_size
    results: list[dict] = []
    for batch, prompt, text in zip(batches, prompts, texts, strict=True):
        if text.startswith(prompt):
            text = text[len(prompt) :]
        rooms = parse_rooms(text, image_size=cfg.image_size, bins=cfg.loc_bins)
        for index, (x, y) in enumerate(batch):
            room = rooms[index] if index < len(rooms) else None
            if room is not None and len(room.get("polygon") or []) >= 3:
                room["sample_point"] = [x * point_scale, y * point_scale]
            else:
                room = None
            results.append({"prompt": prompt, "raw": text, "room": room})
    return results

PANEL_WIDTH_RATIO = 0.6
SIDE_BG = (28, 28, 28)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
TEXT_COLOR = (235, 235, 235)
STATUS_COLOR = (120, 180, 255)
PENDING_COLOR = (255, 200, 0)
DISPLAY_SCALE = 0.8


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
    def __init__(self, cfg, model, processor, image: Image.Image) -> None:
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.base_image = image
        self.rooms: list[dict] = []
        self.pending_points: list[tuple[int, int]] = []
        self.mode: Literal["single", "multiple"] = "single"
        self.last_prompt = ""
        self.last_raw = ""
        self.status = "Single mode — click the floor plan to infer immediately."
        self.image_w, self.image_h = self.base_image.size
        self.display_scale = DISPLAY_SCALE
        self.display_w = int(self.image_w * self.display_scale)
        self.display_h = int(self.image_h * self.display_scale)
        self.panel_w = int(self.image_w * PANEL_WIDTH_RATIO)
        self.window_w = self.display_w + self.panel_w
        self.window = pygame.display.set_mode((self.window_w, self.display_h))
        pygame.display.set_caption("Florence infer UI")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self._layout_buttons()
        self._rebuild_surface()

    def _layout_buttons(self) -> None:
        x = self.display_w + 10
        btn_w = self.panel_w - 20
        y = 10
        self.clear_btn = Button(pygame.Rect(x, y, btn_w, 28), "Clear all")
        y += 34
        self.mode_btn = Button(pygame.Rect(x, y, btn_w, 28), "Mode: single")
        y += 34
        self.infer_btn = Button(pygame.Rect(x, y, btn_w, 28), "Infer", visible=False)
        self.text_top = y + 42

    def _in_image(self, pos: tuple[int, int]) -> bool:
        mx, my = pos
        return mx < self.display_w and my < self.display_h

    def _rebuild_surface(self) -> None:
        overlay = draw_rooms(self.base_image.copy(), self.rooms)
        self.image_surface = _pil_to_surface(overlay)
        self.display_surface = pygame.transform.smoothscale(
            self.image_surface,
            (self.display_w, self.display_h),
        )

    def _screen_to_image(self, sx: int, sy: int) -> tuple[int, int]:
        ix = int(sx / self.display_scale)
        iy = int(sy / self.display_scale)
        return max(0, min(self.image_w - 1, ix)), max(0, min(self.image_h - 1, iy))

    def _image_to_screen(self, ix: int, iy: int) -> tuple[int, int]:
        return int(ix * self.display_scale), int(iy * self.display_scale)

    def _image_to_canvas(self, x: int, y: int) -> tuple[float, float]:
        scale = self.cfg.canvas_size / self.cfg.image_size
        return x * scale, y * scale

    def _set_mode(self, mode: Literal["single", "multiple"]) -> None:
        self.mode = mode
        self.pending_points.clear()
        self.infer_btn.visible = mode == "multiple"
        self.mode_btn.label = f"Mode: {mode}"
        if mode == "single":
            self.status = "Single mode — click the floor plan to infer immediately."
        else:
            self.status = "Multiple mode — click up to 6 points per Infer batch."

    def _toggle_mode(self) -> None:
        self._set_mode("multiple" if self.mode == "single" else "single")

    def _clear_all(self) -> None:
        self.rooms.clear()
        self.pending_points.clear()
        self.last_prompt = ""
        self.last_raw = ""
        self.status = "Cleared all polygons."
        self._rebuild_surface()

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

    def _on_image_click(self, x: int, y: int) -> None:
        if self.mode == "single":
            canvas_x, canvas_y = self._image_to_canvas(x, y)
            self.status = f"Inferring point ({x}, {y})..."
            self._draw()
            pygame.display.flip()
            result = infer_point(
                self.model,
                self.processor,
                self.cfg,
                self.base_image,
                canvas_x,
                canvas_y,
            )
            self._apply_result(result, x, y)
            return

        if (x, y) not in self.pending_points:
            self.pending_points.append((x, y))
        self.status = f"{len(self.pending_points)} point(s) queued — press Infer (up to 6 per batch)."

    def _on_infer_batch(self) -> None:
        if not self.pending_points:
            self.status = "No points queued."
            return

        canvas_points = [self._image_to_canvas(x, y) for x, y in self.pending_points]
        image_points = list(self.pending_points)
        count = len(canvas_points)
        self.status = f"Inferring {count} point(s)..."
        self._draw()
        pygame.display.flip()

        results = infer_points(
            self.model,
            self.processor,
            self.cfg,
            self.base_image,
            canvas_points,
        )
        self.pending_points.clear()

        prompts: list[str] = []
        raws: list[str] = []
        ok = 0
        seen_batches: set[tuple[str, str]] = set()
        for (ix, iy), result in zip(image_points, results, strict=True):
            batch_key = (result["prompt"], result["raw"])
            if batch_key not in seen_batches:
                prompts.append(result["prompt"])
                raws.append(result["raw"])
                seen_batches.add(batch_key)
            room = result["room"]
            if room is not None and len(room.get("polygon") or []) >= 3:
                self.rooms.append(room)
                ok += 1

        self.last_prompt = "\n---\n".join(prompts)
        self.last_raw = "\n---\n".join(raws)
        self.status = f"Inferred {ok}/{count} polygon(s) — {len(self.rooms)} total shown"
        self._rebuild_surface()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.clear_btn.hit(pos):
            self._clear_all()
            return
        if self.mode_btn.hit(pos):
            self._toggle_mode()
            return
        if self.infer_btn.hit(pos):
            self._on_infer_batch()
            return

        mx, my = pos
        if self._in_image(pos):
            self._on_image_click(*self._screen_to_image(mx, my))

    def _draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(self.display_w, 0, self.panel_w, self.display_h)
        pygame.draw.rect(self.window, SIDE_BG, panel_rect)
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.display_w, 0),
            (self.display_w, self.display_h),
            1,
        )

        mouse = pygame.mouse.get_pos()
        for btn in (self.clear_btn, self.mode_btn, self.infer_btn):
            btn.draw(self.window, self.btn_font, hover=btn.hit(mouse))

        x = self.display_w + 10
        y = self.text_top
        max_width = self.panel_w - 20

        for line in _wrap_text(self.font, self.status, max_width):
            status = self.label_font.render(line, True, STATUS_COLOR)
            self.window.blit(status, (x, y))
            y += 18
        y += 8

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
        self.window.blit(self.display_surface, (0, 0))
        for px, py in self.pending_points:
            sx, sy = self._image_to_screen(px, py)
            pygame.draw.circle(self.window, PENDING_COLOR, (sx, sy), 5)
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
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            self._draw()
            pygame.display.flip()
            clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Click floor-plan points to infer room polygons")
    parser.add_argument("--image", type=Path, required=True, help="Source floor-plan image")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    adapter = args.adapter or cfg.train.session_dir

    print(f"Loading model from {adapter}...")
    model, processor = load_model_and_processor(cfg, adapter=adapter)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")

    image = prepare_image(pre_clean(Image.open(args.image)), cfg.image_size)
    print("Ready — click on the floor plan to infer polygons.")
    InferUI(cfg, model, processor, image).run()
    pygame.quit()


if __name__ == "__main__":
    main()
