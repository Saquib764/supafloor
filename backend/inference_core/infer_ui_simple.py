#!/usr/bin/env python3
"""Simple pygame UI for the FPN inference pipeline.

Left panel lists test_image/ files; click to preview the source floor plan.
Right panel has Run old / Run latest and Clear buttons for the selected image.
The viewer compares old vs latest checkpoints side by side for wall-subtract
(per-room colored components on the raw input), wall centerline, and skimage
skeleton. The background image is never region-masked in the viewer.

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python infer_ui_simple.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CheckpointVariant = Literal["old", "latest"]

import pygame
from PIL import Image

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_rgb

DOOR_COLOR = (220, 50, 50)
WINDOW_COLOR = (64, 255, 64)
WALL_OVERLAY_COLOR = (0, 100, 0)

LEGEND_ENTRIES: list[tuple[str, tuple[int, int, int]]] = [
    ("Door", DOOR_COLOR),
    ("Window", WINDOW_COLOR),
    ("Wall overlay", WALL_OVERLAY_COLOR),
]

_ROOT = Path(__file__).resolve().parent
TEST_DIR = _ROOT / "test_image"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LIST_WIDTH = 200
PANEL_WIDTH = 220
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
VIEWER_GAP = 8
DISPLAY_SCALE = 0.28

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


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def wall_subtract_overlay_from_jpg(overlays_path: Path, panel_size: int) -> Image.Image:
    """Middle panel of the first row in room-mask-overlays.jpg (wall-subtract)."""
    image = Image.open(overlays_path)
    return image.crop((panel_size, 0, 2 * panel_size, panel_size))


@dataclass
class ViewerPanel:
    label: str
    image: Image.Image | None = None


@dataclass
class DisplayRow:
    kind: Literal["pair", "single"]
    panels: list[tuple[str, pygame.Surface, int]]


def saved_output_paths(
    output_dir: Path,
    image_path: Path,
    *,
    variant: CheckpointVariant,
) -> dict[str, Path]:
    step_dir = output_dir / variant / image_path.stem / "step-003"
    structure_dir = step_dir / "structure-polygons"
    return {
        "overlays": step_dir / "room-mask-overlays.jpg",
        "wall_centerline": structure_dir / "wall-centerline.jpg",
        "skimage_skeleton": structure_dir / "wall-centerline-skimage-skeleton.jpg",
    }


def panels_from_source(
    source: Image.Image,
    paths: dict[str, Path],
    *,
    component_size: int,
) -> list[ViewerPanel]:
    panels = [
        ViewerPanel("Wall-subtract", None),
        ViewerPanel("Wall centerline", None),
        ViewerPanel("Skimage skeleton", None),
    ]
    if paths["overlays"].is_file():
        panels[0].image = wall_subtract_overlay_from_jpg(paths["overlays"], component_size)
    if paths["wall_centerline"].is_file():
        panels[1].image = Image.open(paths["wall_centerline"]).convert("RGB")
    else:
        panels[1].label = "Source"
        panels[1].image = source
    if paths["skimage_skeleton"].is_file():
        panels[2].image = Image.open(paths["skimage_skeleton"]).convert("RGB")
    return panels


def load_result_panels(
    paths: dict[str, Path],
    *,
    component_size: int,
) -> list[ViewerPanel]:
    blank = Image.new("RGB", (component_size, component_size), EMPTY_SLOT_BG)
    return panels_from_source(blank, paths, component_size=component_size)


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgb = image.convert("RGB")
    return pygame.image.frombuffer(rgb.tobytes(), rgb.size, "RGB").copy()


def _scaled_size(image: Image.Image, target_width: int) -> tuple[int, int]:
    width, height = image.size
    if width <= 0:
        return target_width, target_width
    scale = target_width / width
    return target_width, max(1, int(height * scale))


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


class InferUISimple:
    def __init__(self, cfg, test_images: list[Path], *, config_path: Path) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.config_path = config_path
        self.test_images = test_images
        self.selected_index = 0
        self.list_scroll = 0
        self.status = f"Loaded {test_images[0].name} — run old or latest."
        self.running_inference: CheckpointVariant | None = None
        self.checkpoint_by_variant: dict[CheckpointVariant, Path] = {
            "old": cfg.fpn.checkpoint_old,
            "latest": cfg.fpn.checkpoint_latest,
        }
        self.output_dir_by_variant: dict[CheckpointVariant, Path] = {
            variant: cfg.fpn.output_dir / variant for variant in ("old", "latest")
        }
        self.component_size = cfg.fpn.size
        self.col_width = max(1, int(self.component_size * DISPLAY_SCALE))
        self.viewer_area_width = self.col_width
        self.list_w = LIST_WIDTH
        self.panel_w = PANEL_WIDTH
        self.viewer_x = self.list_w
        self._load_test_image(0, initial=True)
        self._update_layout_geometry()
        self.window_h = self._content_height()
        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Infer UI — FPN pipeline")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self._layout_buttons()

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = self.panel_w - 20
        self.infer_old_btn = Button(pygame.Rect(x, 10, btn_w, 32), "Run old")
        self.infer_latest_btn = Button(pygame.Rect(x, 48, btn_w, 32), "Run latest")
        self.clear_btn = Button(pygame.Rect(x, 86, btn_w, 32), "Clear")
        self.status_y = 132

    def _update_layout_geometry(self) -> None:
        self.panel_x = self.viewer_x + self.viewer_area_width
        self.window_w = self.panel_x + self.panel_w
        if hasattr(self, "infer_old_btn"):
            self._layout_buttons()

    def _scale_panel(self, image: Image.Image, width: int) -> tuple[pygame.Surface, int]:
        surface = _pil_to_surface(image)
        target_w, target_h = _scaled_size(image, width)
        return pygame.transform.smoothscale(surface, (target_w, target_h)), target_h

    def _empty_slot_surface(self, width: int, height: int) -> pygame.Surface:
        surface = pygame.Surface((width, height))
        surface.fill(EMPTY_SLOT_BG)
        pygame.draw.rect(surface, EMPTY_SLOT_BORDER, surface.get_rect(), 1)
        return surface

    def _slot_surface(
        self,
        image: Image.Image | None,
        width: int,
        *,
        fallback_height: int,
    ) -> tuple[pygame.Surface, int]:
        if image is not None:
            return self._scale_panel(image, width)
        height = max(1, fallback_height)
        return self._empty_slot_surface(width, height), height

    def _skimage_row_height(self, full_width: int) -> int:
        return max(1, int(full_width * self.component_size / (self.component_size * 5)))

    def _content_height(self) -> int:
        height = 0
        for index, row in enumerate(self.display_rows):
            if row.kind == "pair":
                row_h = max(surface.get_height() for _label, surface, _h in row.panels)
            else:
                _label, surface, _h = row.panels[0]
                row_h = surface.get_height()
            height += 18 + row_h
            if index + 1 < len(self.display_rows):
                height += VIEWER_GAP
        return max(height, 480)

    def _resize_window(self) -> None:
        self._update_layout_geometry()
        self.window_h = self._content_height()
        self.window = pygame.display.set_mode((self.window_w, self.window_h))

    def _rebuild_surfaces(self) -> None:
        full_width = self.col_width * 2 + VIEWER_GAP
        self.viewer_area_width = full_width
        self.display_rows = []
        for panel_index, panel_name in enumerate(("Wall-subtract", "Wall centerline", "Skimage skeleton")):
            old_panel = self.panels_by_variant["old"][panel_index]
            latest_panel = self.panels_by_variant["latest"][panel_index]
            fallback_height = (
                self._skimage_row_height(full_width)
                if panel_name == "Skimage skeleton"
                else self.col_width
            )
            old_surface, old_h = self._slot_surface(
                old_panel.image,
                self.col_width,
                fallback_height=fallback_height,
            )
            latest_surface, latest_h = self._slot_surface(
                latest_panel.image,
                self.col_width,
                fallback_height=fallback_height,
            )
            self.display_rows.append(
                DisplayRow(
                    "pair",
                    [
                        (f"Old · {panel_name}", old_surface, old_h),
                        (f"Latest · {panel_name}", latest_surface, latest_h),
                    ],
                )
            )
        self._update_layout_geometry()

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

    def _output_dir_for(self, image_path: Path, variant: CheckpointVariant) -> Path:
        return self.output_dir_by_variant[variant] / image_path.stem

    def _saved_output_paths(
        self,
        image_path: Path,
        variant: CheckpointVariant,
    ) -> dict[str, Path]:
        return saved_output_paths(self.cfg.fpn.output_dir, image_path, variant=variant)

    def _source_canvas(self, image_path: Path) -> Image.Image:
        return fit_to_canvas_rgb(Image.open(image_path), size=self.cfg.canvas_size)

    def _panels_for_path(self, image_path: Path) -> dict[CheckpointVariant, list[ViewerPanel]]:
        source = self._source_canvas(image_path)
        return {
            variant: panels_from_source(
                source,
                self._saved_output_paths(image_path, variant),
                component_size=self.component_size,
            )
            for variant in ("old", "latest")
        }

    def _variant_has_outputs(self, image_path: Path, variant: CheckpointVariant) -> bool:
        paths = self._saved_output_paths(image_path, variant)
        return all(path_entry.is_file() for path_entry in paths.values())

    def _load_test_image(self, index: int, *, initial: bool = False) -> None:
        path = self.test_images[index]
        self.selected_index = index
        if not initial:
            self._ensure_list_visible(index)
        self.source_image = self._source_canvas(path)
        self.panels_by_variant = self._panels_for_path(path)
        has_old = self._variant_has_outputs(path, "old")
        has_latest = self._variant_has_outputs(path, "latest")
        if has_old and has_latest:
            self.status = f"Loaded old + latest outputs — {path.name}"
        elif has_old or has_latest:
            ready = "old" if has_old else "latest"
            self.status = f"Loaded {ready} output — {path.name}"
        else:
            self.status = f"Loaded {path.name} — run old or latest."
        self._rebuild_surfaces()
        if not initial:
            self._resize_window()

    def _reload_inference_results(self, variant: CheckpointVariant) -> None:
        path = self.test_images[self.selected_index]
        paths = self._saved_output_paths(path, variant)
        missing = [name for name, output_path in paths.items() if not output_path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {variant} pipeline outputs: {', '.join(missing)}")
        self.panels_by_variant[variant] = load_result_panels(
            paths,
            component_size=self.component_size,
        )
        self.status = f"{variant.capitalize()} inference complete — {path.name}"
        self._rebuild_surfaces()
        self._resize_window()

    def _on_run_inference(self, variant: CheckpointVariant) -> None:
        if self.running_inference is not None:
            return
        path = self.test_images[self.selected_index]
        self.running_inference = variant
        self.status = f"Running {variant} checkpoint on {path.name}..."
        self._draw()
        pygame.display.flip()
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "infer_fpn.py"),
                    "--config",
                    str(self.config_path),
                    "--image",
                    str(path),
                    "--checkpoint",
                    str(self.checkpoint_by_variant[variant]),
                    "--output-dir",
                    str(self.output_dir_by_variant[variant]),
                ],
                check=True,
                cwd=str(_ROOT),
            )
            self._reload_inference_results(variant)
        except subprocess.CalledProcessError as exc:
            self.status = f"{variant.capitalize()} inference failed (exit {exc.returncode})"
        except Exception as exc:
            self.status = f"{variant.capitalize()} inference failed: {exc}"
        finally:
            self.running_inference = None

    def _on_clear(self) -> None:
        if self.running_inference is not None:
            return
        path = self.test_images[self.selected_index]
        cleared = False
        for variant in ("old", "latest"):
            out_dir = self._output_dir_for(path, variant)
            if out_dir.is_dir():
                shutil.rmtree(out_dir)
                cleared = True
        if cleared:
            self.status = f"Cleared old + latest outputs — {path.name}"
        else:
            self.status = f"No outputs to clear — {path.name}"
        self.source_image = self._source_canvas(path)
        self.panels_by_variant = self._panels_for_path(path)
        self._rebuild_surfaces()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_test_image(list_index)
            return
        if self.infer_old_btn.hit(pos):
            self._on_run_inference("old")
            return
        if self.infer_latest_btn.hit(pos):
            self._on_run_inference("latest")
            return
        if self.clear_btn.hit(pos):
            self._on_clear()

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
        y = 0
        for row in self.display_rows:
            if row.kind == "pair":
                (left_label, left_surface, _left_h), (right_label, right_surface, _right_h) = row.panels
                self.window.blit(
                    self.label_font.render(left_label, True, PANEL_LABEL_COLOR),
                    (self.viewer_x, y),
                )
                self.window.blit(
                    self.label_font.render(right_label, True, PANEL_LABEL_COLOR),
                    (self.viewer_x + self.col_width + VIEWER_GAP, y),
                )
                y += 18
                self.window.blit(left_surface, (self.viewer_x, y))
                self.window.blit(
                    right_surface,
                    (self.viewer_x + self.col_width + VIEWER_GAP, y),
                )
                y += max(left_surface.get_height(), right_surface.get_height()) + VIEWER_GAP
                continue

            label, surface, _height = row.panels[0]
            self.window.blit(
                self.label_font.render(label, True, PANEL_LABEL_COLOR),
                (self.viewer_x, y),
            )
            y += 18
            self.window.blit(surface, (self.viewer_x, y))
            y += surface.get_height() + VIEWER_GAP

    def _draw_side_panel(self) -> None:
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_w, self.window_h)
        pygame.draw.rect(self.window, SIDE_BG, panel_rect)
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        disabled = self.running_inference is not None
        self.infer_old_btn.draw(
            self.window,
            self.btn_font,
            hover=self.infer_old_btn.hit(mouse) and not disabled,
        )
        self.infer_latest_btn.draw(
            self.window,
            self.btn_font,
            hover=self.infer_latest_btn.hit(mouse) and not disabled,
        )
        self.clear_btn.draw(
            self.window,
            self.btn_font,
            hover=self.clear_btn.hit(mouse) and not disabled,
        )
        status = self.status
        if self.running_inference is not None:
            status = f"{status} (please wait)"
        wrapped = self._wrap_text(status, self.panel_w - 20)
        y = self.status_y
        for line in wrapped:
            text = self.font.render(line, True, STATUS_COLOR)
            self.window.blit(text, (self.panel_x + 10, y))
            y += 18
        self._draw_legend(y + 10)

    def _draw_legend(self, start_y: int) -> None:
        header = self.label_font.render("Legend", True, PANEL_LABEL_COLOR)
        self.window.blit(header, (self.panel_x + 10, start_y))
        y = start_y + 20
        for label, color in LEGEND_ENTRIES:
            swatch = pygame.Rect(self.panel_x + 10, y + 2, 14, 14)
            pygame.draw.rect(self.window, color, swatch)
            pygame.draw.rect(self.window, (90, 90, 90), swatch, 1)
            text = self.font.render(label, True, LIST_TEXT)
            self.window.blit(text, (self.panel_x + 30, y))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple FPN pipeline inference UI")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    print(f"Ready — {len(test_images)} test image(s), showing {test_images[0].name}.")
    ui = InferUISimple(cfg, test_images, config_path=args.config)
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
