#!/usr/bin/env python3
"""Pygame UI to generate similar floor plans via Replicate image edit.

Left panel lists test_image/ files; center viewer shows the source floor plan.
Right panel has prompt, Generate, and the latest output thumbnail.

Usage (from backend/inference_core):

    uv sync --group dev
    uv run python infer_ui_similar.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pygame
from PIL import Image

from config import CONFIG_PATH, load_config
from util.canvas import fit_to_canvas_rgb
from util.replicate_client import run_replicate_edit

DEFAULT_PROMPT = (
    "Create a new floor plan in the same style as the reference image. "
    "Preserve the wall thickness, door symbols, window symbols, line weights, colors, "
    "and overall CAD/architectural drawing aesthetic. "
    "Design a completely new room layout and configuration — different room arrangement, "
    "sizes, and labels — while keeping the same visual style for walls, doors, and windows."
)

_ROOT = Path(__file__).resolve().parent
TEST_DIR = _ROOT / "test_image"
OUTPUT_DIR = _ROOT / "test_floorplan"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LIST_WIDTH = 200
PANEL_WIDTH = 340
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
VIEWER_PADDING_Y = 6
DISPLAY_SCALE = 0.8
PROMPT_LABEL_HEIGHT = 20
PROMPT_MARGIN = 10
BTN_HEIGHT = 36
STATUS_TOP_GAP = 12
THUMB_SIZE = 72
THUMB_HEADER = 18
THUMB_TOP_GAP = 10
THUMB_BG = (32, 32, 32)
THUMB_SELECTED = (100, 140, 200)

LIST_BG = (22, 22, 22)
LIST_HOVER = (45, 45, 45)
LIST_SELECTED = (60, 80, 120)
LIST_TEXT = (220, 220, 220)
SIDE_BG = (28, 28, 28)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
STATUS_COLOR = (120, 180, 255)
PANEL_LABEL_COLOR = (200, 200, 200)
PROMPT_BG = (18, 18, 18)
PROMPT_BORDER = (90, 90, 90)
PROMPT_TEXT = (230, 230, 230)
PROMPT_FOCUS_BORDER = (100, 140, 200)
CURSOR_COLOR = (240, 240, 240)


def list_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _generated_index(path: Path, stem: str) -> int | None:
    prefix = f"{stem}_"
    if not path.stem.startswith(prefix):
        return None
    suffix = path.stem[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    if len(suffix) >= 3 and suffix[0] == "[" and suffix[-1] == "]" and suffix[1:-1].isdigit():
        return int(suffix[1:-1])
    return None


def _generated_paths_for(image_path: Path) -> list[Path]:
    stem = image_path.stem
    paths = [p for p in OUTPUT_DIR.glob(f"{stem}_*.jpg") if _generated_index(p, stem) is not None]
    return sorted(paths, key=lambda p: _generated_index(p, stem) or 0)


def next_output_path(image_path: Path) -> Path:
    existing = _generated_paths_for(image_path)
    last_index = _generated_index(existing[-1], image_path.stem) if existing else 0
    return OUTPUT_DIR / f"{image_path.stem}_{last_index + 1:03d}.jpg"


def latest_output_path(image_path: Path) -> Path | None:
    paths = _generated_paths_for(image_path)
    return paths[-1] if paths else None


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
class ThumbEntry:
    path: Path
    surface: pygame.Surface
    rect: pygame.Rect


class Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect = rect
        self.label = label

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False) -> None:
        color = BTN_HOVER if hover else BTN_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (90, 90, 90), self.rect, 1, border_radius=4)
        text = font.render(self.label, True, BTN_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class TextBox:
    def __init__(
        self,
        rect: pygame.Rect,
        font: pygame.font.Font,
        *,
        text: str = "",
        line_height: int = 18,
        padding: int = 8,
    ) -> None:
        self.rect = rect
        self.font = font
        self.text = text
        self.line_height = line_height
        self.padding = padding
        self.focused = False
        self.cursor = len(text)
        self.cursor_on = True
        self._cursor_ms = 0

    def set_text(self, text: str) -> None:
        self.text = text
        self.cursor = len(text)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            if self.focused:
                self.cursor = self._cursor_from_pos(event.pos)
                self.cursor_on = True
                self._cursor_ms = 0
            return
        if not self.focused:
            return
        if event.type == pygame.TEXTINPUT:
            self.text = self.text[: self.cursor] + event.text + self.text[self.cursor :]
            self.cursor += len(event.text)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE and self.cursor > 0:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
            elif event.key == pygame.K_RETURN:
                self.text = self.text[: self.cursor] + "\n" + self.text[self.cursor :]
                self.cursor += 1
            elif event.key == pygame.K_ESCAPE:
                self.focused = False

    def tick(self, dt_ms: int) -> None:
        if not self.focused:
            self.cursor_on = False
            return
        self._cursor_ms += dt_ms
        if self._cursor_ms >= 500:
            self.cursor_on = not self.cursor_on
            self._cursor_ms = 0

    def _wrap_layout(self) -> list[tuple[str, int, int]]:
        max_width = self.rect.width - 2 * self.padding
        layout: list[tuple[str, int, int]] = []
        i = 0
        n = len(self.text)
        while i <= n:
            if i == n:
                break
            nl = self.text.find("\n", i)
            if nl == -1:
                nl = n
            words: list[tuple[str, int]] = []
            j = i
            while j < nl:
                while j < nl and self.text[j] == " ":
                    j += 1
                if j >= nl:
                    break
                wstart = j
                while j < nl and self.text[j] != " ":
                    j += 1
                words.append((self.text[wstart:j], wstart))
            if not words:
                layout.append(("", i, i))
                i = nl + 1 if nl < n else n
                continue
            line_start = words[0][1]
            current = words[0][0]
            current_end = words[0][1] + len(words[0][0])
            for wi in range(1, len(words)):
                word, wstart = words[wi]
                candidate = f"{current} {word}"
                if self.font.size(candidate)[0] <= max_width:
                    current = candidate
                    current_end = wstart + len(word)
                else:
                    layout.append((current, line_start, current_end))
                    current = word
                    line_start = wstart
                    current_end = wstart + len(word)
            layout.append((current, line_start, current_end))
            i = nl + 1 if nl < n else n
        return layout or [("", 0, 0)]

    def _cursor_from_pos(self, pos: tuple[int, int]) -> int:
        rel_x = pos[0] - (self.rect.x + self.padding)
        rel_y = pos[1] - (self.rect.y + self.padding)
        line_idx = max(0, rel_y // self.line_height)
        layout = self._wrap_layout()
        if line_idx >= len(layout):
            return len(self.text)
        line_text, start, end = layout[line_idx]
        if rel_x <= 0 or not line_text:
            return start
        best = start
        for offset in range(len(line_text) + 1):
            if self.font.size(line_text[:offset])[0] <= rel_x:
                best = start + offset
            else:
                break
        if best < len(self.text) and best > start:
            prev_w = self.font.size(line_text[: best - start])[0]
            next_w = self.font.size(line_text[: best - start + 1])[0]
            if abs(rel_x - next_w) < abs(rel_x - prev_w):
                best += 1
        return min(best, end if line_idx < len(layout) - 1 else len(self.text))

    def _cursor_xy(self) -> tuple[int, int]:
        layout = self._wrap_layout()
        for line_idx, (line_text, start, end) in enumerate(layout):
            if self.cursor < start:
                break
            if self.cursor <= end or line_idx == len(layout) - 1:
                col = min(max(0, self.cursor - start), len(line_text))
                x = self.rect.x + self.padding + self.font.size(line_text[:col])[0]
                y = self.rect.y + self.padding + line_idx * self.line_height
                return x, y
        return self.rect.x + self.padding, self.rect.y + self.padding

    def draw(self, surface: pygame.Surface) -> None:
        border = PROMPT_FOCUS_BORDER if self.focused else PROMPT_BORDER
        pygame.draw.rect(surface, PROMPT_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=4)

        clip = surface.get_clip()
        surface.set_clip(self.rect.inflate(-2, -2))
        x = self.rect.x + self.padding
        y = self.rect.y + self.padding
        for line_text, _start, _end in self._wrap_layout():
            if y + self.line_height > self.rect.bottom - self.padding:
                break
            text_surf = self.font.render(line_text, True, PROMPT_TEXT)
            surface.blit(text_surf, (x, y))
            y += self.line_height

        if self.focused and self.cursor_on:
            cursor_x, cursor_y = self._cursor_xy()
            if cursor_y + self.line_height <= self.rect.bottom - self.padding:
                pygame.draw.line(
                    surface,
                    CURSOR_COLOR,
                    (cursor_x, cursor_y + 2),
                    (cursor_x, cursor_y + self.line_height - 2),
                    1,
                )
        surface.set_clip(clip)


class SimilarFloorplanUI:
    def __init__(self, cfg, test_images: list[Path]) -> None:
        if not test_images:
            raise ValueError(f"No images found in {TEST_DIR}")
        self.cfg = cfg
        self.test_images = test_images
        self.selected_index = 0
        self.list_scroll = 0
        self.status = f"Loaded {test_images[0].name}"
        self.generating = False
        self.latest_output: Path | None = None
        self.thumb_entry: ThumbEntry | None = None
        self.thumbs_y = 0

        self.list_w = LIST_WIDTH
        self.panel_w = PANEL_WIDTH
        self.viewer_x = self.list_w
        self.viewer_w = max(1, int(self.cfg.canvas_size * DISPLAY_SCALE))
        self.panel_x = self.viewer_x + self.viewer_w

        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.prompt_font = pygame.font.SysFont("monospace", 13)
        self.prompt_box = TextBox(pygame.Rect(0, 0, 1, 1), self.prompt_font, text=DEFAULT_PROMPT)
        self.generate_btn = Button(pygame.Rect(0, 0, 1, 1), "Generate")
        self.status_y = 0

        self.window_w = self.list_w + self.viewer_w + self.panel_w
        self._load_test_image(0, initial=True)
        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Similar Floor Plan Generator")

    def _viewer_content_height(self) -> int:
        return VIEWER_PADDING_Y + 18 + self.source_surface.get_height() + VIEWER_PADDING_Y

    def _rebuild_thumb_entry(self) -> None:
        self.thumb_entry = None
        if self.latest_output is None or not self.latest_output.is_file():
            return
        x0 = self.panel_x + PROMPT_MARGIN
        y0 = self.thumbs_y + THUMB_HEADER
        image = Image.open(self.latest_output).convert("RGB")
        surface = pygame.transform.smoothscale(_pil_to_surface(image), (THUMB_SIZE, THUMB_SIZE))
        self.thumb_entry = ThumbEntry(
            self.latest_output,
            surface,
            pygame.Rect(x0, y0, THUMB_SIZE, THUMB_SIZE),
        )

    def _layout_side_panel(self) -> None:
        prompt_y = PROMPT_MARGIN + PROMPT_LABEL_HEIGHT
        thumb_block_min = (
            THUMB_TOP_GAP + THUMB_HEADER + THUMB_SIZE if self.latest_output is not None else 0
        )
        bottom_reserved = PROMPT_MARGIN + 28 + 54
        prompt_h = max(
            80,
            self.window_h
            - prompt_y
            - BTN_HEIGHT
            - 10
            - thumb_block_min
            - STATUS_TOP_GAP
            - bottom_reserved,
        )
        box_w = self.panel_w - 2 * PROMPT_MARGIN
        self.prompt_box.rect = pygame.Rect(self.panel_x + PROMPT_MARGIN, prompt_y, box_w, prompt_h)
        self.generate_btn.rect = pygame.Rect(
            self.panel_x + PROMPT_MARGIN,
            prompt_y + prompt_h + 10,
            box_w,
            BTN_HEIGHT,
        )
        self.thumbs_y = self.generate_btn.rect.bottom + THUMB_TOP_GAP
        self._rebuild_thumb_entry()
        thumb_block_h = THUMB_HEADER + THUMB_SIZE if self.latest_output is not None else 0
        self.status_y = self.thumbs_y + thumb_block_h + STATUS_TOP_GAP

    def _refresh_outputs(self, *, select_path: Path | None = None) -> None:
        if select_path is not None and select_path.is_file():
            self.latest_output = select_path
            return
        path = self.test_images[self.selected_index]
        self.latest_output = latest_output_path(path)

    def _apply_window_size(self) -> None:
        self.window_h = self._viewer_content_height()
        self._layout_side_panel()

    def _resize_window(self) -> None:
        self._apply_window_size()
        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        self._ensure_list_visible(self.selected_index)

    def _source_canvas(self, image_path: Path) -> Image.Image:
        return fit_to_canvas_rgb(Image.open(image_path), size=self.cfg.canvas_size)

    def _display_width(self) -> int:
        return max(1, int(self.cfg.canvas_size * DISPLAY_SCALE))

    def _scale_image(self, image: Image.Image) -> pygame.Surface:
        surface = _pil_to_surface(image)
        target_w, target_h = _scaled_size(image, self._display_width())
        return pygame.transform.smoothscale(surface, (target_w, target_h))

    def _load_test_image(self, index: int, *, initial: bool = False) -> None:
        path = self.test_images[index]
        self.selected_index = index
        if not initial:
            self._ensure_list_visible(index)

        self.source_image = self._source_canvas(path)
        self.source_surface = self._scale_image(self.source_image)

        self._refresh_outputs(select_path=latest_output_path(path))
        if self.latest_output is not None:
            self.status = f"Loaded {path.name} + {self.latest_output.name}"
        else:
            self.status = f"Loaded {path.name}"

        self._apply_window_size()
        if not initial:
            self._resize_window()

    def _ensure_list_visible(self, index: int) -> None:
        visible = self._list_visible_count()
        if index < self.list_scroll:
            self.list_scroll = index
        elif index >= self.list_scroll + visible:
            self.list_scroll = index - visible + 1
        self.list_scroll = max(0, min(self._max_list_scroll(), self.list_scroll))

    def _list_visible_count(self) -> int:
        return max(1, (self.window_h - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT)

    def _max_list_scroll(self) -> int:
        return max(0, len(self.test_images) - self._list_visible_count())

    def _list_index_at(self, pos: tuple[int, int]) -> int | None:
        mx, my = pos
        if mx >= self.list_w or my < LIST_HEADER_HEIGHT:
            return None
        row = (my - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT
        index = self.list_scroll + row
        if index < 0 or index >= len(self.test_images):
            return None
        return index

    def _on_generate(self) -> None:
        if self.generating:
            return
        if not self.cfg.replicate_token:
            self.status = "Missing Replicate token (REPLICATE_API_TOKEN or config)"
            return

        path = self.test_images[self.selected_index]
        prompt = self.prompt_box.text.strip()
        if not prompt:
            self.status = "Prompt is empty"
            return

        self.generating = True
        self.status = f"Generating {path.name}..."
        self._draw()
        pygame.display.flip()

        try:
            edit = self.cfg.image_edit
            result = run_replicate_edit(
                self.source_image,
                prompt,
                model=edit.model,
                token=self.cfg.replicate_token,
                quality=edit.quality,
                aspect_ratio=edit.aspect_ratio,
                output_format=edit.output_format,
                background=edit.background,
                moderation=edit.moderation,
                number_of_images=edit.number_of_images,
            )
            out_path = next_output_path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(out_path, quality=95)
            self._refresh_outputs(select_path=out_path)
            self.status = f"Saved {out_path.relative_to(_ROOT)}"
            self._layout_side_panel()
        except Exception as exc:
            self.status = f"Generate failed: {exc}"
        finally:
            self.generating = False

    def _handle_click(self, pos: tuple[int, int]) -> None:
        list_index = self._list_index_at(pos)
        if list_index is not None:
            if list_index != self.selected_index:
                self._load_test_image(list_index)
            return
        self.prompt_box.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1}))
        if self.generate_btn.hit(pos) and not self.generating:
            self._on_generate()

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, self.list_w, self.window_h))
        header = self.label_font.render("test_image/", True, LABEL_COLOR)
        self.window.blit(header, (8, 4))
        pygame.draw.line(self.window, (70, 70, 70), (self.list_w - 1, 0), (self.list_w - 1, self.window_h), 1)

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
            self.window.blit(self.font.render(label, True, LIST_TEXT), (8, y + 3))

    def _draw_viewer(self) -> None:
        y = VIEWER_PADDING_Y
        self.window.blit(
            self.label_font.render("Source", True, PANEL_LABEL_COLOR),
            (self.viewer_x, y),
        )
        y += 18
        self.window.blit(self.source_surface, (self.viewer_x, y))

    def _wrap_status(self, text: str, max_width: int) -> list[str]:
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

    def _draw_side_panel(self) -> None:
        pygame.draw.rect(self.window, SIDE_BG, (self.panel_x, 0, self.panel_w, self.window_h))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )

        self.window.blit(
            self.label_font.render("Prompt", True, PANEL_LABEL_COLOR),
            (self.panel_x + PROMPT_MARGIN, PROMPT_MARGIN),
        )
        self.prompt_box.draw(self.window)

        mouse = pygame.mouse.get_pos()
        self.generate_btn.draw(
            self.window,
            self.btn_font,
            hover=self.generate_btn.hit(mouse) and not self.generating,
        )

        if self.thumb_entry is not None:
            self.window.blit(
                self.label_font.render("Latest", True, PANEL_LABEL_COLOR),
                (self.panel_x + PROMPT_MARGIN, self.thumbs_y),
            )
            entry = self.thumb_entry
            pygame.draw.rect(self.window, THUMB_BG, entry.rect, border_radius=3)
            self.window.blit(entry.surface, entry.rect)
            pygame.draw.rect(self.window, THUMB_SELECTED, entry.rect, 2, border_radius=3)
            if entry.rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, entry.rect, 1, border_radius=3)

        status = self.status
        if self.generating:
            status = f"{status} (please wait)"
        y = self.status_y
        for line in self._wrap_status(status, self.panel_w - 2 * PROMPT_MARGIN):
            self.window.blit(
                self.font.render(line, True, STATUS_COLOR),
                (self.panel_x + PROMPT_MARGIN, y),
            )
            y += 18

        model_label = f"Model: {self.cfg.image_edit.model}"
        self.window.blit(
            self.font.render(model_label, True, LABEL_COLOR),
            (self.panel_x + PROMPT_MARGIN, self.window_h - 28),
        )

    def _draw(self) -> None:
        self.window.fill((18, 18, 18))
        self._draw_file_list()
        self._draw_viewer()
        self._draw_side_panel()

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(60)
            self.prompt_box.tick(dt)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self.prompt_box.focused:
                        self.prompt_box.focused = False
                    else:
                        running = False
                elif event.type == pygame.MOUSEWHEEL:
                    mx, _my = pygame.mouse.get_pos()
                    if mx < self.list_w:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.TEXTINPUT, pygame.KEYDOWN):
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                    else:
                        self.prompt_box.handle_event(event)
            self._draw()
            pygame.display.flip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate similar floor plans via Replicate")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config path")
    args = parser.parse_args()

    pygame.init()
    cfg = load_config(args.config)
    test_images = list_test_images()
    if not test_images:
        raise SystemExit(f"No images found in {TEST_DIR}")

    print(f"Ready — {len(test_images)} test image(s), model={cfg.image_edit.model}")
    ui = SimilarFloorplanUI(cfg, test_images)
    ui.run()
    pygame.quit()


if __name__ == "__main__":
    main()
