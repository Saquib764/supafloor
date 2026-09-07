#!/usr/bin/env python3
"""Interactive pygame UI: manually label room polygons on floor-plan images.

Source images live in ``floorplan_dataset/manual/``. Saves into
``floorplan-manual/``:

  input/<stem>.jpg    — 1024×1024 canvas image
  output/<stem>.json  — room polygons (extraction target)
  output2/<stem>.jpg  — simplified render from annotated rooms

Predict mode uses ``extraction/florence_config.yaml`` for model_id and adapter_dir.

Usage (from backend/floorplan):

    uv sync --group dev
    uv run python dev/annotate_rooms.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parents[1]
SOURCE_DIR = _REPO_ROOT / "floorplan_dataset" / "manual"
DATASET_ROOT = _REPO_ROOT / "floorplan-manual"
INPUT_DIR = DATASET_ROOT / "input"
OUTPUT_DIR = DATASET_ROOT / "output"
OUTPUT2_DIR = DATASET_ROOT / "output2"
EXPORT_IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95
FLORENCE_CONFIG = _SERVICE_ROOT / "extraction" / "florence_config.yaml"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (_SERVICE_ROOT / path).resolve()


def _florence_adapter_dir(config_path: Path) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _resolve_path(Path(raw["adapter_dir"]))


def _florence_model_label(config_path: Path) -> str:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_id = str(raw.get("model_id") or "unknown")
    return f"{model_id} ({_florence_adapter_dir(config_path).name})"


def _bootstrap() -> None:
    root = str(_SERVICE_ROOT)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    cached = sys.modules.get("config")
    if cached is not None and not hasattr(cached, "__path__"):
        for name in list(sys.modules):
            if name == "config" or name.startswith("config."):
                del sys.modules[name]


_bootstrap()

import pygame
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from extraction.transform import export_room_polygon

CANVAS_SIZE = 1024
DISPLAY_SCALE = 0.7
LIST_WIDTH = 200
PANEL_WIDTH = 280
LIST_ITEM_HEIGHT = 22
LIST_HEADER_HEIGHT = 24
ROOM_ROW_HEIGHT = 28
ROOM_ROW_GAP = 4
CLOSE_RADIUS = 8
POINT_HIT_RADIUS = 12
SELECTED_POINT_COLOR = (220, 50, 50)
ROOM_POINT_COLOR = (50, 120, 220)
BTN_ACTIVE = (70, 90, 70)
BTN_ACTIVE_HOVER = (90, 110, 90)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SIDE_BG = (28, 28, 28)
LIST_BG = (22, 22, 22)
LIST_HOVER = (45, 45, 45)
LIST_SELECTED = (60, 80, 120)
LIST_TEXT = (220, 220, 220)
BTN_BG = (55, 55, 55)
BTN_HOVER = (75, 75, 75)
BTN_DANGER = (120, 45, 45)
BTN_DANGER_HOVER = (150, 55, 55)
BTN_TEXT = (240, 240, 240)
LABEL_COLOR = (160, 160, 160)
TEXT_COLOR = (235, 235, 235)
STATUS_COLOR = (120, 180, 255)
DRAFT_COLOR = (255, 200, 0, 220)
_FILL_COLORS = (
    (232, 213, 183, 180),
    (212, 230, 195, 180),
    (201, 214, 223, 180),
    (184, 197, 214, 180),
    (240, 237, 230, 180),
    (221, 213, 200, 180),
    (168, 197, 184, 180),
    (229, 229, 229, 180),
)

ROOM_PRESETS = [
    "Living",
    "Dining",
    "Kitchen",
    "Bedroom",
    "Master Bedroom",
    "Bathroom",
    "Powder Room",
    "Hallway",
    "Closet",
    "Storage",
    "Utility",
    "Balcony",
    "Entry",
    "Study",
    "Laundry",
    "Room",
]


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def fit_to_canvas(image: Image.Image, size: int = CANVAS_SIZE) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    orig_w, orig_h = image.size
    scale = size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgb = image.convert("RGB")
    return pygame.image.frombuffer(rgb.tobytes(), rgb.size, "RGB").copy()


def _point_dict(x: float, y: float) -> dict[str, float]:
    return {"x": round(float(x), 3), "y": round(float(y), 3)}


def _polygon_ring(polygon: list[dict[str, float]]) -> list[tuple[int, int]]:
    return [(int(round(point["x"])), int(round(point["y"]))) for point in polygon]


def _valid_polygon(points: list[dict[str, float]]) -> bool:
    if len(points) < 3:
        return False
    ring = [(point["x"], point["y"]) for point in points]
    poly = Polygon(ring)
    return poly.is_valid and poly.area > 0


def _point_segment_dist_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _nearest_edge_insert_index(polygon: list[dict[str, float]], x: float, y: float) -> int:
    best_i = 0
    best_dist = float("inf")
    for i, start in enumerate(polygon):
        end = polygon[(i + 1) % len(polygon)]
        dist = _point_segment_dist_sq(
            x,
            y,
            float(start["x"]),
            float(start["y"]),
            float(end["x"]),
            float(end["y"]),
        )
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def _find_vertex_at(
    x: int,
    y: int,
    rooms: list[dict[str, object]],
    *,
    prefer_room: int | None = None,
) -> tuple[int, int] | None:
    hit_radius_sq = POINT_HIT_RADIUS ** 2
    best: tuple[int, int] | None = None
    best_dist = hit_radius_sq

    room_order = list(range(len(rooms)))
    if prefer_room is not None and prefer_room in room_order:
        room_order.remove(prefer_room)
        room_order.insert(0, prefer_room)

    for room_index in room_order:
        polygon = rooms[room_index].get("polygon") or []
        for point_index, point in enumerate(polygon):  # type: ignore[arg-type]
            px = int(round(float(point["x"])))
            py = int(round(float(point["y"])))
            dist = (x - px) ** 2 + (y - py) ** 2
            if dist <= best_dist:
                best_dist = dist
                best = (room_index, point_index)
    return best


def _load_rooms_from_json(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms: list[dict[str, object]] = []
    for raw in data.get("rooms") or []:
        polygon = [
            _point_dict(float(point["x"]), float(point["y"]))
            for point in raw.get("polygon") or []
        ]
        if len(polygon) < 3:
            continue
        rooms.append({"name": str(raw.get("name") or "Room"), "polygon": polygon})
    return rooms


def _find_existing_json(stem: str, output_dir: Path) -> Path | None:
    path = output_dir / f"{stem}.json"
    return path if path.is_file() else None


def _save_canvas_jpg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=JPEG_QUALITY)


def _save_simple_render(extracted: dict[str, object], path: Path) -> None:
    _bootstrap()
    from extraction.walls import render_extraction_simple_png

    rooms = extracted.get("rooms") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rooms:
        Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white").save(
            path,
            format="JPEG",
            quality=JPEG_QUALITY,
        )
        return
    render_extraction_simple_png(extracted, path)  # type: ignore[arg-type]


def _draw_vertex_handle(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    fill: tuple[int, ...],
    radius: int = 5,
) -> None:
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=(26, 26, 26),
        width=2,
    )


def _draw_overlay(
    base: Image.Image,
    rooms: list[dict[str, object]],
    *,
    draft: list[dict[str, float]] | None,
    selected_index: int | None,
    selected_vertex: tuple[int, int] | None,
    cursor: tuple[int, int] | None,
) -> Image.Image:
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, room in enumerate(rooms):
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        points = _polygon_ring(polygon)  # type: ignore[arg-type]
        fill = _FILL_COLORS[index % len(_FILL_COLORS)]
        outline = (44, 44, 44, 255)
        width = 3 if index == selected_index else 2
        draw.polygon(points, fill=fill, outline=outline, width=width)
        name = str(room.get("name") or "")
        if name:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            draw.text((cx, cy), name, fill=(26, 26, 26))

    if draft:
        draft_points = _polygon_ring(draft)
        if len(draft_points) >= 2:
            draw.line(draft_points, fill=DRAFT_COLOR[:3], width=2)
        if cursor is not None and draft_points:
            draw.line([draft_points[-1], cursor], fill=DRAFT_COLOR[:3], width=1)
        for point in draft_points:
            r = 4
            draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=DRAFT_COLOR[:3])

    if draft is None and selected_index is not None and 0 <= selected_index < len(rooms):
        polygon = rooms[selected_index].get("polygon") or []
        active_room, active_point = selected_vertex if selected_vertex is not None else (-1, -1)
        for point_index, point in enumerate(polygon):  # type: ignore[arg-type]
            if selected_index == active_room and point_index == active_point:
                continue
            px, py = int(round(float(point["x"]))), int(round(float(point["y"])))
            _draw_vertex_handle(draw, px, py, fill=ROOM_POINT_COLOR)

    if draft is None and selected_vertex is not None:
        room_index, point_index = selected_vertex
        if 0 <= room_index < len(rooms):
            polygon = rooms[room_index].get("polygon") or []
            if 0 <= point_index < len(polygon):
                point = polygon[point_index]
                px, py = int(round(float(point["x"]))), int(round(float(point["y"])))
                _draw_vertex_handle(draw, px, py, fill=SELECTED_POINT_COLOR, radius=7)
    return overlay


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    visible: bool = True
    danger: bool = False
    active: bool = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, hover: bool = False) -> None:
        if not self.visible:
            return
        if self.danger:
            color = BTN_DANGER_HOVER if hover else BTN_DANGER
        elif self.active:
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
class RoomRow:
    index: int
    rect: pygame.Rect
    delete_rect: pygame.Rect


class AnnotateUI:
    def __init__(
        self,
        images: list[Path],
        *,
        source_dir: Path,
        input_dir: Path,
        output_dir: Path,
        output2_dir: Path,
    ) -> None:
        if not images:
            raise ValueError(f"No images found in {source_dir}")
        self.images = images
        self.source_dir = source_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output2_dir = output2_dir
        self.florence_model = _florence_model_label(FLORENCE_CONFIG)
        for directory in (self.input_dir, self.output_dir, self.output2_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.selected_image_index = 0
        self.list_scroll = 0
        self.room_scroll = 0

        self.base_image = fit_to_canvas(Image.open(images[0]))
        self.rooms: list[dict[str, object]] = []
        self.draft: list[dict[str, float]] | None = None
        self.selected_room: int | None = None
        self.selected_vertex: tuple[int, int] | None = None
        self.dragging_vertex = False
        self.add_point_mode = False
        self.predict_mode = False
        self.dirty = False
        self.status = f"Loaded {images[0].name}"

        self.viewer_x = LIST_WIDTH
        self.display_scale = DISPLAY_SCALE
        self.display_w = int(CANVAS_SIZE * DISPLAY_SCALE)
        self.display_h = int(CANVAS_SIZE * DISPLAY_SCALE)
        self.panel_x = self.viewer_x + self.display_w
        self.window_w = self.panel_x + PANEL_WIDTH
        self.window_h = self.display_h

        self.window = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Annotate rooms")
        self.font = pygame.font.SysFont("monospace", 14)
        self.label_font = pygame.font.SysFont("monospace", 13, bold=True)
        self.btn_font = pygame.font.SysFont("monospace", 12, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)

        self.room_rows: list[RoomRow] = []
        self._layout_buttons()
        self._load_image(0, force=True)

    def _layout_buttons(self) -> None:
        x = self.panel_x + 10
        btn_w = PANEL_WIDTH - 20
        y = 10
        self.mode_btn = Button(pygame.Rect(x, y, btn_w, 28), "Mode: draw")
        y += 34
        self.new_room_btn = Button(pygame.Rect(x, y, btn_w, 28), "New room")
        y += 34
        self.close_room_btn = Button(pygame.Rect(x, y, btn_w, 28), "Close room", visible=False)
        y += 34
        self.add_point_btn = Button(pygame.Rect(x, y, btn_w, 28), "Add point")
        y += 34
        self.save_btn = Button(pygame.Rect(x, y, btn_w, 28), "Save JSON")
        self.room_list_top = y + 42
        self.status_top = self.window_h - 72

    def _save_path(self, image_path: Path) -> Path:
        return self.output_dir / f"{image_path.stem}.json"

    def _image_path(self, image_path: Path) -> Path:
        return self.input_dir / f"{image_path.stem}{EXPORT_IMAGE_EXT}"

    def _list_visible_count(self) -> int:
        return max(1, (self.window_h - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT)

    def _max_list_scroll(self) -> int:
        return max(0, len(self.images) - self._list_visible_count())

    def _room_list_height(self) -> int:
        return max(80, self.status_top - self.room_list_top - 12)

    def _room_visible_count(self) -> int:
        return max(1, self._room_list_height() // (ROOM_ROW_HEIGHT + ROOM_ROW_GAP))

    def _max_room_scroll(self) -> int:
        return max(0, len(self.rooms) + (1 if self.draft else 0) - self._room_visible_count())

    def _ensure_list_visible(self, index: int) -> None:
        if index < self.list_scroll:
            self.list_scroll = index
        elif index >= self.list_scroll + self._list_visible_count():
            self.list_scroll = index - self._list_visible_count() + 1
        self.list_scroll = max(0, min(self._max_list_scroll(), self.list_scroll))

    def _ensure_room_visible(self, index: int) -> None:
        if index < self.room_scroll:
            self.room_scroll = index
        elif index >= self.room_scroll + self._room_visible_count():
            self.room_scroll = index - self._room_visible_count() + 1
        self.room_scroll = max(0, min(self._max_room_scroll(), self.room_scroll))

    def _load_image(self, index: int, *, force: bool = False) -> None:
        if not force and index == self.selected_image_index:
            return
        if not force and self._has_unsaved_work():
            self.status = "Save or discard changes before switching images."
            return

        path = self.images[index]
        self.selected_image_index = index
        self._ensure_list_visible(index)
        self.base_image = fit_to_canvas(Image.open(path))
        self.rooms = []
        self.draft = None
        self.selected_room = None
        self.selected_vertex = None
        self.dragging_vertex = False
        self.add_point_mode = False
        self.add_point_btn.active = False
        self.dirty = False

        existing = _find_existing_json(path.stem, self.output_dir)
        if existing is not None:
            self.rooms = _load_rooms_from_json(existing)
            self.status = f"Loaded {path.name} — {len(self.rooms)} room(s) from {existing.name}"
        else:
            self.status = f"Loaded {path.name} — click New room, then add vertices."
        self._rebuild_surface()

    def _has_unsaved_work(self) -> bool:
        return self.dirty or self.draft is not None

    def _clear_edit_modes(self) -> None:
        self.selected_vertex = None
        self.dragging_vertex = False
        self.add_point_mode = False
        self.add_point_btn.active = False

    def _set_predict_mode(self, enabled: bool) -> None:
        if enabled and self.draft is not None:
            self.status = "Finish or cancel the current room before predict mode."
            return
        if enabled:
            self._clear_edit_modes()
        self.predict_mode = enabled
        self.mode_btn.active = enabled
        self.mode_btn.label = "Mode: predict" if enabled else "Mode: draw"
        if enabled:
            self.status = f"Predict mode — {self.florence_model}. Click to infer."
        else:
            self.status = "Draw mode — manual polygon editing."

    def _toggle_mode(self) -> None:
        self._set_predict_mode(not self.predict_mode)

    def _rebuild_surface(self, cursor: tuple[int, int] | None = None) -> None:
        overlay = _draw_overlay(
            self.base_image,
            self.rooms,
            draft=self.draft,
            selected_index=self.selected_room,
            selected_vertex=self.selected_vertex,
            cursor=cursor,
        )
        self.image_surface = _pil_to_surface(overlay)
        self.display_surface = pygame.transform.smoothscale(
            self.image_surface,
            (self.display_w, self.display_h),
        )

    def _in_viewer(self, pos: tuple[int, int]) -> bool:
        mx, my = pos
        return self.viewer_x <= mx < self.panel_x and 0 <= my < self.display_h

    def _viewer_point(self, pos: tuple[int, int]) -> tuple[int, int]:
        mx, my = pos
        ix = int((mx - self.viewer_x) / self.display_scale)
        iy = int(my / self.display_scale)
        return max(0, min(CANVAS_SIZE - 1, ix)), max(0, min(CANVAS_SIZE - 1, iy))

    def _next_room_name(self) -> str:
        return f"Room {len(self.rooms) + 1}"

    def _start_new_room(self) -> None:
        if self.predict_mode:
            self.status = "Switch to draw mode to create rooms manually."
            return
        if self.draft:
            self.status = "Finish or cancel the current room first."
            return
        self._clear_edit_modes()
        self.draft = []
        self.selected_room = None
        self.close_room_btn.visible = True
        self.status = "Placing vertices — Enter or click first point to close."

    def _cancel_draft(self) -> None:
        if not self.draft:
            return
        self.draft = None
        self.close_room_btn.visible = False
        self.status = "Draft cancelled."

    def _close_draft(self) -> bool:
        if not self.draft or len(self.draft) < 3:
            self.status = "Need at least 3 vertices to close a room."
            return False
        if not _valid_polygon(self.draft):
            self.status = "Invalid polygon — adjust vertices and try again."
            return False
        self.rooms.append({"name": self._next_room_name(), "polygon": export_room_polygon(self.draft)})
        self.draft = None
        self.close_room_btn.visible = False
        self.selected_room = len(self.rooms) - 1
        self.dirty = True
        self._ensure_room_visible(self.selected_room)
        self.status = f"Added {self.rooms[-1]['name']} — {len(self.rooms)} room(s)."
        return True

    def _add_vertex(self, x: int, y: int) -> None:
        if self.draft is None:
            self.status = "Click New room before placing vertices."
            return
        x = max(0, min(CANVAS_SIZE - 1, x))
        y = max(0, min(CANVAS_SIZE - 1, y))
        if len(self.draft) >= 3:
            first = self.draft[0]
            fx, fy = int(round(first["x"])), int(round(first["y"]))
            if (x - fx) ** 2 + (y - fy) ** 2 <= CLOSE_RADIUS ** 2:
                self._close_draft()
                return
        self.draft.append(_point_dict(x, y))
        self.status = f"Vertex {len(self.draft)} — Enter or click first point to close."

    def _toggle_add_point_mode(self) -> None:
        if self.predict_mode:
            self.status = "Switch to draw mode to edit vertices."
            return
        if self.draft:
            self.status = "Finish or cancel the current room first."
            return
        if self.selected_room is None:
            self.status = "Select a room from the list first."
            return
        self.add_point_mode = not self.add_point_mode
        self.add_point_btn.active = self.add_point_mode
        self.selected_vertex = None
        if self.add_point_mode:
            name = self.rooms[self.selected_room]["name"]
            self.status = f"Add point mode — click an edge on {name}."
        else:
            self.status = "Add point mode off."

    def _insert_point_on_room(self, x: int, y: int) -> None:
        if self.selected_room is None:
            return
        room = self.rooms[self.selected_room]
        polygon: list[dict[str, float]] = list(room["polygon"])  # type: ignore[assignment]
        insert_after = _nearest_edge_insert_index(polygon, x, y)
        polygon.insert(insert_after + 1, _point_dict(x, y))
        room["polygon"] = polygon
        self.selected_vertex = (self.selected_room, insert_after + 1)
        self.dirty = True
        self.status = f"Added point to {room['name']} — {len(polygon)} vertices."

    def _select_vertex(self, room_index: int, point_index: int) -> None:
        self.selected_room = room_index
        self.selected_vertex = (room_index, point_index)
        self.add_point_mode = False
        self.add_point_btn.active = False
        name = self.rooms[room_index]["name"]
        self.status = f"Selected vertex on {name} — drag, Delete to remove."

    def _move_selected_vertex(self, x: int, y: int) -> None:
        if self.selected_vertex is None:
            return
        room_index, point_index = self.selected_vertex
        polygon = self.rooms[room_index]["polygon"]  # type: ignore[index]
        polygon[point_index] = _point_dict(x, y)

    def _finish_vertex_drag(self) -> None:
        if not self.dragging_vertex:
            return
        self.dragging_vertex = False
        if self.selected_vertex is None:
            return
        room_index, _point_index = self.selected_vertex
        polygon = self.rooms[room_index]["polygon"]  # type: ignore[index]
        if not _valid_polygon(polygon):
            self.status = "Invalid polygon after move — adjust or undo with Save reload."
        else:
            self.dirty = True
            self.status = f"Moved vertex on {self.rooms[room_index]['name']}."

    def _delete_selected_vertex(self) -> None:
        if self.selected_vertex is None or self.draft is not None:
            return
        room_index, point_index = self.selected_vertex
        polygon: list[dict[str, float]] = list(self.rooms[room_index]["polygon"])  # type: ignore[index]
        if len(polygon) <= 3:
            self.status = "Cannot delete — room needs at least 3 vertices."
            return
        del polygon[point_index]
        self.rooms[room_index]["polygon"] = polygon
        self.selected_vertex = None
        self.dirty = True
        self.status = f"Deleted vertex — {len(polygon)} remaining on {self.rooms[room_index]['name']}."

    def _predict_at(self, x: int, y: int) -> None:
        path = self.images[self.selected_image_index]
        self.status = f"Inferring room at ({x}, {y})..."
        self._draw()
        pygame.display.flip()
        try:
            from extraction.florence import FlorenceUnavailableError, infer_room_at_canvas_point

            room = infer_room_at_canvas_point(
                path,
                float(x),
                float(y),
                config_path=FLORENCE_CONFIG,
            )
        except FlorenceUnavailableError as exc:
            self.status = f"Florence unavailable: {exc}"
            return
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
            return
        if room is None:
            self.status = f"No polygon at ({x}, {y})"
            return
        polygon = export_room_polygon(room["polygon"])
        self.rooms.append({"name": str(room["name"]), "polygon": polygon})
        self.selected_room = len(self.rooms) - 1
        self.selected_vertex = None
        self.dirty = True
        self._ensure_room_visible(self.selected_room)
        self.status = f"Predicted {room['name']} — {len(self.rooms)} room(s)"
        self._rebuild_surface()

    def _delete_room(self, index: int) -> None:
        if index < 0 or index >= len(self.rooms):
            return
        name = self.rooms[index]["name"]
        del self.rooms[index]
        if self.selected_room == index:
            self.selected_room = None
            self.selected_vertex = None
        elif self.selected_room is not None and self.selected_room > index:
            self.selected_room -= 1
        if self.selected_vertex is not None:
            room_index, point_index = self.selected_vertex
            if room_index == index:
                self.selected_vertex = None
            elif room_index > index:
                self.selected_vertex = (room_index - 1, point_index)
        self.dirty = True
        self.status = f"Deleted {name} — {len(self.rooms)} room(s) remaining."

    def _cycle_room_name(self, delta: int) -> None:
        if self.selected_room is None:
            return
        room = self.rooms[self.selected_room]
        current = str(room.get("name") or "Room")
        try:
            index = ROOM_PRESETS.index(current)
        except ValueError:
            index = len(ROOM_PRESETS) - 1
        room["name"] = ROOM_PRESETS[(index + delta) % len(ROOM_PRESETS)]
        self.dirty = True
        self.status = f"Renamed to {room['name']}."

    def _save(self) -> None:
        if self.draft:
            self.status = "Close or cancel the draft room before saving."
            return
        path = self.images[self.selected_image_index]
        json_path = self._save_path(path)
        payload = {
            "id": path.stem,
            "rooms": [
                {
                    "name": str(room["name"]),
                    "polygon": export_room_polygon(room["polygon"]),  # type: ignore[arg-type]
                }
                for room in self.rooms
            ],
            "meta": {"canvas_size": CANVAS_SIZE, "source": "manual"},
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        jpg_path = self._image_path(path)
        _save_canvas_jpg(self.base_image, jpg_path)
        simple_path = self.output2_dir / f"{path.stem}{EXPORT_IMAGE_EXT}"
        _save_simple_render(payload, simple_path)
        self.dirty = False
        self.status = f"Saved {len(self.rooms)} room(s) → {json_path.name}"

    def _list_index_at(self, pos: tuple[int, int]) -> int | None:
        mx, my = pos
        if mx >= LIST_WIDTH or my < LIST_HEADER_HEIGHT:
            return None
        row = (my - LIST_HEADER_HEIGHT) // LIST_ITEM_HEIGHT
        index = self.list_scroll + row
        if index < 0 or index >= len(self.images):
            return None
        return index

    def _layout_room_rows(self) -> None:
        self.room_rows = []
        x = self.panel_x + 10
        delete_w = 56
        row_w = PANEL_WIDTH - 20
        y = self.room_list_top
        for offset in range(self._room_visible_count()):
            index = self.room_scroll + offset
            if index >= len(self.rooms):
                break
            rect = pygame.Rect(x, y, row_w, ROOM_ROW_HEIGHT)
            delete_rect = pygame.Rect(rect.right - delete_w, rect.y + 4, delete_w, rect.height - 8)
            self.room_rows.append(RoomRow(index=index, rect=rect, delete_rect=delete_rect))
            y += ROOM_ROW_HEIGHT + ROOM_ROW_GAP

    def _handle_click(self, pos: tuple[int, int]) -> None:
        list_index = self._list_index_at(pos)
        if list_index is not None:
            self._load_image(list_index)
            return

        if self.mode_btn.hit(pos):
            self._toggle_mode()
            return
        if self.new_room_btn.hit(pos):
            self._start_new_room()
            return
        if self.close_room_btn.hit(pos):
            self._close_draft()
            return
        if self.add_point_btn.hit(pos):
            self._toggle_add_point_mode()
            return
        if self.save_btn.hit(pos):
            self._save()
            return

        for row in self.room_rows:
            if row.delete_rect.collidepoint(pos):
                self._delete_room(row.index)
                return
            if row.rect.collidepoint(pos):
                self.selected_room = row.index
                self.selected_vertex = None
                self.add_point_mode = False
                self.add_point_btn.active = False
                self._ensure_room_visible(row.index)
                self.status = f"Selected {self.rooms[row.index]['name']} — [ ] to rename."
                return

        if self._in_viewer(pos):
            self._handle_viewer_click(pos)

    def _handle_viewer_click(self, pos: tuple[int, int]) -> None:
        x, y = self._viewer_point(pos)
        if self.predict_mode:
            self._predict_at(x, y)
            return
        if self.draft is not None:
            self._add_vertex(x, y)
            return
        if self.add_point_mode:
            self._insert_point_on_room(x, y)
            return
        hit = _find_vertex_at(x, y, self.rooms, prefer_room=self.selected_room)
        if hit is not None:
            self._select_vertex(*hit)
            self.dragging_vertex = True
            return
        self.selected_vertex = None
        self.status = "Click a vertex to select, or select a room and use Add point."

    def _draw_file_list(self) -> None:
        pygame.draw.rect(self.window, LIST_BG, (0, 0, LIST_WIDTH, self.window_h))
        header = self.label_font.render("manual/", True, LABEL_COLOR)
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
            if index >= len(self.images):
                break
            y = LIST_HEADER_HEIGHT + row * LIST_ITEM_HEIGHT
            rect = pygame.Rect(0, y, LIST_WIDTH, LIST_ITEM_HEIGHT)
            if index == self.selected_image_index:
                pygame.draw.rect(self.window, LIST_SELECTED, rect)
            elif rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, rect)
            label = self.images[index].name
            if self.font.size(label)[0] > LIST_WIDTH - 12:
                while label and self.font.size(label + "…")[0] > LIST_WIDTH - 12:
                    label = label[:-1]
                label += "…"
            self.window.blit(self.font.render(label, True, LIST_TEXT), (8, y + 3))

    def _draw_side_panel(self) -> None:
        pygame.draw.rect(self.window, SIDE_BG, (self.panel_x, 0, PANEL_WIDTH, self.window_h))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )
        mouse = pygame.mouse.get_pos()
        for btn in (self.mode_btn, self.new_room_btn, self.close_room_btn, self.add_point_btn, self.save_btn):
            btn.draw(self.window, self.btn_font, hover=btn.hit(mouse))

        model_label = self.small_font.render(f"Florence: {self.florence_model}", True, LABEL_COLOR)
        self.window.blit(model_label, (self.panel_x + 10, self.room_list_top - 38))

        rooms_label = self.label_font.render("ROOMS", True, LABEL_COLOR)
        self.window.blit(rooms_label, (self.panel_x + 10, self.room_list_top - 20))

        self._layout_room_rows()
        for row in self.room_rows:
            room = self.rooms[row.index]
            if row.index == self.selected_room:
                pygame.draw.rect(self.window, LIST_SELECTED, row.rect, border_radius=4)
            elif row.rect.collidepoint(mouse):
                pygame.draw.rect(self.window, LIST_HOVER, row.rect, border_radius=4)

            swatch = pygame.Rect(row.rect.x + 6, row.rect.y + 8, 12, 12)
            color = _FILL_COLORS[row.index % len(_FILL_COLORS)][:3]
            pygame.draw.rect(self.window, color, swatch)
            pygame.draw.rect(self.window, (44, 44, 44), swatch, 1)

            name = str(room["name"])
            polygon = room.get("polygon") or []
            suffix = f" ({len(polygon)} pts)"
            label = name + suffix
            max_w = row.delete_rect.x - row.rect.x - 28
            while label and self.small_font.size(label + "…")[0] > max_w:
                label = label[:-1]
            if label != name + suffix:
                label += "…"
            self.window.blit(
                self.small_font.render(label, True, TEXT_COLOR),
                (row.rect.x + 24, row.rect.y + 7),
            )

            danger = row.delete_rect.collidepoint(mouse)
            pygame.draw.rect(
                self.window,
                BTN_DANGER_HOVER if danger else BTN_DANGER,
                row.delete_rect,
                border_radius=4,
            )
            del_text = self.btn_font.render("Delete", True, BTN_TEXT)
            self.window.blit(del_text, del_text.get_rect(center=row.delete_rect.center))

        if self.draft is not None:
            draft_y = self.room_list_top + len(self.room_rows) * (ROOM_ROW_HEIGHT + ROOM_ROW_GAP)
            draft_rect = pygame.Rect(self.panel_x + 10, draft_y, PANEL_WIDTH - 20, ROOM_ROW_HEIGHT)
            pygame.draw.rect(self.window, (70, 60, 20), draft_rect, border_radius=4)
            draft_label = f"Draft ({len(self.draft)} pts)"
            self.window.blit(self.small_font.render(draft_label, True, DRAFT_COLOR[:3]), (draft_rect.x + 8, draft_rect.y + 7))

        status_y = self.status_top
        for line in self._wrap_status(self.status):
            self.window.blit(self.label_font.render(line, True, STATUS_COLOR), (self.panel_x + 10, status_y))
            status_y += 18

        hints = "Draw: edit verts  Predict: click infer  [ ] rename  Del vertex  Esc cancel"
        self.window.blit(self.small_font.render(hints, True, LABEL_COLOR), (self.panel_x + 10, self.window_h - 22))

    def _wrap_status(self, text: str) -> list[str]:
        max_width = PANEL_WIDTH - 20
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self.font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _draw(self, cursor: tuple[int, int] | None = None) -> None:
        self.window.fill((18, 18, 18))
        self._draw_file_list()
        self._rebuild_surface(cursor if self.draft else None)
        self.window.blit(self.display_surface, (self.viewer_x, 0))
        pygame.draw.line(
            self.window,
            (70, 70, 70),
            (self.panel_x, 0),
            (self.panel_x, self.window_h),
            1,
        )
        self._draw_side_panel()

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            cursor = None
            mouse = pygame.mouse.get_pos()
            if self.draft and self._in_viewer(mouse):
                cursor = self._viewer_point(mouse)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.draft:
                            self._cancel_draft()
                        elif self.predict_mode:
                            self._set_predict_mode(False)
                        elif self.add_point_mode:
                            self._clear_edit_modes()
                            self.status = "Add point mode off."
                        elif self.selected_vertex is not None:
                            self.selected_vertex = None
                            self.status = "Vertex deselected."
                        else:
                            running = False
                    elif event.key == pygame.K_RETURN:
                        self._close_draft()
                    elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        if self.draft:
                            self.draft.pop()
                            if not self.draft:
                                self.close_room_btn.visible = False
                                self.status = "Draft cleared."
                            else:
                                self.status = f"Vertex {len(self.draft)} — keep placing points."
                        elif self.selected_vertex is not None:
                            self._delete_selected_vertex()
                    elif event.key == pygame.K_LEFTBRACKET:
                        self._cycle_room_name(-1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self._cycle_room_name(1)
                elif event.type == pygame.MOUSEMOTION:
                    if (
                        not self.predict_mode
                        and self.dragging_vertex
                        and self.selected_vertex is not None
                        and self._in_viewer(event.pos)
                    ):
                        self._move_selected_vertex(*self._viewer_point(event.pos))
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.dragging_vertex:
                        self._finish_vertex_drag()
                elif event.type == pygame.MOUSEWHEEL:
                    mx, _my = pygame.mouse.get_pos()
                    if mx < LIST_WIDTH:
                        self.list_scroll = max(
                            0,
                            min(self._max_list_scroll(), self.list_scroll - event.y),
                        )
                    elif mx >= self.panel_x:
                        self.room_scroll = max(
                            0,
                            min(self._max_room_scroll(), self.room_scroll - event.y),
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._draw(cursor)
            pygame.display.flip()
            clock.tick(60)


def main() -> int:
    if not FLORENCE_CONFIG.is_file():
        print(f"Florence config not found: {FLORENCE_CONFIG}", file=sys.stderr)
        return 1
    florence_adapter = _florence_adapter_dir(FLORENCE_CONFIG)
    if not florence_adapter.is_dir():
        print(f"Florence adapter not found: {florence_adapter}", file=sys.stderr)
        return 1

    _bootstrap()
    images = list_images(SOURCE_DIR)
    if not images:
        print(f"No images found in {SOURCE_DIR}", file=sys.stderr)
        print(f"Add floor-plan images to {SOURCE_DIR.relative_to(_REPO_ROOT)}/", file=sys.stderr)
        return 1

    print(f"Florence predict: {_florence_model_label(FLORENCE_CONFIG)}")

    pygame.init()
    try:
        AnnotateUI(
            images,
            source_dir=SOURCE_DIR,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            output2_dir=OUTPUT2_DIR,
        ).run()
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
