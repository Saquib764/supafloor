"""Place symbol PNGs inside rooms during realistic export."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from PIL import Image, ImageChops

from floorplan.requirements.program import space_role

Coord = tuple[float, float]

SYMBOLS_DIR = Path(__file__).resolve().parent / "symbols"
_ROOM_IMAGES = sorted(SYMBOLS_DIR.glob("room-*.png"))
_DINING_IMAGES = sorted(SYMBOLS_DIR.glob("dining-*.png"))
_IMAGE_BY_NAME = {path.name: path for path in SYMBOLS_DIR.glob("*.png")}


def symbol_kind_for_room(name: str) -> str | None:
    lower = name.lower()
    if "stair" in lower:
        return None
    role = space_role(name)
    if role in ("master_bedroom", "bedroom", "living", "dining", "bathroom", "toilet", "kitchen"):
        return role
    return None


def _open_ring(polygon: list[Coord]) -> list[Coord]:
    if len(polygon) >= 2 and polygon[0] == polygon[-1]:
        return list(polygon[:-1])
    return list(polygon)


def _fit_box(polygon: list[Coord], *, inset: float = 0.18) -> tuple[float, float, float, float] | None:
    ring = _open_ring(polygon)
    if len(ring) < 3:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width = maxx - minx
    height = maxy - miny
    if min(width, height) < 24:
        return None
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half_w = (width * (1.0 - 2.0 * inset)) / 2.0
    half_h = (height * (1.0 - 2.0 * inset)) / 2.0
    return cx, cy, half_w, half_h


def _room_bounds(ring: list[Coord]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _wall_stuck_center(
    polygon: list[Coord],
    draw_w: float,
    draw_h: float,
    rng: random.Random,
    *,
    wall_thickness: float,
) -> tuple[float, float] | None:
    ring = _open_ring(polygon)
    if len(ring) < 3:
        return None

    minx, miny, maxx, maxy = _room_bounds(ring)
    width = maxx - minx
    height = maxy - miny
    pad = wall_thickness

    if width >= height:
        wall = rng.choice(("top", "bottom"))
    else:
        wall = rng.choice(("left", "right"))

    if wall == "top":
        cy = miny + pad + draw_h / 2.0
        span_min = minx + pad + draw_w / 2.0
        span_max = maxx - pad - draw_w / 2.0
        cx = (minx + maxx) / 2.0 if span_min > span_max else rng.uniform(span_min, span_max)
    elif wall == "bottom":
        cy = maxy - pad - draw_h / 2.0
        span_min = minx + pad + draw_w / 2.0
        span_max = maxx - pad - draw_w / 2.0
        cx = (minx + maxx) / 2.0 if span_min > span_max else rng.uniform(span_min, span_max)
    elif wall == "left":
        cx = minx + pad + draw_w / 2.0
        span_min = miny + pad + draw_h / 2.0
        span_max = maxy - pad - draw_h / 2.0
        cy = (miny + maxy) / 2.0 if span_min > span_max else rng.uniform(span_min, span_max)
    else:
        cx = maxx - pad - draw_w / 2.0
        span_min = miny + pad + draw_h / 2.0
        span_max = maxy - pad - draw_h / 2.0
        cy = (miny + maxy) / 2.0 if span_min > span_max else rng.uniform(span_min, span_max)

    return cx, cy


@lru_cache(maxsize=32)
def _load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def _pick_image(name: str, rng: random.Random) -> Path | None:
    role = space_role(name)

    if role == "master_bedroom":
        pool = [_IMAGE_BY_NAME[n] for n in ("room-03.png", "room-02.png") if n in _IMAGE_BY_NAME]
        pool = pool or _ROOM_IMAGES
    elif role == "bedroom":
        pool = [_IMAGE_BY_NAME[n] for n in ("room-01.png", "room-02.png", "room-03.png") if n in _IMAGE_BY_NAME]
    elif role == "living":
        pool = [_IMAGE_BY_NAME["room-04.png"]] if "room-04.png" in _IMAGE_BY_NAME else []
    elif role == "dining":
        pool = _DINING_IMAGES
    elif role == "kitchen":
        pool = [_IMAGE_BY_NAME["bath-01.png"]] if "bath-01.png" in _IMAGE_BY_NAME else []
    elif role == "toilet":
        pool = [_IMAGE_BY_NAME["bath-04.png"]] if "bath-04.png" in _IMAGE_BY_NAME else []
    elif role == "bathroom":
        pool = [_IMAGE_BY_NAME[n] for n in ("bath-02.png", "bath-03.png", "bath-04.png") if n in _IMAGE_BY_NAME]
    else:
        return None

    if not pool:
        return None
    return rng.choice(pool)


def _symbol_size(img_w: int, img_h: int, hw: float, hh: float) -> tuple[float, float]:
    max_w = hw * 1.5
    max_h = hh * 1.5
    scale = min(max_w / img_w, max_h / img_h)
    return img_w * scale, img_h * scale


def _tint_rgba(rgba: np.ndarray, blackness: float) -> np.ndarray:
    out = rgba.astype(np.float32)
    white = 255.0 * (1.0 - blackness)
    out[..., :3] = out[..., :3] * blackness + white
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return rgba
    resized = Image.fromarray(rgba).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def _canvas_region(ax: Axes, left: float, right: float, top: float, bottom: float) -> np.ndarray | None:
    from matplotlib.transforms import Bbox, TransformedBbox

    fig = ax.figure
    fig.canvas.draw()
    canvas = np.asarray(fig.canvas.buffer_rgba())
    ch, cw = canvas.shape[:2]

    bbox_data = Bbox.from_extents(left, top, right, bottom)
    bbox_disp = TransformedBbox(bbox_data, ax.transData)
    x0 = int(max(0, np.floor(bbox_disp.x0)))
    x1 = int(min(cw, np.ceil(bbox_disp.x1)))
    row0 = int(max(0, ch - np.ceil(bbox_disp.y1)))
    row1 = int(min(ch, ch - np.floor(bbox_disp.y0)))
    if x1 <= x0 or row1 <= row0:
        return None
    return canvas[row0:row1, x0:x1, :3].copy()


def _composite_symbol(bg_rgb: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    bg = Image.fromarray(bg_rgb, mode="RGB")
    fg = Image.fromarray(fg_rgba[..., :3], mode="RGB")
    alpha = Image.fromarray(fg_rgba[..., 3], mode="L")
    blended = ImageChops.multiply(bg, fg)
    return np.asarray(Image.composite(blended, bg, alpha))


def _blit_symbol(
    ax: Axes,
    rgba: np.ndarray,
    cx: float,
    cy: float,
    draw_w: float,
    draw_h: float,
    *,
    blackness: float = 1.0,
) -> None:
    tinted = _tint_rgba(rgba, blackness)
    left = cx - draw_w / 2.0
    right = cx + draw_w / 2.0
    top = cy - draw_h / 2.0
    bottom = cy + draw_h / 2.0

    bg_rgb = _canvas_region(ax, left, right, top, bottom)
    if bg_rgb is None:
        ax.imshow(tinted, extent=(left, right, bottom, top), zorder=3)
        return

    region_h, region_w = bg_rgb.shape[:2]
    fg_rgba = _resize_rgba(tinted, region_w, region_h)
    composited = _composite_symbol(bg_rgb, fg_rgba)
    out_rgba = np.dstack([composited, np.full((region_h, region_w), 255, dtype=np.uint8)])
    ax.imshow(out_rgba, extent=(left, right, bottom, top), zorder=3)


def draw_room_symbol(
    ax: Axes,
    polygon: list[Coord],
    name: str,
    *,
    wall_thickness: float = 4.0,
    blackness: float = 1.0,
    rng: random.Random | None = None,
) -> bool:
    if symbol_kind_for_room(name) is None:
        return False
    box = _fit_box(polygon)
    if box is None:
        return False
    source_rng = rng or random.Random()
    path = _pick_image(name, source_rng)
    if path is None:
        return False
    cx, cy, hw, hh = box
    rgba = _load_rgba(path)
    img_h, img_w = rgba.shape[:2]
    draw_w, draw_h = _symbol_size(img_w, img_h, hw, hh)

    wall_center = _wall_stuck_center(
        polygon,
        draw_w,
        draw_h,
        source_rng,
        wall_thickness=wall_thickness,
    )
    if wall_center is not None:
        cx, cy = wall_center

    _blit_symbol(ax, rgba, cx, cy, draw_w, draw_h, blackness=blackness)
    return True
