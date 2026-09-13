"""Shared grayscale text layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import ImageDraw, ImageFont

_PRINTED_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

_BOLD_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_ITALIC_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
)

_HANDWRITTEN_FONTS = (
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)

SERIF_FONTS = (
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
)


def _existing_fonts(candidates: tuple[str, ...]) -> list[str]:
    return [path for path in candidates if Path(path).is_file()]


PRINTED_FONT_PATHS = _existing_fonts(_PRINTED_FONTS)
BOLD_FONT_PATHS = _existing_fonts(_BOLD_FONTS)
ITALIC_FONT_PATHS = _existing_fonts(_ITALIC_FONTS)
HANDWRITTEN_FONT_PATHS = _existing_fonts(_HANDWRITTEN_FONTS)
SERIF_FONT_PATHS = _existing_fonts(SERIF_FONTS)
_FONT_CACHE: dict[tuple[str, int], ImageFont.ImageFont] = {}


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (path, size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(path, size=size)
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


@dataclass(frozen=True)
class RenderedSample:
    image: Image.Image
    text: str
    width: int
    height: int


def text_width(font: ImageFont.ImageFont, text: str, *, letter_spacing: float = 0.0) -> float:
    if not text:
        return 0.0
    getlength = getattr(font, "getlength", None)
    if getlength is not None:
        width = float(getlength(text))
    elif hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        width = float(bbox[2] - bbox[0])
    else:
        width = float(len(text) * 8)
    if letter_spacing > 0 and len(text) > 1:
        width += letter_spacing * (len(text) - 1)
    return width


def text_bbox(font: ImageFont.ImageFont, text: str) -> tuple[int, int, int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return left, top, right - left, bottom - top
    width = int(text_width(font, text))
    return 0, 0, width, 12


def draw_text_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: int,
    letter_spacing: float = 0.0,
) -> None:
    if letter_spacing <= 0:
        draw.text((x, y), text, fill=fill, font=font)
        return
    cursor = float(x)
    for character in text:
        draw.text((cursor, y), character, fill=fill, font=font)
        cursor += text_width(font, character) + letter_spacing


def _wrap_long_word(
    word: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    letter_spacing: float,
) -> list[str]:
    rows: list[str] = []
    row = ""
    for character in word:
        candidate = row + character
        if row and text_width(font, candidate, letter_spacing=letter_spacing) > max_width:
            rows.append(row)
            row = character
        else:
            row = candidate
    if row:
        rows.append(row)
    return rows


def wrap_line(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    *,
    letter_spacing: float = 0.0,
) -> list[str]:
    if not text:
        return []
    rows: list[str] = []
    for word in text.split(" "):
        if not word:
            continue
        if text_width(font, word, letter_spacing=letter_spacing) > max_width:
            if rows:
                rows.extend(_wrap_long_word(word, font, max_width, letter_spacing=letter_spacing))
            else:
                rows = _wrap_long_word(word, font, max_width, letter_spacing=letter_spacing)
            continue
        if not rows:
            rows.append(word)
            continue
        candidate = f"{rows[-1]} {word}"
        if text_width(font, candidate, letter_spacing=letter_spacing) > max_width:
            rows.append(word)
        else:
            rows[-1] = candidate
    return rows
