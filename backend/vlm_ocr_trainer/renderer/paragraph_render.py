"""Render sampled paragraph documents with visual variation."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from config import VariationSettings
from renderer.text import (
    BOLD_FONT_PATHS,
    HANDWRITTEN_FONT_PATHS,
    ITALIC_FONT_PATHS,
    PRINTED_FONT_PATHS,
    SERIF_FONT_PATHS,
    RenderedSample,
    draw_text_line,
    load_font,
    text_bbox,
    wrap_line,
)
from renderer.variations import apply_image_effects, apply_image_variations


def _pick_font(
    rng: random.Random,
    size: int,
    settings: VariationSettings,
) -> tuple[ImageFont.ImageFont, str | None, str]:
    roll = rng.random()
    if roll < settings.handwritten_prob and HANDWRITTEN_FONT_PATHS:
        pool = HANDWRITTEN_FONT_PATHS
        style = "handwritten"
    elif roll < settings.handwritten_prob + settings.bold_prob and BOLD_FONT_PATHS:
        pool = BOLD_FONT_PATHS
        style = "bold"
    elif (
        roll < settings.handwritten_prob + settings.bold_prob + settings.italic_prob
        and ITALIC_FONT_PATHS
    ):
        pool = ITALIC_FONT_PATHS
        style = "italic"
    else:
        pool = PRINTED_FONT_PATHS + SERIF_FONT_PATHS
        style = "printed"
    if not pool:
        return ImageFont.load_default(), None, style
    path = rng.choice(pool)
    return load_font(path, size), path, style


def _sample_padding(
    rng: random.Random,
    settings: VariationSettings,
) -> tuple[int, int, int, int]:
    return (
        rng.randint(settings.padding_top_min, settings.padding_top_max),
        rng.randint(settings.padding_right_min, settings.padding_right_max),
        rng.randint(settings.padding_bottom_min, settings.padding_bottom_max),
        rng.randint(settings.padding_left_min, settings.padding_left_max),
    )


def _draw_text_layer(
    width: int,
    height: int,
    paragraphs: list[str],
    *,
    padding: tuple[int, int, int, int],
    bg_value: int,
    ink_value: int,
    font: ImageFont.ImageFont,
    letter_spacing: float,
    line_gap: int,
    paragraph_gap: int,
) -> tuple[Image.Image, str]:
    padding_top, padding_right, padding_bottom, padding_left = padding
    image = Image.new("L", (width, height), color=bg_value)
    draw = ImageDraw.Draw(image)
    rendered_paragraphs: list[list[str]] = []
    y_cursor = padding_top
    text_width = max(1, width - padding_left - padding_right)

    for paragraph in paragraphs:
        rows = wrap_line(
            paragraph,
            font,
            text_width,
            letter_spacing=letter_spacing,
        )
        rendered_rows: list[str] = []
        for row in rows:
            _, _, _row_width, row_height = text_bbox(font, row)
            if y_cursor + row_height > height - padding_bottom:
                break
            draw_text_line(
                draw,
                padding_left,
                y_cursor,
                row,
                font=font,
                fill=ink_value,
                letter_spacing=letter_spacing,
            )
            rendered_rows.append(row)
            y_cursor += row_height + line_gap
        if not rendered_rows:
            break
        rendered_paragraphs.append(rendered_rows)
        if y_cursor >= height - padding_bottom:
            break
        y_cursor += paragraph_gap

    text = "\n\n".join("\n".join(rows) for rows in rendered_paragraphs)
    return image, text


def _render_text_layer(
    width: int,
    height: int,
    rng: random.Random,
    paragraphs: list[str],
    settings: VariationSettings,
) -> tuple[Image.Image, str, int, dict[str, Any]]:
    padding = _sample_padding(rng, settings)
    bg_value = rng.randint(settings.bg_min, settings.bg_max)
    contrast = rng.randint(settings.contrast_min, settings.contrast_max)
    ink_value = rng.randint(0, max(0, bg_value - contrast))
    size = rng.randint(settings.font_size_min, settings.font_size_max)
    font, font_path, font_style = _pick_font(rng, size, settings)
    letter_spacing = (
        rng.uniform(0.0, settings.letter_spacing_max)
        if rng.random() < settings.letter_spacing_prob
        else 0.0
    )
    line_gap = max(1, int(size * rng.uniform(settings.line_gap_min, settings.line_gap_max)))
    paragraph_gap = rng.randint(max(size // 2, line_gap), size * 2)
    image, text = _draw_text_layer(
        width,
        height,
        paragraphs,
        padding=padding,
        bg_value=bg_value,
        ink_value=ink_value,
        font=font,
        letter_spacing=letter_spacing,
        line_gap=line_gap,
        paragraph_gap=paragraph_gap,
    )
    layout_variations: dict[str, Any] = {
        "padding": list(padding),
        "bg_value": bg_value,
        "contrast": contrast,
        "ink_value": ink_value,
        "font_size": size,
        "font_style": font_style,
        "font_path": font_path,
        "letter_spacing": letter_spacing,
        "line_gap": line_gap,
        "paragraph_gap": paragraph_gap,
    }
    return image, text, bg_value, layout_variations


def render_paragraph_document(
    width: int,
    height: int,
    rng: random.Random,
    paragraphs: list[str],
    settings: VariationSettings,
) -> RenderedSample:
    """Render paragraphs; label contains only text that fit on the canvas."""
    image, text, bg_value, variations = _render_text_layer(
        width,
        height,
        rng,
        paragraphs,
        settings,
    )
    image, image_effects = apply_image_variations(
        image,
        rng,
        bg_value=bg_value,
        settings=settings,
    )
    variations["image_effects"] = image_effects
    return RenderedSample(
        image=image,
        text=text,
        width=width,
        height=height,
        variations=variations,
    )


def render_paragraph_from_variations(
    width: int,
    height: int,
    paragraphs: list[str],
    variations: dict[str, Any],
) -> RenderedSample:
    """Render using saved variation values from samples.json."""
    padding_raw = variations["padding"]
    padding = (
        int(padding_raw[0]),
        int(padding_raw[1]),
        int(padding_raw[2]),
        int(padding_raw[3]),
    )
    bg_value = int(variations["bg_value"])
    ink_value = int(variations["ink_value"])
    font_size = int(variations["font_size"])
    font_path = variations.get("font_path")
    font = (
        load_font(str(font_path), font_size)
        if font_path
        else ImageFont.load_default()
    )
    image, text = _draw_text_layer(
        width,
        height,
        paragraphs,
        padding=padding,
        bg_value=bg_value,
        ink_value=ink_value,
        font=font,
        letter_spacing=float(variations.get("letter_spacing", 0.0)),
        line_gap=int(variations["line_gap"]),
        paragraph_gap=int(variations["paragraph_gap"]),
    )
    image_effects = variations.get("image_effects")
    if isinstance(image_effects, dict) and image_effects:
        image = apply_image_effects(image, image_effects, bg_value=bg_value)
    return RenderedSample(
        image=image,
        text=text,
        width=width,
        height=height,
        variations=variations,
    )
