"""Render sampled paragraph documents with visual variation."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

from renderer.corpus import WordCorpus, sample_document_paragraphs
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
from renderer.variations import apply_image_variations

FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 16


def _pick_font(rng: random.Random, size: int) -> ImageFont.ImageFont:
    roll = rng.random()
    if roll < 0.08 and HANDWRITTEN_FONT_PATHS:
        pool = HANDWRITTEN_FONT_PATHS
    elif roll < 0.22 and BOLD_FONT_PATHS:
        pool = BOLD_FONT_PATHS
    elif roll < 0.36 and ITALIC_FONT_PATHS:
        pool = ITALIC_FONT_PATHS
    else:
        pool = PRINTED_FONT_PATHS + SERIF_FONT_PATHS
    if not pool:
        return ImageFont.load_default()
    return load_font(rng.choice(pool), size)


def _render_text_layer(
    width: int,
    height: int,
    rng: random.Random,
    paragraphs: list[str],
    *,
    scale: int = 1,
) -> tuple[Image.Image, str, int]:
    short_side = min(width, height)
    margin = rng.randint(max(8, short_side // 28), max(12, short_side // 16))
    bg_value = rng.randint(210, 255)
    contrast = rng.randint(35, 120)
    ink_value = rng.randint(0, max(0, bg_value - contrast))
    size = rng.randint(FONT_SIZE_MIN * scale, FONT_SIZE_MAX * scale)
    font = _pick_font(rng, size)
    letter_spacing = (
        rng.uniform(0.0, 1.8 * scale) if rng.random() < 0.55 else 0.0
    )
    line_gap = max(1, int(size * rng.uniform(0.15, 0.55)))
    paragraph_gap = rng.randint(max(size // 2, line_gap), size * 2)

    image = Image.new("L", (width, height), color=bg_value)
    draw = ImageDraw.Draw(image)
    rendered_paragraphs: list[list[str]] = []
    y_cursor = margin

    for paragraph in paragraphs:
        rows = wrap_line(
            paragraph,
            font,
            width - 2 * margin,
            letter_spacing=letter_spacing,
        )
        rendered_rows: list[str] = []
        for row in rows:
            _, _, _row_width, row_height = text_bbox(font, row)
            if y_cursor + row_height > height - margin:
                break
            draw_text_line(
                draw,
                margin,
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
        if y_cursor >= height - margin:
            break
        y_cursor += paragraph_gap

    text = "\n\n".join("\n".join(rows) for rows in rendered_paragraphs)
    return image, text, bg_value


def render_paragraph_document(
    width: int,
    height: int,
    rng: random.Random,
    paragraphs: list[str],
) -> RenderedSample:
    """Render paragraphs; label contains only text that fit on the canvas."""
    aa_scale = 2 if rng.random() < 0.35 else 1
    render_w = width * aa_scale
    render_h = height * aa_scale
    image, text, bg_value = _render_text_layer(
        render_w,
        render_h,
        rng,
        paragraphs,
        scale=aa_scale,
    )
    if aa_scale > 1:
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    image = apply_image_variations(image, rng, bg_value=bg_value)
    return RenderedSample(image=image, text=text, width=width, height=height)


def render_paragraph_sample(
    width: int,
    height: int,
    rng: random.Random,
    corpus: WordCorpus,
    *,
    paragraph_counts: tuple[int, ...],
    paragraph_len_min: int,
    paragraph_len_max: int,
) -> RenderedSample:
    paragraphs = sample_document_paragraphs(
        rng,
        corpus,
        paragraph_counts=paragraph_counts,
        paragraph_len_min=paragraph_len_min,
        paragraph_len_max=paragraph_len_max,
    )
    return render_paragraph_document(width, height, rng, paragraphs)
