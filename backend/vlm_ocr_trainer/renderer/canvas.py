"""Render OCR preview and training viz panels."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HEADER_H = 28


def tensor_to_rgb(image: torch.Tensor) -> Image.Image:
    array = ((image.detach().float().cpu().squeeze(0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def with_label(image: Image.Image, label: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, HEADER_H), fill=(255, 255, 255))
    draw.text((8, 6), label, fill=(0, 0, 0), font=ImageFont.load_default())
    return labeled


def _wrap_text_lines(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
        if current:
            lines.append(current)
        if not paragraph and text.endswith("\n"):
            lines.append("")
    return lines or [""]


def text_panel(
    width: int,
    height: int,
    text: str,
    *,
    label: str | None = None,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    margin = 12
    max_width = width - margin * 2
    y = HEADER_H + 4 if label is not None else margin
    for line in _wrap_text_lines(text, draw, font, max_width):
        draw.text((margin, y), line, fill=fill, font=font)
        y += 14
        if y > height - margin:
            break
    if label is None:
        return panel
    return with_label(panel, label)


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("rows must not be empty")
    mode = rows[0].mode
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    merged = Image.new(mode, (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        merged.paste(row.convert(mode), (0, y))
        y += row.height
    return merged


def input_output_row(input_image: Image.Image | torch.Tensor, output_text: str) -> Image.Image:
    if isinstance(input_image, torch.Tensor):
        input_rgb = tensor_to_rgb(input_image)
    else:
        input_rgb = input_image.convert("RGB")
    output_panel = text_panel(input_rgb.width, input_rgb.height, output_text, fill=(0, 80, 0))
    row = Image.new("RGB", (input_rgb.width * 2, input_rgb.height), (255, 255, 255))
    row.paste(input_rgb, (0, 0))
    row.paste(output_panel, (input_rgb.width, 0))
    return row


def training_viz_row(
    input_image: Image.Image | torch.Tensor,
    ground_truth: str,
    prediction: str,
) -> Image.Image:
    if isinstance(input_image, torch.Tensor):
        input_rgb = tensor_to_rgb(input_image)
    else:
        input_rgb = input_image.convert("RGB")
    label_panel = text_panel(input_rgb.width, input_rgb.height, ground_truth)
    pred_panel = text_panel(
        input_rgb.width,
        input_rgb.height,
        prediction,
        fill=(0, 80, 0),
    )
    row = Image.new("RGB", (input_rgb.width * 3, input_rgb.height), (255, 255, 255))
    row.paste(input_rgb, (0, 0))
    row.paste(label_panel, (input_rgb.width, 0))
    row.paste(pred_panel, (input_rgb.width * 2, 0))
    return row
