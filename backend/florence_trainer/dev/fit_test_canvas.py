#!/usr/bin/env python3
"""Fit test/ floor-plan images onto a 1024×1024 white canvas (in place).

Scale to fit (aspect preserved, upscale allowed), center on white square,
overwrite each source file.

Usage (from backend/florence_trainer):

    uv run python dev/fit_test_canvas.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

CANVAS_SIZE = 1024
TEST_DIR = Path(__file__).resolve().parents[1] / "test"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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


def main() -> None:
    paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {TEST_DIR}")
    for path in paths:
        image = Image.open(path)
        fitted = fit_to_canvas(image)
        fmt = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else path.suffix.lstrip(".").upper()
        save_kw = {"quality": 95} if fmt == "JPEG" else {}
        fitted.save(path, format=fmt, **save_kw)
        print(f"{path.name}: {image.size} → {fitted.size}")


if __name__ == "__main__":
    main()
