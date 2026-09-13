"""Random post-render image degradations for OCR training."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def apply_image_variations(
    image: Image.Image,
    rng: random.Random,
    *,
    bg_value: int,
) -> Image.Image:
    """Apply blur, noise, JPEG, rotation, and resize artifacts."""
    if rng.random() < 0.45:
        radius = rng.uniform(0.25, 0.9)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    array = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.85:
        sigma = rng.uniform(1.0, 5.0)
        noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        array += noise_rng.normal(0.0, sigma, array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")

    if rng.random() < 0.4:
        quality = rng.randint(35, 88)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("L")

    if rng.random() < 0.5:
        angle = rng.uniform(-0.5, 0.5)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=bg_value,
        )

    if rng.random() < 0.45:
        width, height = image.size
        scale = rng.uniform(0.82, 0.96)
        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))
        down_filter = rng.choice(
            [Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BOX],
        )
        up_filter = rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR])
        image = image.resize((small_w, small_h), resample=down_filter)
        image = image.resize((width, height), resample=up_filter)

    return image
