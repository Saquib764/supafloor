"""Random post-render image degradations for OCR training."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

from config import VariationSettings


def apply_image_variations(
    image: Image.Image,
    rng: random.Random,
    *,
    bg_value: int,
    settings: VariationSettings,
) -> Image.Image:
    """Apply blur, noise, JPEG, rotation, and resize artifacts."""
    if rng.random() < settings.blur_prob:
        radius = rng.uniform(settings.blur_radius_min, settings.blur_radius_max)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    array = np.asarray(image, dtype=np.float32)
    if rng.random() < settings.noise_prob:
        sigma = rng.uniform(settings.noise_sigma_min, settings.noise_sigma_max)
        noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        array += noise_rng.normal(0.0, sigma, array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")

    if rng.random() < settings.jpeg_prob:
        quality = rng.randint(settings.jpeg_quality_min, settings.jpeg_quality_max)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("L")

    if rng.random() < settings.rotation_prob:
        angle = rng.uniform(-settings.rotation_degrees, settings.rotation_degrees)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=bg_value,
        )

    if rng.random() < settings.resize_prob:
        width, height = image.size
        scale = rng.uniform(settings.resize_scale_min, settings.resize_scale_max)
        small_w = max(1, int(width * scale))
        small_h = max(1, int(height * scale))
        down_filter = rng.choice(
            [Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BOX],
        )
        up_filter = rng.choice([Image.Resampling.NEAREST, Image.Resampling.BILINEAR])
        image = image.resize((small_w, small_h), resample=down_filter)
        image = image.resize((width, height), resample=up_filter)

    return image
