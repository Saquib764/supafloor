"""Random post-render image degradations for OCR training."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

from config import VariationSettings

_RESAMPLING = {
    "NEAREST": Image.Resampling.NEAREST,
    "BILINEAR": Image.Resampling.BILINEAR,
    "BOX": Image.Resampling.BOX,
}


def apply_image_effects(
    image: Image.Image,
    effects: dict[str, object],
    *,
    bg_value: int,
) -> Image.Image:
    """Apply saved post-render effects from a variations dict."""
    if "blur" in effects:
        blur = effects["blur"]
        if isinstance(blur, dict) and "radius" in blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(blur["radius"])))

    array = np.asarray(image, dtype=np.float32)
    if "noise" in effects:
        noise = effects["noise"]
        if isinstance(noise, dict) and "sigma" in noise:
            sigma = float(noise["sigma"])
            seed = int(noise.get("seed", 0))
            noise_rng = np.random.default_rng(seed)
            array += noise_rng.normal(0.0, sigma, array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")

    if "jpeg" in effects:
        jpeg = effects["jpeg"]
        if isinstance(jpeg, dict) and "quality" in jpeg:
            quality = int(jpeg["quality"])
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("L")

    if "rotation" in effects:
        rotation = effects["rotation"]
        if isinstance(rotation, dict) and "degrees" in rotation:
            image = image.rotate(
                float(rotation["degrees"]),
                resample=Image.Resampling.BICUBIC,
                fillcolor=bg_value,
            )

    if "resize" in effects:
        resize = effects["resize"]
        if isinstance(resize, dict) and "scale" in resize:
            width, height = image.size
            scale = float(resize["scale"])
            small_w = max(1, int(width * scale))
            small_h = max(1, int(height * scale))
            down_filter = _RESAMPLING[str(resize.get("down_filter", "BILINEAR"))]
            up_filter = _RESAMPLING[str(resize.get("up_filter", "BILINEAR"))]
            image = image.resize((small_w, small_h), resample=down_filter)
            image = image.resize((width, height), resample=up_filter)

    return image


def apply_image_variations(
    image: Image.Image,
    rng: random.Random,
    *,
    bg_value: int,
    settings: VariationSettings,
) -> tuple[Image.Image, dict[str, object]]:
    """Apply blur, noise, JPEG, and rotation artifacts."""
    effects: dict[str, object] = {}

    if rng.random() < settings.blur_prob:
        radius = rng.uniform(settings.blur_radius_min, settings.blur_radius_max)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        effects["blur"] = {"radius": radius}

    array = np.asarray(image, dtype=np.float32)
    if rng.random() < settings.noise_prob:
        sigma = rng.uniform(settings.noise_sigma_min, settings.noise_sigma_max)
        noise_seed = rng.randint(0, 2**31 - 1)
        noise_rng = np.random.default_rng(noise_seed)
        array += noise_rng.normal(0.0, sigma, array.shape)
        effects["noise"] = {"sigma": sigma, "seed": noise_seed}
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")

    if rng.random() < settings.jpeg_prob:
        quality = rng.randint(settings.jpeg_quality_min, settings.jpeg_quality_max)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("L")
        effects["jpeg"] = {"quality": quality}

    if rng.random() < settings.rotation_prob:
        angle = rng.uniform(-settings.rotation_degrees, settings.rotation_degrees)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=bg_value,
        )
        effects["rotation"] = {"degrees": angle}

    return image, effects
