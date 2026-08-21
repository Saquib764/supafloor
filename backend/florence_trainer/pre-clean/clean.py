"""Floor-plan image pre-cleaning: contrast stretch and isolated dark-patch removal."""

from __future__ import annotations

import numpy as np
from PIL import Image

import cv2


def _histogram_stretch_bounds(channel: np.ndarray) -> tuple[float, float]:
    """Use histogram peaks below/above 127 as stretch min and max."""
    hist = np.bincount(channel.astype(np.uint8).ravel(), minlength=256)
    below = hist[:127]
    above = hist[128:]

    pmin = float(np.argmax(below)) if below.any() else float(channel.min())
    pmax = float(np.argmax(above) + 128) if above.any() else float(channel.max())
    return pmin, pmax


def to_bw_stretched(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert to grayscale, stretch using histogram peaks, then clip to [0, 255]."""
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.float64)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            gray = np.array(Image.fromarray(arr).convert("L"), dtype=np.float64)
        else:
            gray = arr.astype(np.float64)

    pmin, pmax = _histogram_stretch_bounds(gray)
    if pmax > pmin:
        gray = (gray - pmin) / (pmax - pmin) * 255.0
    else:
        gray = np.zeros_like(gray)

    return np.clip(gray, 0, 255).astype(np.uint8)


def zero_isolated_dark_patches(
    gray: np.ndarray,
    *,
    patch_size: int = 2,
    threshold: float = 150,
) -> np.ndarray:
    """Zero 4×4 patches whose average and all neighbour patch averages are below threshold."""
    if gray.ndim != 2:
        raise ValueError("expected 2D grayscale array")

    out = 255-gray.copy()
    h, w = out.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph == 0 or pw == 0:
        return out

    region = out[: ph * patch_size, : pw * patch_size]
    patches = region.reshape(ph, patch_size, pw, patch_size)
    avgs = patches.mean(axis=(1, 3))

    for i in range(ph):
        for j in range(pw):
            if avgs[i, j] >= threshold:
                continue
            neighbours_dark = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ph and 0 <= nj < pw and avgs[ni, nj] >= threshold:
                        neighbours_dark = False
                        break
                if not neighbours_dark:
                    break
            if neighbours_dark:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                region[y0:y1, x0:x1] = 0
    out = 255-out

    # closing operation
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=2)
    return out


def pre_clean(image: Image.Image | np.ndarray) -> Image.Image:
    """Run both pre-clean steps and return an L-mode PIL image."""
    gray = to_bw_stretched(image)
    gray = zero_isolated_dark_patches(gray)
    return Image.fromarray(gray, mode="L")
