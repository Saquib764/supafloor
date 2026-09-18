"""Clean binary masks for FPN inference and polygon fitting."""

from __future__ import annotations

import cv2
import numpy as np


def clean_binary_mask(
    mask: np.ndarray,
    *,
    open_px: int = 2,
    close_px: int = 2,
    min_area_px: int = 64,
) -> np.ndarray:
    """Open to drop speckles, close to fill gaps, then drop small components."""
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    if open_px > 0:
        k = 2 * open_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if min_area_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept = np.zeros_like(binary)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_area_px:
                kept[labels == index] = 1
        binary = kept

    return binary.astype(bool)


def morph_erode_dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode then dilate — smooths boundary before polygon fitting."""
    if px <= 0:
        return np.asarray(mask, dtype=bool)
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated.astype(bool)
