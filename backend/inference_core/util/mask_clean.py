"""Clean inferred SAM masks — largest component and smoothed boundary."""

from __future__ import annotations

import cv2
import numpy as np


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _smooth_boundary_mask(
    binary: np.ndarray,
    *,
    close_px: int,
    smooth_sigma: float,
    approx_epsilon: float,
) -> np.ndarray:
    cleaned = binary
    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if smooth_sigma > 0:
        blurred = cv2.GaussianBlur(cleaned.astype(np.float32), (0, 0), smooth_sigma)
        cleaned = (blurred >= 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return cleaned

    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, approx_epsilon, True)
    if len(approx) < 3:
        approx = contour

    smoothed = np.zeros_like(cleaned)
    cv2.fillPoly(smoothed, [approx], 1)
    return smoothed


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


def clean_mask(
    mask: np.ndarray,
    *,
    close_px: int = 2,
    smooth_sigma: float = 1.0,
    approx_epsilon: float = 2.0,
) -> np.ndarray:
    """Keep the largest connected component and return a smoothed boolean mask."""
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    largest = _largest_connected_component(binary)
    smoothed = _smooth_boundary_mask(
        largest,
        close_px=close_px,
        smooth_sigma=smooth_sigma,
        approx_epsilon=approx_epsilon,
    )
    return smoothed.astype(bool)
