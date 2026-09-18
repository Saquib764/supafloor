"""Fit binary masks with simplified polygons."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FitVariant:
    sides: int
    polygon: list[list[int]]
    mask_minus_poly: int
    poly_minus_mask: int
    score: float


def fit_score(
    mask_minus_poly: int,
    poly_minus_mask: int,
    *,
    width: int,
    height: int,
    sides: int,
) -> float:
    image_pixels = width * height
    if image_pixels <= 0:
        return sides * 0.001
    diff_score = (mask_minus_poly + poly_minus_mask) / image_pixels
    return diff_score + sides * 0.001


def pick_best_variant(variants: list[FitVariant]) -> FitVariant | None:
    if not variants:
        return None
    return min(variants, key=lambda variant: variant.score)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _polygon_mask(polygon: list[list[int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def polygon_mask_diff(
    polygon: list[list[int]],
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any() or len(polygon) < 3:
        return None
    poly_mask = _polygon_mask(polygon, width, height)
    mask_minus_poly = int((mask_bool & ~poly_mask).sum())
    poly_minus_mask = int((poly_mask & ~mask_bool).sum())
    return mask_minus_poly, poly_minus_mask


def _approx_with_sides(contour: np.ndarray, target_sides: int) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    lo = 0.001 * peri
    hi = 0.5 * peri
    best: np.ndarray | None = None
    best_diff = 10_000

    for _ in range(40):
        eps = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, eps, True)
        n = len(approx)
        if n == target_sides:
            return approx
        diff = abs(n - target_sides)
        if diff < best_diff and n >= 3:
            best_diff = diff
            best = approx
        if n > target_sides:
            lo = eps
        else:
            hi = eps

    return best


def fit_mask_variants(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    min_sides: int = 4,
    max_sides: int = 10,
) -> list[FitVariant]:
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 3:
        return []

    variants: list[FitVariant] = []
    for sides in range(min_sides, max_sides + 1):
        approx = _approx_with_sides(contour, sides)
        if approx is None or len(approx) < 3:
            continue
        polygon = [[int(x), int(y)] for x, y in approx[:, 0]]
        diff = polygon_mask_diff(polygon, mask, width=width, height=height)
        if diff is None:
            continue
        mask_minus_poly, poly_minus_mask = diff
        variants.append(
            FitVariant(
                sides=sides,
                polygon=polygon,
                mask_minus_poly=mask_minus_poly,
                poly_minus_mask=poly_minus_mask,
                score=fit_score(
                    mask_minus_poly,
                    poly_minus_mask,
                    width=width,
                    height=height,
                    sides=sides,
                ),
            )
        )
    return variants
