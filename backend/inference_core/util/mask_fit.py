"""Fit a SAM mask with simplified polygons of 4–10 sides."""

from __future__ import annotations

import random
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


def snap_point_to_nearest_corner(
    x: int,
    y: int,
    corners: list[tuple[int, int]],
) -> list[int]:
    if not corners:
        return [int(x), int(y)]
    corner_arr = np.array(corners, dtype=np.float64)
    dx = corner_arr[:, 0] - x
    dy = corner_arr[:, 1] - y
    idx = int(np.argmin(dx * dx + dy * dy))
    cx, cy = corners[idx]
    return [int(cx), int(cy)]


def snap_polygon_to_corners(
    polygon: list[list[int]],
    corners: list[tuple[int, int]],
) -> list[list[int]]:
    if not corners or len(polygon) < 3:
        return polygon
    return [snap_point_to_nearest_corner(x, y, corners) for x, y in polygon]


def line_sobel_score(
    p0: list[int],
    p1: list[int],
    sobel: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.line(
        canvas,
        (int(p0[0]), int(p0[1])),
        (int(p1[0]), int(p1[1])),
        255,
        1,
    )
    line_mask = canvas > 0
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if length <= 0:
        return 0.0
    intersection = line_mask & (np.asarray(sobel, dtype=np.uint8) > 0)
    return float(intersection.sum()) / length


def polygon_line_sobel_scores(
    polygon: list[list[int]],
    sobel: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[float]:
    if len(polygon) < 2:
        return []
    scores: list[float] = []
    for i in range(len(polygon)):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % len(polygon)]
        scores.append(line_sobel_score(p0, p1, sobel, width=width, height=height))
    return scores


def polygon_total_sobel_score(
    polygon: list[list[int]],
    sobel: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    return sum(polygon_line_sobel_scores(polygon, sobel, width=width, height=height))


def _locked_vertex_indices(line_scores: list[float], *, threshold: float = 0.99) -> set[int]:
    locked: set[int] = set()
    n = len(line_scores)
    for i, score in enumerate(line_scores):
        if score > threshold:
            locked.add(i)
            locked.add((i + 1) % n)
    return locked


def _movable_vertex_indices(n: int, locked: set[int], excluded: set[int]) -> list[int]:
    return [i for i in range(n) if i not in locked and i not in excluded]


def refine_polygon_by_sobel(
    polygon: list[list[int]],
    corners: list[tuple[int, int]],
    sobel: np.ndarray,
    rng: random.Random,
    *,
    width: int,
    height: int,
    max_iters: int = 100,
    lock_threshold: float = 0.99,
) -> tuple[list[list[int]], list[float]]:
    poly = [point[:] for point in polygon]
    excluded: set[int] = set()

    for iteration in range(1, max_iters + 1):
        line_scores = polygon_line_sobel_scores(poly, sobel, width=width, height=height)
        locked = _locked_vertex_indices(line_scores, threshold=lock_threshold)
        movable = _movable_vertex_indices(len(poly), locked, excluded)
        total = sum(line_scores)
        print(f"refine iter {iteration}: score={total:.4f}")
        if not movable:
            break

        idx = rng.choice(movable)
        old_pt = poly[idx][:]
        poly[idx] = snap_point_to_nearest_corner(poly[idx][0], poly[idx][1], corners)

        if poly[idx] == old_pt:
            excluded.add(idx)
            print(f"  vertex {idx}: unchanged")
            continue

        new_total = polygon_total_sobel_score(poly, sobel, width=width, height=height)
        if new_total > total:
            excluded.discard(idx)
            print(f"  vertex {idx}: accepted score={new_total:.4f}")
        else:
            poly[idx] = old_pt
            print(f"  vertex {idx}: rejected score={new_total:.4f}")

    final_scores = polygon_line_sobel_scores(poly, sobel, width=width, height=height)
    return poly, final_scores


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
