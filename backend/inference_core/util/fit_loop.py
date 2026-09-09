"""Interactive SAM mask-fit loop — clean input image after each mask."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from config import Config
from util.grid import sample_bright_grid_canvas_points
from util.mask_clean import clean_mask, morph_erode_dilate
from util.mask_fit import (
    FitVariant,
    fit_mask_variants,
    pick_best_variant,
    refine_polygon_by_sobel,
    snap_polygon_to_corners,
)
from util.pool import (
    drop_cross_white_canvas_points,
    point_filter_gray,
    remove_from_grid_pool,
    remove_grid_points_inside_polygon,
)
from util.pre_clean import find_corners, pre_clean, save_corners_overlay, sobel_edges
from util.render import draw_mask, draw_polygon_outline
from util.sam_runtime import load_sam_predictor, load_viewer_image


def _clean_working_gray(working_gray: np.ndarray, mask: np.ndarray) -> None:
    """Paint the detected room as wall (dark) so point filter excludes it."""
    working_gray[np.asarray(mask, dtype=bool)] = 0


def _save_fit_overlay(
    image: Image.Image,
    *,
    mask: np.ndarray,
    variant: FitVariant,
    out_dir: Path,
    attempt: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_mask(draw_polygon_outline(image, variant.polygon, width=5), mask)
    out_path = out_dir / f"step-{attempt:03d}.png"
    overlay.save(out_path)
    print(f"wrote {out_path}")


def run_fit_loop(
    cfg: Config,
    image_path: Path,
    *,
    wait_continue_or_esc: Callable[[], bool],
) -> None:
    predictor, sam_cfg, encode_image, predict_point_encoded = load_sam_predictor(cfg.sam_config)
    image = load_viewer_image(
        image_path,
        canvas_size=cfg.canvas_size,
        margin=cfg.margin,
    )
    sobel_img = sobel_edges(image)
    sobel = np.array(sobel_img, dtype=np.uint8)
    sobel_path = cfg.output / "sobel.png"
    sobel_path.parent.mkdir(parents=True, exist_ok=True)
    sobel_img.save(sobel_path)
    print(f"wrote {sobel_path}")
    corners = find_corners(image)
    save_corners_overlay(image, cfg.output / "corners.png", corners=corners)
    width, height = image.size
    working_image = pre_clean(image).convert("L")
    working_gray = np.array(working_image, dtype=np.uint8)
    filter_gray = point_filter_gray(working_gray, dilate_px=cfg.point_filter.dilate_px)

    grid = cfg.grid
    fit = cfg.fit
    rng = random.Random(grid.seed)
    out_dir = fit.output_dir

    grid_points = sample_bright_grid_canvas_points(
        rng,
        working_image.convert("RGB"),
        n=grid.size,
        canvas_size=cfg.canvas_size,
        tries_per_cell=grid.tries_per_cell,
    )
    before = len(grid_points)
    grid_points = drop_cross_white_canvas_points(filter_gray, grid_points)
    if before > len(grid_points):
        print(f"point filter: removed {before - len(grid_points)} grid points")

    print(f"grid pool: {len(grid_points)} points after filter")
    print("encoding image (backbone)...")
    encoded = encode_image(predictor, image)

    attempt = 0
    while grid_points:
        attempt += 1
        point = rng.choice(grid_points)
        remove_from_grid_pool(grid_points, [point])
        print(f"\n--- attempt {attempt} ---")
        print(f"point ({int(point[0])}, {int(point[1])}), pool left: {len(grid_points)}")

        mask, _sam_score = predict_point_encoded(
            predictor,
            encoded,
            point,
            multimask_output=sam_cfg.infer.multimask_output,
        )
        mask = clean_mask(
            mask,
            close_px=cfg.mask_smooth.close_px,
            smooth_sigma=cfg.mask_smooth.smooth_sigma,
            approx_epsilon=cfg.mask_smooth.approx_epsilon,
        )
        if not mask.any():
            print("empty mask — skipping")
            if not wait_continue_or_esc():
                break
            continue

        fit_mask = morph_erode_dilate(mask, 3)
        if not fit_mask.any():
            print("empty mask after morph — skipping")
            if not wait_continue_or_esc():
                break
            continue

        variants = fit_mask_variants(
            fit_mask,
            width=width,
            height=height,
            min_sides=fit.min_sides,
            max_sides=fit.max_sides,
        )
        if not variants:
            print("no polygon fits — skipping")
            if not wait_continue_or_esc():
                break
            continue

        best = pick_best_variant(variants)
        snapped = snap_polygon_to_corners(best.polygon, corners)
        refined, line_scores = refine_polygon_by_sobel(
            snapped,
            corners,
            sobel,
            rng,
            width=width,
            height=height,
        )
        best = FitVariant(
            sides=best.sides,
            polygon=refined,
            mask_minus_poly=best.mask_minus_poly,
            poly_minus_mask=best.poly_minus_mask,
            score=best.score,
        )
        for i, line_score in enumerate(line_scores):
            p0 = refined[i]
            p1 = refined[(i + 1) % len(refined)]
            print(
                f"line {i + 1}: ({p0[0]},{p0[1]})-({p1[0]},{p1[1]}) "
                f"score={line_score:.4f}"
            )
        removed = remove_grid_points_inside_polygon(grid_points, refined)
        if removed:
            print(f"removed {removed} pool point(s) inside polygon")
        _save_fit_overlay(
            image,
            mask=fit_mask,
            variant=best,
            out_dir=out_dir,
            attempt=attempt,
        )
        if not wait_continue_or_esc():
            break

        _clean_working_gray(working_gray, mask)
        filter_gray = point_filter_gray(working_gray, dilate_px=cfg.point_filter.dilate_px)
        before = len(grid_points)
        grid_points[:] = drop_cross_white_canvas_points(filter_gray, grid_points)
        if before > len(grid_points):
            print(f"re-filter after clean: removed {before - len(grid_points)} point(s)")

    print(f"done: {attempt} attempt(s)")
