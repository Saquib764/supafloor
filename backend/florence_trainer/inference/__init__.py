"""Shared Florence inference — config-free entry points."""

from inference.grid import sample_bright_grid_canvas_points, sample_grid_canvas_points
from inference.params import InferParams, infer_params_from_config, infer_params_from_mapping
from inference.progressive import (
    GRID_SIZE,
    MAX_POINTS_PER_PROMPT,
    MAX_STALE_ATTEMPTS,
    detect_rooms_at_points,
    drop_cross_white_canvas_points,
    expand_polygon,
    format_perimeter_counts,
    infer_at_points,
    infer_automatic,
    perimeter_pixel_counts,
    pick_canvas_points,
    point_in_polygon,
    remove_grid_points_inside_polygon,
    white_black_ratio,
)

__all__ = [
    "GRID_SIZE",
    "MAX_POINTS_PER_PROMPT",
    "MAX_STALE_ATTEMPTS",
    "InferParams",
    "detect_rooms_at_points",
    "drop_cross_white_canvas_points",
    "expand_polygon",
    "format_perimeter_counts",
    "infer_at_points",
    "infer_automatic",
    "infer_params_from_config",
    "infer_params_from_mapping",
    "perimeter_pixel_counts",
    "pick_canvas_points",
    "point_in_polygon",
    "remove_grid_points_inside_polygon",
    "sample_bright_grid_canvas_points",
    "sample_grid_canvas_points",
    "white_black_ratio",
]
