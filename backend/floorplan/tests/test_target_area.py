"""Tests for target floor-area options used during generation."""

from floorplan.dataset import (
    TARGET_AREAS_SQFT,
    dims_from_area_sqft,
    generate_sample,
    resolve_plan_dims,
)
from floorplan.geometry import area
from floorplan.services.boundary_adapt import generate_boundary


def test_target_area_options():
    assert TARGET_AREAS_SQFT == (900, 2000, 2500, 3000, 4000, 5000, 6000)


def test_dims_from_area_sqft_matches_rect_footprint():
    w, d = dims_from_area_sqft(2000, aspect=1.25, boundary_shape="rect")
    footprint_m2 = w * d
    footprint_sqft = footprint_m2 / 0.09290304
    assert abs(footprint_sqft - 2000) < 1.0


def test_dims_from_area_sqft_matches_l_footprint():
    w, d = dims_from_area_sqft(2500, aspect=1.2, boundary_shape="l")
    boundary = generate_boundary(width=w, depth=d, shape="l")
    footprint_m2 = area(boundary.coords())
    footprint_sqft = footprint_m2 / 0.09290304
    assert abs(footprint_sqft - 2500) < 2.0


def test_resolve_picks_from_options():
    areas = {resolve_plan_dims(seed=s)[2] for s in range(40)}
    assert areas <= set(TARGET_AREAS_SQFT)
    assert len(areas) >= 3


def test_resolve_explicit_target():
    w, d, sqft = resolve_plan_dims(seed=0, target_area_sqft=4000)
    assert sqft == 4000
    assert abs(w * d / 0.09290304 - 4000) < 1.0


def test_generate_sample_records_target_area():
    sample = generate_sample("studio", seed=11, mutate=False, include_renders=False)
    assert sample.metadata["target_area_sqft"] in TARGET_AREAS_SQFT
    assert sample.requirements["target_area_sqft"] == sample.metadata["target_area_sqft"]


def test_generate_sample_honors_explicit_dims():
    sample = generate_sample(
        "studio",
        width=10.0,
        depth=8.0,
        seed=0,
        mutate=False,
        include_renders=False,
    )
    # 80 m² ≈ 861 sqft
    assert sample.metadata["width_m"] == 10.0
    assert sample.metadata["depth_m"] == 8.0
