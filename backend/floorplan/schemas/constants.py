"""Shared constants with no internal floorplan imports."""

from __future__ import annotations

import random

TARGET_AREAS_SQFT: tuple[float, ...] = (150, 400, 600, 900, 1200, 1500, 2000, 2500, 3000, 4000, 4500, 5000)


def pick_target_area(*, index: int | None = None, rng: random.Random | None = None) -> float:
    if index is not None:
        return TARGET_AREAS_SQFT[index % len(TARGET_AREAS_SQFT)]
    rng = rng or random.Random()
    return float(rng.choice(TARGET_AREAS_SQFT))
