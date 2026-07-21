"""Decide which named spaces to create from a sleeping-room count."""

from __future__ import annotations

import random
from typing import Any


def space_names_for(total_rooms: int, rng: random.Random) -> list[str]:
    """total_rooms = count of Room N (or Living/Sleeping when 1)."""
    n = max(1, int(total_rooms))
    names: list[str] = []

    if n == 1:
        names.append("Living/Sleeping")
    else:
        names.append("Living")
        for i in range(1, n + 1):
            names.append(f"Room {i}")

    names.append("Kitchen")
    if n >= 2:
        names.append("Dining")
        names.append("Hall")

    names.append("Bathroom")
    if n >= 3:
        names.append("Toilet")

    if n >= 2 and rng.random() < 0.5:
        names.append("Closet")
    if rng.random() < (0.55 if n >= 2 else 0.35):
        names.append("Balcony")
    if n >= 3 and rng.random() < 0.35:
        names.append("Laundry")
    if n >= 3 and rng.random() < 0.3:
        names.append("Garage")

    return names


def build_requirements(
    total_rooms: int,
    *,
    usable_area: float,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Backward-compatible thin wrapper: names + counts only."""
    rng = rng or random.Random()
    n = max(1, int(total_rooms))
    names = space_names_for(n, rng)
    return {
        "total_rooms": n,
        "usable_area": float(usable_area),
        "space_names": names,
    }
