"""Decide which named spaces to create from a sleeping-room count."""

from __future__ import annotations

import math
import random
from typing import Any

from floorplan.requirements.areas import area_options_for, name_root

_ENSUITE_KINDS = ("Bath", "Toilet", "Closet")


def bedroom_name(index: int) -> str:
    return f"Room {index}"


def is_ensuite(name: str) -> bool:
    return any(name.endswith(f" {kind}") for kind in _ENSUITE_KINDS)


def ensuite_kind(name: str) -> str | None:
    for kind in _ENSUITE_KINDS:
        if name.endswith(f" {kind}"):
            return kind
    return None


def is_bedroom_name(name: str) -> bool:
    return name_root(name) == "Room" and not is_ensuite(name)


def bedroom_parent(name: str) -> str | None:
    """Bedroom host for ``Room N Bath`` / ``Toilet`` / ``Closet``."""
    kind = ensuite_kind(name)
    if kind is None:
        return None
    parent = name[: -(len(kind) + 1)]
    return parent if is_bedroom_name(parent) else None


def ensuite_name(bedroom: str, kind: str) -> str:
    return f"{bedroom} {kind}"


def _bedroom_names(n: int) -> list[str]:
    """Sleeping rooms for n bedrooms."""
    return [bedroom_name(i) for i in range(1, n + 1)]


def _ensuite_for(bedroom: str, rng: random.Random) -> list[str]:
    """Random bath / toilet / closet; Room 1 tends to get more."""
    primary = bedroom == "Room 1"
    out: list[str] = []
    if rng.random() < (0.9 if primary else 0.55):
        out.append(ensuite_name(bedroom, "Bath"))
    if rng.random() < (0.5 if primary else 0.25):
        out.append(ensuite_name(bedroom, "Toilet"))
    if rng.random() < (0.6 if primary else 0.2):
        out.append(ensuite_name(bedroom, "Closet"))
    return out


def space_names_for(bedroom_count: int, rng: random.Random) -> list[str]:
    """Bedroom count (or Living/Sleeping when 1)."""
    n = max(1, int(bedroom_count))
    names: list[str] = []

    if n == 1:
        names.append("Living/Sleeping")
        names.append("Kitchen")
        names.append("Bathroom")
    else:
        names.append("Living")
        beds = _bedroom_names(n)
        names.extend(beds)
        for bed in beds:
            names.extend(_ensuite_for(bed, rng))
        names.append("Kitchen")
        names.append("Dining")
        names.append("Hallway")
        if n >= 3 and rng.random() < 0.45:
            names.append("Bathroom")

    if n >= 2 and rng.random() < 0.5:
        names.append("Closet")
    if rng.random() < (0.55 if n >= 2 else 0.35):
        names.append("Balcony")
    if n >= 3 and rng.random() < 0.35:
        names.append("Laundry")
    if n >= 3 and rng.random() < 0.3:
        names.append("Garage")

    return names


def _floor_area(area: float) -> float:
    return float(max(1, math.floor(area)))


def scale_selected_areas(
    selected: dict[str, float],
    boundary_area: float,
    *,
    fill_ratio: float = 1.0,
) -> dict[str, float]:
    """Scale soft-minimum room areas so their sum fits inside the boundary."""
    total = sum(selected.values())
    cap = boundary_area * fill_ratio
    if total <= cap or total <= 0:
        return {name: _floor_area(area) for name, area in selected.items()}
    factor = cap / total
    return {name: _floor_area(area * factor) for name, area in selected.items()}


def pick_selected_areas(names: list[str], rng: random.Random) -> dict[str, float]:
    return {name: float(rng.choice(area_options_for(name))) for name in names}


def space_role(name: str) -> str:
    """Coarse role for placement / graph logic."""
    if name in ("Living", "Living/Sleeping"):
        return "living"
    if name == "Room 1":
        return "master_bedroom"
    if is_bedroom_name(name):
        return "bedroom"
    if name.endswith(" Bath") or name == "Bathroom":
        return "bathroom"
    if name.endswith(" Toilet"):
        return "toilet"
    if name.endswith(" Closet") or name == "Closet":
        return "closet"
    if name == "Kitchen":
        return "kitchen"
    if name == "Dining":
        return "dining"
    if name == "Hallway":
        return "hall"
    if name == "Balcony":
        return "balcony"
    if name == "Garage":
        return "garage"
    if name == "Laundry":
        return "laundry"
    return "other"


def is_bedroom(name: str) -> bool:
    return space_role(name) in ("master_bedroom", "bedroom")


def is_toilet(name: str) -> bool:
    return space_role(name) == "toilet"


def toilet_parent(name: str) -> str | None:
    """Bedroom that owns a private toilet; None for shared toilets."""
    return bedroom_parent(name) if ensuite_kind(name) == "Toilet" else None


def closet_parent(name: str) -> str | None:
    """Bedroom that owns a private closet; None for shared closets."""
    return bedroom_parent(name) if ensuite_kind(name) == "Closet" else None

