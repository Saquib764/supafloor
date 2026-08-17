"""Imperial units (feet) shared across the engine."""

from __future__ import annotations

FT_PER_M = 3.280839895

INTERIOR_DOOR_FT = 2.75
EXTERIOR_DOOR_FT = 3.0
BATH_DOOR_FT = 2.5
MIN_OPENING_EDGE_FT = 3.0

WINDOW_HEIGHT_FT = 4.5
WINDOW_SILL_FT = 3.0


def window_length_ft(room_area_sqft: float, edge_length: float) -> float:
    """Glazing width from room area (~10% floor area at 4.5 ft height), capped by wall."""
    target = max(2.0, min(8.0, room_area_sqft / 40.0))
    return min(target, edge_length * 0.4)
