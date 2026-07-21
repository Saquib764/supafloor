"""Three area options (m²) per room-name root. Pick one at random when placing."""

from __future__ import annotations

# Keys are name roots: "Room 1" → "Room", "Living/Sleeping" → "Living/Sleeping"
AREA_OPTIONS: dict[str, tuple[float, float, float]] = {
    "Living/Sleeping": (18.0, 22.0, 28.0),
    "Living": (16.0, 20.0, 26.0),
    "Room": (9.0, 12.0, 15.0),
    "Toilet": (2.0, 3.0, 4.0),
    "Bathroom": (4.0, 5.0, 6.0),
    "Kitchen": (5.0, 7.0, 9.0),
    "Dining": (6.0, 8.0, 10.0),
    "Hall": (3.0, 4.0, 6.0),
    "Entry": (1.5, 2.0, 3.0),
    "Closet": (1.0, 2.0, 3.0),
    "Balcony": (3.0, 5.0, 7.0),
    "Laundry": (2.0, 3.0, 4.0),
    "Garage": (10.0, 14.0, 18.0),
}


def name_root(name: str) -> str:
    """'Room 2' → 'Room'; bare names pass through."""
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def area_options_for(name: str) -> tuple[float, float, float]:
    return AREA_OPTIONS.get(name_root(name), (6.0, 8.0, 10.0))
