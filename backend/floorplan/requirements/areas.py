"""Three area options (ft²) per room-name root. Pick one at random when placing."""

from __future__ import annotations

# Keys are name roots: "Room 2" → "Room", "Living/Sleeping" → "Living/Sleeping"
# Values are absolute ft²; one option is chosen at random when placing.
AREA_OPTIONS: dict[str, tuple[float, float, float]] = {
    "Living/Sleeping": (200.0, 250.0, 320.0),
    "Living": (180.0, 220.0, 280.0),
    "Room 1": (130.0, 160.0, 180.0),
    "Room": (100.0, 130.0, 160.0),
    "Toilet": (25.0, 35.0, 45.0),
    "Bathroom": (45.0, 60.0, 75.0),
    "Kitchen": (80.0, 100.0, 120.0),
    "Dining": (100.0, 120.0, 150.0),
    "Hallway": (35.0, 50.0, 70.0),
    "Closet": (12.0, 20.0, 30.0),
    "Balcony": (40.0, 60.0, 90.0),
    "Laundry": (25.0, 35.0, 50.0),
    "Garage": (200.0, 280.0, 360.0),
}

_ENSUITE_AREA: dict[str, str] = {
    "Bath": "Bathroom",
    "Toilet": "Toilet",
    "Closet": "Closet",
}


def name_root(name: str) -> str:
    """'Room 2' → 'Room'; bare names pass through."""
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def area_options_for(name: str) -> tuple[float, float, float]:
    for kind, key in _ENSUITE_AREA.items():
        if name.endswith(f" {kind}"):
            return AREA_OPTIONS[key]
    if name == "Room 1":
        return AREA_OPTIONS["Room 1"]
    key = name_root(name)
    return AREA_OPTIONS.get(key, (80.0, 100.0, 120.0))
