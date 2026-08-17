"""Geometry helpers for authoring closed rooms and opening lines."""

from __future__ import annotations

from schemas.models import Door, Point2D, Room, Window
from schemas.enums import RoomType, ZoneType


def make_room(
    name: str,
    coords: list[tuple[float, float]],
    *,
    room_type: RoomType = RoomType.OTHER,
    zone: ZoneType = ZoneType.PUBLIC,
    **kwargs,
) -> Room:
    """Create a room with a closed, geometrically valid polygon.

    ``coords`` may be open or closed; Room validation closes and validates.
    """
    return Room(
        name=name,
        room_type=room_type,
        zone=zone,
        polygon=[Point2D(x=float(x), y=float(y)) for x, y in coords],
        **kwargs,
    )


def make_door_line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connects: list[str] | None = None,
    is_exterior: bool = False,
    swing: str = "in",
) -> Door:
    return Door(
        start=Point2D(x=start[0], y=start[1]),
        end=Point2D(x=end[0], y=end[1]),
        connects=connects or [],
        is_exterior=is_exterior,
        swing=swing,
    )


def make_window_line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    room_ids: list[str] | None = None,
) -> Window:
    return Window(
        start=Point2D(x=start[0], y=start[1]),
        end=Point2D(x=end[0], y=end[1]),
        room_ids=room_ids or [],
    )


def horizontal_opening(
    x_center: float,
    y: float,
    length: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    half = length / 2.0
    return (x_center - half, y), (x_center + half, y)


def vertical_opening(
    x: float,
    y_center: float,
    length: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    half = length / 2.0
    return (x, y_center - half), (x, y_center + half)
