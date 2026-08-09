"""Native matplotlib drawing for realistic floor plan export."""

from __future__ import annotations

import math

from matplotlib.axes import Axes
from matplotlib.patches import Arc, Polygon as MplPolygon, Rectangle

Coord = tuple[float, float]


def _shift(origin: Coord, distance: float, angle_deg: float) -> Coord:
    rad = math.radians(angle_deg)
    return (origin[0] + distance * math.cos(rad), origin[1] + distance * math.sin(rad))


def _segment(start: Coord, end: Coord) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


def draw_polygon(
    ax: Axes,
    polygon: list[Coord],
    *,
    facecolor: str = "#F7F7F7",
    edgecolor: str = "none",
    linewidth: float = 0.0,
) -> None:
    if len(polygon) < 3:
        return
    closed = list(polygon)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    ax.add_patch(
        MplPolygon(
            closed,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def draw_wall(ax: Axes, start: Coord, end: Coord, thickness: float, *, color: str = "black") -> None:
    length, angle = _segment(start, end)
    if length < 1e-6:
        return
    ax.add_patch(
        Rectangle(start, length, thickness, angle=angle, facecolor=color, edgecolor=color)
    )


def draw_window(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    overall_thickness: float,
    line_thickness: float,
    color: str = "black",
) -> None:
    length, angle = _segment(start, end)
    if length < 1e-6 or overall_thickness <= 2 * line_thickness:
        return
    ax.add_patch(Rectangle(start, length, line_thickness, angle=angle, facecolor=color, edgecolor=color))
    inner = _shift(start, overall_thickness - line_thickness, angle + 90)
    ax.add_patch(Rectangle(inner, length, line_thickness, angle=angle, facecolor=color, edgecolor=color))


def draw_door(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    door_width: float,
    thickness: float = 2.0,
    to_the_right: bool = True,
    color: str = "black",
) -> None:
    length, angle = _segment(start, end)
    if length < 1e-6:
        return
    doorway_width = length
    frame_width = max((doorway_width - door_width) / 2, 0.5)
    door_width = min(door_width, doorway_width - 2 * frame_width) if doorway_width > 2 * frame_width else door_width

    ax.add_patch(Rectangle(start, frame_width, thickness, angle=angle, facecolor=color, edgecolor=color))

    far_frame = _shift(start, frame_width + door_width, angle)
    ax.add_patch(Rectangle(far_frame, frame_width, thickness, angle=angle, facecolor=color, edgecolor=color))

    hinges = _shift(start, frame_width, angle)
    if to_the_right:
        ax.add_patch(
            Rectangle(hinges, door_width, thickness, angle=angle - 90, facecolor=color, edgecolor=color)
        )
    else:
        hinges = _shift(hinges, thickness, angle + 90)
        ax.add_patch(
            Rectangle(hinges, thickness, door_width, angle=angle, facecolor=color, edgecolor=color)
        )

    arc_center = _shift(hinges, thickness, angle)
    swing = door_width - thickness
    if swing <= 0:
        return
    if to_the_right:
        theta1, theta2 = -92, 0
    else:
        theta1, theta2 = 0, 92
    ax.add_patch(
        Arc(
            arc_center,
            2 * swing,
            2 * door_width,
            angle=angle,
            theta1=theta1,
            theta2=theta2,
            color=color,
            linewidth=1,
        )
    )
