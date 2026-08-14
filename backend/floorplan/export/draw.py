"""Native matplotlib drawing for realistic floor plan export."""

from __future__ import annotations

import math

from matplotlib.axes import Axes
from matplotlib.patches import Arc, Polygon as MplPolygon

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
    facecolor: str | None = "none",
    edgecolor: str = "black",
    linewidth: float = 1.5,
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


def _opening_cut_polygon(start: Coord, end: Coord, thickness: float) -> list[Coord]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    offset_x = -dy / length * (thickness / 2.0)
    offset_y = dx / length * (thickness / 2.0)
    return [
        (start[0] + offset_x, start[1] + offset_y),
        (end[0] + offset_x, end[1] + offset_y),
        (end[0] - offset_x, end[1] - offset_y),
        (start[0] - offset_x, start[1] - offset_y),
    ]


def draw_door(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    wall_thickness: float,
    to_the_right: bool = True,
    color: str | tuple[float, float, float] = "black",
    draw_arc: bool = True,
) -> None:
    length, angle = _segment(start, end)
    if length < 1e-6:
        return

    cut = _opening_cut_polygon(start, end, wall_thickness)
    if cut:
        draw_polygon(ax, cut, facecolor="white", edgecolor="white", linewidth=0.0)

    if not draw_arc:
        return

    door_width = length
    hinges = start
    leaf_angle = angle - 90 if to_the_right else angle + 90
    leaf_end = _shift(hinges, door_width, leaf_angle)
    ax.plot(
        [hinges[0], leaf_end[0]],
        [hinges[1], leaf_end[1]],
        color=color,
        linewidth=1.5,
        solid_capstyle="butt",
    )

    arc_center = _shift(hinges, 1.5, angle)
    swing = max(door_width - 1.5, length * 0.5)
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


def draw_window(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    wall_thickness: float,
) -> None:
    cut = _opening_cut_polygon(start, end, wall_thickness - 4.0)
    if cut:
        draw_polygon(ax, cut, facecolor="white", edgecolor="white", linewidth=0.0)
