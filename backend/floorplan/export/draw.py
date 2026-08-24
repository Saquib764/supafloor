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


def _draw_door_swing(
    ax: Axes,
    hinge: Coord,
    wall_angle: float,
    leaf_length: float,
    *,
    arc_angle_deg: float,
    to_the_right: bool,
    color: str | tuple[float, float, float],
) -> None:
    if leaf_length < 1e-6 or arc_angle_deg < 1e-6:
        return

    if to_the_right:
        open_angle = wall_angle - arc_angle_deg
        theta1, theta2 = -(arc_angle_deg + 2), 0
    else:
        open_angle = wall_angle + arc_angle_deg
        theta1, theta2 = 0, arc_angle_deg + 2

    leaf_end = _shift(hinge, leaf_length, open_angle)
    ax.plot(
        [hinge[0], leaf_end[0]],
        [hinge[1], leaf_end[1]],
        color=color,
        linewidth=1.5,
        solid_capstyle="butt",
    )

    arc_center = _shift(hinge, 1.5, wall_angle)
    swing = max(leaf_length - 1.5, leaf_length * 0.5)
    if swing <= 0:
        return
    ax.add_patch(
        Arc(
            arc_center,
            2 * swing,
            2 * leaf_length,
            angle=wall_angle,
            theta1=theta1,
            theta2=theta2,
            color=color,
            linewidth=1,
        )
    )


def draw_door(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    wall_thickness: float,
    to_the_right: bool = True,
    color: str | tuple[float, float, float] = "black",
    draw_arc: bool = True,
    door_type: str = "single",
    arc_angle_deg: float = 90.0,
) -> None:
    length, angle = _segment(start, end)
    if length < 1e-6:
        return

    cut = _opening_cut_polygon(start, end, wall_thickness)
    if cut:
        draw_polygon(ax, cut, facecolor="white", edgecolor="white", linewidth=0.0)

    if not draw_arc:
        return

    if door_type == "double":
        half = length / 2.0
        _draw_door_swing(
            ax,
            start,
            angle,
            half,
            arc_angle_deg=arc_angle_deg,
            to_the_right=to_the_right,
            color=color,
        )
        _draw_door_swing(
            ax,
            end,
            angle + 180.0,
            half,
            arc_angle_deg=arc_angle_deg,
            to_the_right=not to_the_right,
            color=color,
        )
        return

    _draw_door_swing(
        ax,
        start,
        angle,
        length,
        arc_angle_deg=arc_angle_deg,
        to_the_right=to_the_right,
        color=color,
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


def draw_pillar(
    ax: Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation_deg: float,
    *,
    facecolor: str | tuple[float, float, float] = "black",
    edgecolor: str | tuple[float, float, float] = "black",
    linewidth: float = 1.0,
) -> None:
    if width <= 0 or height <= 0:
        return
    hw, hh = width / 2.0, height / 2.0
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    corners = [
        (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
        for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]
    draw_polygon(ax, corners, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
