"""Native matplotlib drawing for realistic floor plan export."""

from __future__ import annotations

import math
from typing import Any

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
    hatch: str | None = None,
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
            hatch=hatch,
        )
    )


def extend_wall_endpoints(
    start: Coord,
    end: Coord,
    *,
    thickness: float,
    extend_start: bool,
    extend_end: bool,
) -> tuple[Coord, Coord]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return start, end
    half = thickness / 2.0
    ux, uy = dx / length, dy / length
    sx, sy = start
    ex, ey = end
    if extend_start:
        sx -= ux * half
        sy -= uy * half
    if extend_end:
        ex += ux * half
        ey += uy * half
    return (sx, sy), (ex, ey)


def draw_room_interior_textured(
    ax: Axes,
    polygon: list[Coord],
    *,
    ink: str | tuple[float, float, float],
    hatch: str,
) -> None:
    draw_polygon(ax, polygon, facecolor="white", edgecolor=ink, linewidth=0.0, hatch=hatch)


def draw_wall_textured(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    thickness: float,
    ink: str | tuple[float, float, float],
    wall_hatch: str | None = None,
) -> None:
    strip = _opening_cut_polygon(start, end, thickness)
    if not strip:
        return
    if wall_hatch:
        draw_polygon(ax, strip, facecolor="white", edgecolor=ink, linewidth=0.0, hatch=wall_hatch)
    else:
        draw_polygon(ax, strip, facecolor=ink, edgecolor="none", linewidth=0.0)


def draw_wall_strip_outline(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    thickness: float,
    ink: str | tuple[float, float, float],
    linewidth: float = 1.0,
) -> None:
    strip = _opening_cut_polygon(start, end, thickness)
    if not strip:
        return
    draw_polygon(ax, strip, facecolor="none", edgecolor=ink, linewidth=linewidth)


def draw_wall_centerline(
    ax: Axes,
    start: Coord,
    end: Coord,
    *,
    ink: str | tuple[float, float, float],
    linewidth: float = 1.0,
) -> None:
    length, _ = _segment(start, end)
    if length < 1e-6:
        return
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=ink,
        linewidth=linewidth,
        solid_capstyle="butt",
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


def _pillar_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation_deg: float,
) -> list[Coord]:
    if width <= 0 or height <= 0:
        return []
    hw, hh = width / 2.0, height / 2.0
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    return [
        (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
        for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def draw_pillar_textured(
    ax: Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation_deg: float,
    *,
    ink: str | tuple[float, float, float],
    wall_hatch: str | None = None,
) -> None:
    corners = _pillar_corners(cx, cy, width, height, rotation_deg)
    if not corners:
        return
    if wall_hatch:
        draw_polygon(ax, corners, facecolor="white", edgecolor=ink, linewidth=0.0, hatch=wall_hatch)
    else:
        draw_polygon(ax, corners, facecolor=ink, edgecolor="none", linewidth=0.0)


def draw_pillar_outline(
    ax: Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation_deg: float,
    *,
    ink: str | tuple[float, float, float],
    linewidth: float = 1.0,
) -> None:
    corners = _pillar_corners(cx, cy, width, height, rotation_deg)
    if not corners:
        return
    draw_polygon(ax, corners, facecolor="none", edgecolor=ink, linewidth=linewidth)


def draw_wall_rect_outline(
    ax: Axes,
    polygon: list[Coord],
    *,
    ink: str | tuple[float, float, float],
    linewidth: float = 1.0,
) -> None:
    if len(polygon) < 3:
        return
    draw_polygon(ax, polygon, facecolor="none", edgecolor=ink, linewidth=linewidth)


def _coord_from_dict(point: dict[str, Any]) -> Coord:
    return (float(point["x"]), float(point["y"]))


def draw_dimension_tick(
    ax: Axes,
    center: Coord,
    line_angle_deg: float,
    *,
    length: float,
    ink: str | tuple[float, float, float],
    linewidth: float = 0.75,
) -> None:
    half = length / 2.0
    tick_angle = line_angle_deg + 45.0
    a = _shift(center, half, tick_angle)
    b = _shift(center, half, tick_angle + 180.0)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=ink, linewidth=linewidth, solid_capstyle="butt")


def draw_dimension(
    ax: Axes,
    spec: dict[str, Any],
    *,
    ink: str | tuple[float, float, float],
    linewidth: float = 0.75,
    tick_length: float = 6.0,
    font_size: float = 10.0,
) -> None:
    try:
        dim_a = _coord_from_dict(spec["dim_a"])
        dim_b = _coord_from_dict(spec["dim_b"])
        witness_a = _coord_from_dict(spec["witness_a"])
        witness_b = _coord_from_dict(spec["witness_b"])
        witness_c = _coord_from_dict(spec["witness_c"])
        witness_d = _coord_from_dict(spec["witness_d"])
        label = str(spec["label"])
        label_x = float(spec["label_x"])
        label_y = float(spec["label_y"])
        label_rotation = float(spec.get("label_rotation") or 0.0)
    except (KeyError, TypeError, ValueError):
        return

    _, line_angle = _segment(dim_a, dim_b)
    ax.plot(
        [witness_a[0], witness_b[0]],
        [witness_a[1], witness_b[1]],
        color=ink,
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    ax.plot(
        [witness_c[0], witness_d[0]],
        [witness_c[1], witness_d[1]],
        color=ink,
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    ax.plot(
        [dim_a[0], dim_b[0]],
        [dim_a[1], dim_b[1]],
        color=ink,
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    draw_dimension_tick(ax, dim_a, line_angle, length=tick_length, ink=ink, linewidth=linewidth)
    draw_dimension_tick(ax, dim_b, line_angle, length=tick_length, ink=ink, linewidth=linewidth)
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        color=ink,
        fontsize=font_size,
        rotation=label_rotation,
        family="monospace",
    )
