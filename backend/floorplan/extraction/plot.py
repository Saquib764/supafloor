"""Plot extracted room polygons on the extraction canvas."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from extraction.constants import CANVAS_SIZE

_FILL_COLORS = (
    "#E8D5B7",
    "#D4E6C3",
    "#C9D6DF",
    "#B8C5D6",
    "#F0EDE6",
    "#DDD5C8",
    "#A8C5B8",
    "#E5E5E5",
)


def _setup_ax(ax: plt.Axes) -> None:
    ax.set_xlim(1, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_polygons(ax: plt.Axes, rooms: list[dict]) -> None:
    for i, room in enumerate(rooms):
        pts = room.get("polygon") or []
        if len(pts) < 3:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        color = _FILL_COLORS[i % len(_FILL_COLORS)]
        patch = mpatches.Polygon(
            list(zip(xs, ys)),
            closed=True,
            facecolor=color,
            edgecolor="#2C2C2C",
            linewidth=1.5,
            alpha=0.85,
        )
        ax.add_patch(patch)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        ax.text(cx, cy, room.get("name", ""), ha="center", va="center", fontsize=8, color="#1A1A1A")


def plot_rooms(
    rooms: list[dict],
    *,
    background: Image.Image | None = None,
    out_path: Path | str | None = None,
) -> bytes:
    """Render room polygons. With background: side-by-side canvas + polygons. Returns PNG bytes."""
    panel_in = CANVAS_SIZE / 100
    if background is not None:
        fig, (ax_img, ax_poly) = plt.subplots(1, 2, figsize=(2 * panel_in, panel_in), dpi=100)
        _setup_ax(ax_img)
        ax_img.imshow(background, extent=(1, CANVAS_SIZE, CANVAS_SIZE, 1))
        _setup_ax(ax_poly)
        _draw_polygons(ax_poly, rooms)
    else:
        fig, ax_poly = plt.subplots(figsize=(panel_in, panel_in), dpi=100)
        _setup_ax(ax_poly)
        _draw_polygons(ax_poly, rooms)

    fig.patch.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    png_bytes = buf.getvalue()

    if out_path is not None:
        Path(out_path).write_bytes(png_bytes)
    return png_bytes
