"""Sample center locations per space from type-specific weight grids."""

from __future__ import annotations

import math
import random
from typing import Any

from shapely.geometry import Point, Polygon

from floorplan.requirements.program import bedroom_parent, is_bedroom_name, space_role

_ANCHOR_ROLES = frozenset(
    {
        "living",
        "master_bedroom",
        "primary_bedroom",
        "bedroom",
        "kitchen",
        "dining",
        "hall",
        "garage",
    }
)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _near_point(p: tuple[float, float], q: tuple[float, float], sigma: float) -> float:
    return math.exp(-_dist(p, q) / max(sigma, 0.5))


def _boundary_dist(p: tuple[float, float], boundary_poly: Polygon) -> float:
    return float(boundary_poly.boundary.distance(Point(p)))


def _near_boundary(p: tuple[float, float], boundary_poly: Polygon, sigma: float = 2.5) -> float:
    d = _boundary_dist(p, boundary_poly)
    return math.exp(-d / max(sigma, 0.5))


def _grid_points(boundary_poly: Polygon, step: int = 1) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = (int(v) for v in boundary_poly.bounds)
    pts: list[tuple[int, int]] = []
    for x in range(minx, maxx + 1, step):
        for y in range(miny, maxy + 1, step):
            if boundary_poly.contains(Point(x + 0.5, y + 0.5)):
                pts.append((x, y))
    return pts


def _sample_weighted(
    weights: dict[tuple[int, int], float],
    rng: random.Random,
) -> tuple[float, float]:
    total = sum(weights.values())
    if total <= 0:
        keys = list(weights.keys())
        x, y = rng.choice(keys)
        return float(x) + 0.5, float(y) + 0.5
    pick = rng.uniform(0, total)
    acc = 0.0
    for (x, y), w in weights.items():
        acc += w
        if pick <= acc:
            return float(x) + 0.5, float(y) + 0.5
    x, y = next(iter(weights))
    return float(x) + 0.5, float(y) + 0.5


def _away_from_entry(
    p: tuple[float, float],
    entry: tuple[float, float],
    boundary_poly: Polygon,
) -> float:
    """Prefer points deeper inside when away from the entry."""
    cx = sum(c[0] for c in boundary_poly.exterior.coords[:-1]) / max(len(boundary_poly.exterior.coords) - 1, 1)
    cy = sum(c[1] for c in boundary_poly.exterior.coords[:-1]) / max(len(boundary_poly.exterior.coords) - 1, 1)
    inward = _dist(p, (cx, cy)) / max(_dist(entry, (cx, cy)), 1.0)
    return min(2.0, 0.5 + inward)


def _garage_block_penalty(
    p: tuple[float, float],
    entry: tuple[float, float],
    living_center: tuple[float, float] | None,
) -> float:
    """Penalize garage centers that sit on the entry corridor."""
    seg = _dist(p, entry)
    if seg < 3.0:
        return 0.05
    if living_center and _dist(p, living_center) < 2.5:
        return 0.2
    return 1.0


def center_sampling_order(names: list[str]) -> list[str]:
    role_rank = {
        "living": 0,
        "dining": 1,
        "kitchen": 2,
        "hall": 3,
        "master_bedroom": 4,
        "bedroom": 5,
        "closet": 7,
        "bathroom": 8,
        "toilet": 9,
        "laundry": 10,
        "balcony": 11,
        "garage": 12,
        "other": 13,
    }
    return sorted(names, key=lambda n: (role_rank.get(space_role(n), 99), n))


def _center_parent(name: str) -> str | None:
    parent = bedroom_parent(name)
    if parent:
        return parent
    if name == "Laundry":
        return "Kitchen"
    return None


def _is_anchor(name: str) -> bool:
    return space_role(name) in _ANCHOR_ROLES


def _clamp_inside(p: tuple[float, float], boundary_poly: Polygon) -> tuple[float, float]:
    if boundary_poly.contains(Point(p)):
        return p
    c = boundary_poly.centroid
    for t in (0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1):
        q = (c.x + (p[0] - c.x) * t, c.y + (p[1] - c.y) * t)
        if boundary_poly.contains(Point(q)):
            return q
    rp = boundary_poly.representative_point()
    return (float(rp.x), float(rp.y))


def _coverage_domain(boundary_poly: Polygon) -> Polygon:
    """Inset slightly so thin notches don't dominate empty-distance."""
    eroded = boundary_poly.buffer(-1.25)
    if eroded.is_empty or eroded.area < boundary_poly.area * 0.35:
        return boundary_poly
    if eroded.geom_type == "MultiPolygon":
        return max(eroded.geoms, key=lambda g: g.area)
    return eroded


def _ideal_radius(boundary_area: float, n: int) -> float:
    return 0.5 * math.sqrt(boundary_area / max(n, 1))


def _empty_distances(
    centers: dict[str, list[float] | tuple[float, float]],
    domain: Polygon,
    *,
    step: int = 2,
) -> list[tuple[tuple[float, float], float]]:
    pts = [(float(c[0]), float(c[1])) for c in centers.values()]
    if not pts:
        return []
    out: list[tuple[tuple[float, float], float]] = []
    for cell in _grid_points(domain, step=step):
        p = (float(cell[0]) + 0.5, float(cell[1]) + 0.5)
        d = min(_dist(p, c) for c in pts)
        out.append((p, d))
    return out


def center_coverage_score(
    centers: dict[str, list[float] | tuple[float, float]],
    boundary_poly: Polygon,
    *,
    step: int = 2,
) -> float:
    """How well center points span the layout, in [0, 1].

    Combines mean and max distance from interior samples to the nearest
    center, normalized by the ideal packing radius ``0.5 * sqrt(A / n)``.
    Low scores mean centers are clumped and leave large empty pockets.
    """
    if not centers:
        return 0.0
    domain = _coverage_domain(boundary_poly)
    samples = _empty_distances(centers, domain, step=step)
    if not samples:
        return 0.0
    dists = [d for _, d in samples]
    mean_d = sum(dists) / len(dists)
    max_d = max(dists)
    r = _ideal_radius(boundary_poly.area, len(centers))
    mean_term = r / max(mean_d, 1e-6)
    max_term = (2.0 * r) / max(max_d, 1e-6)
    return max(0.0, min(1.0, 0.45 * mean_term + 0.55 * max_term))


def _stretch_anchors(
    centers: dict[str, list[float]],
    boundary_poly: Polygon,
    *,
    fill: float = 0.88,
) -> dict[str, list[float]]:
    """Expand anchor cloud toward the boundary bbox; dependents keep offsets."""
    anchors = [n for n in centers if _is_anchor(n)]
    if len(anchors) < 2:
        return {k: list(v) for k, v in centers.items()}

    xs = [float(centers[n][0]) for n in anchors]
    ys = [float(centers[n][1]) for n in anchors]
    cx0, cx1 = min(xs), max(xs)
    cy0, cy1 = min(ys), max(ys)
    span_x = max(cx1 - cx0, 1e-6)
    span_y = max(cy1 - cy0, 1e-6)

    minx, miny, maxx, maxy = boundary_poly.bounds
    pad = 1.5
    tx0, ty0, tx1, ty1 = minx + pad, miny + pad, maxx - pad, maxy - pad
    sx = min(max((tx1 - tx0) * fill / span_x, 1.0), 2.4)
    sy = min(max((ty1 - ty0) * fill / span_y, 1.0), 2.4)
    if sx <= 1.02 and sy <= 1.02:
        return {k: list(v) for k, v in centers.items()}

    mx, my = (cx0 + cx1) * 0.5, (cy0 + cy1) * 0.5
    tmx, tmy = (tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5
    out = {k: list(v) for k, v in centers.items()}
    deltas: dict[str, tuple[float, float]] = {}

    for name in anchors:
        x, y = float(centers[name][0]), float(centers[name][1])
        nx, ny = _clamp_inside((tmx + (x - mx) * sx, tmy + (y - my) * sy), boundary_poly)
        deltas[name] = (nx - x, ny - y)
        out[name] = [round(nx, 3), round(ny, 3)]

    for name, pt in centers.items():
        parent = _center_parent(name)
        if not parent or parent not in deltas:
            continue
        dx, dy = deltas[parent]
        nx, ny = _clamp_inside((float(pt[0]) + dx, float(pt[1]) + dy), boundary_poly)
        out[name] = [round(nx, 3), round(ny, 3)]

    return out


def _nudge_toward_voids(
    centers: dict[str, list[float]],
    boundary_poly: Polygon,
    *,
    min_score: float,
    max_iters: int = 10,
) -> dict[str, list[float]]:
    """Move clustered anchors toward the largest empty interior region."""
    out = {k: list(v) for k, v in centers.items()}
    domain = _coverage_domain(boundary_poly)
    min_sep = math.sqrt(boundary_poly.area / max(len(out), 1)) * 0.32

    for _ in range(max_iters):
        samples = _empty_distances(out, domain, step=2)
        if not samples:
            break
        void, max_d = max(samples, key=lambda item: item[1])
        r = _ideal_radius(boundary_poly.area, len(out))
        if center_coverage_score(out, boundary_poly) >= min_score and max_d <= 2.8 * r:
            break

        anchors = [n for n in out if _is_anchor(n)]
        if not anchors:
            break

        def redundancy(name: str) -> float:
            p = (float(out[name][0]), float(out[name][1]))
            others = [_dist(p, (float(out[o][0]), float(out[o][1]))) for o in anchors if o != name]
            return -min(others) if others else 0.0

        # Prefer a clustered anchor; fall back to the one nearest the void.
        cand = max(anchors, key=redundancy)
        px, py = float(out[cand][0]), float(out[cand][1])
        step = 0.42
        nx, ny = _clamp_inside((px + step * (void[0] - px), py + step * (void[1] - py)), boundary_poly)

        crowded = any(
            _dist((nx, ny), (float(out[o][0]), float(out[o][1]))) < min_sep * 0.55
            for o in anchors
            if o != cand
        )
        if crowded:
            cand = min(anchors, key=lambda n: _dist((float(out[n][0]), float(out[n][1])), void))
            px, py = float(out[cand][0]), float(out[cand][1])
            nx, ny = _clamp_inside(
                (px + 0.35 * (void[0] - px), py + 0.35 * (void[1] - py)),
                boundary_poly,
            )

        dx, dy = nx - px, ny - py
        if abs(dx) + abs(dy) < 0.15:
            break
        out[cand] = [round(nx, 3), round(ny, 3)]
        for name, pt in list(out.items()):
            if _center_parent(name) != cand:
                continue
            qx, qy = _clamp_inside((float(pt[0]) + dx, float(pt[1]) + dy), boundary_poly)
            out[name] = [round(qx, 3), round(qy, 3)]

    return out


def _max_empty_distance(
    centers: dict[str, list[float] | tuple[float, float]],
    boundary_poly: Polygon,
    *,
    step: int = 2,
) -> float:
    samples = _empty_distances(centers, _coverage_domain(boundary_poly), step=step)
    return max((d for _, d in samples), default=0.0)


def _coverage_ok(
    centers: dict[str, list[float] | tuple[float, float]],
    boundary_poly: Polygon,
    *,
    min_score: float,
) -> bool:
    if center_coverage_score(centers, boundary_poly) < min_score:
        return False
    r = _ideal_radius(boundary_poly.area, len(centers))
    return _max_empty_distance(centers, boundary_poly) <= 2.8 * r


def improve_center_coverage(
    centers: dict[str, list[float]],
    boundary_poly: Polygon,
    *,
    min_score: float = 0.78,
) -> dict[str, list[float]]:
    """If centers are clumped, stretch then nudge them to raise coverage."""
    if len(centers) < 2:
        return {k: list(v) for k, v in centers.items()}
    if _coverage_ok(centers, boundary_poly, min_score=min_score):
        return {k: list(v) for k, v in centers.items()}
    stretched = _stretch_anchors(centers, boundary_poly)
    if _coverage_ok(stretched, boundary_poly, min_score=min_score):
        return stretched
    return _nudge_toward_voids(stretched, boundary_poly, min_score=min_score)


def _weight_for_space(
    name: str,
    p: tuple[float, float],
    *,
    boundary_poly: Polygon,
    entry: tuple[float, float],
    placed: dict[str, tuple[float, float]],
    graph: dict[str, Any],
    target_area: float,
) -> float:
    role = space_role(name)
    w = 1.0

    if role == "living":
        w *= 4.0 * _near_point(p, entry, 4.0)
        w *= 1.5 * _near_boundary(p, boundary_poly, 3.0)
    elif role in ("master_bedroom", "primary_bedroom", "bedroom"):
        w *= 2.5 * _near_boundary(p, boundary_poly, 2.5)
        w *= _away_from_entry(p, entry, boundary_poly)
        hall = placed.get("Hallway")
        if hall:
            w *= 1.5 * _near_point(p, hall, 5.0)
        if "Kitchen" in placed:
            w *= max(0.05, 1.0 - 0.92 * _near_point(p, placed["Kitchen"], 4.0))
    elif role == "kitchen":
        w *= 2.0 * _near_boundary(p, boundary_poly, 2.5)
        for key in ("Dining", "Living", "Living/Sleeping"):
            if key in placed:
                w *= 2.0 * _near_point(p, placed[key], 4.0)
                break
        for wet in placed:
            if space_role(wet) in ("bathroom", "toilet", "laundry"):
                w *= 1.3 * _near_point(p, placed[wet], 5.0)
        # Prefer kitchen away from bedrooms (no direct bedroom adjacency).
        for other, oc in placed.items():
            if space_role(other) in ("master_bedroom", "bedroom"):
                w *= max(0.05, 1.0 - 0.92 * _near_point(p, oc, 4.0))
    elif role == "dining":
        for key in ("Living", "Living/Sleeping"):
            if key in placed:
                w *= 3.0 * _near_point(p, placed[key], 4.0)
                break
        w *= 1.2 * _near_boundary(p, boundary_poly, 4.0)
    elif role == "hall":
        for key in ("Living", "Living/Sleeping"):
            if key in placed:
                w *= 2.5 * _near_point(p, placed[key], 5.0)
                break
        w *= 0.8 * _near_boundary(p, boundary_poly, 5.0)
    elif role in ("bathroom", "toilet", "closet"):
        parent_name = bedroom_parent(name)
        if parent_name and parent_name in placed:
            w *= 4.0 * _near_point(p, placed[parent_name], 3.0)
        w *= 0.5 * _near_boundary(p, boundary_poly, 1.5)
    elif role == "balcony":
        w *= 3.5 * _near_boundary(p, boundary_poly, 1.5)
        balcony_hosts = ["Living", "Living/Sleeping", "Dining"]
        balcony_hosts.extend(n for n in placed if is_bedroom_name(n))
        for key in balcony_hosts:
            if key in placed:
                w *= 2.0 * _near_point(p, placed[key], 5.0)
        for other, oc in placed.items():
            if space_role(other) == "balcony":
                w *= max(0.05, 1.0 - 2.5 * _near_point(p, oc, 4.0))
    elif role == "garage":
        w *= 3.0 * _near_point(p, entry, 5.0)
        w *= 2.5 * _near_boundary(p, boundary_poly, 2.0)
        living = placed.get("Living") or placed.get("Living/Sleeping")
        w *= _garage_block_penalty(p, entry, living)
    elif role == "laundry":
        if "Kitchen" in placed:
            w *= 3.0 * _near_point(p, placed["Kitchen"], 4.0)

    min_sep = math.sqrt(max(target_area, 1.0)) * 0.45
    for other, oc in placed.items():
        if other == name:
            continue
        d = _dist(p, oc)
        if d < min_sep:
            w *= max(0.05, d / min_sep)

    return max(w, 0.0)


def sample_centers(
    names: list[str],
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
    entry_point: list[float],
    graph: dict[str, Any],
    rng: random.Random,
) -> dict[str, list[float]]:
    """Sample one center per space in dependency order, then spread if clumped."""
    grid = _grid_points(boundary_poly)
    if not grid:
        minx, miny, maxx, maxy = boundary_poly.bounds
        grid = [(int(minx), int(miny))]

    entry = (float(entry_point[0]), float(entry_point[1]))
    placed: dict[str, tuple[float, float]] = {}
    centers: dict[str, list[float]] = {}

    for name in center_sampling_order(names):
        weights: dict[tuple[int, int], float] = {}
        target = selected_areas.get(name, 80.0)
        for cell in grid:
            p = (float(cell[0]) + 0.5, float(cell[1]) + 0.5)
            wt = _weight_for_space(
                name,
                p,
                boundary_poly=boundary_poly,
                entry=entry,
                placed=placed,
                graph=graph,
                target_area=target,
            )
            if wt > 0:
                weights[cell] = wt

        if not weights:
            x, y = rng.choice(grid)
            cx, cy = float(x) + 0.5, float(y) + 0.5
        else:
            cx, cy = _sample_weighted(weights, rng)

        placed[name] = (cx, cy)
        centers[name] = [round(cx, 3), round(cy, 3)]

    return improve_center_coverage(centers, boundary_poly)
