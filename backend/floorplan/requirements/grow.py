"""Grow axis-aligned seed rectangles from sampled centers."""

from __future__ import annotations

import math
import random
from typing import Any

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from floorplan.requirements.centers import center_sampling_order
from floorplan.requirements.program import bedroom_parent, ensuite_kind, is_bedroom, is_bedroom_name, space_role

# Soft minimums from selected_areas; residual free space is claimed past those mins.
_MAX_ASPECT = 2.8
_MAX_ASPECT_HALL = 4.0
_ROLE_EXPAND_WEIGHT = {
    "living": 10.0,
    "master_bedroom": 9.0,
    "primary_bedroom": 8.0,
    "bedroom": 8.0,
    "dining": 7.0,
    "kitchen": 6.0,
    "hall": 5.0,
    "garage": 5.0,
    "bathroom": 4.0,
    "balcony": 4.0,
    "toilet": 3.0,
    "closet": 3.0,
    "laundry": 3.0,
    "other": 2.0,
}


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        poly = max(geom.geoms, key=lambda g: g.area)
        return poly if poly.area > 0 else None
    return None


def _rect_for_area(area: float, rng: random.Random) -> tuple[int, int]:
    aspect = rng.uniform(0.7, 1.5)
    w = max(1, int(round(math.sqrt(area * aspect))))
    h = max(1, int(round(area / max(w, 1))))
    if w * h < area * 0.85:
        h = max(h, int(round(area / w)))
    return w, h


def _rect_dict(x: int, y: int, w: int, h: int) -> dict[str, Any]:
    return {"x": x, "y": y, "w": w, "h": h, "area": float(w * h)}


def _free_polygon(boundary_poly: Polygon, occupied: list[Polygon]) -> Polygon | None:
    if not occupied:
        return boundary_poly
    return _largest_polygon(boundary_poly.difference(unary_union(occupied)))


def _touches_boundary(rect: Polygon, boundary_poly: Polygon, tol: float = 0.05) -> bool:
    return rect.boundary.intersection(boundary_poly.boundary).length > tol


def _attach_to_parent(
    parent: dict[str, Any],
    area: float,
    rng: random.Random,
    free_poly: Polygon,
    *,
    boundary_poly: Polygon | None = None,
    prefer_boundary: bool = False,
) -> dict[str, Any] | None:
    """Place a small room flush against a parent rectangle (slide along edges)."""
    sizes = [_rect_for_area(area, rng), _rect_for_area(area * 0.85, rng)]
    side = max(2, int(round(math.sqrt(max(area, 1.0)))))
    sizes.extend([(side, max(2, int(round(area / side)))), (max(2, int(round(area / side))), side)])
    px, py, pw, ph = parent["x"], parent["y"], parent["w"], parent["h"]

    def _collect(*, require_boundary: bool) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for w0, h0 in sizes:
            w = max(2, min(int(w0), 10))
            h = max(2, min(int(h0), 10))
            opts: list[tuple[int, int, int, int]] = []
            for y in range(py - h + 1, py + ph):
                opts.append((px + pw, y, w, h))
                opts.append((px - w, y, w, h))
            for x in range(px - w + 1, px + pw):
                opts.append((x, py + ph, w, h))
                opts.append((x, py - h, w, h))
            for x, y, rw, rh in opts:
                key = (x, y, rw, rh)
                if key in seen:
                    continue
                seen.add(key)
                geom = box(x, y, x + rw, y + rh)
                if not free_poly.covers(geom):
                    continue
                if require_boundary and boundary_poly is not None and not _touches_boundary(geom, boundary_poly):
                    continue
                found.append(_rect_dict(int(x), int(y), rw, rh))
        return found

    candidates = _collect(require_boundary=prefer_boundary and boundary_poly is not None)
    if not candidates and prefer_boundary:
        candidates = _collect(require_boundary=False)
    if not candidates:
        return None
    return rng.choice(candidates)


def _neighbor_names(name: str, graph: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for a, b in graph.get("edges", []):
        if a == name:
            out.append(b)
        elif b == name:
            out.append(a)
    return out


def _score_candidate(
    rect: dict[str, Any],
    *,
    cx: float,
    cy: float,
    target_area: float,
    boundary_poly: Polygon,
    name: str,
    neighbor_boxes: dict[str, Polygon],
    neighbors: list[str],
) -> float:
    geom = box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])
    rcx = rect["x"] + rect["w"] / 2.0
    rcy = rect["y"] + rect["h"] / 2.0
    score = 0.0
    score -= math.hypot(rcx - cx, rcy - cy) * 0.15
    ratio = geom.area / max(target_area, 1.0)
    if 0.8 <= ratio <= 1.3:
        score += 4.0
    elif ratio >= 0.5:
        score += ratio * 2.0
    else:
        score += ratio

    role = space_role(name)
    if role in ("master_bedroom", "bedroom", "living", "balcony"):
        if _touches_boundary(geom, boundary_poly):
            score += 3.0
        else:
            score -= 2.0

    for nb in neighbors:
        nb_geom = neighbor_boxes.get(nb)
        if nb_geom is None:
            continue
        if geom.boundary.intersection(nb_geom.boundary).length > 0.4:
            score += 5.0

    # Kitchen must not sit flush against a bedroom.
    if role == "kitchen":
        for nb_name, nb_geom in neighbor_boxes.items():
            if is_bedroom(nb_name) and geom.boundary.intersection(nb_geom.boundary).length > 0.4:
                return float("-inf")
    elif role in ("master_bedroom", "bedroom"):
        kit = neighbor_boxes.get("Kitchen")
        if kit is not None and geom.boundary.intersection(kit.boundary).length > 0.4:
            return float("-inf")

    return score


def _candidates_near_center(
    cx: float,
    cy: float,
    w: int,
    h: int,
    free_poly: Polygon,
    radius: int = 8,
) -> list[dict[str, Any]]:
    ix = int(round(cx - w / 2))
    iy = int(round(cy - h / 2))
    seen: set[tuple[int, int, int, int]] = set()
    out: list[dict[str, Any]] = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x = ix + dx
            y = iy + dy
            key = (x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            geom = box(x, y, x + w, y + h)
            if free_poly.covers(geom):
                out.append(_rect_dict(x, y, w, h))

    if out:
        return out

    minx, miny, maxx, maxy = (int(v) for v in free_poly.bounds)
    w = min(w, max(1, maxx - minx))
    h = min(h, max(1, maxy - miny))
    for x in range(minx, max(minx, maxx - w) + 1):
        for y in range(miny, max(miny, maxy - h) + 1):
            geom = box(x, y, x + w, y + h)
            if free_poly.covers(geom):
                out.append(_rect_dict(x, y, w, h))
    return out


def _place_hall_corridor(
    living: dict[str, Any],
    bedroom_centers: list[tuple[float, float]],
    area: float,
    free_poly: Polygon,
    rng: random.Random,
) -> dict[str, Any] | None:
    if not bedroom_centers:
        return None
    bx = sum(c[0] for c in bedroom_centers) / len(bedroom_centers)
    by = sum(c[1] for c in bedroom_centers) / len(bedroom_centers)
    lx = living["x"] + living["w"] / 2.0
    ly = living["y"] + living["h"] / 2.0
    side = max(3, int(round(math.sqrt(area))))
    depth = max(3, int(round(area / side)))
    midx = int(round((lx + bx) / 2))
    midy = int(round((ly + by) / 2))
    candidates: list[dict[str, Any]] = []
    for w, h in ((side, depth), (depth, side)):
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                x = midx - w // 2 + dx
                y = midy - h // 2 + dy
                geom = box(x, y, x + w, y + h)
                if not free_poly.covers(geom):
                    continue
                living_box = box(living["x"], living["y"], living["x"] + living["w"], living["y"] + living["h"])
                if geom.boundary.intersection(living_box.boundary).length < 0.4:
                    continue
                candidates.append(_rect_dict(x, y, w, h))
    if not candidates:
        return None
    return rng.choice(candidates)


def _aspect_ok(name: str, w: int, h: int, *, under_minimum: bool = False) -> bool:
    if w <= 0 or h <= 0:
        return False
    if under_minimum:
        return True
    limit = _MAX_ASPECT_HALL if space_role(name) == "hall" else _MAX_ASPECT
    return max(w / h, h / w) <= limit


def _aabb_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (
        a["x"] + a["w"] <= b["x"]
        or b["x"] + b["w"] <= a["x"]
        or a["y"] + a["h"] <= b["y"]
        or b["y"] + b["h"] <= a["y"]
    )


def _aabb_shares_edge(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    vertical = (ax1 == bx0 or bx1 == ax0) and max(ay0, by0) < min(ay1, by1)
    horizontal = (ay1 == by0 or by1 == ay0) and max(ax0, bx0) < min(ax1, bx1)
    return vertical or horizontal


def _kitchen_bedroom_contact(rect: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    name = rect["name"]
    if name == "Kitchen":
        return any(is_bedroom(o["name"]) and _aabb_shares_edge(rect, o) for o in others)
    if is_bedroom(name):
        return any(o["name"] == "Kitchen" and _aabb_shares_edge(rect, o) for o in others)
    return False


def _cell_box(x: int, y: int) -> Polygon:
    return box(x, y, x + 1, y + 1)


def _rect_cells(rect: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (x, y)
        for x in range(int(rect["x"]), int(rect["x"] + rect["w"]))
        for y in range(int(rect["y"]), int(rect["y"] + rect["h"]))
    }


def _cells_share_edge(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ax, ay = a
    bx, by = b
    if ax + 1 == bx and ay < by + 1 and by < ay + 1:
        return True
    if bx + 1 == ax and ay < by + 1 and by < ay + 1:
        return True
    if ay + 1 == by and ax < bx + 1 and bx < ax + 1:
        return True
    if by + 1 == ay and ax < bx + 1 and bx < ax + 1:
        return True
    return False


def _kitchen_bedroom_cell_contact(
    name: str,
    cell: tuple[int, int],
    room_cells: dict[str, set[tuple[int, int]]],
) -> bool:
    if name == "Kitchen":
        return any(
            is_bedroom(other) and any(_cells_share_edge(cell, c) for c in cells)
            for other, cells in room_cells.items()
            if other != "Kitchen"
        )
    if is_bedroom(name):
        kitchen = room_cells.get("Kitchen", set())
        return any(_cells_share_edge(cell, c) for c in kitchen)
    return False


def _absorb_free_cells(
    rects: list[dict[str, Any]],
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
) -> list[dict[str, Any]]:
    """Claim adjacent 1×1 free cells when full-edge AABB growth cannot."""
    if not rects:
        return []

    room_cells = {rect["name"]: _rect_cells(rect) for rect in rects}
    minx, miny, maxx, maxy = (int(v) for v in boundary_poly.bounds)
    occupied = set().union(*room_cells.values())
    free_cells = {
        (x, y)
        for x in range(minx, maxx)
        for y in range(miny, maxy)
        if (x, y) not in occupied and boundary_poly.covers(_cell_box(x, y))
    }

    max_steps = max(1, len(free_cells) * len(rects))
    for _ in range(max_steps):
        best: tuple[float, str, tuple[int, int]] | None = None
        for name, cells in room_cells.items():
            cur_area = float(len(cells))
            minimum = float(selected_areas.get(name, 0.0))
            deficit = max(0.0, minimum - cur_area)
            role_w = _ROLE_EXPAND_WEIGHT.get(space_role(name), 2.0)

            adjacent: set[tuple[int, int]] = set()
            for cx, cy in cells:
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (nx, ny) in free_cells:
                        adjacent.add((nx, ny))

            for cell in adjacent:
                if _kitchen_bedroom_cell_contact(name, cell, room_cells):
                    continue
                score = (1000.0 + deficit * 10.0 + role_w * 5.0 + 1.0) if deficit > 0 else role_w * 5.0 + 0.5
                if best is None or score > best[0]:
                    best = (score, name, cell)

        if best is None:
            break
        _, name, cell = best
        room_cells[name].add(cell)
        free_cells.remove(cell)

    out: list[dict[str, Any]] = []
    for rect in rects:
        name = rect["name"]
        cells = room_cells[name]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
        row = {**rect, "x": x0, "y": y0, "w": w, "h": h, "area": float(len(cells))}

        unit_geom = _largest_polygon(
            unary_union([_cell_box(cx, cy) for cx, cy in cells]).intersection(boundary_poly)
        )
        bbox_geom = box(x0, y0, x0 + w, y0 + h)
        if unit_geom is not None and abs(unit_geom.area - bbox_geom.area) > 0.01:
            row["footprint"] = [(float(x), float(y)) for x, y in unit_geom.exterior.coords[:-1]]
        out.append(row)
    return out


def _expand_legal(
    rect: dict[str, Any],
    others: list[dict[str, Any]],
    boundary_poly: Polygon,
    *,
    under_minimum: bool = False,
) -> bool:
    if not _aspect_ok(rect["name"], rect["w"], rect["h"], under_minimum=under_minimum):
        return False
    geom = box(rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])
    if not boundary_poly.covers(geom):
        return False
    return not any(_aabb_overlap(rect, other) for other in others)


def expand_rectangles_into_free(
    rectangles: list[dict[str, Any]],
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
) -> list[dict[str, Any]]:
    """Expand AABB edges into free space; selected_areas are soft minimums."""
    if not rectangles:
        return []

    rects: list[dict[str, Any]] = []
    for rect in rectangles:
        x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
        row = _rect_dict(x, y, w, h)
        row["name"] = rect["name"]
        rects.append(row)

    max_steps = max(1, int(boundary_poly.area))
    for _ in range(max_steps):
        best: tuple[float, int, dict[str, Any]] | None = None
        for idx, rect in enumerate(rects):
            others = [rects[j] for j in range(len(rects)) if j != idx]
            name = rect["name"]
            cur_area = float(rect["w"] * rect["h"])
            minimum = float(selected_areas.get(name, 0.0))
            deficit = max(0.0, minimum - cur_area)
            role_w = _ROLE_EXPAND_WEIGHT.get(space_role(name), 2.0)

            grown = (
                {**rect, "x": rect["x"] - 1, "w": rect["w"] + 1},
                {**rect, "w": rect["w"] + 1},
                {**rect, "y": rect["y"] - 1, "h": rect["h"] + 1},
                {**rect, "h": rect["h"] + 1},
            )
            for cand in grown:
                if not _expand_legal(cand, others, boundary_poly, under_minimum=deficit > 0):
                    continue
                contact = cand["h"] if cand["w"] != rect["w"] else cand["w"]
                if _kitchen_bedroom_contact(cand, others):
                    continue
                if deficit > 0:
                    score = 1000.0 + deficit * 10.0 + role_w * 5.0 + contact
                else:
                    score = role_w * 5.0 + contact * 0.5
                if best is None or score > best[0]:
                    best = (score, idx, cand)

        if best is None:
            break
        _, idx, chosen = best
        chosen["area"] = float(chosen["w"] * chosen["h"])
        rects[idx] = chosen

    return _absorb_free_cells(rects, selected_areas, boundary_poly)


def _try_attach(
    name: str,
    host: str,
    *,
    rectangles_by_name: dict[str, dict[str, Any]],
    selected_areas: dict[str, float],
    occupied: list[Polygon],
    neighbor_boxes: dict[str, Polygon],
    boundary_poly: Polygon,
    rng: random.Random,
    prefer_boundary: bool = False,
) -> bool:
    if name in rectangles_by_name or host not in rectangles_by_name:
        return False
    free_poly = _free_polygon(boundary_poly, occupied)
    if free_poly is None:
        return False
    attached = _attach_to_parent(
        rectangles_by_name[host],
        selected_areas.get(name, 40.0),
        rng,
        free_poly,
        boundary_poly=boundary_poly,
        prefer_boundary=prefer_boundary,
    )
    if attached is None:
        return False
    attached["name"] = name
    rectangles_by_name[name] = attached
    geom = box(attached["x"], attached["y"], attached["x"] + attached["w"], attached["y"] + attached["h"])
    occupied.append(geom)
    neighbor_boxes[name] = geom
    return True


def grow_rectangles(
    names: list[str],
    centers: dict[str, list[float]],
    selected_areas: dict[str, float],
    boundary_poly: Polygon,
    graph: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Place seed rectangles near centers, then expand into free space.

    ``selected_areas`` are soft minimums: seeds aim near them, then edges keep
    growing into leftover free cells (past the minimum when space remains).
    """
    occupied: list[Polygon] = []
    neighbor_boxes: dict[str, Polygon] = {}
    rectangles_by_name: dict[str, dict[str, Any]] = {}

    for name in center_sampling_order(names):
        if name in rectangles_by_name:
            continue
        center = centers.get(name)
        if not center:
            continue
        cx, cy = float(center[0]), float(center[1])
        target = selected_areas.get(name, 80.0)
        free_poly = _free_polygon(boundary_poly, occupied)
        if free_poly is None or free_poly.area < max(1.0, target * 0.25):
            continue

        role = space_role(name)
        parent_key = bedroom_parent(name)

        # Ensuite rooms require their bedroom host; never free-float without it.
        if parent_key and parent_key not in rectangles_by_name:
            continue

        if parent_key and role in ("bathroom", "toilet", "closet"):
            parent_rect = rectangles_by_name[parent_key]
            attached = _attach_to_parent(
                parent_rect,
                target,
                rng,
                free_poly,
                boundary_poly=boundary_poly,
            )
            if attached is None and ensuite_kind(name) == "Closet":
                attached = _attach_to_parent(
                    parent_rect,
                    4.0,
                    rng,
                    free_poly,
                    boundary_poly=boundary_poly,
                )
            if attached is not None:
                attached["name"] = name
                rectangles_by_name[name] = attached
                geom = box(attached["x"], attached["y"], attached["x"] + attached["w"], attached["y"] + attached["h"])
                occupied.append(geom)
                neighbor_boxes[name] = geom
                continue
            if ensuite_kind(name) == "Closet":
                continue

        if role == "hall" and "Living" in rectangles_by_name:
            bed_centers = [
                (float(centers[n][0]), float(centers[n][1]))
                for n in names
                if n in centers and space_role(n) in ("master_bedroom", "bedroom")
            ]
            living_key = "Living" if "Living" in rectangles_by_name else "Living/Sleeping"
            corridor = _place_hall_corridor(
                rectangles_by_name[living_key],
                bed_centers,
                target,
                free_poly,
                rng,
            )
            if corridor is not None:
                corridor["name"] = name
                rectangles_by_name[name] = corridor
                geom = box(corridor["x"], corridor["y"], corridor["x"] + corridor["w"], corridor["y"] + corridor["h"])
                occupied.append(geom)
                neighbor_boxes[name] = geom
                continue

        if role == "hall" and "Living/Sleeping" in rectangles_by_name and "Living" not in rectangles_by_name:
            bed_centers = [
                (float(centers[n][0]), float(centers[n][1]))
                for n in names
                if n in centers and space_role(n) in ("master_bedroom", "bedroom")
            ]
            corridor = _place_hall_corridor(
                rectangles_by_name["Living/Sleeping"],
                bed_centers,
                target,
                free_poly,
                rng,
            )
            if corridor is not None:
                corridor["name"] = name
                rectangles_by_name[name] = corridor
                geom = box(corridor["x"], corridor["y"], corridor["x"] + corridor["w"], corridor["y"] + corridor["h"])
                occupied.append(geom)
                neighbor_boxes[name] = geom
                continue

        attach_host = None
        prefer_boundary = False
        if role == "balcony":
            balcony_hosts = ["Living", "Living/Sleeping", "Dining"]
            balcony_hosts.extend(n for n in names if is_bedroom_name(n))
            for host in balcony_hosts:
                if host in rectangles_by_name:
                    attach_host = host
                    prefer_boundary = True
                    break
        elif role == "laundry" and "Kitchen" in rectangles_by_name:
            attach_host = "Kitchen"

        if attach_host and attach_host in rectangles_by_name:
            attached = _attach_to_parent(
                rectangles_by_name[attach_host],
                target,
                rng,
                free_poly,
                boundary_poly=boundary_poly,
                prefer_boundary=prefer_boundary,
            )
            if attached is not None:
                attached["name"] = name
                rectangles_by_name[name] = attached
                geom = box(attached["x"], attached["y"], attached["x"] + attached["w"], attached["y"] + attached["h"])
                occupied.append(geom)
                neighbor_boxes[name] = geom
                continue

        neighbors = _neighbor_names(name, graph)
        best_rect: dict[str, Any] | None = None
        best_score = float("-inf")

        for w, h in [(w, h) for w, h in [_rect_for_area(target, rng), _rect_for_area(target * 0.85, rng)]]:
            w = min(w, int(max(1, free_poly.bounds[2] - free_poly.bounds[0])))
            h = min(h, int(max(1, free_poly.bounds[3] - free_poly.bounds[1])))
            candidates = _candidates_near_center(cx, cy, w, h, free_poly)
            if role in ("master_bedroom", "bedroom") and "Hallway" in neighbor_boxes:
                hall_geom = neighbor_boxes["Hallway"]
                touching = [
                    c
                    for c in candidates
                    if box(c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]).boundary.intersection(hall_geom.boundary).length
                    > 0.4
                ]
                if touching:
                    candidates = touching
            for cand in candidates:
                s = _score_candidate(
                    cand,
                    cx=cx,
                    cy=cy,
                    target_area=target,
                    boundary_poly=boundary_poly,
                    name=name,
                    neighbor_boxes=neighbor_boxes,
                    neighbors=neighbors,
                )
                if s > best_score:
                    best_score = s
                    best_rect = cand

        if best_rect is None:
            continue

        best_rect["name"] = name
        rectangles_by_name[name] = best_rect
        geom = box(best_rect["x"], best_rect["y"], best_rect["x"] + best_rect["w"], best_rect["y"] + best_rect["h"])
        occupied.append(geom)
        neighbor_boxes[name] = geom

        if is_bedroom_name(name):
            for ens in names:
                if bedroom_parent(ens) != name or ensuite_kind(ens) != "Closet":
                    continue
                if ens in rectangles_by_name:
                    continue
                attached = _try_attach(
                    ens,
                    name,
                    rectangles_by_name=rectangles_by_name,
                    selected_areas=selected_areas,
                    occupied=occupied,
                    neighbor_boxes=neighbor_boxes,
                    boundary_poly=boundary_poly,
                    rng=rng,
                )
                if attached:
                    continue
                free_now = _free_polygon(boundary_poly, occupied)
                if free_now is None:
                    continue
                tiny = _attach_to_parent(
                    rectangles_by_name[name],
                    4.0,
                    rng,
                    free_now,
                    boundary_poly=boundary_poly,
                )
                if tiny is None:
                    continue
                tiny["name"] = ens
                rectangles_by_name[ens] = tiny
                tgeom = box(tiny["x"], tiny["y"], tiny["x"] + tiny["w"], tiny["y"] + tiny["h"])
                occupied.append(tgeom)
                neighbor_boxes[ens] = tgeom

        if name in ("Living", "Living/Sleeping"):
            for bal in [n for n in names if space_role(n) == "balcony"]:
                _try_attach(
                    bal,
                    name,
                    rectangles_by_name=rectangles_by_name,
                    selected_areas=selected_areas,
                    occupied=occupied,
                    neighbor_boxes=neighbor_boxes,
                    boundary_poly=boundary_poly,
                    rng=rng,
                    prefer_boundary=True,
                )
        if name == "Kitchen":
            for laun in [n for n in names if space_role(n) == "laundry"]:
                _try_attach(
                    laun,
                    "Kitchen",
                    rectangles_by_name=rectangles_by_name,
                    selected_areas=selected_areas,
                    occupied=occupied,
                    neighbor_boxes=neighbor_boxes,
                    boundary_poly=boundary_poly,
                    rng=rng,
                )

    seeds = [rectangles_by_name[name] for name in names if name in rectangles_by_name]
    return expand_rectangles_into_free(seeds, selected_areas, boundary_poly)
