"""Path analysis on door adjacency graphs."""

from __future__ import annotations

from collections import deque

from geometry import area, min_width
from schemas.models import FloorPlan, Room


def shortest_path(adj: dict[str, list[str]], start: str, target: str) -> list[str] | None:
    if start == target:
        return [start]
    parent: dict[str, str | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == target:
                path = []
                node: str | None = target
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path))
            q.append(nxt)
    return None


def shortest_hop(adj: dict[str, list[str]], starts: list[str], target: str) -> int | None:
    best: int | None = None
    for s in starts:
        path = shortest_path(adj, s, target)
        if path is None:
            continue
        hops = len(path) - 1
        if best is None or hops < best:
            best = hops
    return best


def path_min_width(path: list[str], rooms_by_id: dict[str, Room]) -> float:
    if not path:
        return 0.0
    return min(min_width(rooms_by_id[rid].coords()) for rid in path if rid in rooms_by_id)


def entry_start_ids(plan: FloorPlan, exterior_room_ids: list[str]) -> list[str]:
    entries = [r.id for r in plan.rooms if r.room_type.value == "entry"]
    if entries:
        return entries
    starts: list[str] = []
    for door in plan.doors:
        if door.is_exterior:
            starts.extend(door.connects)
    if starts:
        return list(dict.fromkeys(starts))
    if exterior_room_ids:
        return list(exterior_room_ids)
    return [plan.rooms[0].id] if plan.rooms else []


def passageway_rooms(
    adj: dict[str, list[str]],
    rooms_by_id: dict[str, Room],
    *,
    skip_types: set[str],
) -> list[str]:
    """Non-circulation rooms with degree >= 3 used on multiple shortest paths."""
    hubs = [rid for rid, nbrs in adj.items() if len(nbrs) >= 3 and rooms_by_id[rid].room_type.value not in skip_types]
    if not hubs:
        return []
    ids = list(rooms_by_id)
    usage: dict[str, int] = {h: 0 for h in hubs}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if rooms_by_id[a].room_type.value in skip_types or rooms_by_id[b].room_type.value in skip_types:
                continue
            path = shortest_path(adj, a, b)
            if path is None or len(path) < 3:
                continue
            for mid in path[1:-1]:
                if mid in usage:
                    usage[mid] += 1
    return [rid for rid, cnt in usage.items() if cnt >= 2]


def circ_dead_end_ids(adj: dict[str, list[str]], circ_ids: set[str]) -> list[str]:
    return [rid for rid in circ_ids if len(adj.get(rid, [])) <= 1]
