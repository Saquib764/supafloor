"""Topology helpers: adjacency graphs, reachability, hashes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from typing import Optional

from floorplan.geometry import (
    area,
    aspect_ratio,
    perimeter,
    polygon_from_boundary,
    polygon_from_room,
    shared_edge,
    touches,
)
from floorplan.schemas.models import ConstraintGraph, FloorPlan, Room


def build_adjacency(plan: FloorPlan, tol: float = 0.08) -> dict[str, list[str]]:
    """Undirected adjacency from shared edges / touching rooms."""
    adj: dict[str, set[str]] = defaultdict(set)
    rooms = plan.rooms
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            if touches(a.coords(), b.coords(), tol=tol) or shared_edge(a.coords(), b.coords(), tol=tol):
                adj[a.id].add(b.id)
                adj[b.id].add(a.id)
    return {k: sorted(v) for k, v in adj.items()}


def adjacency_via_doors(plan: FloorPlan) -> dict[str, list[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for door in plan.doors:
        if len(door.connects) >= 2:
            a, b = door.connects[0], door.connects[1]
            adj[a].add(b)
            adj[b].add(a)
    # Fall back to geometric adjacency if no doors annotated
    if not adj:
        return build_adjacency(plan)
    return {k: sorted(v) for k, v in adj.items()}


def reachable(plan: FloorPlan, start_id: Optional[str] = None) -> set[str]:
    """BFS reachability through door connections (or geometric adjacency)."""
    adj = adjacency_via_doors(plan)
    if not plan.rooms:
        return set()
    if start_id is None:
        # Prefer entry, else first room
        entry = next((r for r in plan.rooms if r.room_type.value == "entry"), plan.rooms[0])
        start_id = entry.id
    seen = {start_id}
    q = deque([start_id])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def exterior_touching_rooms(plan: FloorPlan, tol: float = 0.08) -> list[str]:
    boundary = polygon_from_boundary(plan.ensure_boundary())
    result = []
    for room in plan.rooms:
        poly = polygon_from_room(room)
        inter = poly.boundary.intersection(boundary.boundary)
        length = 0.0
        if inter.geom_type == "LineString":
            length = inter.length
        elif inter.geom_type == "MultiLineString":
            length = sum(g.length for g in inter.geoms)
        elif not inter.is_empty:
            buffered = poly.buffer(tol).intersection(boundary.buffer(tol))
            length = buffered.area
        if length > tol:
            result.append(room.id)
    return result


def constraint_graph_from_plan(plan: FloorPlan) -> ConstraintGraph:
    adj = build_adjacency(plan)
    edges: list[tuple[str, str]] = []
    for a, neighbors in adj.items():
        for b in neighbors:
            if a < b:
                edges.append((a, b))
    return ConstraintGraph(
        nodes=[r.id for r in plan.rooms],
        edges=edges,
        required_exterior=exterior_touching_rooms(plan),
        preferred_adjacency=edges,
    )


def topology_hash(plan: FloorPlan) -> str:
    """Hash of room types + undirected adjacency by type multiset."""
    adj = build_adjacency(plan)
    id_to_type = {r.id: r.room_type.value for r in plan.rooms}
    type_edges = sorted(
        tuple(sorted((id_to_type[a], id_to_type[b])))
        for a, neighbors in adj.items()
        for b in neighbors
        if a < b and a in id_to_type and b in id_to_type
    )
    payload = {
        "types": sorted(id_to_type.values()),
        "edges": type_edges,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def geometry_hash(plan: FloorPlan, precision: int = 1) -> str:
    """Quantized geometry hash to catch near-duplicates."""

    def q(v: float) -> float:
        return round(v, precision)

    rooms = []
    for room in sorted(plan.rooms, key=lambda r: (r.room_type.value, area(r.coords()))):
        coords = [(q(x), q(y)) for x, y in room.coords()]
        rooms.append({"t": room.room_type.value, "c": coords, "a": q(area(room.coords()))})
    b = [(q(x), q(y)) for x, y in plan.ensure_boundary().coords()]
    payload = {"b": b, "r": rooms}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def adjacency_signature(plan: FloorPlan) -> str:
    return topology_hash(plan)


def room_ratio_signature(plan: FloorPlan) -> str:
    total = sum(area(r.coords()) for r in plan.rooms) or 1.0
    ratios = {}
    for room in plan.rooms:
        key = room.room_type.value
        ratios[key] = ratios.get(key, 0.0) + area(room.coords()) / total
    quantized = {k: round(v, 2) for k, v in sorted(ratios.items())}
    return hashlib.sha256(json.dumps(quantized, sort_keys=True).encode()).hexdigest()[:16]


def room_metrics(room: Room) -> dict:
    coords = room.coords()
    return {
        "id": room.id,
        "name": room.name,
        "room_type": room.room_type.value,
        "zone": room.zone.value,
        "area": area(coords),
        "aspect_ratio": aspect_ratio(coords),
        "perimeter": perimeter(coords),
    }
