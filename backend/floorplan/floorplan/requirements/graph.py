"""Build connection graph and circulation tree from space names."""

from __future__ import annotations

from collections import deque
from typing import Any

from floorplan.requirements.areas import name_root
from floorplan.requirements.program import bedroom_parent, is_bedroom, is_bedroom_name


def _living_name(names: list[str]) -> str:
    if "Living" in names:
        return "Living"
    return "Living/Sleeping"


def _add_edge(edges: set[tuple[str, str]], a: str, b: str) -> None:
    if a and b and a != b:
        edges.add(tuple(sorted((a, b))))


def build_connection_graph(names: list[str]) -> dict[str, Any]:
    """Adjacency edges + spanning tree rooted at the living room."""
    living = _living_name(names) if any(n in names for n in ("Living", "Living/Sleeping")) else names[0]
    edges: set[tuple[str, str]] = set()
    name_set = set(names)

    if "Hallway" in name_set:
        _add_edge(edges, living, "Hallway")
    if "Dining" in name_set:
        _add_edge(edges, living, "Dining")
        _add_edge(edges, "Dining", "Kitchen")
    elif "Kitchen" in name_set:
        _add_edge(edges, living, "Kitchen")

    hall_or_living = "Hallway" if "Hallway" in name_set else living
    for name in names:
        if is_bedroom(name):
            _add_edge(edges, hall_or_living, name)

    for name in names:
        parent = bedroom_parent(name)
        if parent and parent in name_set:
            _add_edge(edges, parent, name)

    if "Kitchen" in name_set and "Laundry" in name_set:
        _add_edge(edges, "Kitchen", "Laundry")

    if "Bathroom" in name_set:
        _add_edge(edges, hall_or_living, "Bathroom")

    if "Balcony" in name_set:
        balcony_hosts = ["Living", "Living/Sleeping", "Dining"]
        balcony_hosts.extend(n for n in names if is_bedroom_name(n))
        for host in balcony_hosts:
            if host in name_set:
                _add_edge(edges, host, "Balcony")
                break

    if "Garage" in name_set:
        _add_edge(edges, "Garage", living)
        if "Hallway" in name_set:
            _add_edge(edges, "Garage", "Hallway")

    for name in names:
        if name_root(name) == "Room" and "Closet" in name_set and name == "Room 2":
            _add_edge(edges, name, "Closet")

    # Kitchen must not connect directly to a bedroom (use dining/living/hall instead).
    edges = {
        (a, b)
        for a, b in edges
        if not (
            (a == "Kitchen" and is_bedroom(b))
            or (b == "Kitchen" and is_bedroom(a))
        )
    }

    adj: dict[str, list[str]] = {n: [] for n in names}
    for a, b in edges:
        if a not in adj or b not in adj:
            continue
        adj[a].append(b)
        adj[b].append(a)

    tree: list[list[str]] = []
    seen = {living}
    q = deque([living])
    while q:
        cur = q.popleft()
        for nxt in sorted(adj.get(cur, [])):
            if nxt in seen:
                continue
            seen.add(nxt)
            tree.append([cur, nxt])
            q.append(nxt)

    for name in names:
        if name not in seen and is_bedroom(name):
            tree.append([hall_or_living, name])
            seen.add(name)

    return {
        "root": living,
        "edges": [list(e) for e in sorted(edges)],
        "tree": tree,
    }
