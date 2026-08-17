"""Build connection graph and circulation tree from space names."""

from __future__ import annotations

from collections import deque
from typing import Any

from requirements.areas import name_root
from requirements.program import bedroom_parent, is_bedroom, is_bedroom_name


def living_name(names: list[str]) -> str:
    if "Living" in names:
        return "Living"
    return "Living/Sleeping"


def hall_or_living(names: list[str], living: str | None = None) -> str:
    if "Hallway" in names:
        return "Hallway"
    return living or living_name(names)


def add_edge(edges: set[tuple[str, str]], a: str, b: str) -> None:
    if a and b and a != b:
        edges.add(tuple(sorted((a, b))))


def normalize_edge(a: str, b: str) -> tuple[str, str] | None:
    if not a or not b or a == b:
        return None
    return tuple(sorted((a, b)))


def is_kitchen_bedroom_edge(a: str, b: str) -> bool:
    return (a == "Kitchen" and is_bedroom(b)) or (b == "Kitchen" and is_bedroom(a))


def reachable_from(root: str, edges: set[tuple[str, str]], names: list[str]) -> set[str]:
    adj: dict[str, list[str]] = {n: [] for n in names}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    seen = {root}
    q = deque([root])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def derive_tree(living: str, edges: set[tuple[str, str]], names: list[str]) -> list[list[str]]:
    adj: dict[str, list[str]] = {n: [] for n in names}
    for a, b in edges:
        if a not in adj or b not in adj:
            continue
        adj[a].append(b)
        adj[b].append(a)

    hall = hall_or_living(names, living)
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
            tree.append([hall, name])
            seen.add(name)

    return tree


def graph_payload(root: str, edges: set[tuple[str, str]], names: list[str]) -> dict[str, Any]:
    return {
        "root": root,
        "edges": [list(e) for e in sorted(edges)],
        "tree": derive_tree(root, edges, names),
    }


def build_connection_graph(names: list[str]) -> dict[str, Any]:
    """Adjacency edges + spanning tree rooted at the living room."""
    living = living_name(names) if any(n in names for n in ("Living", "Living/Sleeping")) else names[0]
    edges: set[tuple[str, str]] = set()
    name_set = set(names)

    if "Hallway" in name_set:
        add_edge(edges, living, "Hallway")
    if "Dining" in name_set:
        add_edge(edges, living, "Dining")
        add_edge(edges, "Dining", "Kitchen")
    elif "Kitchen" in name_set:
        add_edge(edges, living, "Kitchen")

    hub = hall_or_living(names, living)
    for name in names:
        if is_bedroom(name):
            add_edge(edges, hub, name)

    for name in names:
        parent = bedroom_parent(name)
        if parent and parent in name_set:
            add_edge(edges, parent, name)

    if "Kitchen" in name_set and "Laundry" in name_set:
        add_edge(edges, "Kitchen", "Laundry")

    if "Bathroom" in name_set:
        add_edge(edges, hub, "Bathroom")

    if "Balcony" in name_set:
        balcony_hosts = ["Living", "Living/Sleeping", "Dining"]
        balcony_hosts.extend(n for n in names if is_bedroom_name(n))
        for host in balcony_hosts:
            if host in name_set:
                add_edge(edges, host, "Balcony")
                break

    if "Garage" in name_set:
        add_edge(edges, "Garage", living)
        if "Hallway" in name_set:
            add_edge(edges, "Garage", "Hallway")

    for name in names:
        if name_root(name) == "Room" and "Closet" in name_set and name == "Room 2":
            add_edge(edges, name, "Closet")

    edges = {(a, b) for a, b in edges if not is_kitchen_bedroom_edge(a, b)}
    return graph_payload(living, edges, names)
