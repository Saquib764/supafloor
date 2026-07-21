"""Topology package."""

from .graph import (
    adjacency_signature,
    adjacency_via_doors,
    build_adjacency,
    constraint_graph_from_plan,
    exterior_touching_rooms,
    geometry_hash,
    reachable,
    room_metrics,
    room_ratio_signature,
    topology_hash,
)

__all__ = [
    "adjacency_signature",
    "adjacency_via_doors",
    "build_adjacency",
    "constraint_graph_from_plan",
    "exterior_touching_rooms",
    "geometry_hash",
    "reachable",
    "room_metrics",
    "room_ratio_signature",
    "topology_hash",
]
