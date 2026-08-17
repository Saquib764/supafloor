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
from .paths import (
    circ_dead_end_ids,
    entry_start_ids,
    passageway_rooms,
    path_min_width,
    shortest_hop,
    shortest_path,
)

__all__ = [
    "adjacency_signature",
    "adjacency_via_doors",
    "build_adjacency",
    "circ_dead_end_ids",
    "constraint_graph_from_plan",
    "entry_start_ids",
    "exterior_touching_rooms",
    "geometry_hash",
    "passageway_rooms",
    "path_min_width",
    "reachable",
    "room_metrics",
    "room_ratio_signature",
    "shortest_hop",
    "shortest_path",
    "topology_hash",
]
