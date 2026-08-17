"""Shared prompt context for LLM pipeline substeps."""

from __future__ import annotations

from typing import Any

from requirements.context import _room_type_for_name, _zone_for_name
from requirements.program import bedroom_parent, space_role


def format_user_metadata(req: dict[str, Any]) -> str:
    meta = req.get("user_metadata") or req.get("metadata") or {}
    if not meta:
        return ""
    lines = ["", "  ## Additional user metadata"]
    for key, value in meta.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def spaces_table(names: list[str]) -> str:
    rows = []
    for name in names:
        rows.append(
            f"- {name} | role={space_role(name)} | type={_room_type_for_name(name).value} | zone={_zone_for_name(name).value}"
        )
    return "\n".join(rows)


def target_areas_summary(selected_areas: dict[str, float]) -> str:
    if not selected_areas:
        return "- (none)"
    return "\n".join(f"- {name}: {area:.1f} ft²" for name, area in selected_areas.items())


def connection_edges_table(graph: dict[str, Any]) -> str:
    edges = graph.get("edges") or []
    if not edges:
        return "- (none)"
    return "\n".join(f"- {a} ↔ {b}" for a, b in edges)


def ensuite_pairs_table(names: list[str]) -> str:
    rows = []
    for name in names:
        parent = bedroom_parent(name)
        if parent:
            rows.append(f"- {name} must connect to {parent}")
    return "\n".join(rows) if rows else "- (none)"
