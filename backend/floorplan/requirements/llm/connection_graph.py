"""Resolve connection graph via deterministic rules or LLM with validation/repair."""

from __future__ import annotations

from typing import Any

from floorplan.config.pipeline import get_pipeline_config, prompt_id_for, substep_provider
from floorplan.llm._context import (
    ensuite_pairs_table,
    format_user_metadata,
    spaces_table,
    target_areas_summary,
)
from floorplan.llm.client import openai_model, parse_structured
from floorplan.llm.schemas import ConnectionGraphResponse
from floorplan.prompts import render_prompt
from floorplan.requirements.graph import (
    add_edge,
    build_connection_graph,
    graph_payload,
    hall_or_living,
    is_kitchen_bedroom_edge,
    living_name,
    reachable_from,
)
from floorplan.requirements.program import bedroom_parent, is_bedroom


def _normalize_edges(
    raw_edges: list[Any],
    names: list[str],
    corrections: list[str],
) -> set[tuple[str, str]]:
    name_set = set(names)
    out: set[tuple[str, str]] = set()
    for pair in raw_edges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            corrections.append(f"dropped invalid edge: {pair!r}")
            continue
        a, b = str(pair[0]), str(pair[1])
        if a not in name_set or b not in name_set:
            corrections.append(f"dropped unknown endpoint: {a}–{b}")
            continue
        if a == b:
            continue
        out.add(tuple(sorted((a, b))))
    if not out and len(names) > 1:
        raise ValueError("LLM returned no valid edges")
    return out


def _remove_kitchen_bedroom_edges(
    edges: set[tuple[str, str]],
    corrections: list[str],
) -> set[tuple[str, str]]:
    kept: set[tuple[str, str]] = set()
    for a, b in edges:
        if is_kitchen_bedroom_edge(a, b):
            corrections.append(f"removed kitchen-bedroom: {a}–{b}")
        else:
            kept.add((a, b))
    return kept


def _repair_ensuite_edges(
    edges: set[tuple[str, str]],
    names: list[str],
    corrections: list[str],
) -> set[tuple[str, str]]:
    out = set(edges)
    name_set = set(names)
    for name in names:
        parent = bedroom_parent(name)
        if parent and parent in name_set:
            edge = tuple(sorted((parent, name)))
            if edge not in out:
                add_edge(out, parent, name)
                corrections.append(f"added ensuite→parent: {parent}–{name}")
    return out


def _repair_connectivity(
    edges: set[tuple[str, str]],
    names: list[str],
    living: str,
    corrections: list[str],
) -> set[tuple[str, str]]:
    out = set(edges)
    det_edges = {tuple(sorted(e)) for e in build_connection_graph(names)["edges"]}
    hub = hall_or_living(names, living)
    reachable = reachable_from(living, out, names)

    for name in names:
        if name in reachable:
            continue
        candidates = sorted(
            e for e in det_edges if name in e and (e[0] in reachable or e[1] in reachable)
        )
        if candidates:
            edge = candidates[0]
            out.add(edge)
            corrections.append(f"added connectivity (from template): {edge[0]}–{edge[1]}")
            reachable = reachable_from(living, out, names)
            continue
        if is_bedroom(name):
            add_edge(out, hub, name)
            corrections.append(f"added bedroom hub: {hub}–{name}")
            reachable = reachable_from(living, out, names)

    return out


def _finalize_graph(edges: set[tuple[str, str]], names: list[str]) -> dict[str, Any]:
    living = living_name(names) if any(n in names for n in ("Living", "Living/Sleeping")) else names[0]
    reachable = reachable_from(living, edges, names)
    missing = set(names) - reachable
    if missing:
        raise ValueError(f"graph still disconnected: unreachable={sorted(missing)}")
    return graph_payload(living, edges, names)


def _llm_propose_edges(
    names: list[str],
    doc: dict[str, Any],
    *,
    selected_areas: dict[str, float] | None,
    layout_total_area_sqft: float,
    prompt_id: str,
) -> list[Any]:
    req = doc["requirements"]
    sampling = doc["sampling"]
    summary = doc["summary"]
    living = living_name(names) if any(n in names for n in ("Living", "Living/Sleeping")) else names[0]
    system, user = render_prompt(
        prompt_id,
        layout_total_area_sqft=round(float(layout_total_area_sqft), 1),
        living_root=living,
        hall_or_living=hall_or_living(names, living),
        bedroom_count=req.get("bedroom_count"),
        seed=sampling.get("seed", 0),
        plan_id=summary.get("id", ""),
        user_metadata_block=format_user_metadata(req),
        spaces_table=spaces_table(names),
        target_areas_summary=target_areas_summary(selected_areas or {}),
        ensuite_pairs_table=ensuite_pairs_table(names),
    )
    parsed = parse_structured(system, user, ConnectionGraphResponse)
    return parsed.edges


def _process_llm_edges(
    raw_edges: list[Any],
    names: list[str],
) -> tuple[dict[str, Any], list[str], list[list[str]]]:
    corrections: list[str] = []
    edges = _normalize_edges(raw_edges, names, corrections)
    llm_pairs = [list(pair) for pair in sorted(edges)]
    edges = _remove_kitchen_bedroom_edges(edges, corrections)
    edges = _repair_ensuite_edges(edges, names, corrections)
    living = living_name(names) if any(n in names for n in ("Living", "Living/Sleeping")) else names[0]
    edges = _repair_connectivity(edges, names, living, corrections)
    graph = _finalize_graph(edges, names)
    return graph, corrections, llm_pairs


def resolve_connection_graph(
    names: list[str],
    doc: dict[str, Any],
    *,
    selected_areas: dict[str, float] | None = None,
    layout_total_area_sqft: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build or LLM-propose a connection graph with validation and repair."""
    configured = substep_provider("sample_into_document", "connection_graph")
    prompt_id = prompt_id_for("connection_graph")
    model = openai_model()
    cfg = get_pipeline_config()
    fallback = cfg["llm"].get("fallback", "deterministic")

    meta: dict[str, Any] = {
        "prompt_id": prompt_id if configured == "llm" else None,
        "model": model if configured == "llm" else None,
        "source": "deterministic",
        "fallback": False,
    }

    if configured != "llm":
        graph = build_connection_graph(names)
        meta["final_edge_count"] = len(graph["edges"])
        return graph, meta

    area = layout_total_area_sqft
    if area is None:
        area = float((doc.get("summary") or {}).get("target_area_sqft") or 0)

    try:
        raw_edges = _llm_propose_edges(
            names,
            doc,
            selected_areas=selected_areas,
            layout_total_area_sqft=area,
            prompt_id=prompt_id,
        )
        graph, corrections, llm_pairs = _process_llm_edges(raw_edges, names)
        meta.update(
            source="llm",
            llm_edge_count=len(raw_edges),
            final_edge_count=len(graph["edges"]),
            llm_edges=llm_pairs,
            final_edges=graph["edges"],
            corrections=corrections,
        )
        return graph, meta
    except Exception as exc:
        if fallback != "deterministic":
            raise
        meta.update(
            source="deterministic_fallback",
            fallback=True,
            error=str(exc),
        )
        graph = build_connection_graph(names)
        meta["final_edge_count"] = len(graph["edges"])
        return graph, meta
