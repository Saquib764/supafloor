"""Dataset generator — parameters → generated → FloorPlan; optional disk save."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from floorplan.dataset.store import delete_document, list_documents, load_document, save_document
from floorplan.evaluator import hard_failures, score_plan
from floorplan.renderer import render_plan
from floorplan.requirements import (
    build_document,
    document_to_plan,
    recreate_document,
)
from floorplan.schemas.models import DatasetSample
from floorplan.topology import (
    constraint_graph_from_plan,
    geometry_hash,
    room_ratio_signature,
    topology_hash,
)


def _sample_from_document(document: dict[str, Any], *, include_renders: bool = True) -> DatasetSample:
    plan = document_to_plan(document)
    scores = score_plan(plan)
    topology = constraint_graph_from_plan(plan)
    failures = hard_failures(scores.rules)

    svg = None
    png_b64 = None
    if include_renders:
        rendered = render_plan(plan)
        svg = rendered["svg"]
        png_b64 = rendered["png"]

    params = document["parameters"]
    return DatasetSample(
        id=document["id"],
        boundary=plan.ensure_boundary(),
        requirements={
            "parameters": params,
            "generated": document["generated"],
            "id": document["id"],
            "label": document.get("label"),
        },
        topology=topology,
        floor_plan=plan,
        scores=scores,
        rule_breakdown=scores.rules,
        mutation_history=[],
        svg=svg,
        png_b64=png_b64,
        metadata={
            "seed": params.get("seed"),
            "total_rooms": params.get("total_rooms"),
            "target_area": params.get("target_area"),
            "topology_hash": topology_hash(plan),
            "geometry_hash": geometry_hash(plan),
            "room_ratio_signature": room_ratio_signature(plan),
            "hard_failures": [r.rule_id for r in failures],
            "valid": len(failures) == 0,
            "plan_id": document["id"],
        },
    )


def generate_sample(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
    mutate: bool = False,
    include_renders: bool = True,
    save: bool = False,
) -> DatasetSample:
    """Sample parameters → run generation → FloorPlan. Optionally persist JSON."""
    del mutate
    document = build_document(total_rooms, target_area=target_area, seed=seed)
    if save:
        save_document(document)
    return _sample_from_document(document, include_renders=include_renders)


def recreate_sample(plan_id: str, *, include_renders: bool = True, save: bool = True) -> DatasetSample:
    """Load saved parameters, re-run generation algorithm, optionally overwrite JSON."""
    document = recreate_document(load_document(plan_id))
    if save:
        save_document(document)
    return _sample_from_document(document, include_renders=include_renders)


def generate_dataset(
    n: int = 10,
    *,
    total_rooms: int | None = None,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    include_renders: bool = False,
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    seen_hashes: set[str] = set()

    attempts = 0
    while len(samples) < n and attempts < n * 5:
        rooms = total_rooms if total_rooms is not None else 1 + (attempts % 4)
        sample = generate_sample(
            rooms,
            target_area=45.0 + (attempts % 7) * 8.0,
            seed=seed + attempts,
            include_renders=include_renders,
            save=False,
        )
        attempts += 1
        key = f"{sample.metadata.get('topology_hash')}:{sample.metadata.get('geometry_hash')}"
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        samples.append(sample)

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            doc = {
                "id": sample.id,
                "label": sample.requirements.get("label"),
                "parameters": sample.requirements.get("parameters"),
                "generated": sample.requirements.get("generated"),
            }
            (path / f"{sample.id}.json").write_text(json.dumps(doc, indent=2))
            if sample.svg:
                (path / f"{sample.id}.svg").write_text(sample.svg)
        index = [
            {
                "id": s.id,
                "total_rooms": (s.requirements.get("parameters") or {}).get("total_rooms"),
                "overall": s.scores.overall if s.scores else None,
                "valid": s.metadata.get("valid"),
                "topology_hash": s.metadata.get("topology_hash"),
            }
            for s in samples
        ]
        (path / "index.json").write_text(json.dumps(index, indent=2))

    return samples


def list_saved_plans() -> list[dict[str, Any]]:
    return list_documents()


def get_saved_plan(plan_id: str) -> dict[str, Any]:
    return load_document(plan_id)


def delete_saved_plan(plan_id: str) -> None:
    delete_document(plan_id)


def load_sample(plan_id: str, *, include_renders: bool = True) -> DatasetSample:
    """Load saved JSON and materialize geometry from generated (no recompute)."""
    return _sample_from_document(load_document(plan_id), include_renders=include_renders)
