"""Dataset generator pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from floorplan.evaluator import hard_failures, score_plan
from floorplan.mutations import mutate_pipeline
from floorplan.renderer import render_plan
from floorplan.schemas.enums import MutationOp
from floorplan.schemas.models import DatasetSample, FloorPlan
from floorplan.services.boundary_adapt import adapt_to_boundary, generate_boundary
from floorplan.templates import get_template, instantiate_template, list_templates
from floorplan.topology import (
    adjacency_signature,
    constraint_graph_from_plan,
    geometry_hash,
    room_ratio_signature,
    topology_hash,
)


def generate_sample(
    template_id: str = "compact_2bed",
    *,
    width: float = 10.0,
    depth: float = 8.0,
    boundary_shape: str = "rect",
    seed: int = 0,
    mutate: bool = True,
    include_renders: bool = True,
) -> DatasetSample:
    """Choose Template → Boundary → Fit → Mutate → Validate → Evaluate → Render → Export."""
    template = get_template(template_id)
    plan = instantiate_template(template_id, width=width, depth=depth)
    boundary = generate_boundary(width=width, depth=depth, shape=boundary_shape)
    plan = adapt_to_boundary(plan, boundary)

    history: list[dict[str, Any]] = []
    if mutate:
        plan, history, _ = mutate_pipeline(
            plan,
            ops=template.mutation_rules or list(MutationOp)[:5],
            seed=seed,
            require_valid=True,
        )

    scores = score_plan(plan)
    topology = constraint_graph_from_plan(plan)

    svg = None
    png_b64 = None
    if include_renders:
        rendered = render_plan(plan)
        svg = rendered["svg"]
        png_b64 = rendered["png"]

    return DatasetSample(
        boundary=boundary,
        requirements={
            "template_id": template.id,
            "template_name": template.name,
            "rooms": [r.model_dump() for r in template.rooms],
        },
        topology=topology,
        floor_plan=plan,
        scores=scores,
        rule_breakdown=scores.rules,
        mutation_history=history,
        svg=svg,
        png_b64=png_b64,
        metadata={
            "seed": seed,
            "topology_hash": topology_hash(plan),
            "geometry_hash": geometry_hash(plan),
            "adjacency_signature": adjacency_signature(plan),
            "room_ratio_signature": room_ratio_signature(plan),
            "hard_failures": [r.rule_id for r in hard_failures(scores.rules)],
            "valid": len(hard_failures(scores.rules)) == 0,
        },
    )


def generate_dataset(
    n: int = 10,
    template_ids: Optional[list[str]] = None,
    *,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    include_renders: bool = False,
) -> list[DatasetSample]:
    template_ids = template_ids or [t["id"] for t in list_templates()]
    samples: list[DatasetSample] = []
    seen_hashes: set[str] = set()

    i = 0
    attempts = 0
    while len(samples) < n and attempts < n * 5:
        tid = template_ids[attempts % len(template_ids)]
        w = 8.0 + (attempts % 5)
        d = 7.0 + (attempts % 4)
        shape = "rect" if attempts % 3 else "l"
        sample = generate_sample(
            tid,
            width=w,
            depth=d,
            boundary_shape=shape if tid != "studio" else "rect",
            seed=seed + attempts,
            mutate=True,
            include_renders=include_renders,
        )
        attempts += 1
        ghash = sample.metadata.get("geometry_hash")
        thash = sample.metadata.get("topology_hash")
        key = f"{thash}:{ghash}"
        if key in seen_hashes:
            continue
        # Prefer valid samples but allow some invalid for diversity of failures
        seen_hashes.add(key)
        samples.append(sample)
        i += 1

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            (path / f"{sample.id}.json").write_text(sample.model_dump_json(indent=2))
            if sample.svg:
                (path / f"{sample.id}.svg").write_text(sample.svg)
        index = [
            {
                "id": s.id,
                "template": s.requirements.get("template_id"),
                "overall": s.scores.overall if s.scores else None,
                "valid": s.metadata.get("valid"),
                "topology_hash": s.metadata.get("topology_hash"),
            }
            for s in samples
        ]
        (path / "index.json").write_text(json.dumps(index, indent=2))

    return samples
