"""Export floor plans as Unsloth / Gemma-4 LoRA SFT rows.

Unsloth Studio accepts local JSONL. For Gemma 4 IT LoRA, use ChatML-style
``messages`` with roles ``system`` / ``user`` / ``model`` (not ``assistant``).

Each row also carries Alpaca-style columns so Studio can map:
  instruction ← prompt, input ← schema, output ← output

Builds on ``build_training_example`` (prompt / schema / output), then wraps
those fields into Unsloth-ready chat or Alpaca text columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

from floorplan.dataset.layout_prompt import build_training_example
from floorplan.requirements.plan_io import document_to_plan
from floorplan.schemas.models import FloorPlan

FormatName = Literal["messages", "alpaca", "both"]

SYSTEM_PROMPT = (
    "You generate residential floor-plan room polygons as JSON. "
    "Given a layout description and a schema with room target areas plus the "
    "boundary polygon (feet), return only a JSON array of rooms. Each room must "
    "include name, actual_area_sqft, polygon (closed {x,y} ring), and optional "
    "room_type and zone. Fit rooms inside layout_polygon and respect adjacencies "
    "described in the prompt."
)

_SKIP_NAMES = frozenset({"index.json", "manifest.json", "scores_index.json"})


def _user_content(prompt: str, schema: str) -> str:
    return f"Layout prompt:\n{prompt}\n\nSchema:\n{schema}"


def build_sft_parts(plan: FloorPlan) -> dict[str, str] | None:
    """Return prompt / schema / output strings, or None if the plan has no rooms."""
    example = build_training_example(plan)
    if example is None:
        return None
    prompt = example["input"]["prompt"]
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "schema": json.dumps(example["input"]["schema"], indent=2),
        "output": json.dumps(example["output"], indent=2),
    }


def build_unsloth_row(
    plan: FloorPlan,
    *,
    plan_id: str | None = None,
    fmt: FormatName = "both",
) -> dict[str, Any] | None:
    """One Unsloth-ready training row from a FloorPlan."""
    parts = build_sft_parts(plan)
    if parts is None:
        return None

    row: dict[str, Any] = {
        "id": plan_id or plan.metadata.get("plan_id") or plan.id,
        "prompt": parts["prompt"],
        "schema": parts["schema"],
        "output": parts["output"],
    }

    if fmt in ("alpaca", "both"):
        row["instruction"] = parts["prompt"]
        row["input"] = parts["schema"]

    if fmt in ("messages", "both"):
        row["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(parts["prompt"], parts["schema"])},
            {"role": "model", "content": parts["output"]},
        ]

    return row


def iter_plan_documents(directory: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (plan_id, document) for each saved plan (flat JSON or ``{id}/floorplan.json``)."""
    from floorplan.dataset.store import PLAN_JSON

    for path in sorted(directory.iterdir()):
        if path.is_dir():
            json_path = path / PLAN_JSON
            if not json_path.exists():
                continue
            document = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or "rooms" not in document:
                continue
            yield path.name, document
            continue
        if path.suffix != ".json":
            continue
        if path.name in _SKIP_NAMES or path.name.endswith(".score.json"):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "rooms" not in document:
            continue
        yield path.stem, document


def export_unsloth_dataset(
    plans_dir: Path,
    out_path: Path,
    *,
    fmt: FormatName = "both",
) -> dict[str, Any]:
    """Convert a folder of canonical plan JSON into Unsloth JSONL.

    Returns a small manifest summary.
    """
    plans_dir = Path(plans_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for stem, document in iter_plan_documents(plans_dir):
        plan = document_to_plan(document)
        row = build_unsloth_row(plan, plan_id=stem, fmt=fmt)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_target": "google/gemma-4-E4B-it (Unsloth LoRA)",
        "format": fmt,
        "chat_roles": ["system", "user", "model"],
        "count": len(rows),
        "skipped": skipped,
        "source": str(plans_dir.resolve()),
        "output": str(out_path.resolve()),
        "fields": {
            "prompt": "Natural-language room locations / adjacencies",
            "schema": "Room target areas + boundary layout_polygon JSON",
            "output": "Room polygon JSON (training target)",
            "messages": "Gemma-4 ChatML with role=model for the response",
            "instruction/input": "Alpaca aliases of prompt/schema for Studio mapping",
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2))
    return summary
