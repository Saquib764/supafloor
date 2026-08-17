"""Persist canonical plan JSON under backend/floorplan/dev/output/plans/{id}/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from requirements.plan_io import plan_document_id

_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "dev" / "output"
_DEFAULT_PLANS_DIR = _OUTPUT_ROOT / "plans"
_DEFAULT_SCORES_DIR = _OUTPUT_ROOT / "score"

PLAN_JSON = "floorplan.json"
PLAN_PNG = "plan.png"
SCORE_JSON = "score.json"
SCORE_MD = "score.md"


def plans_dir(root: Path | None = None) -> Path:
    path = root or _DEFAULT_PLANS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def scores_dir(root: Path | None = None) -> Path:
    path = root or _DEFAULT_SCORES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def plan_dir(plan_id: str, root: Path | None = None) -> Path:
    return plans_dir(root) / plan_id


def plan_json_path(plan_id: str, root: Path | None = None) -> Path:
    return plan_dir(plan_id, root) / PLAN_JSON


def save_document(document: dict[str, Any], root: Path | None = None) -> Path:
    plan_id = plan_document_id(document)
    if not plan_id:
        raise ValueError("Plan document missing summary.id")
    directory = plan_dir(plan_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PLAN_JSON
    path.write_text(json.dumps(document, indent=2))
    _write_index(root)
    return path


def load_document(plan_id: str, root: Path | None = None) -> dict[str, Any]:
    path = plan_json_path(plan_id, root)
    if not path.exists():
        raise FileNotFoundError(plan_id)
    return json.loads(path.read_text())


def delete_document(plan_id: str, root: Path | None = None) -> None:
    directory = plan_dir(plan_id, root)
    if not directory.is_dir():
        raise FileNotFoundError(plan_id)
    shutil.rmtree(directory)
    _write_index(root)


def list_documents(root: Path | None = None) -> list[dict[str, Any]]:
    directory = plans_dir(root)
    items: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        json_path = path / PLAN_JSON
        if not json_path.exists():
            continue
        try:
            doc = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        summary = doc.get("summary") or {}
        requirements = doc.get("requirements") or {}
        items.append(
            {
                "id": summary.get("id") or path.name,
                "label": summary.get("name") or f"{requirements.get('bedroom_count', '?')}-room",
                "bedroom_count": requirements.get("bedroom_count"),
                "room_count": summary.get("room_count") or len(doc.get("rooms") or []),
                "path": path.name,
            }
        )
    return items


def _write_index(root: Path | None = None) -> None:
    directory = plans_dir(root)
    (directory / "index.json").write_text(json.dumps(list_documents(root), indent=2))
