"""Persist canonical plan JSON under dev/output/plans/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from floorplan.requirements.plan_io import plan_document_id

# Repo: backend/floorplan/floorplan/dataset/store.py → ../../../.. = repo root
_OUTPUT_ROOT = Path(__file__).resolve().parents[4] / "dev" / "output"
_DEFAULT_PLANS_DIR = _OUTPUT_ROOT / "plans"
_DEFAULT_SCORES_DIR = _OUTPUT_ROOT / "score"


def plans_dir(root: Path | None = None) -> Path:
    path = root or _DEFAULT_PLANS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def scores_dir(root: Path | None = None) -> Path:
    path = root or _DEFAULT_SCORES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_document(document: dict[str, Any], root: Path | None = None) -> Path:
    plan_id = plan_document_id(document)
    if not plan_id:
        raise ValueError("Plan document missing summary.id")
    path = plans_dir(root) / f"{plan_id}.json"
    path.write_text(json.dumps(document, indent=2))
    _write_index(root)
    return path


def load_document(plan_id: str, root: Path | None = None) -> dict[str, Any]:
    path = plans_dir(root) / f"{plan_id}.json"
    if not path.exists():
        raise FileNotFoundError(plan_id)
    return json.loads(path.read_text())


def delete_document(plan_id: str, root: Path | None = None) -> None:
    path = plans_dir(root) / f"{plan_id}.json"
    if not path.exists():
        raise FileNotFoundError(plan_id)
    path.unlink()
    for name in (f"{plan_id}.png", f"{plan_id}.score.md", f"{plan_id}.score.json"):
        artifact = scores_dir() / name
        if artifact.exists():
            artifact.unlink()
    _write_index(root)


def list_documents(root: Path | None = None) -> list[dict[str, Any]]:
    directory = plans_dir(root)
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name == "index.json":
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summary = doc.get("summary") or {}
        requirements = doc.get("requirements") or {}
        items.append(
            {
                "id": summary.get("id") or path.stem,
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
