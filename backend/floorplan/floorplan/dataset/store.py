"""Persist generation documents as JSON (parameters + generated)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Repo: backend/floorplan/floorplan/dataset/store.py → ../../../.. = repo root
_DEFAULT_DIR = Path(__file__).resolve().parents[4] / "dev" / "output" / "plans"


def plans_dir(root: Path | None = None) -> Path:
    path = root or _DEFAULT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_document(document: dict[str, Any], root: Path | None = None) -> Path:
    path = plans_dir(root) / f"{document['id']}.json"
    payload = {
        "id": document["id"],
        "label": document.get("label"),
        "parameters": document["parameters"],
        "generated": document["generated"],
    }
    path.write_text(json.dumps(payload, indent=2))
    _write_index(root)
    return path


def load_document(plan_id: str, root: Path | None = None) -> dict[str, Any]:
    path = plans_dir(root) / f"{plan_id}.json"
    if not path.exists():
        raise FileNotFoundError(plan_id)
    return json.loads(path.read_text())


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
        params = doc.get("parameters") or {}
        generated = doc.get("generated") or {}
        spaces = generated.get("spaces") or []
        items.append(
            {
                "id": doc.get("id") or path.stem,
                "label": doc.get("label") or f"{params.get('total_rooms', '?')}-room",
                "total_rooms": params.get("total_rooms"),
                "seed": params.get("seed"),
                "space_count": sum(1 for s in spaces if s.get("polygon")),
                "path": path.name,
            }
        )
    return items


def _write_index(root: Path | None = None) -> None:
    directory = plans_dir(root)
    (directory / "index.json").write_text(json.dumps(list_documents(root), indent=2))
