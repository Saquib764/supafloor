"""Browse image-to-floorplan/basic_5000 training pairs."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
CORPUS_ROOT = REPO_ROOT / "image-to-floorplan" / "basic_5000"
_STEM_RE = re.compile(r"^[0-9a-f]+_\d+$")


def _validate_stem(stem: str) -> None:
    if not _STEM_RE.fullmatch(stem):
        raise ValueError(f"Invalid dataset item id: {stem}")


@lru_cache(maxsize=1)
def _all_stems() -> tuple[str, ...]:
    output_dir = CORPUS_ROOT / "output"
    if not output_dir.is_dir():
        return ()
    return tuple(sorted(path.stem for path in output_dir.glob("*.json")))


def list_items(offset: int, limit: int) -> dict:
    stems = _all_stems()
    total = len(stems)
    if offset < 0:
        offset = 0
    if limit < 1:
        limit = 1
    page = list(stems[offset : offset + limit])
    return {
        "items": [{"stem": stem} for stem in page],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
    }


def load_json(stem: str) -> dict:
    _validate_stem(stem)
    path = CORPUS_ROOT / "output" / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(stem)
    return json.loads(path.read_text())


def image_path(stem: str) -> Path:
    _validate_stem(stem)
    path = CORPUS_ROOT / "input" / f"{stem}.jpg"
    if not path.is_file():
        raise FileNotFoundError(stem)
    return path
