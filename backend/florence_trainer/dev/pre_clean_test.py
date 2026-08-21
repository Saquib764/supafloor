#!/usr/bin/env python3
"""Run pre-clean on every image in test/ and write results to dev/output/pre-clean/.

Usage (from backend/florence_trainer):

    uv run python dev/pre_clean_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "pre-clean") not in sys.path:
    sys.path.insert(0, str(_ROOT / "pre-clean"))

from clean import pre_clean

_TEST_DIR = _ROOT / "test"
_OUT_DIR = Path(__file__).resolve().parent / "output" / "pre-clean"
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    paths = sorted(p for p in _TEST_DIR.iterdir() if p.suffix.lower() in _EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {_TEST_DIR}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = Image.open(path)
        cleaned = pre_clean(image)
        out_path = _OUT_DIR / f"{path.stem}.png"
        cleaned.save(out_path)
        print(f"{path.name}: {image.size} → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
