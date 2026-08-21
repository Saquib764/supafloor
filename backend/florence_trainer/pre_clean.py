"""Import shim for the pre-clean/ package (hyphenated folder name)."""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "pre-clean"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from clean import pre_clean

__all__ = ["pre_clean"]
