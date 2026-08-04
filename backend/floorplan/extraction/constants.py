"""Shared constants for LLM floor plan image extraction."""

from __future__ import annotations

CANVAS_SIZE: int = 1024
EXTRACTION_MODEL: str = "gpt-5.6-sol"
LLM_TIMEOUT_SECONDS: float = 120.0
MORPH_KERNEL_SIZE: int = 3
MORPH_ITERATIONS: int = 1
THRESHOLD_VALUE: int = 200
