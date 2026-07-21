"""Evaluator package — rules, furniture, scoring."""

from .furniture import score_furniture
from .rules import evaluate_rules, hard_failures, is_valid
from .scoring import (
    DEFAULT_WEIGHTS,
    build_summary,
    format_score_report,
    score_label,
    score_plan,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "build_summary",
    "evaluate_rules",
    "format_score_report",
    "hard_failures",
    "is_valid",
    "score_furniture",
    "score_label",
    "score_plan",
]
