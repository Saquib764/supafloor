"""Evaluator package — rules and scoring."""

from .rules import evaluate_rules, hard_failures, is_valid
from .scoring import (
    DEFAULT_WEIGHTS,
    SCORE_LEVELS,
    SCORE_LEVEL_CATEGORY_ADDS,
    SCORE_LEVEL_RULE_ADDS,
    active_categories,
    active_rules,
    build_summary,
    format_score_report,
    overall_for_level,
    rule_active_at_level,
    score_label,
    score_plan,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "SCORE_LEVELS",
    "SCORE_LEVEL_CATEGORY_ADDS",
    "SCORE_LEVEL_RULE_ADDS",
    "active_categories",
    "active_rules",
    "build_summary",
    "evaluate_rules",
    "format_score_report",
    "hard_failures",
    "is_valid",
    "overall_for_level",
    "rule_active_at_level",
    "score_label",
    "score_plan",
]
