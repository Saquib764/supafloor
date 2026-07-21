"""Human-readable scoring report tests."""

from floorplan.evaluator import format_score_report, score_label, score_plan
from floorplan.templates import instantiate_template


def test_score_label_bands():
    assert score_label(9.5) == "Excellent"
    assert score_label(8.0) == "Good"
    assert score_label(6.5) == "Fair"
    assert score_label(4.5) == "Weak"
    assert score_label(2.0) == "Poor"


def test_score_plan_includes_summary_and_report():
    plan = instantiate_template("studio")
    scores = score_plan(plan)
    assert scores.summary
    assert "/10" in scores.summary
    assert "Category scores" in scores.report
    md = format_score_report(scores, plan=plan)
    assert plan.name in md
    assert "Hard" in md or "hard" in md.lower()
