"""Tests for the retrieval regression gate.

The gate is what turns "we measured it once" into "it cannot silently get
worse", so the comparison rules are tested with hand-written numbers.
"""

from __future__ import annotations

from clausewright.eval.retrieval import baseline_regressions

BASELINE = {
    "configs": {
        "bm25": {"recall_at_5": 0.75, "mrr": 0.60},
        "hybrid": {"recall_at_5": 0.60, "mrr": 0.45},
    }
}


def test_no_regression_when_metrics_hold() -> None:
    report = {
        "configs": {
            "bm25": {"recall_at_5": 0.75, "mrr": 0.60},
            "hybrid": {"recall_at_5": 0.60, "mrr": 0.45},
        }
    }
    assert baseline_regressions(report, BASELINE, tolerance=0.0) == []


def test_improvement_is_not_a_regression() -> None:
    report = {
        "configs": {
            "bm25": {"recall_at_5": 0.90, "mrr": 0.80},
            "hybrid": {"recall_at_5": 0.60, "mrr": 0.45},
        }
    }
    assert baseline_regressions(report, BASELINE, tolerance=0.0) == []


def test_drop_beyond_tolerance_is_reported() -> None:
    report = {
        "configs": {
            "bm25": {"recall_at_5": 0.70, "mrr": 0.60},
            "hybrid": {"recall_at_5": 0.60, "mrr": 0.45},
        }
    }
    drops = baseline_regressions(report, BASELINE, tolerance=0.02)
    assert len(drops) == 1
    assert "bm25.recall_at_5" in drops[0]
    assert "0.700" in drops[0]


def test_drop_within_tolerance_passes() -> None:
    report = {
        "configs": {
            "bm25": {"recall_at_5": 0.74, "mrr": 0.60},
            "hybrid": {"recall_at_5": 0.60, "mrr": 0.45},
        }
    }
    assert baseline_regressions(report, BASELINE, tolerance=0.02) == []


def test_missing_config_is_a_regression() -> None:
    report = {"configs": {"bm25": {"recall_at_5": 0.75, "mrr": 0.60}}}
    drops = baseline_regressions(report, BASELINE, tolerance=0.0)
    assert any("hybrid: missing" in d for d in drops)


def test_non_numeric_baseline_fields_are_ignored() -> None:
    baseline = {"configs": {"bm25": {"recall_at_5": 0.75, "note": "hand-written"}}}
    report = {"configs": {"bm25": {"recall_at_5": 0.75}}}
    assert baseline_regressions(report, baseline, tolerance=0.0) == []
