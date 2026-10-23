"""Tests for clausewright.eval.calibration."""

from __future__ import annotations

import math

import pytest

from clausewright.eval.calibration import (
    CalibrationRow,
    _format_val,
    print_calibration_table,
)


class TestCalibrationRow:
    def test_dataclass_fields(self) -> None:
        row = CalibrationRow(
            threshold=0.5,
            abstention_precision=0.8,
            abstention_recall=0.9,
            false_abstention_rate=0.05,
            answer_coverage=0.95,
            mean_recall_at_5=float("nan"),
            n_queries=100,
        )
        assert row.threshold == 0.5
        assert row.abstention_precision == 0.8
        assert row.abstention_recall == 0.9
        assert row.false_abstention_rate == 0.05
        assert row.answer_coverage == 0.95
        assert math.isnan(row.mean_recall_at_5)
        assert row.n_queries == 100


class TestFormatVal:
    def test_none(self) -> None:
        assert _format_val(None) == " None"

    def test_nan(self) -> None:
        assert _format_val(float("nan")) == "  nan"

    def test_float(self) -> None:
        assert _format_val(0.1234) == "0.123"


class TestPrintCalibrationTable:
    def test_print_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [
            {
                "threshold": 0.5,
                "abstention_precision": 0.8,
                "abstention_recall": 0.9,
                "false_abstention_rate": 0.05,
                "answer_coverage": 0.95,
                "n_queries": 100,
            },
            {
                "threshold": None,
                "abstention_precision": 0.0,
                "abstention_recall": 0.0,
                "false_abstention_rate": 0.0,
                "answer_coverage": 1.0,
                "n_queries": 100,
            },
        ]
        print_calibration_table(rows)
        captured = capsys.readouterr().out
        assert "threshold" in captured
        assert "0.500" in captured
        assert "None" in captured
