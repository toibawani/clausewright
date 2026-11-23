"""Hand-computed tests for the paired bootstrap.

These pin the definition, not the exact bootstrap noise: cases where every
resample must give the same answer, cases where the interval must straddle
zero, and the reproducibility guarantee.
"""

from __future__ import annotations

import math

import pytest

from clausewright.eval.stats import PairedDifference, mean_or_nan, paired_bootstrap


class TestConstantDifference:
    def test_identical_gap_gives_degenerate_interval(self) -> None:
        # Two contracts, every query improves by exactly 0.5. No resample can
        # produce anything but 0.5, so the interval must collapse.
        pairs = [
            ("c1", 1.0, 0.5),
            ("c1", 0.5, 0.0),
            ("c2", 0.8, 0.3),
        ]
        result = paired_bootstrap(
            metric="recall_at_5",
            left_label="struct",
            right_label="window",
            pairs=pairs,
            n_resamples=200,
        )
        assert result.point == pytest.approx(0.5)
        assert result.low == pytest.approx(0.5)
        assert result.high == pytest.approx(0.5)
        assert result.excludes_zero

    def test_counts_are_reported(self) -> None:
        pairs = [("c1", 1.0, 0.0), ("c1", 0.0, 0.0), ("c2", 1.0, 1.0)]
        result = paired_bootstrap(
            metric="mrr", left_label="a", right_label="b", pairs=pairs, n_resamples=50
        )
        assert result.n_contracts == 2
        assert result.n_pairs == 3


class TestIntervalStraddlesZero:
    def test_opposite_contracts_include_zero(self) -> None:
        # Contract c1 gains 1.0, c2 loses 1.0. Resamples of two contracts can
        # be all-c1 or all-c2, so the interval must include zero.
        pairs = [
            ("c1", 1.0, 0.0),
            ("c2", 0.0, 1.0),
        ]
        result = paired_bootstrap(
            metric="recall_at_1",
            left_label="left",
            right_label="right",
            pairs=pairs,
            n_resamples=500,
        )
        assert result.point == pytest.approx(0.0)
        assert result.low <= 0.0 <= result.high
        assert not result.excludes_zero


class TestReproducibility:
    def test_same_seed_same_interval(self) -> None:
        pairs = [
            ("c1", 1.0, 0.2),
            ("c1", 0.4, 0.1),
            ("c2", 0.0, 0.6),
            ("c3", 0.9, 0.9),
        ]
        kwargs: dict[str, object] = {
            "metric": "mrr",
            "left_label": "a",
            "right_label": "b",
            "pairs": pairs,
            "n_resamples": 300,
        }
        first = paired_bootstrap(**kwargs, seed=7)
        second = paired_bootstrap(**kwargs, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_different_seed_may_move_the_edges(self) -> None:
        pairs = [("c1", 1.0, 0.0), ("c2", 0.0, 1.0), ("c3", 0.5, 0.5)]
        a = paired_bootstrap(
            metric="m", left_label="a", right_label="b", pairs=pairs, n_resamples=200, seed=1
        )
        b = paired_bootstrap(
            metric="m", left_label="a", right_label="b", pairs=pairs, n_resamples=200, seed=2
        )
        # Both are valid intervals; the point estimate is seed-independent.
        assert a.point == b.point
        assert a.low <= a.point <= a.high
        assert b.low <= b.point <= b.high


class TestValidation:
    def test_empty_pairs_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            paired_bootstrap(metric="m", left_label="a", right_label="b", pairs=[])

    def test_bad_resample_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            paired_bootstrap(
                metric="m", left_label="a", right_label="b", pairs=[("c", 1.0, 0.0)], n_resamples=0
            )

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            paired_bootstrap(
                metric="m", left_label="a", right_label="b", pairs=[("c", 1.0, 0.0)], alpha=0.0
            )


class TestSerialisation:
    def test_as_dict_rounds_to_four_places(self) -> None:
        result = PairedDifference(
            metric="m",
            left_label="a",
            right_label="b",
            n_contracts=2,
            n_pairs=3,
            point=0.123456,
            low=-0.5,
            high=0.7,
        )
        payload = result.as_dict()
        assert payload["point"] == 0.1235
        assert payload["ci_low"] == -0.5
        assert payload["excludes_zero"] is False


def test_mean_or_nan() -> None:
    assert mean_or_nan([1.0, 2.0]) == 1.5
    assert math.isnan(mean_or_nan([]))
