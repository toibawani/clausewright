"""Tests for clausewright.eval.metrics.

Every case here is hand-computed.  Exact values, no randomness — a metrics bug
means fake numbers, so these pin down the definitions.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    abstention_scores,
    first_hit_rank,
    mean,
    mrr_score,
    overlap_ratio,
    percentile,
    recall_at_k,
)


class TestOverlapRatio:
    def test_full_containment_is_one(self) -> None:
        gold = "thirty days prior written notice"
        assert overlap_ratio(gold, "requires " + gold + " to terminate") == 1.0

    def test_partial_containment(self) -> None:
        # 3 of 4 gold tokens present -> 0.75
        assert overlap_ratio("a b c d", "a b x c") == pytest.approx(0.75)

    def test_repetition_counts(self) -> None:
        # gold repeats "notice"; chunk has it once -> 1 of 3 matched
        assert overlap_ratio("notice and notice", "notice") == pytest.approx(1 / 3)

    def test_empty_gold_is_zero(self) -> None:
        assert overlap_ratio("", "anything") == 0.0

    def test_threshold_default_is_half(self) -> None:
        assert SPAN_OVERLAP_THRESHOLD == 0.5


class TestRetrievalMetrics:
    GOLD: ClassVar[list[str]] = ["Governance Section Text Here"]
    RANKED: ClassVar[list[str]] = [
        "totally unrelated indemnity words",
        "more words about Governance Section Text Here inside a clause",
    ]

    def test_first_hit_rank(self) -> None:
        assert first_hit_rank(self.GOLD, self.RANKED) == 2
        assert first_hit_rank(self.GOLD, ["none", "nothing"]) is None

    def test_mrr(self) -> None:
        assert mrr_score(self.GOLD, self.RANKED) == pytest.approx(0.5)
        assert mrr_score(self.GOLD, ["none"]) == 0.0

    def test_recall_at_k_counts_spans(self) -> None:
        gold = ["alpha beta gamma", "delta epsilon"]
        ranked = ["delta epsilon found", "alpha beta gamma found"]
        assert recall_at_k(gold, ranked, 1) == pytest.approx(0.5)
        assert recall_at_k(gold, ranked, 2) == pytest.approx(1.0)

    def test_recall_at_k_rejects(self) -> None:
        with pytest.raises(ValueError, match="no gold spans"):
            recall_at_k([], ["text"], 5)
        with pytest.raises(ValueError, match="k must be"):
            recall_at_k(["a"], ["a"], 0)


class TestAbstentionScores:
    def test_hand_computed_case(self) -> None:
        # 3 absent (T), 3 answerable (F); model abstained on items 0,1,3
        truth = [True, True, True, False, False, False]
        predicted = [True, True, False, True, False, False]
        out = abstention_scores(true_absent=truth, predicted_abstain=predicted)
        assert out["precision"] == pytest.approx(2 / 3)   # 2 correct of 3 abstentions
        assert out["recall"] == pytest.approx(2 / 3)      # 2 of 3 absent caught
        assert out["false_abstention_rate"] == pytest.approx(1 / 3)

    def test_perfect(self) -> None:
        truth = [True, True, False, False]
        out = abstention_scores(true_absent=truth, predicted_abstain=truth)
        assert out == {"precision": 1.0, "recall": 1.0, "false_abstention_rate": 0.0}

    def test_never_abstains(self) -> None:
        out = abstention_scores(true_absent=[True, False], predicted_abstain=[False, False])
        assert out == {"precision": 0.0, "recall": 0.0, "false_abstention_rate": 0.0}

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            abstention_scores(true_absent=[True], predicted_abstain=[True, False])


class TestAggregates:
    def test_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert math.isnan(mean([]))

    def test_percentile_nearest_rank(self) -> None:
        vals = [5.0, 1.0, 3.0, 2.0, 4.0]  # ordered: 1,2,3,4,5
        assert percentile(vals, 50) == 3.0
        assert percentile(vals, 95) == 5.0
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)
