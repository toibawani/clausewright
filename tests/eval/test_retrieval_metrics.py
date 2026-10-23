"""Tests for retrieval evaluation metrics and span overlap.

Every case here is hand-computed. Exact values, no randomness — a metrics bug
means fake numbers, so these pin down the definitions.
"""

from __future__ import annotations

import math

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
    span_hit,
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


class TestSpanHit:
    def test_meets_threshold(self) -> None:
        assert span_hit("a b c d", "a b c x", threshold=0.75)

    def test_misses_threshold(self) -> None:
        assert not span_hit("a b c d", "a b x y", threshold=0.75)


class TestFirstHitRank:
    def test_hit_at_rank_one(self) -> None:
        gold = ["notice period is 30 days"]
        chunks = [
            "the notice period is 30 days prior",
            "unrelated liability text",
        ]
        assert first_hit_rank(gold, chunks) == 1

    def test_hit_at_rank_two(self) -> None:
        gold = ["governing law is new york"]
        chunks = [
            "payment terms net 30",
            "this agreement is governed by the laws of new york",
        ]
        assert first_hit_rank(gold, chunks) == 2

    def test_no_hit_returns_none(self) -> None:
        gold = ["non-compete for 2 years"]
        chunks = ["clause a", "clause b"]
        assert first_hit_rank(gold, chunks) is None


class TestRecallAtK:
    def test_single_gold_hit_at_k1(self) -> None:
        gold = ["arbitration in delaware"]
        chunks = ["disputes shall be resolved by arbitration in delaware", "other"]
        assert recall_at_k(gold, chunks, k=1) == 1.0

    def test_single_gold_miss_at_k1_hit_at_k2(self) -> None:
        gold = ["arbitration in delaware"]
        chunks = ["other text", "disputes shall be resolved by arbitration in delaware"]
        assert recall_at_k(gold, chunks, k=1) == 0.0
        assert recall_at_k(gold, chunks, k=2) == 1.0

    def test_multiple_golds_partial_recall(self) -> None:
        gold = ["alpha beta gamma", "delta epsilon zeta"]
        chunks = ["alpha beta gamma omega", "unrelated text"]
        assert recall_at_k(gold, chunks, k=1) == 0.5
        assert recall_at_k(gold, chunks, k=2) == 0.5

    def test_empty_gold_raises(self) -> None:
        with pytest.raises(ValueError, match="no gold spans"):
            recall_at_k([], ["text"], k=1)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(["gold"], ["text"], k=0)


class TestMRRScore:
    def test_rank_one_mrr_is_one(self) -> None:
        gold = ["indemnification obligation"]
        chunks = ["indemnification obligation covers losses"]
        assert mrr_score(gold, chunks) == 1.0

    def test_rank_two_mrr_is_half(self) -> None:
        gold = ["indemnification obligation"]
        chunks = ["other", "indemnification obligation covers losses"]
        assert mrr_score(gold, chunks) == 0.5

    def test_rank_four_mrr_is_quarter(self) -> None:
        gold = ["indemnification obligation"]
        chunks = ["1", "2", "3", "indemnification obligation covers losses"]
        assert mrr_score(gold, chunks) == 0.25

    def test_no_hit_mrr_is_zero(self) -> None:
        gold = ["indemnification obligation"]
        chunks = ["1", "2", "3"]
        assert mrr_score(gold, chunks) == 0.0


class TestAbstentionScores:
    def test_perfect_abstention(self) -> None:
        # 2 absent (both abstained), 2 answerable (neither abstained)
        true_absent = [True, True, False, False]
        pred_abstain = [True, True, False, False]
        scores = abstention_scores(true_absent=true_absent, predicted_abstain=pred_abstain)
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["false_abstention_rate"] == 0.0

    def test_false_abstention(self) -> None:
        # 1 absent (abstained), 1 answerable (abstained wrongly)
        true_absent = [True, False]
        pred_abstain = [True, True]
        scores = abstention_scores(true_absent=true_absent, predicted_abstain=pred_abstain)
        assert scores["precision"] == 0.5
        assert scores["recall"] == 1.0
        assert scores["false_abstention_rate"] == 1.0


class TestStatisticalHelpers:
    def test_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0
        assert math.isnan(mean([]))

    def test_percentile(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert percentile(data, 50.0) == 30.0
        assert percentile(data, 90.0) == 50.0
