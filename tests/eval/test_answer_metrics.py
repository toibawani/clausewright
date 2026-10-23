"""Tests for clausewright.eval.answer_metrics — all pure functions, exact values.

Every assertion uses an exact or near-exact expected value computed by hand to
ensure the metric formulae are correct.  These tests must never hit a model,
the filesystem, or the network.
"""

from __future__ import annotations

import math

import pytest

from clausewright.eval.answer_metrics import (
    abstention_metrics,
    answer_quality_report,
    citation_quote_rate,
    citation_validity_rate,
    faithfulness_rate,
    mean_claims_per_answer,
    qualifier_guard_rate,
)

# ---------------------------------------------------------------------------
# Helpers: stub AnswerResult objects (no model involved)
# ---------------------------------------------------------------------------


class _FakeAnswer:
    status = "answered"

    def __init__(self, n_claims: int = 1) -> None:
        self.claims = [None] * n_claims  # length is all we need


class _FakeAbstention:
    status = "abstained"


# ---------------------------------------------------------------------------
# citation_validity_rate
# ---------------------------------------------------------------------------


class TestCitationValidityRate:
    def test_all_valid(self) -> None:
        rate = citation_validity_rate(n_claims_total=10, n_claims_unknown_id=0)
        assert rate == 1.0

    def test_all_invalid(self) -> None:
        rate = citation_validity_rate(n_claims_total=4, n_claims_unknown_id=4)
        assert rate == 0.0

    def test_partial(self) -> None:
        # 3 of 5 claims had valid ids → rate = 0.6
        rate = citation_validity_rate(n_claims_total=5, n_claims_unknown_id=2)
        assert rate == pytest.approx(0.6, abs=1e-9)

    def test_zero_claims_returns_nan(self) -> None:
        rate = citation_validity_rate(n_claims_total=0, n_claims_unknown_id=0)
        assert math.isnan(rate)

    def test_negative_total_raises(self) -> None:
        with pytest.raises(ValueError):
            citation_validity_rate(n_claims_total=-1, n_claims_unknown_id=0)

    def test_unknown_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError):
            citation_validity_rate(n_claims_total=3, n_claims_unknown_id=5)


# ---------------------------------------------------------------------------
# citation_quote_rate
# ---------------------------------------------------------------------------


class TestCitationQuoteRate:
    def test_all_found(self) -> None:
        assert citation_quote_rate(n_claims_valid_id=8, n_claims_quote_not_found=0) == 1.0

    def test_none_found(self) -> None:
        assert citation_quote_rate(n_claims_valid_id=3, n_claims_quote_not_found=3) == 0.0

    def test_partial(self) -> None:
        # 7 of 10 quotes found → rate = 0.7
        rate = citation_quote_rate(n_claims_valid_id=10, n_claims_quote_not_found=3)
        assert rate == pytest.approx(0.7, abs=1e-9)

    def test_zero_valid_id_returns_nan(self) -> None:
        assert math.isnan(citation_quote_rate(n_claims_valid_id=0, n_claims_quote_not_found=0))

    def test_quote_exceeds_valid_raises(self) -> None:
        with pytest.raises(ValueError):
            citation_quote_rate(n_claims_valid_id=2, n_claims_quote_not_found=3)


# ---------------------------------------------------------------------------
# abstention_metrics
# ---------------------------------------------------------------------------


class TestAbstentionMetrics:
    def test_perfect_classifier(self) -> None:
        results = [_FakeAbstention(), _FakeAnswer(), _FakeAbstention()]
        true_absent = [True, False, True]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["false_abstention_rate"] == 0.0

    def test_never_abstains_on_absent(self) -> None:
        # Model always answers; all absent questions are missed
        results = [_FakeAnswer(), _FakeAnswer(), _FakeAnswer()]
        true_absent = [True, True, False]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["recall"] == 0.0
        assert m["false_abstention_rate"] == 0.0  # never abstained at all

    def test_always_abstains(self) -> None:
        # Model always abstains: abstention recall perfect, FAR = 1.0
        results = [_FakeAbstention(), _FakeAbstention(), _FakeAbstention()]
        true_absent = [True, False, False]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["precision"] == pytest.approx(1 / 3, abs=1e-9)
        assert m["recall"] == 1.0
        assert m["false_abstention_rate"] == 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            abstention_metrics([_FakeAnswer()], true_absent=[True, False])

    def test_abstention_rate(self) -> None:
        results = [_FakeAbstention(), _FakeAnswer(), _FakeAbstention(), _FakeAnswer()]
        true_absent = [True, False, False, True]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["abstention_rate"] == pytest.approx(0.5, abs=1e-9)

    def test_no_absent_items_recall_is_one(self) -> None:
        # Edge case: all items answerable; recall is 1 (vacuously true)
        results = [_FakeAnswer(), _FakeAnswer()]
        true_absent = [False, False]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["recall"] == 1.0

    def test_all_absent_no_false_abstention(self) -> None:
        results = [_FakeAbstention(), _FakeAbstention()]
        true_absent = [True, True]
        m = abstention_metrics(results, true_absent=true_absent)
        assert m["false_abstention_rate"] == 0.0  # no answerable items → 0


# ---------------------------------------------------------------------------
# qualifier_guard_rate
# ---------------------------------------------------------------------------


class TestQualifierGuardRate:
    def test_empty_returns_nan(self) -> None:
        assert math.isnan(qualifier_guard_rate([]))

    def test_all_clean(self) -> None:
        pairs = [
            ("The contract shall terminate on 30 days notice.", "The contract shall terminate on 30 days notice."),
            ("Either party may exit.", "Either party may exit."),
        ]
        assert qualifier_guard_rate(pairs) == 1.0

    def test_flipped_qualifier_caught(self) -> None:
        # Claim says "shall" but quote only has "may"
        pairs = [("The licensee shall pay immediately.", "The licensee may pay immediately.")]
        rate = qualifier_guard_rate(pairs)
        # "shall" not in quote → violation → rate = 0.0
        assert rate == 0.0

    def test_mixed(self) -> None:
        pairs = [
            ("The party shall comply.", "The party shall comply."),  # clean
            ("The vendor must deliver.", "The vendor may deliver."),  # "must" violated
        ]
        rate = qualifier_guard_rate(pairs)
        assert rate == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# faithfulness_rate
# ---------------------------------------------------------------------------


class TestFaithfulnessRate:
    def test_empty_returns_nan(self) -> None:
        assert math.isnan(faithfulness_rate([]))

    def test_all_faithful(self) -> None:
        pairs = [
            ("The fee is $5,000.", "The fee is $5,000."),
            ("Notice period is 30 days.", "Notice period is 30 days."),
        ]
        assert faithfulness_rate(pairs) == 1.0

    def test_wrong_number_caught(self) -> None:
        # Claim says $10,000 but quote has $5,000
        pairs = [("The fee is $10,000.", "The fee is $5,000.")]
        assert faithfulness_rate(pairs) == 0.0

    def test_correct_number_passes(self) -> None:
        # Comma formatting differs: $5,000 vs $5000 — should normalize to same
        pairs = [("The fee is $5,000.", "The fee is $5000.")]
        assert faithfulness_rate(pairs) == 1.0

    def test_mixed_faithful_unfaithful(self) -> None:
        pairs = [
            ("Term is 2 years.", "Term is 2 years."),      # faithful
            ("Penalty is $99.", "Penalty is $100."),         # unfaithful (wrong number)
            ("Party may exit.", "Party may exit."),          # faithful
        ]
        rate = faithfulness_rate(pairs)
        assert rate == pytest.approx(2 / 3, abs=1e-9)


# ---------------------------------------------------------------------------
# mean_claims_per_answer
# ---------------------------------------------------------------------------


class TestMeanClaimsPerAnswer:
    def test_no_results_returns_nan(self) -> None:
        assert math.isnan(mean_claims_per_answer([]))

    def test_only_abstentions_returns_nan(self) -> None:
        results = [_FakeAbstention(), _FakeAbstention()]
        assert math.isnan(mean_claims_per_answer(results))

    def test_single_answer(self) -> None:
        results = [_FakeAnswer(n_claims=3)]
        assert mean_claims_per_answer(results) == 3.0

    def test_mixed(self) -> None:
        results = [_FakeAnswer(n_claims=2), _FakeAbstention(), _FakeAnswer(n_claims=4)]
        # mean over answered: (2 + 4) / 2 = 3.0
        assert mean_claims_per_answer(results) == 3.0


# ---------------------------------------------------------------------------
# answer_quality_report
# ---------------------------------------------------------------------------


class TestAnswerQualityReport:
    def test_basic_report_structure(self) -> None:
        results = [_FakeAnswer(n_claims=2), _FakeAbstention()]
        true_absent = [False, True]
        report = answer_quality_report(results, true_absent=true_absent)

        assert report["n_questions"] == 2
        assert report["n_answered"] == 1
        assert report["n_abstained"] == 1
        assert report["abstention_precision"] == 1.0
        assert report["abstention_recall"] == 1.0
        assert report["false_abstention_rate"] == 0.0
        assert report["mean_claims_per_answer"] == 2.0

    def test_with_verification_stats(self) -> None:
        results = [_FakeAnswer()]
        true_absent = [False]
        stats = [{"n_total": 4, "n_unknown_id": 1, "n_valid_id": 3, "n_quote_not_found": 1}]
        report = answer_quality_report(results, true_absent=true_absent, verification_stats=stats)
        # validity = (4-1)/4 = 0.75
        assert report["citation_validity_rate"] == pytest.approx(0.75, abs=1e-4)
        # quote rate = (3-1)/3 ≈ 0.6667
        assert report["citation_quote_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError):
            answer_quality_report([], true_absent=[])
