"""Tests for clausewright.eval.faithfulness — deterministic literal checks.

Every case is hand-computed.  A metrics bug means fake numbers, so these pin
down the extraction and the comparison rules.
"""

from __future__ import annotations

from clausewright.eval.faithfulness import (
    QUALIFIER_WORDS,
    extract_literals,
    is_faithful,
    violations,
)


class TestExtractLiterals:
    def test_numbers(self) -> None:
        assert extract_literals("30 days") == ["30"]

    def test_currency_with_commas(self) -> None:
        assert extract_literals("cap of $2,000,000") == ["2000000"]

    def test_iso_date(self) -> None:
        assert extract_literals("Signed 2024-01-05.") == ["2024-01-05"]

    def test_month_date(self) -> None:
        assert extract_literals("Effective January 5, 2024.") == ["January 5, 2024"]

    def test_qualifiers(self) -> None:
        assert extract_literals("shall not subject to provided that") == [
            "shall not", "subject to", "provided that"
        ]

    def test_dedup_not_deduped(self) -> None:
        # Same number twice backs two facts; both are kept.
        assert extract_literals("30 days then 30 more days") == ["30", "30"]

    def test_currency_code(self) -> None:
        assert extract_literals("price is EUR 50000") == ["50000"]

    def test_year_only(self) -> None:
        assert extract_literals("In 2024 the cap changed") == ["2024"]


class TestViolations:
    def test_clean(self) -> None:
        assert violations(
            "Cap is $2,000,000 per Section 4.",
            "...Capped at $2,000,000 per Section 4...",
        ) == []

    def test_qualifier_flip(self) -> None:
        assert violations(
            "Party shall terminate with 30 days notice.",
            "Party may terminate with 30 days notice.",
        ) == ["shall"]

    def test_invented_number(self) -> None:
        assert violations(
            "Termination requires 90 days notice.",
            "Party may terminate with 30 days notice.",
        ) == ["90"]

    def test_missing_qualifier(self) -> None:
        # "shall not" in claim; quote has "may" — "shall not" is missing.
        # "shall" is shadowed by the multi-word phrase and not double-reported.
        assert violations(
            "Party shall not terminate.",
            "Party may terminate.",
        ) == ["shall not"]

    def test_comma_normalisation(self) -> None:
        assert violations("Cap is $1,000 total.", "Cap is $1000 total.") == []

    def test_iso_date_present(self) -> None:
        assert violations("Signed 2024-01-05.", "Agreement dated 2024-01-05...") == []

    def test_missing_multiword_qualifier(self) -> None:
        assert violations(
            "Subject to Section 9, payment is due.",
            "Payment is due under Section 9.",
        ) == ["subject to"]

    def test_currency_symbol_preserved(self) -> None:
        assert violations("Cost is $50.", "Cost is $50.") == []

    def test_partial_date_missing(self) -> None:
        assert violations(
            "Effective January 5, 2024.", "Effective January 5."
        ) == ["January 5, 2024"]


class TestIsFaithful:
    def test_faithful_no_violations(self) -> None:
        assert is_faithful("Fee is $500.", "The fee is $500 per month.") is True

    def test_unfaithful_wrong_number(self) -> None:
        assert is_faithful("Fee is $900.", "The fee is $500 per month.") is False

    def test_faithful_no_literals(self) -> None:
        # Claim has no numbers/dates/qualifiers — nothing to check; trivially faithful
        assert is_faithful("Payment is due promptly.", "Payment is due promptly.") is True


class TestQualifierWords:
    def test_list_contains_all_required(self) -> None:
        required = {
            "shall not", "subject to", "provided that",
            "shall", "must", "may", "unless", "except",
        }
        assert set(QUALIFIER_WORDS) == required
