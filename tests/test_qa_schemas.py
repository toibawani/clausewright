"""Tests for clausewright.qa.schemas.

Pure validation checks — no models, no network.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clausewright.qa.schemas import (
    NOTICE,
    Abstention,
    AbstentionReason,
    Answer,
    Citation,
    Claim,
)


def _citation(**overrides: object) -> Citation:
    base: dict = {
        "evidence_id": 1,
        "doc_id": "doc",
        "section_number": "12.3",
        "page_range": (3, 4),
        "score": 1.5,
    }
    base.update(overrides)
    return Citation(**base)  # type: ignore[arg-type]


def _claim() -> Claim:
    return Claim(
        text="The notice period is 30 days.",
        quote="thirty (30) days prior written notice",
        citations=(_citation(),),
    )


class TestCitation:
    def test_valid(self) -> None:
        c = _citation()
        assert c.section_number == "12.3"
        assert c.page_range == (3, 4)

    def test_evidence_id_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _citation(evidence_id=0)

    def test_frozen_and_no_extra_fields(self) -> None:
        c = _citation()
        with pytest.raises(ValidationError):
            c.score = 2.0  # type: ignore[misc]
        with pytest.raises(ValidationError):
            _citation(bogus=1)


class TestClaim:
    def test_requires_at_least_one_citation(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="t", quote="q", citations=())


class TestAnswer:
    def test_answer_carries_single_notice(self) -> None:
        a = Answer(question="q?", text="t", claims=(_claim(),), model="test-model")
        assert a.notice == NOTICE
        assert a.status == "answered"

    def test_answer_requires_claims(self) -> None:
        with pytest.raises(ValidationError):
            Answer(question="q?", text="t", claims=(), model="m")


class TestAbstention:
    def test_abstention_defaults(self) -> None:
        a = Abstention(
            question="q?",
            reason=AbstentionReason.NO_EVIDENCE,
            detail="nothing matched",
        )
        assert a.status == "abstained"
        assert a.top_score is None
        assert a.searched_sections == ()
        assert a.notice == NOTICE

    def test_reason_is_enum_only(self) -> None:
        with pytest.raises(ValidationError):
            Abstention(question="q?", reason="guessing", detail="d")  # type: ignore[arg-type]
