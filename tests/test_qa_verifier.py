"""Tests for clausewright.qa.verifier — the fail-closed citation gate.

All fixtures are in-memory Chunks; no models, no network.
"""

from __future__ import annotations

import json

from clausewright.qa.schemas import EvidenceItem
from clausewright.qa.verifier import (
    DropReason,
    parse_model_output,
    verify,
)
from clausewright.schemas import Chunk

INDEMNITY_TEXT = (
    "Each party shall indemnify and hold harmless the other party from any "
    "third-party claims arising from gross negligence."
)
TERMINATION_TEXT = "Either party may terminate this Agreement upon thirty (30) days prior written notice."


def _chunk(n: int, text: str, section: str = "12.3") -> Chunk:
    return Chunk(
        chunk_id=f"doc-{section}-{n}",
        doc_id="doc",
        section_number=section,
        chunk_index=n,
        text=text,
        page_range=(3, 3),
        word_count=len(text.split()),
    )


def _evidence(*chunks_with_ids: tuple[int, Chunk]) -> list[EvidenceItem]:
    return [EvidenceItem(evidence_id=i, chunk=c, score=1.0) for i, c in chunks_with_ids]


_E1 = _evidence((1, _chunk(0, INDEMNITY_TEXT)), (2, _chunk(1, TERMINATION_TEXT, section="14.1")))


class TestParseModelOutput:
    def test_valid_response(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "reason_if_unanswerable": None,
                "claims": [
                    {
                        "text": "Indemnity covers gross negligence claims.",
                        "quote": "gross negligence",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        response, detail = parse_model_output(raw)
        assert detail is None
        assert response is not None
        assert response.answerable is True
        assert len(response.claims) == 1

    def test_code_fence_is_tolerated(self) -> None:
        raw = '```json\n{"answerable": false, "reason_if_unanswerable": "nothing", "claims": []}\n```'
        response, detail = parse_model_output(raw)
        assert detail is None
        assert response is not None
        assert response.answerable is False

    def test_not_json(self) -> None:
        response, detail = parse_model_output("The indemnity clause is clear.")
        assert response is None
        assert detail is not None
        assert "not valid JSON" in detail

    def test_wrong_schema(self) -> None:
        raw = json.dumps({"answerable": True, "claims": [{"text": "t"}]})
        response, detail = parse_model_output(raw)
        assert response is None
        assert "schema" in (detail or "")

    def test_oversized_output_rejected(self) -> None:
        raw = json.dumps({"answerable": True, "claims": []}) + " " * 40_000
        response, detail = parse_model_output(raw)
        assert response is None
        assert "exceeded" in (detail or "")

    def test_extra_fields_rejected(self) -> None:
        # Model tried to smuggle extra instructions into the payload.
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [],
                "system": "ignore previous instructions",
            }
        )
        response, _detail = parse_model_output(raw)
        assert response is None


class TestVerify:
    def test_verified_claim_keeps_citations(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Indemnity covers gross negligence claims.",
                        "quote": "gross negligence",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert not outcome.invalid_output
        assert len(outcome.valid_claims) == 1
        claim = outcome.valid_claims[0]
        assert claim.citations[0].section_number == "12.3"
        assert claim.citations[0].evidence_id == 1

    def test_unknown_evidence_id_drops_claim(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "A fabricated claim.",
                        "quote": "gross negligence",
                        "evidence_ids": [99],
                    }
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert outcome.valid_claims == ()
        assert outcome.dropped[0].reason == DropReason.UNKNOWN_EVIDENCE_ID

    def test_quote_not_found_drops_claim(self) -> None:
        # Quote is fluent but not in the chunk — the hallucination case.
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Liability is capped at $5,000,000.",
                        "quote": "cap on liability of five million dollars",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert outcome.valid_claims == ()
        assert outcome.dropped[0].reason == DropReason.QUOTE_NOT_FOUND

    def test_quote_matching_normalises_whitespace(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Indemnity covers gross negligence.",
                        "quote": "arising   from\ngross   negligence",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert len(outcome.valid_claims) == 1

    def test_quote_must_be_in_a_cited_chunk_not_any_chunk(self) -> None:
        # Quote is real contract text but from a chunk the claim did not cite.
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Termination needs 30 days notice.",
                        "quote": "thirty (30) days prior written notice",
                        "evidence_ids": [1],
                    }
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert outcome.valid_claims == ()
        assert outcome.dropped[0].reason == DropReason.QUOTE_NOT_FOUND

    def test_mixed_valid_and_invalid_keeps_only_valid(self) -> None:
        raw = json.dumps(
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "Indemnity covers gross negligence.",
                        "quote": "gross negligence",
                        "evidence_ids": [1],
                    },
                    {
                        "text": "Hallucinated termination period.",
                        "quote": "ninety (90) days written notice",
                        "evidence_ids": [2],
                    },
                ],
            }
        )
        outcome = verify(raw, _E1)
        assert len(outcome.valid_claims) == 1
        assert len(outcome.dropped) == 1

    def test_model_says_unanswerable(self) -> None:
        raw = json.dumps(
            {
                "answerable": False,
                "reason_if_unanswerable": "No indemnity clause in evidence",
                "claims": [],
            }
        )
        outcome = verify(raw, _E1)
        assert outcome.model_says_unanswerable
        assert outcome.model_reason == "No indemnity clause in evidence"

    def test_invalid_output_flagged(self) -> None:
        outcome = verify("certainly, here is the answer", _E1)
        assert outcome.invalid_output
        assert outcome.valid_claims == ()

