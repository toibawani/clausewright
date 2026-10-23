"""Unit tests for OrderOfPrecedenceResolver (Pillar 1)."""

from __future__ import annotations

from clausewright.extraction.precedence import (
    DocumentTier,
    OrderOfPrecedenceResolver,
    classify_document_tier,
)
from clausewright.schemas import Chunk


def _make_chunk(doc_id: str, chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_number="1",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


def test_classify_document_tier() -> None:
    assert classify_document_tier("Amendment_No_2.pdf") == DocumentTier.AMENDMENT
    assert classify_document_tier("Order_Form_Acme.pdf") == DocumentTier.ORDER_FORM
    assert classify_document_tier("SOW_Phase_1.pdf") == DocumentTier.STATEMENT_OF_WORK
    assert classify_document_tier("Schedule_B_Fees.pdf") == DocumentTier.SCHEDULE
    assert classify_document_tier("Exhibit_A_SLA.pdf") == DocumentTier.EXHIBIT
    assert classify_document_tier("Master_Services_Agreement.pdf") == DocumentTier.BASE_AGREEMENT


def test_order_of_precedence_prioritize() -> None:
    resolver = OrderOfPrecedenceResolver()
    resolver.register_document("doc-base", "MSA.pdf", tier=DocumentTier.BASE_AGREEMENT, version=1)
    resolver.register_document(
        "doc-amend", "Amendment_1.pdf", tier=DocumentTier.AMENDMENT, version=2
    )

    chunk_base = _make_chunk("doc-base", "c-base", "Payment is net 30 days.")
    chunk_amend = _make_chunk("doc-amend", "c-amend", "Payment is net 15 days.")

    # Even if base has slightly higher raw retrieval score, amendment should be boosted
    candidates = [(chunk_base, 0.85), (chunk_amend, 0.83)]
    prioritized = resolver.prioritize_candidates(candidates)

    assert len(prioritized) == 2
    assert prioritized[0][0].chunk_id == "c-amend"
    assert prioritized[1][0].chunk_id == "c-base"
