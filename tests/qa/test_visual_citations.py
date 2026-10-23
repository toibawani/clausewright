"""Unit tests for visual citation anchoring and bounding boxes (Pillar 4)."""

from __future__ import annotations

from clausewright.qa.visual_citations import (
    find_quote_offsets,
    locate_visual_citation,
)
from clausewright.schemas import Block, Chunk


def test_find_quote_offsets_exact() -> None:
    text = "The governing law shall be the laws of the State of New York."
    quote = "laws of the State of New York"
    offsets = find_quote_offsets(quote, text)
    assert offsets is not None
    start, end = offsets
    assert text[start:end] == quote


def test_find_quote_offsets_whitespace_normalized() -> None:
    text = "Payment is due\n    within thirty (30)\n    calendar days."
    quote = "Payment is due within thirty (30) calendar days."
    offsets = find_quote_offsets(quote, text)
    assert offsets is not None


def test_locate_visual_citation_with_blocks() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        section_number="5",
        chunk_index=0,
        text="The Supplier agrees to indemnify Customer against all third party claims.",
        page_range=(2, 2),
        word_count=11,
    )
    block = Block(
        text="The Supplier agrees to indemnify Customer against all third party claims.",
        page_number=2,
        bbox=(72.0, 100.0, 500.0, 120.0),
        font_size=11.0,
    )

    quote = "indemnify Customer against all third party claims"
    anchor = locate_visual_citation(quote, chunk, available_blocks=[block])

    assert anchor.page_number == 2
    assert anchor.char_offset[0] >= 0
    assert len(anchor.bounding_boxes) == 1
    assert anchor.bounding_boxes[0] == (72.0, 100.0, 500.0, 120.0)
