"""Tests for the fixed-window chunker baseline."""

from __future__ import annotations

import pytest

from clausewright.eval.baselines import fixed_window_chunks
from clausewright.ingestion import chunk_sections, parse_sections
from clausewright.schemas import Block, Page, ParsedDocument


def _block(text: str, page: int = 1) -> Block:
    return Block(text=text, page_number=page, bbox=(0.0, 0.0, 10.0, 10.0), font_size=10.0)


def _doc(*page_texts: list[str]) -> ParsedDocument:
    pages = [
        Page(number=i + 1, width=100.0, height=100.0, blocks=[_block(t, i + 1) for t in pt])
        for i, pt in enumerate(page_texts)
    ]
    return ParsedDocument(doc_id="d", source_path="s", pages=pages)


class TestFixedWindowChunks:
    def test_windows_ignore_structure(self) -> None:
        doc = _doc([" ".join(f"w{i}" for i in range(400))])
        out = fixed_window_chunks("d", doc, max_words=100, stride=100)
        assert len(out) == 4
        assert all(c.section_number == "window" for c in out)
        assert out[0].word_count == 100
        # First and last words land in the first and last window.
        assert out[0].text.startswith("w0")
        assert out[-1].text.endswith("w399")

    def test_spans_sections_and_pages(self) -> None:
        doc = _doc(
            [" ".join(f"a{i}" for i in range(60))],
            [" ".join(f"b{i}" for i in range(60))],
        )
        out = fixed_window_chunks("d", doc, max_words=100, stride=100)
        assert len(out) == 2
        assert out[0].page_range == (1, 2)
        assert out[1].page_range == (2, 2)

    def test_same_inputs_as_structure_chunker(self) -> None:
        text = " ".join(f"word{i}" for i in range(250))
        doc = _doc([text])
        window_chunks = fixed_window_chunks("d", doc, max_words=300, stride=150)
        struct_chunks = chunk_sections("d", parse_sections(doc), max_words=300, stride=150)
        assert len(window_chunks) == len(struct_chunks) == 1
        assert window_chunks[0].text == struct_chunks[0].text

    def test_empty_document_rejected(self) -> None:
        doc = ParsedDocument(doc_id="d", source_path="s", pages=[Page(number=1, width=1.0, height=1.0)])
        with pytest.raises(ValueError, match="no text blocks"):
            fixed_window_chunks("d", doc)

    def test_chunk_ids_unique(self) -> None:
        doc = _doc([f"{' '.join(f'w{i}' for i in range(300))}"])
        out = fixed_window_chunks("d", doc, max_words=100, stride=50)
        ids = [c.chunk_id for c in out]
        assert len(ids) == len(set(ids))
