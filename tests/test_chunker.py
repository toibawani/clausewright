"""Tests for clausewright.ingestion.chunker."""

from __future__ import annotations

import pytest

from clausewright.ingestion.chunker import chunk_sections
from clausewright.schemas import Block, Chunk, Section

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    text: str,
    page: int = 1,
    font_size: float = 12.0,
    is_bold: bool = False,
) -> Block:
    return Block(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 100.0, 20.0),
        font_size=font_size,
        is_bold=is_bold,
    )


def _section(
    number: str,
    *blocks: Block,
    level: int = 1,
    page_range: tuple[int, int] = (1, 1),
) -> Section:
    return Section(
        section_number=number,
        level=level,
        blocks=list(blocks),
        page_range=page_range,
    )


def _words(n: int, word: str = "word") -> str:
    """Return a string of *n* repeated words."""
    return " ".join([word] * n)


def _seq_words(n: int, prefix: str = "w") -> str:
    """Return *n* distinct words w0 w1 ... so overlaps can be asserted."""
    return " ".join(f"{prefix}{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_max_words_raises() -> None:
    with pytest.raises(ValueError, match="max_words"):
        chunk_sections("doc", [], max_words=0)


def test_invalid_stride_zero_raises() -> None:
    with pytest.raises(ValueError, match="stride"):
        chunk_sections("doc", [], max_words=100, stride=0)


def test_stride_exceeds_max_words_raises() -> None:
    with pytest.raises(ValueError, match="stride"):
        chunk_sections("doc", [], max_words=100, stride=101)


def test_stride_equal_max_words_is_valid() -> None:
    """stride == max_words means no overlap — should not raise."""
    sec = _section("1", _block(_words(80)))
    result = chunk_sections("doc", [sec], max_words=80, stride=80)
    assert len(result) == 1
    assert result[0].word_count == 80


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------


def test_empty_section_list_returns_empty() -> None:
    assert chunk_sections("doc", []) == []


def test_section_with_no_text_skipped() -> None:
    """Section with blocks that tokenise to zero words should be skipped."""
    sec = _section("PREAMBLE", _block("   "))  # whitespace only → no words
    result = chunk_sections("doc", [sec], max_words=100, stride=50)
    # Whitespace-only blocks produce zero words → section skipped
    assert result == []


# ---------------------------------------------------------------------------
# Single-chunk sections
# ---------------------------------------------------------------------------


def test_short_section_produces_single_chunk() -> None:
    sec = _section("1", _block("Hello world this is a test."))
    chunks = chunk_sections("doc", [sec], max_words=100, stride=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].section_number == "1"
    assert chunks[0].doc_id == "doc"


def test_exact_window_size_produces_single_chunk() -> None:
    sec = _section("2", _block(_words(50)))
    chunks = chunk_sections("doc", [sec], max_words=50, stride=25)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Multi-chunk splitting
# ---------------------------------------------------------------------------


def test_long_section_produces_multiple_chunks() -> None:
    # 160 words, window=80, stride=80 → 2 non-overlapping chunks, both at the floor
    sec = _section("3", _block(_words(160)))
    chunks = chunk_sections("doc", [sec], max_words=80, stride=80)
    assert len(chunks) == 2
    assert [c.word_count for c in chunks] == [80, 80]


def test_overlapping_chunks() -> None:
    # 140 words, window=100, stride=60 → windows at positions 0 and 60
    # window 0: words [0, 100)  → 100 words
    # window 1: words [60, 140) → 80 words (both clear the 80-word floor)
    sec = _section("4", _block(_seq_words(140)))
    chunks = chunk_sections("doc", [sec], max_words=100, stride=60)
    assert len(chunks) == 2
    assert [c.word_count for c in chunks] == [100, 80]
    # The two windows overlap by 40 words: chunk 1 starts at w60.
    assert chunks[0].text.split()[60] == chunks[1].text.split()[0] == "w60"


def test_chunk_indices_are_sequential() -> None:
    # 300 words, window=80, stride=80 → 80/80/80/60; the tail merges, leaving 3
    sec = _section("5", _block(_words(300)))
    chunks = chunk_sections("doc", [sec], max_words=80, stride=80)
    assert len(chunks) > 1
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_all_words_covered_non_overlapping() -> None:
    """With stride == max_words, the union of chunk words equals the section text."""
    text = _words(320, word="tok")
    sec = _section("6", _block(text))
    chunks = chunk_sections("doc", [sec], max_words=80, stride=80)
    reconstructed = " ".join(c.text for c in chunks)
    assert reconstructed == text


# ---------------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------------


def test_word_count_matches_text() -> None:
    sec = _section("7", _block(_words(160)))
    chunks = chunk_sections("doc", [sec], max_words=80, stride=80)
    for chunk in chunks:
        assert chunk.word_count == len(chunk.text.split())


def test_trailing_small_chunk_is_merged_into_previous() -> None:
    # 230 words, window 100, stride 100 → 100, 100, 30; the 30-word tail is
    # below the floor and is folded into the previous chunk instead of kept.
    sec = _section("8", _block(_words(230)))
    chunks = chunk_sections("doc", [sec], max_words=100, stride=100)
    assert len(chunks) == 2
    assert chunks[-1].word_count == 130


# ---------------------------------------------------------------------------
# chunk_id format
# ---------------------------------------------------------------------------


def test_chunk_id_format() -> None:
    sec = _section("1.2", _block("Some text here."))
    chunks = chunk_sections("my-doc", [sec], max_words=100, stride=50)
    assert chunks[0].chunk_id == "my-doc-1_2-0"


def test_chunk_id_unique_across_sections() -> None:
    s1 = _section("1", _block(_words(100)))
    s2 = _section("2", _block(_words(100)))
    chunks = chunk_sections("doc", [s1, s2], max_words=100, stride=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Page range attribution
# ---------------------------------------------------------------------------


def test_single_page_chunk_range() -> None:
    sec = _section("1", _block("Text on page one.", page=1), page_range=(1, 1))
    chunks = chunk_sections("doc", [sec], max_words=100, stride=50)
    assert chunks[0].page_range == (1, 1)


def test_multi_page_block_range() -> None:
    """Chunk spanning blocks on pages 2 and 3 should report (2, 3)."""
    sec = _section(
        "1",
        _block("Page two text.", page=2),
        _block("Page three text.", page=3),
        page_range=(2, 3),
    )
    chunks = chunk_sections("doc", [sec], max_words=100, stride=50)
    assert chunks[0].page_range == (2, 3)


def test_page_range_first_chunk_vs_second() -> None:
    """When blocks split across pages, the two windows report their own pages."""
    # 80 words on page 1, 80 words on page 2 → window=80, stride=80 → 2 chunks
    sec = _section(
        "1",
        _block(_words(80), page=1),
        _block(_words(80), page=2),
        page_range=(1, 2),
    )
    chunks = chunk_sections("doc", [sec], max_words=80, stride=80)
    assert len(chunks) == 2
    assert chunks[0].page_range == (1, 1)
    assert chunks[1].page_range == (2, 2)


# ---------------------------------------------------------------------------
# Multiple sections
# ---------------------------------------------------------------------------


def test_multiple_sections_ordering() -> None:
    s1 = _section("1", _block(_words(100)))
    s2 = _section("2", _block(_words(100)))
    s3 = _section("3", _block(_words(100)))
    chunks = chunk_sections("doc", [s1, s2, s3], max_words=100, stride=50)
    sec_nums = [c.section_number for c in chunks]
    assert sec_nums == ["1", "2", "3"]


def test_returns_list_of_chunk_instances() -> None:
    sec = _section("1", _block("Some text."))
    chunks = chunk_sections("doc", [sec])
    assert all(isinstance(c, Chunk) for c in chunks)


# ---------------------------------------------------------------------------
# Return type immutability (frozen model)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression: merging small chunks must not drop the final chunk
# ---------------------------------------------------------------------------


def test_merge_small_chunks_keeps_final_chunk() -> None:
    """Both chunks already meet the floor, so merging must return both."""
    from clausewright.ingestion.chunker import _merge_small_chunks

    chunks = [
        Chunk(
            chunk_id="d-1-0",
            doc_id="d",
            section_number="1",
            chunk_index=0,
            text=_words(80),
            page_range=(1, 1),
            word_count=80,
        ),
        Chunk(
            chunk_id="d-1-1",
            doc_id="d",
            section_number="1",
            chunk_index=1,
            text=_words(80, word="tail"),
            page_range=(1, 1),
            word_count=80,
        ),
    ]

    merged = _merge_small_chunks(chunks, min_words=80)

    assert len(merged) == 2
    assert merged[-1].word_count == 80
    assert merged[-1].text.split()[0] == "tail"


def test_final_merged_window_is_not_dropped() -> None:
    """A merged final window of >= MIN_CHUNK_WORDS must be returned, not lost."""
    sec = _section("1", _block(_words(100)))
    chunks = chunk_sections("doc", [sec], max_words=50, stride=50)
    assert len(chunks) == 1
    assert chunks[0].word_count == 100


def test_trailing_small_chunk_merges_without_duplicating() -> None:
    """A short trailing window is folded into the previous one exactly once."""
    sec = _section("1", _block(_words(230, word="tok")))
    chunks = chunk_sections("doc", [sec], max_words=100, stride=100)
    assert len(chunks) == 2
    assert chunks[-1].word_count == 130
    assert " ".join(c.text for c in chunks).split() == ["tok"] * 230


def test_chunk_is_frozen() -> None:
    """Pydantic v2 frozen models raise ValidationError on direct field assignment."""
    from pydantic import ValidationError

    sec = _section("1", _block("Immutable chunk text."))
    chunk = chunk_sections("doc", [sec])[0]
    with pytest.raises(ValidationError, match="frozen"):
        chunk.text = "mutated"  # type: ignore[misc]

