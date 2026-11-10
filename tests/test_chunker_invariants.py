"""Invariant tests for the chunker.

These assert properties that must hold for any chunker configuration,
independent of the exact window layout:

* words are neither lost nor duplicated when there is no overlap,
* the original word order is preserved (including overlapping windows),
* every chunk meets the ``MIN_CHUNK_WORDS`` floor unless the document is
  smaller than the floor,
* a chunk's ``section_number`` matches the section that produced its first
  word,
* output is deterministic.
"""

from __future__ import annotations

from clausewright.ingestion.chunker import MIN_CHUNK_WORDS, chunk_sections
from clausewright.schemas import Block, Chunk, Section


def _block(text: str, page: int = 1) -> Block:
    return Block(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 100.0, 20.0),
        font_size=12.0,
    )


def _section(number: str, words: list[str], page: int = 1) -> Section:
    return Section(
        section_number=number,
        blocks=[_block(" ".join(words), page)],
        page_range=(page, page),
    )


def _doc(section_sizes: tuple[int, ...]) -> tuple[list[Section], list[str]]:
    """Build sections whose words are unique tokens ``s<section>_<index>``."""
    sections: list[Section] = []
    expected: list[str] = []
    for section_index, n_words in enumerate(section_sizes, start=1):
        words = [f"s{section_index}_{i}" for i in range(n_words)]
        sections.append(_section(str(section_index), words))
        expected.extend(words)
    return sections, expected


def test_no_words_lost_or_duplicated_without_overlap() -> None:
    sections, expected = _doc((120, 95, 240))
    chunks = chunk_sections("doc", sections, max_words=100, stride=100)

    actual = " ".join(c.text for c in chunks).split()

    assert actual == expected


def test_word_order_preserved_with_overlap() -> None:
    sections, expected = _doc((300, 140))
    chunks = chunk_sections("doc", sections, max_words=100, stride=50)

    first_positions: list[int] = []
    for chunk in chunks:
        pos = 0
        for word in chunk.text.split():
            while pos < len(expected) and expected[pos] != word:
                pos += 1
            assert pos < len(expected), f"{word!r} does not appear in document order"
            pos += 1
        first_positions.append(expected.index(chunk.text.split()[0]))

    assert first_positions == sorted(first_positions)


def test_no_chunk_below_floor_unless_document_smaller() -> None:
    sections, expected = _doc((30, 25, 40))
    assert len(expected) > MIN_CHUNK_WORDS

    chunks = chunk_sections("doc", sections, max_words=100, stride=50)

    assert chunks
    assert all(c.word_count >= MIN_CHUNK_WORDS for c in chunks)


def test_small_document_is_kept_as_one_chunk() -> None:
    sections, expected = _doc((25, 20))
    assert len(expected) < MIN_CHUNK_WORDS

    chunks = chunk_sections("doc", sections, max_words=100, stride=50)

    assert len(chunks) == 1
    assert chunks[0].word_count == len(expected)


def test_section_provenance_matches_first_word() -> None:
    sections, _ = _doc((150, 100, 60))
    chunks = chunk_sections("doc", sections, max_words=100, stride=100)

    for chunk in chunks:
        first_word = chunk.text.split()[0]
        owning_section = first_word.split("_")[0][1:]
        assert chunk.section_number == owning_section


def test_output_is_deterministic() -> None:
    sections, _ = _doc((250, 130))

    first = chunk_sections("doc", sections, max_words=100, stride=50)
    second = chunk_sections("doc", sections, max_words=100, stride=50)

    def signature(chunks: list[Chunk]) -> list[tuple[str, str, tuple[int, int], int, str]]:
        return [
            (c.chunk_id, c.text, c.page_range, c.word_count, c.section_number) for c in chunks
        ]

    assert signature(first) == signature(second)
