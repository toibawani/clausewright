"""Tests for the packed-section chunker: packing, boundaries, splitting."""

from __future__ import annotations

import pytest

from clausewright.ingestion.packed_chunker import pack_sections
from clausewright.schemas import Block, Section


def _block(text: str, page: int = 1) -> Block:
    return Block(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        font_size=10.0,
    )


def _section(
    number: str,
    body: str,
    *,
    parent: str | None = None,
    page: int = 1,
    title: str | None = None,
) -> Section:
    return Section(
        section_number=number,
        title=title,
        level=number.count(".") + 1,
        parent_number=parent,
        blocks=[_block(body, page=page)],
        page_range=(page, page),
    )


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_small_sections_under_one_article_pack_into_a_single_chunk() -> None:
    sections = [
        _section("1", _words(50), title="Definitions"),
        _section("1.1", _words(40), parent="1", title="Affiliate"),
        _section("1.2", _words(40), parent="1", title="Confidential Information"),
    ]

    chunks = pack_sections("doc", sections, max_words=200)

    assert len(chunks) == 1
    assert chunks[0].word_count == 130
    # Provenance is the first section of the window.
    assert chunks[0].section_number == "1"
    assert chunks[0].breadcrumbs == ("Section 1: Definitions",)


def test_packing_stops_at_the_word_budget() -> None:
    sections = [
        _section("1", _words(100)),
        _section("1.1", _words(100), parent="1"),
        _section("1.2", _words(100), parent="1"),
    ]

    chunks = pack_sections("doc", sections, max_words=150)

    assert [c.word_count for c in chunks] == [100, 100, 100]
    assert [c.section_number for c in chunks] == ["1", "1.1", "1.2"]


def test_packing_never_crosses_a_top_level_article_boundary() -> None:
    sections = [
        _section("1", _words(30)),
        _section("1.1", _words(30), parent="1"),
        _section("2", _words(30)),
        _section("2.1", _words(30), parent="2"),
    ]

    chunks = pack_sections("doc", sections, max_words=500)

    # Two articles, so two windows even though both would fit together.
    assert len(chunks) == 2
    assert chunks[0].section_number == "1"
    assert chunks[0].word_count == 60
    assert chunks[1].section_number == "2"
    assert chunks[1].word_count == 60


def test_oversize_section_is_split_and_does_not_absorb_neighbours() -> None:
    sections = [
        _section("1", _words(30)),
        _section("2", _words(250)),
        _section("3", _words(30)),
    ]

    chunks = pack_sections("doc", sections, max_words=100)

    section_numbers = [c.section_number for c in chunks]
    # 250 words at a 100-word window and 50-word stride is four windows.
    assert section_numbers == ["1", "2", "2", "2", "2", "3"]
    # Neither neighbour is absorbed into the split section.
    assert chunks[0].word_count == 30
    assert chunks[-1].word_count == 30
    # Only the oversize section is allowed to appear more than once.
    assert section_numbers.count("1") == 1
    assert section_numbers.count("3") == 1


def test_chunk_ids_are_unique_and_prefixed() -> None:
    sections = [
        _section("1", _words(30)),
        _section("1.1", _words(30), parent="1"),
        _section("2", _words(30)),
    ]

    chunks = pack_sections("doc", sections, max_words=200)

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all("-packed-" in c.chunk_id for c in chunks)


def test_pages_span_the_packed_sections() -> None:
    sections = [
        _section("1", _words(20), page=1),
        _section("1.1", _words(20), parent="1", page=3),
    ]

    chunks = pack_sections("doc", sections, max_words=200)

    assert chunks[0].page_range == (1, 3)


def test_empty_sections_are_skipped() -> None:
    empty = Section(section_number="1", level=1, blocks=[], page_range=(1, 1))
    sections = [empty, _section("1.1", _words(10), parent="1")]

    chunks = pack_sections("doc", sections, max_words=200)

    assert len(chunks) == 1
    assert chunks[0].word_count == 10


def test_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        pack_sections("doc", [_section("1", _words(5))], max_words=0)
