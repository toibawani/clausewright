"""Tests for clausewright.ingestion.section_parser."""

from __future__ import annotations

from clausewright.ingestion.section_parser import parse_sections
from clausewright.schemas import Block, Page, ParsedDocument, Section

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    text: str,
    page: int = 1,
    font_size: float = 12.0,
    is_bold: bool = False,
) -> Block:
    """Convenience constructor for a Block with a simple bounding box."""
    return Block(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 100.0, 20.0),
        font_size=font_size,
        is_bold=is_bold,
    )


def _doc(*pages_blocks: list[Block], doc_id: str = "test-doc") -> ParsedDocument:
    """Build a ParsedDocument from a list-of-lists: one inner list per page."""
    pages = [
        Page(number=i + 1, width=595.0, height=842.0, blocks=blocks)
        for i, blocks in enumerate(pages_blocks)
    ]
    return ParsedDocument(doc_id=doc_id, source_path="/tmp/test.pdf", pages=pages)


# ---------------------------------------------------------------------------
# Empty / trivial documents
# ---------------------------------------------------------------------------


def test_empty_document_returns_empty_list() -> None:
    doc = ParsedDocument(doc_id="empty", source_path="/tmp/empty.pdf", pages=[])
    assert parse_sections(doc) == []


def test_document_with_only_body_text_no_font_heuristic() -> None:
    """With font heuristic disabled and no heading patterns, result is empty."""
    doc = _doc([_block("Just some body text."), _block("More body text.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    # No heading matched → PREAMBLE is created but has blocks; it should be returned
    # only when non-empty (body-only docs produce one PREAMBLE section here).
    assert len(sections) == 1
    assert sections[0].section_number == "PREAMBLE"


def test_single_page_no_headings_returns_preamble() -> None:
    doc = _doc([_block("Introduction text.", page=1)])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert sections[0].section_number == "PREAMBLE"
    assert len(sections[0].blocks) == 1


# ---------------------------------------------------------------------------
# Numeric heading detection
# ---------------------------------------------------------------------------


def test_simple_numeric_headings() -> None:
    doc = _doc([
        _block("1 Definitions"),
        _block("This section defines terms."),
        _block("2 Obligations"),
        _block("Each party shall..."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    nums = [s.section_number for s in sections]
    assert "1" in nums
    assert "2" in nums


def test_dotted_numeric_heading_level() -> None:
    """1.2 should produce level=2 and parent_number='1'."""
    doc = _doc([
        _block("1 Main Section"),
        _block("Body text."),
        _block("1.2 Sub Section"),
        _block("Sub body text."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    sub = next(s for s in sections if s.section_number == "1.2")
    assert sub.level == 2
    assert sub.parent_number == "1"


def test_triple_dotted_heading_level() -> None:
    doc = _doc([
        _block("1.2.3 Deep subsection"),
        _block("Body."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    deep = next(s for s in sections if s.section_number == "1.2.3")
    assert deep.level == 3
    assert deep.parent_number == "1.2"


def test_heading_with_trailing_dot() -> None:
    """'1.' should be treated identically to '1'."""
    doc = _doc([
        _block("1. Definitions"),
        _block("Body."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert any(s.section_number == "1" for s in sections)


def test_title_extracted_from_heading_block() -> None:
    doc = _doc([_block("3 Payment Terms"), _block("Payment body.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    sec = next(s for s in sections if s.section_number == "3")
    assert sec.title == "Payment Terms"


# ---------------------------------------------------------------------------
# Keyword headings  (Article / Section prefix)
# ---------------------------------------------------------------------------


def test_article_keyword_heading() -> None:
    """'Article 1 Definitions' → section_number='A1'."""
    doc = _doc([_block("Article 1 Definitions"), _block("Body.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert any(s.section_number == "A1" for s in sections)


def test_section_keyword_heading() -> None:
    """'Section 2.3 Payments' → section_number='S2.3'."""
    doc = _doc([_block("Section 2.3 Payments"), _block("Body.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert any(s.section_number == "S2.3" for s in sections)


def test_roman_numeral_article() -> None:
    """'Article IV' → section_number='A4'."""
    doc = _doc([_block("Article IV Obligations"), _block("Body.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert any(s.section_number == "A4" for s in sections)


# ---------------------------------------------------------------------------
# Page range
# ---------------------------------------------------------------------------


def test_page_range_single_page() -> None:
    doc = _doc([_block("1 Terms"), _block("Body on page 1.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    sec = next(s for s in sections if s.section_number == "1")
    assert sec.page_range == (1, 1)


def test_page_range_spans_pages() -> None:
    """Section that starts on page 1 and has a block on page 2."""
    doc = _doc(
        [_block("1 Terms", page=1), _block("Body on page 1.", page=1)],
        [_block("Continued on page 2.", page=2)],
    )
    sections = parse_sections(doc, use_font_heuristic=False)
    sec = next(s for s in sections if s.section_number == "1")
    assert sec.page_range == (1, 2)


def test_page_range_second_section() -> None:
    doc = _doc(
        [_block("1 First", page=1), _block("Body 1.", page=1)],
        [_block("2 Second", page=2), _block("Body 2.", page=2)],
    )
    sections = parse_sections(doc, use_font_heuristic=False)
    sec2 = next(s for s in sections if s.section_number == "2")
    assert sec2.page_range == (2, 2)


# ---------------------------------------------------------------------------
# Section text property
# ---------------------------------------------------------------------------


def test_section_text_concatenates_blocks() -> None:
    doc = _doc([
        _block("1 Definitions"),
        _block("First paragraph."),
        _block("Second paragraph."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    sec = next(s for s in sections if s.section_number == "1")
    assert "First paragraph." in sec.text
    assert "Second paragraph." in sec.text


# ---------------------------------------------------------------------------
# Font heuristic
# ---------------------------------------------------------------------------


def test_bold_block_becomes_heading_with_heuristic() -> None:
    doc = _doc([
        _block("Introduction", is_bold=True),
        _block("Some body text."),
    ])
    sections = parse_sections(doc, use_font_heuristic=True)
    # The bold block should produce an H1 section
    assert any(s.section_number.startswith("H") for s in sections)


def test_large_font_block_becomes_heading_with_heuristic() -> None:
    doc = _doc([
        _block("Introduction", font_size=18.0),  # >> avg of 12
        _block("Body text.", font_size=12.0),
    ])
    sections = parse_sections(doc, use_font_heuristic=True)
    assert any(s.section_number.startswith("H") for s in sections)


def test_no_heuristic_heading_for_single_token_bold() -> None:
    """A single-word bold block should NOT become a heading (below min_heading_words)."""
    doc = _doc([
        _block("Note", is_bold=True),
        _block("Body text."),
    ])
    # min_heading_words defaults to 1, so "Note" (1 word) still qualifies.
    # Adjust min_heading_words to 2 to suppress it.
    sections = parse_sections(doc, use_font_heuristic=True, min_heading_words=2)
    h_secs = [s for s in sections if s.section_number.startswith("H")]
    assert len(h_secs) == 0


def test_heuristic_disabled_ignores_bold() -> None:
    doc = _doc([
        _block("Introduction", is_bold=True, font_size=18.0),
        _block("Body."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert all(not s.section_number.startswith("H") for s in sections)


# ---------------------------------------------------------------------------
# Return type & model compliance
# ---------------------------------------------------------------------------


def test_returns_list_of_section_instances() -> None:
    doc = _doc([_block("1 Terms"), _block("Body.")])
    sections = parse_sections(doc, use_font_heuristic=False)
    assert all(isinstance(s, Section) for s in sections)


def test_ordering_preserved() -> None:
    doc = _doc([
        _block("1 First"),
        _block("Body 1."),
        _block("2 Second"),
        _block("Body 2."),
        _block("3 Third"),
        _block("Body 3."),
    ])
    sections = parse_sections(doc, use_font_heuristic=False)
    nums = [s.section_number for s in sections]
    assert nums == ["1", "2", "3"]


def test_multiple_pages_multi_section() -> None:
    doc = _doc(
        [_block("1 Scope", page=1), _block("Scope body.", page=1)],
        [_block("2 Payment", page=2), _block("Payment body.", page=2)],
        [_block("3 Termination", page=3), _block("Termination body.", page=3)],
    )
    sections = parse_sections(doc, use_font_heuristic=False)
    assert len(sections) == 3
    assert sections[0].section_number == "1"
    assert sections[1].section_number == "2"
    assert sections[2].section_number == "3"
