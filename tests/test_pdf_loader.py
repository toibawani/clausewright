"""Tests for PDF text and layout extraction."""

from pathlib import Path

import pymupdf
import pytest

from clausewright.ingestion import load_pdf


def make_pdf(path: Path) -> Path:
    """Generate a 2-page test PDF with styled text blocks."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 100), "1. Termination", fontsize=16, fontname="hebo")
    page1.insert_text(
        (72, 140), "Either party may terminate on notice.", fontsize=11, fontname="helv"
    )
    page2 = doc.new_page()
    page2.insert_text((72, 100), "2. Payment", fontsize=16, fontname="hebo")
    doc.save(path)
    doc.close()
    return path


def test_pages_and_numbers(tmp_path: Path) -> None:
    parsed = load_pdf(make_pdf(tmp_path / "c.pdf"))
    assert parsed.page_count == 2
    assert [p.number for p in parsed.pages] == [1, 2]
    assert parsed.doc_id == "c"


def test_heading_has_larger_bold_font_than_body(tmp_path: Path) -> None:
    parsed = load_pdf(make_pdf(tmp_path / "c.pdf"))
    heading, body = parsed.pages[0].blocks[0], parsed.pages[0].blocks[1]
    assert heading.text == "1. Termination"
    assert heading.font_size > body.font_size
    assert heading.is_bold
    assert not body.is_bold


def test_blocks_in_reading_order(tmp_path: Path) -> None:
    parsed = load_pdf(make_pdf(tmp_path / "c.pdf"))
    ys = [b.bbox[1] for b in parsed.pages[0].blocks]
    assert ys == sorted(ys)


def test_custom_doc_id(tmp_path: Path) -> None:
    parsed = load_pdf(make_pdf(tmp_path / "c.pdf"), doc_id="acme-msa")
    assert parsed.doc_id == "acme-msa"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_pdf_without_text_raises(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError, match="no extractable text"):
        load_pdf(path)


def test_pdf_with_drawings_only_raises(tmp_path: Path) -> None:
    path = tmp_path / "drawings.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError, match="no extractable text"):
        load_pdf(path)


def test_partially_blank_pages_succeed(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    doc.new_page()  # Page 1: blank divider
    p2 = doc.new_page()  # Page 2: text
    p2.insert_text((72, 100), "Clause 1", fontsize=12)
    doc.save(path)
    doc.close()
    parsed = load_pdf(path)
    assert parsed.page_count == 2
    assert len(parsed.pages[0].blocks) == 0
    assert len(parsed.pages[1].blocks) == 1
