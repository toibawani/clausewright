"""Tests for document, page, and block schemas and their validation rules."""

import pytest
from pydantic import ValidationError

from clausewright.schemas import Block, Page, ParsedDocument


def make_block(page_number: int = 1, **overrides: object) -> Block:
    fields: dict[str, object] = {
        "text": "1. Termination",
        "page_number": page_number,
        "bbox": (72.0, 100.0, 300.0, 120.0),
        "font_size": 12.0,
        "is_bold": True,
    }
    fields.update(overrides)
    return Block(**fields)  # type: ignore[arg-type]


def make_page(number: int = 1, **overrides: object) -> Page:
    fields: dict[str, object] = {
        "number": number,
        "width": 612.0,
        "height": 792.0,
        "blocks": [make_block(number)],
    }
    fields.update(overrides)
    return Page(**fields)  # type: ignore[arg-type]


def test_valid_document_builds() -> None:
    doc = ParsedDocument(
        doc_id="acme-msa", source_path="data/raw/acme.pdf", pages=[make_page(1), make_page(2)]
    )
    assert doc.page_count == 2
    assert doc.doc_id == "acme-msa"


def test_empty_document_builds() -> None:
    doc = ParsedDocument(doc_id="empty-doc", source_path="empty.pdf", pages=[])
    assert doc.page_count == 0


@pytest.mark.parametrize(
    "invalid_bbox",
    [
        (300.0, 100.0, 72.0, 120.0),  # x0 > x1
        (72.0, 120.0, 300.0, 100.0),  # y0 > y1
        (72.0, 100.0, 72.0, 120.0),  # x0 == x1 (zero width)
        (72.0, 100.0, 300.0, 100.0),  # y0 == y1 (zero height)
    ],
)
def test_inverted_or_degenerate_bbox_rejected(
    invalid_bbox: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValidationError, match="Inverted or degenerate bounding box"):
        make_block(bbox=invalid_bbox)


def test_block_on_wrong_page_rejected() -> None:
    with pytest.raises(ValidationError, match="Block with page_number=1 assigned to Page 2"):
        Page(number=2, width=612.0, height=792.0, blocks=[make_block(page_number=1)])


def test_page_numbering_gap_rejected() -> None:
    with pytest.raises(ValidationError, match="Page numbers must be sequential"):
        ParsedDocument(doc_id="x", source_path="x.pdf", pages=[make_page(1), make_page(3)])


def test_page_numbering_non_one_start_rejected() -> None:
    with pytest.raises(ValidationError, match="Page numbers must be sequential"):
        ParsedDocument(doc_id="x", source_path="x.pdf", pages=[make_page(2), make_page(3)])


def test_page_numbering_out_of_order_rejected() -> None:
    with pytest.raises(ValidationError, match="Page numbers must be sequential"):
        ParsedDocument(doc_id="x", source_path="x.pdf", pages=[make_page(2), make_page(1)])


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        make_block(text="")


def test_non_positive_font_size_rejected() -> None:
    with pytest.raises(ValidationError):
        make_block(font_size=0.0)
    with pytest.raises(ValidationError):
        make_block(font_size=-2.0)


def test_invalid_page_number_rejected() -> None:
    with pytest.raises(ValidationError):
        make_block(page_number=0)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        make_block(colour="red")


def test_frozen_immutability() -> None:
    block = make_block(1)
    with pytest.raises(ValidationError):
        block.text = "Changed"

