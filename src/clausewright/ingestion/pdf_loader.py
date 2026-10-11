"""Extract text blocks with page numbers and layout hints from a PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from clausewright.schemas import Block, Page, ParsedDocument

_BOLD_FLAG = 16  # PyMuPDF span flag bit for bold


def _block_from_dict(raw: dict[str, Any], page_number: int) -> Block | None:
    """Convert one PyMuPDF text block to a Block, or None if it has no text."""
    lines: list[str] = []
    size_chars: dict[float, int] = {}
    bold_chars = 0
    total_chars = 0

    for line in raw.get("lines", []):
        parts: list[str] = []
        for span in line.get("spans", []):
            text = str(span.get("text", ""))
            parts.append(text)
            n = len(text.strip())
            if n == 0:
                continue
            size = round(float(span.get("size", 10.0)), 1)
            size_chars[size] = size_chars.get(size, 0) + n
            total_chars += n
            flags = int(span.get("flags", 0))
            font = str(span.get("font", "")).lower()
            if (flags & _BOLD_FLAG) or ("bold" in font):
                bold_chars += n
        joined_line = "".join(parts).strip()
        if joined_line:
            lines.append(joined_line)

    text = " ".join(lines)
    if not text or not size_chars:
        return None

    dominant_size = max(size_chars, key=lambda s: size_chars[s])
    bbox_raw = raw.get("bbox", (0.0, 0.0, 0.0, 0.0))
    x0, y0, x1, y1 = (
        float(bbox_raw[0]),
        float(bbox_raw[1]),
        float(bbox_raw[2]),
        float(bbox_raw[3]),
    )
    if x0 >= x1 or y0 >= y1:
        return None

    return Block(
        text=text,
        page_number=page_number,
        bbox=(x0, y0, x1, y1),
        font_size=dominant_size,
        is_bold=bold_chars * 2 > total_chars,
    )


def load_pdf(path: Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse a PDF into pages of text blocks in reading order.

    Raises FileNotFoundError if the file is missing, and ValueError if the
    PDF has no extractable text (for example a scanned PDF; OCR is out of scope).
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    pages: list[Page] = []
    with pymupdf.open(path) as pdf:  # type: ignore[no-untyped-call]
        for index, pdf_page in enumerate(pdf, start=1):
            raw_blocks = pdf_page.get_text("dict", sort=True).get("blocks", [])
            blocks = [
                b
                for raw in raw_blocks
                if raw.get("type") == 0
                if (b := _block_from_dict(raw, index)) is not None
            ]
            pages.append(
                Page(
                    number=index,
                    width=float(pdf_page.rect.width),
                    height=float(pdf_page.rect.height),
                    blocks=blocks,
                )
            )

    if not any(page.blocks for page in pages):
        raise ValueError(f"{path} has no extractable text (scanned PDF?)")

    return ParsedDocument(
        doc_id=doc_id or path.stem,
        source_path=str(path),
        pages=pages,
    )
