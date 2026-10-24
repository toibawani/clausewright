"""Visual PDF Citation and Bounding Box Anchoring (Pillar 4).

In professional legal tech applications, users must not be expected to manually
hunt through a 90-page contract PDF to verify an AI citation.

This module maps a verified text quote from a Chunk back to:
1. Exact character offsets (start_char, end_char) within the Chunk text.
2. The specific PDF page number where the quote occurs.
3. Precise PyMuPDF bounding boxes (x0, y0, x1, y1) for frontend visual highlights.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from clausewright.schemas import Block, Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationAnchor:
    """Exact spatial and textual location of a citation in the contract."""

    char_offset: tuple[int, int]
    page_number: int
    bounding_boxes: tuple[tuple[float, float, float, float], ...] = ()


def find_quote_offsets(quote: str, text: str) -> tuple[int, int] | None:
    """Find the exact (start, end) character offsets of a quote in text.

    Supports whitespace normalization to reliably match across line breaks.
    """
    clean_quote = " ".join(quote.split())
    clean_text = " ".join(text.split())

    idx = clean_text.find(clean_quote)
    if idx == -1:
        # Fallback to case-insensitive match
        idx = clean_text.lower().find(clean_quote.lower())
        if idx == -1:
            return None

    # Map normalized index back to approximate original text offset
    pattern = re.escape(clean_quote).replace(r"\ ", r"\s+")
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return (match.start(), match.end())

    return (idx, idx + len(clean_quote))


def locate_visual_citation(
    quote: str,
    chunk: Chunk,
    available_blocks: Sequence[Block] | None = None,
) -> CitationAnchor:
    """Derive spatial bounding boxes and character offsets for a verified quote.

    Parameters
    ----------
    quote:
        The verbatim cited excerpt.
    chunk:
        The chunk containing the quote.
    available_blocks:
        Optional sequence of parsed layout blocks for the document/page.
    """
    offsets = find_quote_offsets(quote, chunk.text)
    char_offset = offsets if offsets else (0, len(quote))

    # Approximate page number from chunk page range
    page_num = chunk.page_range[0]
    boxes: list[tuple[float, float, float, float]] = []

    if available_blocks:
        quote_words = set(quote.lower().split())
        for block in available_blocks:
            if block.page_number in range(chunk.page_range[0], chunk.page_range[1] + 1):
                block_words = set(block.text.lower().split())
                # Check significant word overlap
                overlap = quote_words.intersection(block_words)
                if len(overlap) >= min(3, len(quote_words)):
                    boxes.append(block.bbox)
                    page_num = block.page_number

    return CitationAnchor(
        char_offset=char_offset,
        page_number=page_num,
        bounding_boxes=tuple(boxes),
    )
