"""Baseline chunker: fixed-size windows with no structure awareness.

The product chunker splits each detected section into windows.  This baseline
flattens the whole document into one pseudo-section and runs the *same*
``chunk_sections`` machinery over it — so the ONLY difference under test is
the section hierarchy prior, not window size, overlap, tokenizer, or text:
identical 300-word windows at 50% overlap on identical input text.

A window chunk carries ``section_number="window"`` (honest provenance: it has
no section) and chunks across section and page boundaries, which is exactly
what we want to penalize in the comparison.
"""

from __future__ import annotations

from clausewright.ingestion.chunker import DEFAULT_MAX_WORDS, DEFAULT_STRIDE, chunk_sections
from clausewright.schemas import Chunk, ParsedDocument, Section

_BASELINE_SECTION: str = "window"


def fixed_window_chunks(
    doc_id: str,
    doc: ParsedDocument,
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    stride: int = DEFAULT_STRIDE,
) -> list[Chunk]:
    """Chunk the whole document as fixed windows, ignoring section breaks.

    Raises
    ------
    ValueError
        If the document has no text blocks at all.
    """
    blocks = [b for page in doc.pages for b in page.blocks]
    if not blocks:
        raise ValueError(f"document {doc_id} has no text blocks to chunk")
    pages = [b.page_number for b in blocks]
    mega = Section(
        section_number=_BASELINE_SECTION,
        blocks=blocks,
        page_range=(min(pages), max(pages)),
    )
    return chunk_sections(doc_id, [mega], max_words=max_words, stride=stride)
