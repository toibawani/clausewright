"""Chunker: splits ``Section`` objects into overlapping ``Chunk`` windows.

Design
------
*  Word-based windowing is used as a lightweight proxy for token count.
   Typical LLM tokenisers produce ~0.75 tokens per word for English text,
   so a ``max_words=300`` window ≈ 400 tokens — well within the limits of
   most embedding models (512–8192 tokens).

*  Overlap via ``stride < max_words``.  When ``stride >= max_words`` there
   is no overlap and chunks are disjoint.

*  Each ``Chunk`` records the *doc_id*, *section_number*, *chunk_index*,
   *page_range*, and *word_count* for full provenance during retrieval.

*  Sections whose total word count fits in a single window produce exactly
   one ``Chunk`` (no splitting required).

*  Page attribution per chunk is approximated: we track which ``Block``
   each word comes from and report the page range spanned by blocks whose
   words appear in the window.

Public API
----------
``chunk_sections(doc_id, sections, *, max_words, stride) -> list[Chunk]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from clausewright.schemas import Block, Chunk, Section

# Default window / stride sizes (in words)
DEFAULT_MAX_WORDS: Final[int] = 300
DEFAULT_STRIDE: Final[int] = 150  # 50 % overlap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _WordSpan:
    """Maps a range of word positions back to the originating Block."""

    start: int  # inclusive
    end: int    # exclusive
    block: Block


def _annotate_words(section: Section) -> tuple[list[str], list[_WordSpan]]:
    """Return (word_list, span_list) for all words in a section's blocks.

    ``word_list`` is the flat sequence of whitespace-split tokens.
    ``span_list`` records which block each contiguous range of words came from.
    """
    words: list[str] = []
    spans: list[_WordSpan] = []
    pos = 0
    for block in section.blocks:
        block_words = block.text.split()
        if not block_words:
            continue
        end = pos + len(block_words)
        spans.append(_WordSpan(start=pos, end=end, block=block))
        words.extend(block_words)
        pos = end
    return words, spans


def _page_range_for_window(
    start_word: int, end_word: int, spans: list[_WordSpan]
) -> tuple[int, int]:
    """Return ``(min_page, max_page)`` for words in ``[start_word, end_word)``."""
    pages: list[int] = []
    for span in spans:
        # Check for overlap between [start_word, end_word) and [span.start, span.end)
        if span.end <= start_word:
            continue
        if span.start >= end_word:
            break
        pages.append(span.block.page_number)
    if not pages:
        return (1, 1)
    return (min(pages), max(pages))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_sections(
    doc_id: str,
    sections: list[Section],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    stride: int = DEFAULT_STRIDE,
) -> list[Chunk]:
    """Convert a list of ``Section`` objects into overlapping ``Chunk`` windows.

    Parameters
    ----------
    doc_id:
        Identifier of the source document; embedded in each ``Chunk.chunk_id``.
    sections:
        Ordered list of sections as produced by
        :func:`clausewright.ingestion.section_parser.parse_sections`.
    max_words:
        Maximum number of words per chunk.  Acts as a proxy for token count
        (≈ 0.75 × ``max_words`` tokens for English text).
    stride:
        Number of words to advance the window at each step.  Must be
        ``1 <= stride <= max_words``.  Setting ``stride == max_words``
        produces non-overlapping chunks.

    Returns
    -------
    list[Chunk]
        All chunks across all sections, in section → window order.

    Raises
    ------
    ValueError
        If ``max_words < 1`` or ``stride < 1`` or ``stride > max_words``.
    """
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > max_words:
        raise ValueError(f"stride ({stride}) cannot exceed max_words ({max_words})")

    chunks: list[Chunk] = []

    for section in sections:
        words, spans = _annotate_words(section)
        total_words = len(words)

        if total_words == 0:
            # Section has no text content — skip silently
            continue

        chunk_index = 0
        window_start = 0

        while window_start < total_words:
            window_end = min(window_start + max_words, total_words)
            window_words = words[window_start:window_end]
            window_text = " ".join(window_words)
            page_start, page_end = _page_range_for_window(window_start, window_end, spans)

            # Sanitize section number for use in chunk_id (replace dots/spaces)
            safe_section = section.section_number.replace(".", "_").replace(" ", "_")
            chunk_id = f"{doc_id}-{safe_section}-{chunk_index}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_number=section.section_number,
                    chunk_index=chunk_index,
                    text=window_text,
                    page_range=(page_start, page_end),
                    word_count=len(window_words),
                )
            )

            chunk_index += 1

            # If we've reached the end of the section, stop
            if window_end >= total_words:
                break

            window_start += stride

    return chunks
