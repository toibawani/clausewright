"""Chunker: splits ``Section`` objects into overlapping ``Chunk`` windows.

Design
------
* Word-based windowing is used as a lightweight proxy for token count.
  Typical LLM tokenisers produce ~0.75 tokens per word for English text,
  so a ``max_words=300`` window ≈ 400 tokens — well within the limits of
  most embedding models (512–8192 tokens).

* Overlap via ``stride < max_words``.  When ``stride >= max_words`` there
  is no overlap and chunks are disjoint.

* Each ``Chunk`` records the *doc_id*, *section_number*, *chunk_index*,
  *page_range*, and *word_count* for full provenance during retrieval.

* Sections whose total word count fits in a single window produce exactly
  one ``Chunk`` (no splitting required).

* Page attribution per chunk is approximated: we track which ``Block``
  each word comes from and report the page range spanned by blocks whose
  words appear in the window.

* A minimum-chunk-size floor prevents retrieving tiny fragments that
  hurt BM25 and dense recall (e.g. section headers, page numbers).

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

# Minimum chunk size in words.  Chunks smaller than this are merged with
# the next window so that every chunk has enough text for reliable
# sparse and dense retrieval.
MIN_CHUNK_WORDS: Final[int] = 80

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

def _build_section_breadcrumbs(section: Section, section_map: dict[str, Section]) -> tuple[str, ...]:
    """Walk up parent_number to construct the hierarchical lineage tuple."""
    crumbs: list[str] = []
    curr: Section | None = section
    visited: set[str] = set()

    while curr is not None and curr.section_number not in visited:
        visited.add(curr.section_number)
        label = f"Section {curr.section_number}"
        if curr.title:
            label = f"{label}: {curr.title}"
        crumbs.append(label)

        if curr.parent_number and curr.parent_number in section_map:
            curr = section_map[curr.parent_number]
        else:
            curr = None

    return tuple(reversed(crumbs))


def _merge_small_chunks(chunks: list[Chunk], min_words: int = MIN_CHUNK_WORDS) -> list[Chunk]:
    """Merge chunks that are smaller than *min_words* with the next chunk.

    This prevents the retriever from seeing many tiny fragments (e.g. section
    headers, page-only blocks) that do not contain enough contextual text for
    BM25 or dense similarity to match well.

    After merging, every chunk has at least *min_words* unless the very last
    chunk of a document is still smaller (kept as-is).
    """
    if not chunks:
        return []

    merged: list[Chunk] = []
    cur = chunks[0]

    for next_c in chunks[1:]:
        if cur.word_count < min_words:
            # Merge next into cur so cur meets the minimum
            cur = Chunk(
                chunk_id=cur.chunk_id,
                doc_id=cur.doc_id,
                section_number=cur.section_number,
                chunk_index=cur.chunk_index,
                text=cur.text + " " + next_c.text,
                page_range=cur.page_range,
                word_count=cur.word_count + next_c.word_count,
                breadcrumbs=cur.breadcrumbs,
                contextual_text=cur.contextual_text,
            )
        else:
            merged.append(cur)
            cur = next_c

    # If the final chunk is still smaller than min_words, try merging it
    # with the previous chunk in the merged list (if any).
    if cur.word_count < min_words and merged:
        prev = merged.pop()
        combined = Chunk(
            chunk_id=prev.chunk_id,
            doc_id=prev.doc_id,
            section_number=prev.section_number,
            chunk_index=prev.chunk_index,
            text=prev.text + " " + cur.text,
            page_range=prev.page_range,
            word_count=prev.word_count + cur.word_count,
            breadcrumbs=prev.breadcrumbs,
            contextual_text=prev.contextual_text,
        )
        merged.append(combined)
    else:
        # Either the final chunk already meets the floor (keep it) or there is
        # no previous chunk to merge a short document into (keep it as-is).
        merged.append(cur)

    return merged


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
    doc_id : str
        Identifier of the source document; embedded in each ``Chunk.chunk_id``.
    sections : list[Section]
        Ordered list of sections as produced by
        :func:`clausewright.ingestion.section_parser.parse_sections`.
    max_words : int, default 300
        Maximum number of words per chunk.  Acts as a proxy for token count
        (≈ 0.75 × ``max_words`` tokens for English text).
    stride : int, default 150
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
    section_map = {s.section_number: s for s in sections}

    for section in sections:
        words, spans = _annotate_words(section)
        total_words = len(words)

        if total_words == 0:
            # Section has no text content — skip silently
            continue

        breadcrumbs = _build_section_breadcrumbs(section, section_map)

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
            contextual_text = f"[{' > '.join(breadcrumbs)}]\n{window_text}" if breadcrumbs else window_text

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_number=section.section_number,
                    chunk_index=chunk_index,
                    text=window_text,
                    page_range=(page_start, page_end),
                    word_count=len(window_words),
                    breadcrumbs=breadcrumbs,
                    contextual_text=contextual_text,
                )
            )

            chunk_index += 1

            # If we've reached the end of the section, stop
            if window_end >= total_words:
                break

            window_start += stride

    # Merge chunks that are too small to be useful for retrieval
    chunks = _merge_small_chunks(chunks, min_words=MIN_CHUNK_WORDS)

    return chunks