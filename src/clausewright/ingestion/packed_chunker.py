"""Packed-section chunker: whole sections packed up to a word budget.

Why this exists
---------------
``chunk_sections`` windows every section independently, so a 40-word section
still becomes its own chunk and a 900-word section is cut in the middle of a
clause. Both behaviours cost either retrieval signal or citation fidelity.

This chunker sits between the two extremes:

* Consecutive sections under the same top-level article are packed into one
  window while the total stays at or below *max_words*.
* A section is split only when it alone exceeds *max_words*; in that case the
  normal windowing machinery handles it.
* A window never crosses a top-level article (or preamble) boundary, so an
  indemnity clause is never glued to a payment clause from a different article.
* The section path of the window's first section is kept as metadata
  (``breadcrumbs``) and the window carries the first section's number.

The result is a chunk list with far fewer fragments than per-section
windowing, without silently merging unrelated parts of the contract.

Public API
----------
``pack_sections(doc_id, sections, *, max_words) -> list[Chunk]``
"""

from __future__ import annotations

from typing import Final

from clausewright.ingestion.chunker import _build_section_breadcrumbs, chunk_sections
from clausewright.schemas import Chunk, Section

#: Default packed window size in words.
DEFAULT_PACKED_WORDS: Final[int] = 400


def _section_words(section: Section) -> int:
    """Word count of a section's text, using the same split as the chunker."""
    return len(section.text.split())


def _root_of(section: Section, section_map: dict[str, Section]) -> str:
    """Return the section number of the top-level ancestor of *section*.

    A section whose ``parent_number`` is absent from *section_map* is treated
    as its own root: the parser can emit parent links to headings it did not
    keep, and a missing parent must not merge two articles.
    """
    current = section
    visited: set[str] = set()
    while (
        current.parent_number
        and current.parent_number in section_map
        and current.parent_number not in visited
    ):
        visited.add(current.parent_number)
        current = section_map[current.parent_number]
    return current.section_number


def _packed_chunk(
    doc_id: str,
    group: list[Section],
    section_map: dict[str, Section],
    index: int,
) -> Chunk:
    """Build one chunk from a run of consecutive sections."""
    text = "\n\n".join(s.text for s in group)
    first = group[0]
    breadcrumbs = _build_section_breadcrumbs(first, section_map)
    safe_section = first.section_number.replace(".", "_").replace(" ", "_")
    pages = [p for s in group for p in s.page_range]
    contextual_text = (
        f"[{' > '.join(breadcrumbs)}]\n{text}" if breadcrumbs else text
    )
    return Chunk(
        chunk_id=f"{doc_id}-packed-{safe_section}-{index}",
        doc_id=doc_id,
        section_number=first.section_number,
        chunk_index=index,
        text=text,
        page_range=(min(pages), max(pages)),
        word_count=len(text.split()),
        breadcrumbs=breadcrumbs,
        contextual_text=contextual_text,
    )


def pack_sections(
    doc_id: str,
    sections: list[Section],
    *,
    max_words: int = DEFAULT_PACKED_WORDS,
) -> list[Chunk]:
    """Pack consecutive sections into windows of at most *max_words* words.

    Parameters
    ----------
    doc_id:
        Identifier of the source document; embedded in each ``chunk_id``.
    sections:
        Ordered sections from :func:`clausewright.ingestion.section_parser.parse_sections`.
    max_words:
        Upper bound on a packed window, in words. A section longer than this
        is split with the per-section winder instead of being packed.

    Returns
    -------
    list[Chunk]
        Chunks in document order, grouped by top-level article.

    Raises
    ------
    ValueError
        If *max_words* < 1.
    """
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")

    section_map = {s.section_number: s for s in sections}

    # Sections that cannot fit in one window are windowed on their own. Passing
    # the whole document here would let the small-chunk merging step glue one
    # section's tail onto the next section, which is the boundary this chunker
    # exists to keep.
    split_by_section: dict[str, list[Chunk]] = {}
    for section in sections:
        if _section_words(section) <= max_words:
            continue
        for chunk in chunk_sections(
            doc_id, [section], max_words=max_words, stride=max(1, max_words // 2)
        ):
            split_by_section.setdefault(section.section_number, []).append(chunk)

    chunks: list[Chunk] = []
    buffer: list[Section] = []
    buffer_words = 0
    buffer_root: str | None = None
    index_in_root: dict[str, int] = {}

    def flush() -> None:
        nonlocal buffer, buffer_words, buffer_root
        if not buffer:
            return
        index = index_in_root.get(buffer_root or "", 0)
        chunks.append(_packed_chunk(doc_id, buffer, section_map, index))
        index_in_root[buffer_root or ""] = index + 1
        buffer = []
        buffer_words = 0
        buffer_root = None

    for section in sections:
        words = _section_words(section)
        if words == 0:
            continue
        root = _root_of(section, section_map)

        if words > max_words:
            flush()
            chunks.extend(split_by_section.get(section.section_number, []))
            continue

        if buffer and (root != buffer_root or buffer_words + words > max_words):
            flush()

        buffer.append(section)
        buffer_words += words
        buffer_root = root

    flush()
    return chunks
