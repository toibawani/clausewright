"""Section parser: converts a ``ParsedDocument`` into a flat list of ``Section`` objects.

Strategy
--------
1. Iterate every ``Block`` across all pages in document order.
2. Classify each block as a *heading* or *body* using:
   - A regex that matches common contract numbering patterns such as
     ``1``, ``1.``, ``1.2``, ``1.2.3``, ``Article I``, ``Section 1``, etc.
   - An optional font-size / bold heuristic as a secondary signal.
3. On each heading detection, close the current open section and open a new one.
4. Determine ``level`` from the depth of the numbering token (dots + 1) or from
   explicit keywords (``Article`` → 1, ``Section`` → 2).
5. Derive ``parent_number`` by stripping the last dotted component.
6. Build ``page_range`` from the first and last ``Block.page_number`` in the section.
"""

from __future__ import annotations

import re
from typing import Final

from clausewright.schemas import Block, ParsedDocument, Section

# ---------------------------------------------------------------------------
# Heading-detection patterns
# ---------------------------------------------------------------------------

# Numeric heading: optional leading keyword, then a dotted number, then optional dot
# Examples: "1", "1.", "1.2", "1.2.3", "Section 1", "Section 1.2", "Article 3"
_NUMERIC: Final = re.compile(
    r"""
    ^\s*
    (?:(?P<keyword>article|section)\s+)?   # optional keyword
    (?P<number>\d+(?:\.\d+)*)              # dotted number  e.g. 1, 1.2, 1.2.3
    \.?                                    # optional trailing dot
    (?:\s+(?P<title>.+))?                  # optional title text after the number
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Roman-numeral article headings: "Article I", "Article IV", etc.
_ROMAN: Final = re.compile(
    r"""
    ^\s*
    (?P<keyword>article)\s+
    (?P<roman>[IVXLCDM]+)                  # Roman numeral
    (?:\.)?                                # optional trailing dot
    (?:\s+(?P<title>.+))?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Minimum font size ratio relative to body average to treat a block as a heading
# (only used when the text does NOT match the regex).
_FONT_RATIO_THRESHOLD: Final[float] = 1.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer."""
    vals: dict[str, int] = {
        "I": 1, "V": 5, "X": 10, "L": 50,
        "C": 100, "D": 500, "M": 1000,
    }
    result = 0
    prev = 0
    for ch in reversed(s.upper()):
        cur = vals.get(ch, 0)
        result += cur if cur >= prev else -cur
        prev = cur
    return result


def _derive_level_and_parent(section_number: str) -> tuple[int, str | None]:
    """Return ``(level, parent_number)`` from a dotted section number like ``1.2.3``."""
    parts = section_number.split(".")
    level = len(parts)
    parent = ".".join(parts[:-1]) if level > 1 else None
    return level, parent


def _average_font_size(doc: ParsedDocument) -> float:
    """Return the average font size across all blocks in the document."""
    sizes = [b.font_size for p in doc.pages for b in p.blocks]
    return sum(sizes) / len(sizes) if sizes else 12.0


def _match_heading(block: Block) -> tuple[str, str | None] | None:
    """Return ``(section_number, title_or_None)`` if *block* looks like a heading.

    Returns ``None`` if the block is not a heading.
    """
    text = block.text.strip()
    # Try numeric pattern first
    m = _NUMERIC.match(text)
    if m:
        number = m.group("number")
        title = (m.group("title") or "").strip() or None
        keyword = (m.group("keyword") or "").lower()
        # Prefix keyword-based numbers so they sort cleanly: "A1", "S1.2", etc.
        if keyword == "article":
            section_number = f"A{number}"
        elif keyword == "section":
            section_number = f"S{number}"
        else:
            section_number = number
        return section_number, title

    # Try Roman numeral article pattern
    m2 = _ROMAN.match(text)
    if m2:
        roman = m2.group("roman").upper()
        int_val = _roman_to_int(roman)
        section_number = f"A{int_val}"
        title = (m2.group("title") or "").strip() or None
        return section_number, title

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sections(
    doc: ParsedDocument,
    *,
    use_font_heuristic: bool = True,
    min_heading_words: int = 1,
    max_heading_words: int = 20,
) -> list[Section]:
    """Parse a ``ParsedDocument`` into an ordered flat list of ``Section`` objects.

    Parameters
    ----------
    doc:
        The parsed document produced by :mod:`clausewright.ingestion.pdf_loader`.
    use_font_heuristic:
        When ``True``, blocks that are bold **or** significantly larger than the
        document average font size are also treated as headings even without a
        matching numbering pattern.  The detected heading gets an auto-generated
        section number of the form ``H<n>``.
    min_heading_words:
        Minimum number of whitespace-separated tokens for a block to qualify as a
        font-heuristic heading (avoids single punctuation marks).
    max_heading_words:
        Maximum number of tokens.  Very long lines are almost certainly body text.

    Returns
    -------
    list[Section]
        Sections in document order.  The first ``Section`` captures any blocks
        that appear before the first detected heading under the synthetic number
        ``"PREAMBLE"``.
    """
    if not doc.pages:
        return []

    avg_font = _average_font_size(doc) if use_font_heuristic else 0.0

    # --- walk all blocks in document order ----------------------------------
    # Each pending section is stored as (section_number, title, level, parent, [blocks])
    pending: list[tuple[str, str | None, int, str | None, list[Block]]] = []
    current_blocks: list[Block] = []
    current_number: str = "PREAMBLE"
    current_title: str | None = None
    current_level: int = 1
    current_parent: str | None = None
    auto_heading_counter: int = 0

    def _flush() -> None:
        """Append the currently open section to *pending* (even if empty)."""
        pending.append(
            (current_number, current_title, current_level, current_parent, current_blocks[:])
        )

    for page in doc.pages:
        for block in page.blocks:
            heading_match = _match_heading(block)

            if heading_match is None and use_font_heuristic:
                word_count = len(block.text.split())
                is_large = block.font_size >= avg_font * _FONT_RATIO_THRESHOLD
                if (block.is_bold or is_large) and min_heading_words <= word_count <= max_heading_words:
                    auto_heading_counter += 1
                    heading_match = (f"H{auto_heading_counter}", block.text.strip())

            if heading_match is not None:
                _flush()
                sec_num, sec_title = heading_match
                level, parent = _derive_level_and_parent(sec_num)
                current_number = sec_num
                current_title = sec_title
                current_level = level
                current_parent = parent
                current_blocks = []
                # The heading block itself belongs to the new section
                current_blocks.append(block)
            else:
                current_blocks.append(block)

    # Flush the last open section
    _flush()

    # --- build Section objects ----------------------------------------------
    sections: list[Section] = []
    for sec_num, sec_title, level, parent, blocks in pending:
        if not blocks and sec_num == "PREAMBLE":
            continue

        start_page = blocks[0].page_number if blocks else 1
        end_page = blocks[-1].page_number if blocks else start_page

        sections.append(
            Section(
                section_number=sec_num,
                title=sec_title,
                level=level,
                parent_number=parent,
                blocks=blocks,
                page_range=(start_page, end_page),
            )
        )

    return sections
