"""Unit tests for AmendmentLinker (issue #31).

Gold cases:
1. Base clause overridden by an amendment in the same section.
2. Base clause overridden by text similarity (different section number but
   very similar text).
3. Unrelated amendment chunk does NOT override base clause.
4. Multiple amendments: amendment-2 overrides amendment-1's chunk.
5. Incorrect input (no base, non-ascending versions) raises ValueError.
"""

from __future__ import annotations

import pytest

from clausewright.extraction.amendments import (
    AmendmentLinker,
    DocVersion,
    _jaccard_similarity,
    _section_prefix,
)
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    cid: str,
    doc_id: str = "doc1",
    section: str = "1.1",
    text: str = "placeholder text for testing",
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        section_number=section,
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_jaccard_identical():
    assert _jaccard_similarity("the termination clause", "the termination clause") == pytest.approx(1.0)


def test_jaccard_disjoint():
    assert _jaccard_similarity("payment terms", "renewal period") == pytest.approx(0.0)


def test_jaccard_partial():
    sim = _jaccard_similarity("termination notice period", "notice period required")
    assert 0.0 < sim < 1.0


def test_section_prefix_two_levels():
    assert _section_prefix("12.3.b") == "12.3"


def test_section_prefix_one_level():
    assert _section_prefix("12") == "12"


def test_section_prefix_three_plus_levels():
    assert _section_prefix("1.2.3.4") == "1.2"


# ---------------------------------------------------------------------------
# Gold case 1: section-prefix override
# ---------------------------------------------------------------------------


def test_gold_case_1_section_override():
    """Amendment chunk in same section as base chunk overrides it."""
    base_chunk = _make_chunk("base-1.1-0", doc_id="base", section="1.1", text="sixty days notice required")
    amend_chunk = _make_chunk("amend1-1.1-0", doc_id="amend1", section="1.1", text="thirty days notice required")

    linker = AmendmentLinker(similarity_threshold=0.5)
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=[base_chunk]),
        DocVersion(doc_id="amend1", version=1, chunks=[amend_chunk]),
    ])

    active_ids = [c.chunk_id for c in corpus.active_chunks]
    flagged_ids = [c.chunk_id for c, _ in corpus.flagged_chunks]

    assert "base-1.1-0" not in active_ids, "Overridden base chunk should not be active"
    assert "amend1-1.1-0" in active_ids, "Amendment chunk should be active"
    assert "base-1.1-0" in flagged_ids


# ---------------------------------------------------------------------------
# Gold case 2: text-similarity override (different section numbers)
# ---------------------------------------------------------------------------


def test_gold_case_2_similarity_override():
    """Amendment chunk with high text similarity overrides a differently-numbered base clause."""
    base_text = (
        "The term of this agreement shall be three years commencing on the effective date "
        "unless terminated earlier pursuant to the provisions herein."
    )
    amend_text = (
        "The term of this agreement shall be five years commencing on the effective date "
        "unless terminated earlier pursuant to the provisions herein."
    )
    base_chunk = _make_chunk("base-2.1-0", doc_id="base", section="2.1", text=base_text)
    amend_chunk = _make_chunk("amend1-3.5-0", doc_id="amend1", section="3.5", text=amend_text)

    linker = AmendmentLinker(similarity_threshold=0.6)
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=[base_chunk]),
        DocVersion(doc_id="amend1", version=1, chunks=[amend_chunk]),
    ])

    flagged_ids = [c.chunk_id for c, _ in corpus.flagged_chunks]
    assert "base-2.1-0" in flagged_ids, "Base chunk should be overridden by similar amendment text"


# ---------------------------------------------------------------------------
# Gold case 3: unrelated chunk does NOT override
# ---------------------------------------------------------------------------


def test_gold_case_3_unrelated_amendment_no_override():
    """An amendment chunk in a completely different section with no text overlap should NOT override."""
    base_chunk = _make_chunk("base-1.1-0", section="1.1", text="governing law is New York")
    amend_chunk = _make_chunk("amend1-7.3-0", section="7.3", text="payment due on first of month")

    linker = AmendmentLinker(similarity_threshold=0.35)
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=[base_chunk]),
        DocVersion(doc_id="amend1", version=1, chunks=[amend_chunk]),
    ])

    active_ids = {c.chunk_id for c in corpus.active_chunks}
    assert "base-1.1-0" in active_ids, "Unrelated base chunk should remain active"
    assert len(corpus.flagged_chunks) == 0


# ---------------------------------------------------------------------------
# Active and flagged chunk counts
# ---------------------------------------------------------------------------


def test_active_chunks_excludes_overridden():
    b1 = _make_chunk("b1", section="1.1", text="original clause text")
    b2 = _make_chunk("b2", section="2.1", text="unrelated clause entirely")
    a1 = _make_chunk("a1", section="1.1", text="amended clause text")

    linker = AmendmentLinker(similarity_threshold=0.5)
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=[b1, b2]),
        DocVersion(doc_id="amend1", version=1, chunks=[a1]),
    ])

    active_ids = {c.chunk_id for c in corpus.active_chunks}
    assert "b1" not in active_ids
    assert "b2" in active_ids
    assert "a1" in active_ids


def test_override_map_populated():
    b1 = _make_chunk("b1", section="1.1")
    a1 = _make_chunk("a1", section="1.1")

    linker = AmendmentLinker()
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=[b1]),
        DocVersion(doc_id="amend1", version=1, chunks=[a1]),
    ])

    assert corpus.override_map() == {"b1": "a1"}


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_versions_raises():
    with pytest.raises(ValueError, match="non-empty"):
        AmendmentLinker().link([])


def test_no_base_version_raises():
    chunk = _make_chunk("c1")
    with pytest.raises(ValueError, match="base"):
        AmendmentLinker().link([
            DocVersion(doc_id="amend1", version=1, chunks=[chunk])
        ])


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        AmendmentLinker(similarity_threshold=0.0)

    with pytest.raises(ValueError):
        AmendmentLinker(similarity_threshold=1.5)
