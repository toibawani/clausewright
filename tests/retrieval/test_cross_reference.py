"""Unit tests for cross-reference normalization and expansion (closes #24)."""

from __future__ import annotations

from clausewright.retrieval.cross_reference import (
    CrossReferenceExpander,
    CrossReferenceRetriever,
    extract_cross_references,
    normalize_reference,
)
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    text: str,
    section_number: str = "1",
    doc_id: str = "doc_1",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_number=section_number,
        chunk_index=0,
        page_range=(1, 1),
        text=text,
        word_count=max(1, len(text.split())),
    )


class _StubRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        return self.results[:top_k]


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_normalize_reference_section():
    assert normalize_reference("sec. 12.3") == "Section 12.3"
    assert normalize_reference("Section 12.3(b)") == "Section 12.3(b)"
    assert normalize_reference("SECTION  8.4") == "Section 8.4"


def test_normalize_reference_clause_article():
    assert normalize_reference("clause 4.1") == "Clause 4.1"
    assert normalize_reference("article IV") == "Article IV"
    assert normalize_reference("Article 5.2") == "Article 5.2"


def test_normalize_reference_exhibit_schedule():
    assert normalize_reference("exhibit a") == "Exhibit A"
    assert normalize_reference("Schedule  b") == "Schedule B"
    assert normalize_reference("attachment 1") == "Attachment 1"
    assert normalize_reference("appendix c") == "Appendix C"


def test_extract_cross_references_multiple():
    text = (
        "Section 14.2 states liability is capped at 12 months fees, subject to "
        "Section 8.4 and in accordance with Schedule B."
    )
    refs = extract_cross_references(text)
    canonicals = [r.canonical for r in refs]

    assert "Section 14.2" in canonicals
    assert "Section 8.4" in canonicals
    assert "Schedule B" in canonicals


def test_extract_cross_references_deduplicated():
    text = "Subject to Section 8.4 and Section 8.4 again."
    refs = extract_cross_references(text)
    assert len(refs) == 1
    assert refs[0].canonical == "Section 8.4"


# ---------------------------------------------------------------------------
# Expander tests
# ---------------------------------------------------------------------------


def test_expander_prepends_referenced_chunk():
    c_14_2 = _make_chunk(
        "c_14_2",
        "Section 14.2 states liability is capped at 12 months fees, subject to Section 8.4.",
        section_number="14.2",
    )
    c_8_4 = _make_chunk(
        "c_8_4",
        "Section 8.4 states indemnification obligations for IP infringement are uncapped.",
        section_number="8.4",
    )

    expander = CrossReferenceExpander([c_14_2, c_8_4])
    candidates = [(c_14_2, 0.9)]

    expanded = expander.expand(candidates)
    assert len(expanded) == 2
    # c_8_4 is prepended before c_14_2
    assert expanded[0][0].chunk_id == "c_8_4"
    assert expanded[1][0].chunk_id == "c_14_2"
    assert expanded[0][1] == 0.9 * 0.95


def test_expander_disabled_returns_unchanged():
    c_14_2 = _make_chunk(
        "c_14_2",
        "Liability is subject to Section 8.4.",
        section_number="14.2",
    )
    c_8_4 = _make_chunk("c_8_4", "Indemnity is uncapped.", section_number="8.4")

    expander = CrossReferenceExpander([c_14_2, c_8_4], enabled=False)
    candidates = [(c_14_2, 0.9)]

    expanded = expander.expand(candidates)
    assert expanded == candidates


def test_expander_does_not_duplicate_existing_candidate():
    c_1 = _make_chunk("c_1", "Subject to Section 2.", section_number="1")
    c_2 = _make_chunk("c_2", "Details of obligations.", section_number="2")

    expander = CrossReferenceExpander([c_1, c_2])
    # Both are already in candidates
    candidates = [(c_1, 0.9), (c_2, 0.8)]

    expanded = expander.expand(candidates)
    assert len(expanded) == 2
    chunk_ids = [c.chunk_id for c, _ in expanded]
    assert chunk_ids == ["c_1", "c_2"]


def test_expander_circular_reference_does_not_loop():
    c_a = _make_chunk("c_a", "Subject to Section 2.", section_number="1")
    c_b = _make_chunk("c_b", "Subject to Section 1.", section_number="2")

    expander = CrossReferenceExpander([c_a, c_b])
    candidates = [(c_a, 0.9)]

    expanded = expander.expand(candidates)
    # c_b is prepended once, c_a is present once
    assert len(expanded) == 2
    ids = [c.chunk_id for c, _ in expanded]
    assert "c_a" in ids
    assert "c_b" in ids


def test_expander_schedule_lookup():
    c_main = _make_chunk("c_main", "Fees per Schedule B.", section_number="5.1")
    c_sched = _make_chunk("c_sched", "Schedule B: Hourly rate is $500.", section_number="B")

    expander = CrossReferenceExpander([c_main, c_sched])
    expanded = expander.expand([(c_main, 0.85)])

    assert len(expanded) == 2
    assert expanded[0][0].chunk_id == "c_sched"
    assert expanded[1][0].chunk_id == "c_main"


# ---------------------------------------------------------------------------
# Retriever wrapper tests
# ---------------------------------------------------------------------------


def test_cross_reference_retriever():
    c_1 = _make_chunk("c_1", "Subject to Section 2.", section_number="1")
    c_2 = _make_chunk("c_2", "Section 2 contents.", section_number="2")

    base = _StubRetriever([(c_1, 0.9)])
    retriever = CrossReferenceRetriever(base, chunks=[c_1, c_2])

    results = retriever.search("query", top_k=5)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c_2"
    assert results[1][0].chunk_id == "c_1"
