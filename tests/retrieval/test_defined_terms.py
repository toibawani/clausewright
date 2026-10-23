"""Unit tests for defined terms resolution and chunk attachment (closes #25)."""

from __future__ import annotations

from clausewright.retrieval.defined_terms import (
    DefinedTermsExpander,
    DefinedTermsRetriever,
    DefinitionsIndex,
    extract_candidate_terms,
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
# Extraction tests
# ---------------------------------------------------------------------------


def test_extract_candidate_terms():
    text = (
        "Either party may terminate this Agreement if a Material Adverse Effect occurs, "
        "or upon a Change of Control."
    )
    terms = extract_candidate_terms(text)
    assert "Material Adverse Effect" in terms
    assert "Change of Control" in terms
    # Stopwords should not be extracted
    assert "Either" not in terms
    assert "Agreement" not in terms


def test_extract_quoted_terms():
    text = 'The Receiving Party shall safeguard all "Confidential Information" received.'
    terms = extract_candidate_terms(text)
    assert "Confidential Information" in terms


# ---------------------------------------------------------------------------
# DefinitionsIndex tests
# ---------------------------------------------------------------------------


def test_definitions_index_lookup():
    c_def = _make_chunk(
        "c_def_1",
        'Section 1.1: "Material Adverse Effect" means any event causing a decrease in revenues > 25%.',
        section_number="1.1",
    )
    c_def_2 = _make_chunk(
        "c_def_2",
        "Section 1.2: Change of Control means the transfer of more than 50% voting stock.",
        section_number="1.2",
    )

    index = DefinitionsIndex([c_def, c_def_2])
    res_mae = index.lookup("Material Adverse Effect")
    assert res_mae is not None
    assert res_mae.chunk_id == "c_def_1"

    res_coc = index.lookup("Change of Control")
    assert res_coc is not None
    assert res_coc.chunk_id == "c_def_2"

    assert index.lookup("NonExistentTerm") is None


# ---------------------------------------------------------------------------
# DefinedTermsExpander tests
# ---------------------------------------------------------------------------


def test_expander_appends_definition_chunk():
    c_def = _make_chunk(
        "c_def",
        "Section 1.1: Material Adverse Effect means any event causing a decrease in consolidated revenues of greater than 25%.",
        section_number="1.1",
    )
    c_substantive = _make_chunk(
        "c_term",
        "Section 9.2: Either party may terminate if a Material Adverse Effect occurs.",
        section_number="9.2",
    )

    expander = DefinedTermsExpander([c_def, c_substantive])
    candidates = [(c_substantive, 0.9)]

    expanded = expander.expand(candidates)
    assert len(expanded) == 2
    # Substantive chunk stays first, definition chunk is appended
    assert expanded[0][0].chunk_id == "c_term"
    assert expanded[1][0].chunk_id == "c_def"
    assert expanded[1][1] == 0.9 * 0.95


def test_expander_disabled_returns_unchanged():
    c_def = _make_chunk("c_def", 'Section 1.1: "Term" means something.', section_number="1.1")
    c_sub = _make_chunk("c_sub", 'Section 2.1: Governed by "Term".', section_number="2.1")

    expander = DefinedTermsExpander([c_def, c_sub], enabled=False)
    candidates = [(c_sub, 0.85)]

    expanded = expander.expand(candidates)
    assert expanded == candidates


def test_expander_does_not_duplicate_existing_candidate():
    c_def = _make_chunk("c_def", "Section 1.1: Material Adverse Effect means ...", section_number="1.1")
    c_sub = _make_chunk("c_sub", "Section 9.2: Triggers on Material Adverse Effect.", section_number="9.2")

    expander = DefinedTermsExpander([c_def, c_sub])
    # Both are already in candidate set
    candidates = [(c_sub, 0.9), (c_def, 0.8)]

    expanded = expander.expand(candidates)
    assert len(expanded) == 2
    ids = [c.chunk_id for c, _ in expanded]
    assert ids == ["c_sub", "c_def"]


# ---------------------------------------------------------------------------
# DefinedTermsRetriever wrapper tests
# ---------------------------------------------------------------------------


def test_defined_terms_retriever():
    c_def = _make_chunk("c_def", "Section 1.1: Change of Control means ...", section_number="1.1")
    c_sub = _make_chunk("c_sub", "Section 14: Change of Control triggers option acceleration.", section_number="14")

    base = _StubRetriever([(c_sub, 0.92)])
    retriever = DefinedTermsRetriever(base, chunks=[c_def, c_sub])

    results = retriever.search("option acceleration", top_k=5)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c_sub"
    assert results[1][0].chunk_id == "c_def"
