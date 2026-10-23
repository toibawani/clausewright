"""Unit tests for MetadataFilter (issue #28)."""

from __future__ import annotations

from clausewright.retrieval.metadata_filter import ChunkMeta, MetadataFilter
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(cid: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc1",
        section_number="1.1",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


class _StubRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# No filters applied — passthrough
# ---------------------------------------------------------------------------


def test_no_filter_returns_all():
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    retriever = _StubRetriever([(c, float(5 - i)) for i, c in enumerate(chunks)])
    meta_filter = MetadataFilter(base_retriever=retriever)
    results = meta_filter.search("query", top_k=10)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# section_type filter
# ---------------------------------------------------------------------------


def test_section_type_filter_keeps_matches():
    c_term = _make_chunk("c_term")
    c_pay = _make_chunk("c_pay")
    retriever = _StubRetriever([(c_term, 0.9), (c_pay, 0.8)])
    chunk_meta = {
        "c_term": ChunkMeta(section_type="termination"),
        "c_pay": ChunkMeta(section_type="payment"),
    }
    meta_filter = MetadataFilter(base_retriever=retriever, chunk_meta=chunk_meta)
    results = meta_filter.search("query", top_k=10, section_type="termination")
    ids = [c.chunk_id for c, _ in results]
    assert "c_term" in ids
    assert "c_pay" not in ids


def test_section_type_filter_none_match_returns_empty():
    c1 = _make_chunk("c1")
    retriever = _StubRetriever([(c1, 0.9)])
    chunk_meta = {"c1": ChunkMeta(section_type="payment")}
    meta_filter = MetadataFilter(base_retriever=retriever, chunk_meta=chunk_meta)
    results = meta_filter.search("query", top_k=10, section_type="termination")
    assert results == []


# ---------------------------------------------------------------------------
# contract_type filter
# ---------------------------------------------------------------------------


def test_contract_type_filter():
    c_nda = _make_chunk("c_nda")
    c_saas = _make_chunk("c_saas")
    retriever = _StubRetriever([(c_nda, 0.9), (c_saas, 0.8)])
    chunk_meta = {
        "c_nda": ChunkMeta(contract_type="NDA"),
        "c_saas": ChunkMeta(contract_type="SaaS"),
    }
    meta_filter = MetadataFilter(base_retriever=retriever, chunk_meta=chunk_meta)
    results = meta_filter.search("confidentiality", top_k=10, contract_type="NDA")
    assert all(c.chunk_id == "c_nda" for c, _ in results)


# ---------------------------------------------------------------------------
# Chunk absent from meta dict passes all filters
# ---------------------------------------------------------------------------


def test_chunk_not_in_meta_passes_all_filters():
    c_no_meta = _make_chunk("c_unknown")
    retriever = _StubRetriever([(c_no_meta, 0.7)])
    meta_filter = MetadataFilter(
        base_retriever=retriever,
        chunk_meta={},  # empty
    )
    results = meta_filter.search("query", top_k=10, section_type="termination")
    # Chunk with no metadata should NOT be filtered out.
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


def test_combined_filters_all_must_match():
    c_match = _make_chunk("c_match")
    c_partial = _make_chunk("c_partial")
    retriever = _StubRetriever([(c_match, 0.9), (c_partial, 0.8)])
    chunk_meta = {
        "c_match": ChunkMeta(section_type="termination", contract_type="NDA"),
        "c_partial": ChunkMeta(section_type="termination", contract_type="SaaS"),
    }
    meta_filter = MetadataFilter(base_retriever=retriever, chunk_meta=chunk_meta)
    results = meta_filter.search(
        "query", top_k=10, section_type="termination", contract_type="NDA"
    )
    assert [c.chunk_id for c, _ in results] == ["c_match"]


# ---------------------------------------------------------------------------
# top_k respected
# ---------------------------------------------------------------------------


def test_top_k_respected_after_filter():
    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    meta = {c.chunk_id: ChunkMeta(section_type="payment") for c in chunks}
    retriever = _StubRetriever([(c, float(10 - i)) for i, c in enumerate(chunks)])
    meta_filter = MetadataFilter(base_retriever=retriever, chunk_meta=meta)
    results = meta_filter.search("query", top_k=3, section_type="payment")
    assert len(results) == 3
