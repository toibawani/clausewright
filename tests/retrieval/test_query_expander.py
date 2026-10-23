"""Unit tests for QueryExpander (issue #26)."""

from __future__ import annotations

import pytest

from clausewright.retrieval.query_expander import QueryExpander, _parse_paraphrases
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(cid: str, text: str) -> Chunk:
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
    """Returns a fixed list keyed by query prefix for testing."""

    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        return self._results[:top_k]


class _StubLLM:
    def __init__(self, paraphrases: list[str]) -> None:
        self._paraphrases = paraphrases

    def complete(self, *, system: str, user: str) -> str:
        return "\n".join(
            f"{i + 1}. {p}" for i, p in enumerate(self._paraphrases)
        )


class _FailingLLM:
    def complete(self, *, system: str, user: str) -> str:
        raise RuntimeError("LLM unreachable")


# ---------------------------------------------------------------------------
# _parse_paraphrases
# ---------------------------------------------------------------------------


def test_parse_paraphrases_numbered():
    raw = "1. First paraphrase\n2. Second paraphrase\n3. Third paraphrase"
    result = _parse_paraphrases(raw)
    assert result == ["First paraphrase", "Second paraphrase", "Third paraphrase"]


def test_parse_paraphrases_with_paren():
    raw = "1) First\n2) Second"
    result = _parse_paraphrases(raw)
    assert result == ["First", "Second"]


def test_parse_paraphrases_empty():
    assert _parse_paraphrases("") == []


def test_parse_paraphrases_skips_blank_lines():
    raw = "1. First\n\n2. Second\n"
    assert _parse_paraphrases(raw) == ["First", "Second"]


# ---------------------------------------------------------------------------
# QueryExpander — disabled mode
# ---------------------------------------------------------------------------


def test_disabled_passes_through():
    chunk = _make_chunk("c1", "termination notice period thirty days")
    retriever = _StubRetriever([(chunk, 0.9)])
    llm = _StubLLM(["paraphrase"])
    expander = QueryExpander(retriever, llm, n_paraphrases=2, enabled=False)
    results = expander.search("termination notice", top_k=5)
    assert results == [(chunk, 0.9)]


def test_zero_paraphrases_passes_through():
    chunk = _make_chunk("c1", "termination notice period")
    retriever = _StubRetriever([(chunk, 0.8)])
    llm = _StubLLM(["should not appear"])
    expander = QueryExpander(retriever, llm, n_paraphrases=0)
    results = expander.search("notice", top_k=5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# QueryExpander — enabled mode
# ---------------------------------------------------------------------------


def test_expander_deduplicates_chunks():
    c1 = _make_chunk("c1", "termination notice thirty days")
    c2 = _make_chunk("c2", "notice period clause")
    # Retriever always returns [c1, c2] regardless of query.
    retriever = _StubRetriever([(c1, 0.9), (c2, 0.7)])
    llm = _StubLLM(["right to cancel", "cancellation clause"])
    expander = QueryExpander(retriever, llm, n_paraphrases=2)
    results = expander.search("termination notice", top_k=10)
    # Both chunks should appear exactly once.
    ids = [c.chunk_id for c, _ in results]
    assert ids.count("c1") == 1
    assert ids.count("c2") == 1


def test_expander_limits_top_k():
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(10)]
    retriever = _StubRetriever([(c, float(10 - i)) for i, c in enumerate(chunks)])
    llm = _StubLLM(["p1"])
    expander = QueryExpander(retriever, llm, n_paraphrases=1)
    results = expander.search("query", top_k=3)
    assert len(results) == 3


def test_expander_llm_failure_falls_back():
    chunk = _make_chunk("c1", "text")
    retriever = _StubRetriever([(chunk, 0.5)])
    expander = QueryExpander(retriever, _FailingLLM(), n_paraphrases=2)
    # Should not raise; should return base results.
    results = expander.search("query", top_k=5)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_n_paraphrases_raises():
    with pytest.raises(ValueError, match="n_paraphrases"):
        QueryExpander(_StubRetriever([]), _StubLLM([]), n_paraphrases=-1)
