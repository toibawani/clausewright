"""Unit tests for HyDERetriever (issue #27)."""

from __future__ import annotations

from clausewright.retrieval.hyde import HyDERetriever
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
    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        return self._results[:top_k]


class _StubEmbedder:
    """Stub embedder: stores the query used for search so we can assert on it."""

    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self._results = results
        self.last_query: str | None = None

    def search(self, query: str, index: object, chunks: list[Chunk], k: int = 10) -> list[tuple[Chunk, float]]:
        self.last_query = query
        return self._results[:k]


class _StubLLM:
    def __init__(self, hypothesis: str) -> None:
        self._hypothesis = hypothesis

    def complete(self, *, system: str, user: str) -> str:
        return self._hypothesis


class _FailingLLM:
    def complete(self, *, system: str, user: str) -> str:
        raise RuntimeError("LLM down")


# ---------------------------------------------------------------------------
# Disabled mode — delegates to base retriever
# ---------------------------------------------------------------------------


def test_disabled_uses_base_retriever():
    chunk = _make_chunk("c1", "renewal clause")
    base = _StubRetriever([(chunk, 0.8)])
    embedder = _StubEmbedder([])
    retriever = HyDERetriever(
        base_retriever=base,
        embedder=embedder,
        index=object(),
        chunks=[chunk],
        llm=_StubLLM("hypothesis"),
        enabled=False,
    )
    results = retriever.search("renewal period", top_k=5)
    assert results == [(chunk, 0.8)]
    # Embedder should NOT have been called.
    assert embedder.last_query is None


# ---------------------------------------------------------------------------
# Enabled mode — uses hypothetical embedding
# ---------------------------------------------------------------------------


def test_enabled_embeds_hypothesis():
    chunk = _make_chunk("c1", "renewal clause thirty days")
    base = _StubRetriever([])
    hypothesis_text = "The agreement shall renew automatically unless notice is given."
    embedder = _StubEmbedder([(chunk, 0.95)])
    retriever = HyDERetriever(
        base_retriever=base,
        embedder=embedder,
        index=object(),
        chunks=[chunk],
        llm=_StubLLM(hypothesis_text),
        enabled=True,
    )
    results = retriever.search("renewal notice period", top_k=5)
    assert embedder.last_query == hypothesis_text
    assert results == [(chunk, 0.95)]


def test_enabled_empty_hypothesis_falls_back():
    chunk = _make_chunk("c1", "text")
    base = _StubRetriever([(chunk, 0.6)])
    embedder = _StubEmbedder([])
    retriever = HyDERetriever(
        base_retriever=base,
        embedder=embedder,
        index=object(),
        chunks=[chunk],
        llm=_StubLLM(""),  # empty hypothesis
        enabled=True,
    )
    results = retriever.search("some query", top_k=5)
    assert results == [(chunk, 0.6)]
    assert embedder.last_query is None


def test_enabled_llm_failure_falls_back():
    chunk = _make_chunk("c1", "text")
    base = _StubRetriever([(chunk, 0.5)])
    embedder = _StubEmbedder([])
    retriever = HyDERetriever(
        base_retriever=base,
        embedder=embedder,
        index=object(),
        chunks=[chunk],
        llm=_FailingLLM(),
        enabled=True,
    )
    results = retriever.search("question", top_k=5)
    assert results == [(chunk, 0.5)]
