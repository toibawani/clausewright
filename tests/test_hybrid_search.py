"""Tests for clausewright.retrieval.hybrid_search.

All tests mock ``SentenceTransformer`` so they run without downloading any
model weights. BM25 runs on real in-memory tokenised data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import (
    HybridRetriever,
    _reciprocal_rank_fusion,
    _tokenize,
)
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DIM = 8


def _make_embedder(tmp_path: Path, dim: int = _DIM) -> tuple[Embedder, MagicMock]:
    """Return an Embedder backed by a fully mocked SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def _fake_encode(texts: list[str], **kwargs: Any) -> np.ndarray:
        rng = np.random.default_rng(seed=sum(len(t) for t in texts))
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    mock_model.encode.side_effect = _fake_encode

    with patch("clausewright.retrieval.embedder.SentenceTransformer", return_value=mock_model):
        embedder = Embedder(model_name="mock-model", index_dir=tmp_path)
    return embedder, mock_model


def _chunk(
    n: int,
    section: str = "1",
    doc: str = "doc",
    page: int = 1,
    text: str | None = None,
) -> Chunk:
    t = text or f"Chunk {n} covers indemnification terms in section {section} of the contract."
    return Chunk(
        chunk_id=f"{doc}-{section}-{n}",
        doc_id=doc,
        section_number=section,
        chunk_index=n,
        text=t,
        page_range=(page, page),
        word_count=len(t.split()),
    )


def _chunks(count: int, **kw: Any) -> list[Chunk]:
    return [_chunk(i, **kw) for i in range(count)]


def _build_retriever(
    tmp_path: Path,
    n_chunks: int = 10,
    dense_candidates: int = 50,
    sparse_candidates: int = 50,
) -> HybridRetriever:
    embedder, _ = _make_embedder(tmp_path)
    chunks = _chunks(n_chunks)
    index, chunks_out = embedder.build_index(chunks)
    return HybridRetriever(
        embedder=embedder,
        index=index,
        chunks=chunks_out,
        dense_candidates=dense_candidates,
        sparse_candidates=sparse_candidates,
    )


# ---------------------------------------------------------------------------
# Unit tests: _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_lowercases() -> None:
    assert _tokenize("Payment TERMS") == ["payment", "terms"]


def test_tokenize_splits_on_whitespace() -> None:
    assert _tokenize("a b c") == ["a", "b", "c"]


def test_tokenize_empty_string() -> None:
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# Unit tests: _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_single_list() -> None:
    """Documents in a single list get scores 1/(k+1), 1/(k+2), ..."""
    result = _reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=60)
    assert result["a"] > result["b"] > result["c"]


def test_rrf_two_lists_agree() -> None:
    """Document ranked first in both lists should score highest."""
    result = _reciprocal_rank_fusion([["x", "y"], ["x", "z"]], rrf_k=60)
    assert result["x"] > result["y"]
    assert result["x"] > result["z"]


def test_rrf_empty_lists_returns_empty() -> None:
    assert _reciprocal_rank_fusion([]) == {}


def test_rrf_all_unique_ids() -> None:
    result = _reciprocal_rank_fusion([["a"], ["b"]], rrf_k=60)
    assert set(result.keys()) == {"a", "b"}


def test_rrf_custom_k() -> None:
    result_k1 = _reciprocal_rank_fusion([["a"]], rrf_k=1)
    result_k100 = _reciprocal_rank_fusion([["a"]], rrf_k=100)
    # k=1 → 1/2 = 0.5; k=100 → 1/101 ≈ 0.0099
    assert result_k1["a"] > result_k100["a"]


def test_rrf_score_values_are_positive() -> None:
    result = _reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=60)
    for v in result.values():
        assert v > 0


# ---------------------------------------------------------------------------
# HybridRetriever construction
# ---------------------------------------------------------------------------


def test_retriever_init_builds_bm25(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=5)
    assert retriever._bm25 is not None


def test_retriever_empty_chunks_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    dummy_index = faiss.IndexFlatIP(_DIM)
    with pytest.raises(ValueError, match="non-empty"):
        HybridRetriever(embedder=embedder, index=dummy_index, chunks=[])


def test_retriever_chunk_map_has_all_ids(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=6)
    expected_ids = {c.chunk_id for c in retriever.chunks}
    assert set(retriever._chunk_map.keys()) == expected_ids


# ---------------------------------------------------------------------------
# HybridRetriever.search — return shape and type
# ---------------------------------------------------------------------------


def test_search_returns_list(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=8)
    result = retriever.search("payment terms", top_k=3)
    assert isinstance(result, list)


def test_search_returns_chunk_float_tuples(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=8)
    result = retriever.search("indemnification", top_k=3)
    for item in result:
        chunk, score = item
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)


def test_search_top_k_respected(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=10)
    result = retriever.search("contract clause", top_k=4)
    assert len(result) == 4


def test_search_top_k_capped_to_num_chunks(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=3)
    result = retriever.search("clause", top_k=100)
    assert len(result) == 3


def test_search_results_descending_score(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=10)
    result = retriever.search("payment terms", top_k=10)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_search_chunk_ids_are_valid(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=8)
    result = retriever.search("clause", top_k=5)
    valid_ids = {c.chunk_id for c in retriever.chunks}
    for chunk, _ in result:
        assert chunk.chunk_id in valid_ids


def test_search_no_duplicate_chunk_ids(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=10)
    result = retriever.search("indemnification", top_k=10)
    ids = [c.chunk_id for c, _ in result]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# HybridRetriever.search — edge cases and error handling
# ---------------------------------------------------------------------------


def test_search_empty_query_raises(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=5)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ", top_k=3)


def test_search_top_k_zero_raises(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=5)
    with pytest.raises(ValueError, match="top_k must be"):
        retriever.search("query", top_k=0)


def test_search_single_chunk(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=1)
    result = retriever.search("any query", top_k=5)
    assert len(result) == 1


def test_search_scores_are_positive(tmp_path: Path) -> None:
    retriever = _build_retriever(tmp_path, n_chunks=5)
    result = retriever.search("contract terms", top_k=5)
    for _, score in result:
        assert score > 0


# ---------------------------------------------------------------------------
# Integration: keyword query benefits from BM25 channel
# ---------------------------------------------------------------------------


def test_exact_keyword_chunk_appears_in_results(tmp_path: Path) -> None:
    """A chunk with the *exact* query keyword should be in the top results.

    We embed a corpus where one chunk contains an unusual legal term that
    would have low semantic similarity to a generic embedding but high BM25
    score because of exact match.
    """
    embedder, _ = _make_embedder(tmp_path)

    keyword = "xyzindemnificationxyz"
    chunks = [
        Chunk(
            chunk_id=f"doc-1-{i}",
            doc_id="doc",
            section_number="1",
            chunk_index=i,
            text=f"generic contract clause number {i}",
            page_range=(1, 1),
            word_count=5,
        )
        for i in range(9)
    ]
    # One chunk has the rare keyword
    keyword_chunk = Chunk(
        chunk_id="doc-1-9",
        doc_id="doc",
        section_number="1",
        chunk_index=9,
        text=f"This section covers {keyword} obligations for all parties.",
        page_range=(1, 1),
        word_count=9,
    )
    all_chunks = chunks + [keyword_chunk]
    index, chunks_out = embedder.build_index(all_chunks)
    retriever = HybridRetriever(
        embedder=embedder,
        index=index,
        chunks=chunks_out,
        dense_candidates=10,
        sparse_candidates=10,
    )
    result = retriever.search(keyword, top_k=10)
    result_ids = [c.chunk_id for c, _ in result]
    assert keyword_chunk.chunk_id in result_ids
