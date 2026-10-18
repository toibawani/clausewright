"""Tests for clausewright.retrieval.reranker.

CrossEncoder is fully mocked so tests run without any model downloads.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _chunk(
    n: int,
    section: str = "1",
    doc: str = "doc",
    text: str | None = None,
) -> Chunk:
    t = text or f"Chunk {n}: This section discusses liability and indemnification clause {n}."
    return Chunk(
        chunk_id=f"{doc}-{section}-{n}",
        doc_id=doc,
        section_number=section,
        chunk_index=n,
        text=t,
        page_range=(1, 1),
        word_count=len(t.split()),
    )


def _candidates(count: int, **kw: Any) -> list[tuple[Chunk, float]]:
    """Return a list of (Chunk, prior_score) pairs with decreasing prior score."""
    return [(_chunk(i, **kw), float(count - i)) for i in range(count)]


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    """Return a CrossEncoderReranker whose model.predict returns *scores*."""
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    with patch("clausewright.retrieval.reranker.CrossEncoder", return_value=mock_model):
        reranker = CrossEncoderReranker(model_name="mock-ce")
    return reranker


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_reranker_loads_model() -> None:
    mock_model = MagicMock()
    with patch("clausewright.retrieval.reranker.CrossEncoder", return_value=mock_model) as cls:
        reranker = CrossEncoderReranker(model_name="test-ce", device="cpu")
        cls.assert_called_once_with("test-ce", device="cpu")
    assert reranker.model_name == "test-ce"


def test_reranker_model_name_property() -> None:
    reranker = _make_reranker([1.0])
    assert reranker.model_name == "mock-ce"


# ---------------------------------------------------------------------------
# rerank — return shape and type
# ---------------------------------------------------------------------------


def test_rerank_returns_list() -> None:
    reranker = _make_reranker([0.9, 0.5, 0.1])
    result = reranker.rerank("payment terms", _candidates(3))
    assert isinstance(result, list)


def test_rerank_returns_chunk_float_tuples() -> None:
    reranker = _make_reranker([0.8, 0.3])
    result = reranker.rerank("indemnification", _candidates(2))
    for item in result:
        chunk, score = item
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)


def test_rerank_without_top_n_returns_all() -> None:
    n = 5
    reranker = _make_reranker([float(i) for i in range(n)])
    result = reranker.rerank("clause", _candidates(n))
    assert len(result) == n


def test_rerank_top_n_limits_results() -> None:
    reranker = _make_reranker([0.9, 0.7, 0.5, 0.3, 0.1])
    result = reranker.rerank("clause", _candidates(5), top_n=3)
    assert len(result) == 3


def test_rerank_top_n_larger_than_candidates_returns_all() -> None:
    reranker = _make_reranker([0.9, 0.5])
    result = reranker.rerank("query", _candidates(2), top_n=100)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# rerank — ordering
# ---------------------------------------------------------------------------


def test_rerank_sorted_descending_by_ce_score() -> None:
    # Scores come back out-of-order from the mock; result should be sorted.
    reranker = _make_reranker([0.3, 0.9, 0.1, 0.7])
    result = reranker.rerank("query", _candidates(4))
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_score_overrides_prior_score() -> None:
    """The returned score should be the CE score (0.99), not the prior (99.0)."""
    reranker = _make_reranker([0.99])
    result = reranker.rerank("query", [(_chunk(0), 99.0)])
    _, returned_score = result[0]
    assert returned_score == pytest.approx(0.99)


def test_rerank_top_n_selects_highest_ce_scores() -> None:
    """When top_n=2, the two chunks with highest CE scores should be returned."""
    # CE scores assigned to chunk indices 0..4: chunk 2 highest, chunk 4 second.
    ce_scores = [0.1, 0.3, 0.95, 0.2, 0.88]
    reranker = _make_reranker(ce_scores)
    candidates = _candidates(5)
    result = reranker.rerank("query", candidates, top_n=2)
    returned_chunk_ids = {c.chunk_id for c, _ in result}
    assert candidates[2][0].chunk_id in returned_chunk_ids  # score 0.95
    assert candidates[4][0].chunk_id in returned_chunk_ids  # score 0.88


# ---------------------------------------------------------------------------
# rerank — error handling
# ---------------------------------------------------------------------------


def test_rerank_empty_query_raises() -> None:
    reranker = _make_reranker([1.0])
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("   ", _candidates(1))


def test_rerank_empty_candidates_raises() -> None:
    reranker = _make_reranker([])
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("payment terms", [])


def test_rerank_top_n_zero_raises() -> None:
    reranker = _make_reranker([1.0])
    with pytest.raises(ValueError, match="top_n must be"):
        reranker.rerank("query", _candidates(1), top_n=0)


def test_rerank_top_n_negative_raises() -> None:
    reranker = _make_reranker([1.0])
    with pytest.raises(ValueError, match="top_n must be"):
        reranker.rerank("query", _candidates(1), top_n=-5)


# ---------------------------------------------------------------------------
# rerank — model interactions
# ---------------------------------------------------------------------------


def test_rerank_calls_model_predict_once() -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.3]
    with patch("clausewright.retrieval.reranker.CrossEncoder", return_value=mock_model):
        reranker = CrossEncoderReranker(model_name="mock-ce")

    reranker.rerank("indemnification", _candidates(2))
    mock_model.predict.assert_called_once()


def test_rerank_passes_correct_pairs_to_model() -> None:
    mock_model = MagicMock()
    query = "payment terms"
    cands = _candidates(2)
    mock_model.predict.return_value = [0.5, 0.3]

    with patch("clausewright.retrieval.reranker.CrossEncoder", return_value=mock_model):
        reranker = CrossEncoderReranker(model_name="mock-ce")

    reranker.rerank(query, cands)

    call_args = mock_model.predict.call_args
    pairs_arg = call_args[0][0]
    assert pairs_arg[0] == (query, cands[0][0].text)
    assert pairs_arg[1] == (query, cands[1][0].text)


def test_rerank_single_candidate() -> None:
    reranker = _make_reranker([0.77])
    result = reranker.rerank("query", _candidates(1))
    assert len(result) == 1
    _, score = result[0]
    assert score == pytest.approx(0.77)
