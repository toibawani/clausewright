"""Tests for clausewright.retrieval.embedder.

All tests mock ``SentenceTransformer`` so they run without downloading any
model weights and complete in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from clausewright.retrieval.embedder import Embedder
from clausewright.schemas import Chunk

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DIM = 8  # tiny embedding dimension for tests


def _make_embedder(tmp_path: Path, dim: int = _DIM) -> tuple[Embedder, MagicMock]:
    """Return an Embedder whose SentenceTransformer is fully mocked.

    The mock model's ``encode`` returns deterministic random float32 vectors
    and ``get_sentence_embedding_dimension`` returns *dim*.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def _fake_encode(texts: list[str], **kwargs: Any) -> np.ndarray:
        rng = np.random.default_rng(seed=len(texts))
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        # Normalise so inner-product == cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

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
    t = text or f"This is chunk {n} with some contract text for section {section}."
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


# ---------------------------------------------------------------------------
# Embedder.__init__ / model loading
# ---------------------------------------------------------------------------


def test_embedder_loads_model(tmp_path: Path) -> None:
    """Constructor should call SentenceTransformer with the given model name."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = _DIM
    mock_model.encode.side_effect = lambda t, **kw: np.zeros((len(t), _DIM), dtype=np.float32)

    with patch("clausewright.retrieval.embedder.SentenceTransformer", return_value=mock_model) as cls:
        Embedder(model_name="test-model", device="cpu", index_dir=tmp_path)
        cls.assert_called_once_with("test-model", device="cpu")


def test_embedding_dim_property(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path, dim=16)
    assert embedder.embedding_dim == 16


# ---------------------------------------------------------------------------
# Embedder.embed
# ---------------------------------------------------------------------------


def test_embed_returns_correct_shape(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    matrix = embedder.embed(["hello world", "foo bar", "baz"])
    assert matrix.shape == (3, _DIM)


def test_embed_dtype_is_float32(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    matrix = embedder.embed(["text"])
    assert matrix.dtype == np.float32


def test_embed_normalized_vectors_unit_norm(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    matrix = embedder.embed(["contract clause one", "indemnification terms"])
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)


def test_embed_empty_texts_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed([])


def test_embed_calls_model_encode(tmp_path: Path) -> None:
    embedder, mock_model = _make_embedder(tmp_path)
    embedder.embed(["hello"])
    mock_model.encode.assert_called_once()


# ---------------------------------------------------------------------------
# Embedder.build_index
# ---------------------------------------------------------------------------


def test_build_index_returns_tuple(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, returned_chunks = embedder.build_index(_chunks(3))
    assert isinstance(idx, faiss.Index)
    assert len(returned_chunks) == 3


def test_build_index_ntotal_matches_chunks(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, _ = embedder.build_index(_chunks(7))
    assert idx.ntotal == 7


def test_build_index_empty_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        embedder.build_index([])


def test_build_index_single_chunk(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(1))
    assert idx.ntotal == 1
    assert len(ch) == 1


def test_build_index_preserves_chunk_order(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    source = _chunks(5)
    _, returned = embedder.build_index(source)
    assert [c.chunk_id for c in returned] == [c.chunk_id for c in source]


# ---------------------------------------------------------------------------
# Embedder.save / load
# ---------------------------------------------------------------------------


def test_save_creates_faiss_file(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(3))
    embedder.save(idx, ch, name="myindex")
    assert (tmp_path / "myindex.faiss").exists()


def test_save_creates_chunks_json(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(3))
    embedder.save(idx, ch, name="myindex")
    assert (tmp_path / "myindex.chunks.json").exists()


def test_save_chunks_json_is_valid(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    chunks = _chunks(2)
    idx, ch = embedder.build_index(chunks)
    embedder.save(idx, ch, name="valid")
    raw = json.loads((tmp_path / "valid.chunks.json").read_text())
    assert isinstance(raw, list)
    assert len(raw) == 2
    assert raw[0]["chunk_id"] == chunks[0].chunk_id


def test_load_returns_index_and_chunks(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    source = _chunks(4)
    idx, ch = embedder.build_index(source)
    embedder.save(idx, ch, name="round_trip")
    loaded_idx, loaded_chunks = embedder.load(name="round_trip")
    assert loaded_idx.ntotal == 4
    assert len(loaded_chunks) == 4


def test_load_reconstructs_chunk_ids(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    source = _chunks(3)
    idx, ch = embedder.build_index(source)
    embedder.save(idx, ch, name="ids")
    _, loaded = embedder.load(name="ids")
    assert [c.chunk_id for c in loaded] == [c.chunk_id for c in source]


def test_load_missing_faiss_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        embedder.load(name="nonexistent")


def test_load_missing_chunks_json_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    # Write a fake faiss file but no chunks json
    idx = faiss.IndexFlatIP(_DIM)
    faiss.write_index(idx, str(tmp_path / "partial.faiss"))
    with pytest.raises(FileNotFoundError, match="Chunk metadata not found"):
        embedder.load(name="partial")


def test_save_creates_index_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    embedder, _ = _make_embedder(tmp_path, dim=_DIM)
    # re-point index_dir to un-created nested path
    embedder._index_dir = nested
    idx, ch = embedder.build_index(_chunks(2))
    embedder.save(idx, ch, name="nested")
    assert (nested / "nested.faiss").exists()
    assert (nested / "nested.chunks.json").exists()



# ---------------------------------------------------------------------------
# Embedder.search
# ---------------------------------------------------------------------------


def test_search_returns_k_results(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    source = _chunks(10)
    idx, ch = embedder.build_index(source)
    results = embedder.search("payment terms", idx, ch, k=3)
    assert len(results) == 3


def test_search_returns_chunk_score_tuples(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(5))
    results = embedder.search("indemnification", idx, ch, k=2)
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)


def test_search_scores_between_minus1_and_1(tmp_path: Path) -> None:
    """Cosine similarity scores are in [-1, 1] for normalised vectors."""
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(5))
    results = embedder.search("contract clause", idx, ch, k=5)
    for _, score in results:
        assert -1.0 <= score <= 1.0 + 1e-5


def test_search_k_capped_to_num_chunks(tmp_path: Path) -> None:
    """Requesting more results than chunks should not crash."""
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(3))
    results = embedder.search("query", idx, ch, k=100)
    assert len(results) == 3


def test_search_empty_query_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(2))
    with pytest.raises(ValueError, match="non-empty"):
        embedder.search("   ", idx, ch, k=1)


def test_search_k_zero_raises(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(2))
    with pytest.raises(ValueError, match="k must be"):
        embedder.search("query", idx, ch, k=0)


def test_search_results_ordered_by_score_descending(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    idx, ch = embedder.build_index(_chunks(8))
    results = embedder.search("any query", idx, ch, k=8)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_chunk_ids_from_original_list(tmp_path: Path) -> None:
    embedder, _ = _make_embedder(tmp_path)
    source = _chunks(5)
    idx, ch = embedder.build_index(source)
    results = embedder.search("clause", idx, ch, k=5)
    result_ids = {c.chunk_id for c, _ in results}
    source_ids = {c.chunk_id for c in source}
    assert result_ids <= source_ids
