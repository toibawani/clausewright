"""Dense embedder and FAISS index management for contract chunk retrieval.

Architecture
------------
``Embedder`` is a thin, stateless wrapper around a ``SentenceTransformer``
model.  It owns three concerns:

1. **Encoding** — convert a list of text strings into a float32 numpy matrix
   (one row per text, shape ``[N, D]``).

2. **Index management** — build an inner-product FAISS index from a list of
   ``Chunk`` objects, then save/load the pair ``(faiss.Index, list[Chunk])``
   to disk so it survives restarts.

3. **Search** — encode a query string and retrieve the *k* most similar
   chunks with their cosine-similarity scores.

Cosine similarity is achieved by L2-normalising all vectors before calling
``faiss.IndexFlatIP`` (inner product on unit vectors = cosine).

Persistence layout (under ``PATHS.index_dir``)::

    index_dir/
        <name>.faiss        # binary FAISS index
        <name>.chunks.json  # JSON-serialised list of Chunk objects

Usage example::

    embedder = Embedder()                         # uses EMBEDDING_MODEL_NAME
    index, chunks = embedder.build_index(chunks)  # encode + build FAISS
    embedder.save(index, chunks, name="v1")       # persist to data/index/
    index, chunks = embedder.load(name="v1")      # reload from disk
    results = embedder.search("payment terms", index, chunks, k=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from clausewright.config import EMBEDDING_MODEL_NAME, PATHS
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# FAISS index file extension
_INDEX_SUFFIX = ".faiss"
_CHUNKS_SUFFIX = ".chunks.json"


class Embedder:
    """Encode contract chunks and manage a FAISS similarity index.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for ``SentenceTransformer``.
        Defaults to ``EMBEDDING_MODEL_NAME`` from ``clausewright.config``.
    device:
        Pytorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Defaults to ``"cpu"`` for portability.
    index_dir:
        Directory where FAISS indexes and chunk lists are saved.
        Defaults to ``PATHS.index_dir``.
    batch_size:
        Number of texts encoded per forward pass.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = "cpu",
        index_dir: Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._index_dir: Path = index_dir if index_dir is not None else PATHS.index_dir
        self._batch_size = batch_size
        logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[no-untyped-call]

    def embed(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        """Encode a list of texts into a float32 embedding matrix.

        Parameters
        ----------
        texts:
            Non-empty list of strings to encode.
        normalize:
            When ``True`` (default) each row is L2-normalised so that inner
            product equals cosine similarity.

        Returns
        -------
        np.ndarray
            Shape ``[len(texts), embedding_dim]``, dtype ``float32``.

        Raises
        ------
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        logger.debug("Encoding %d texts (batch_size=%d)", len(texts), self._batch_size)
        raw: Any = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        matrix = np.array(raw, dtype=np.float32)
        return matrix

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[Chunk]) -> tuple[faiss.Index, list[Chunk]]:
        """Encode *chunks* and build a FAISS inner-product index.

        Parameters
        ----------
        chunks:
            Non-empty list of ``Chunk`` objects.  Each chunk's ``text`` field
            is used for encoding.

        Returns
        -------
        tuple[faiss.Index, list[Chunk]]
            The FAISS index and the *same* list of chunks in the same order,
            so that ``index.search(...)`` result positions map 1-to-1 to chunks.

        Raises
        ------
        ValueError
            If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("chunks must be non-empty to build an index")

        texts = [c.text for c in chunks]
        logger.info("Encoding %d chunks for index build", len(texts))
        vectors = self.embed(texts, normalize=True)

        dim = vectors.shape[1]
        index: faiss.Index = faiss.IndexFlatIP(dim)
        index.add(vectors)  # type: ignore[attr-defined]
        logger.info("FAISS IndexFlatIP built: dim=%d, ntotal=%d", dim, index.ntotal)
        return index, list(chunks)

    def save(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        name: str = "default",
    ) -> None:
        """Persist *index* and *chunks* to ``index_dir/<name>.faiss`` and ``.chunks.json``.

        Parameters
        ----------
        index:
            FAISS index to serialise.
        chunks:
            Corresponding chunk list (same order as index rows).
        name:
            Base filename (no extension) for both files.
        """
        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_dir / f"{name}{_INDEX_SUFFIX}"
        chunks_path = self._index_dir / f"{name}{_CHUNKS_SUFFIX}"

        faiss.write_index(index, str(index_path))
        logger.info("FAISS index saved to %s", index_path)

        chunks_data = [c.model_dump() for c in chunks]
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")
        logger.info("Chunk metadata saved to %s", chunks_path)

    def load(self, name: str = "default") -> tuple[faiss.Index, list[Chunk]]:
        """Load a previously saved FAISS index and its chunk list from disk.

        Parameters
        ----------
        name:
            Base filename used when calling :meth:`save`.

        Returns
        -------
        tuple[faiss.Index, list[Chunk]]
            Loaded index and reconstructed chunk list.

        Raises
        ------
        FileNotFoundError
            If either the index or chunk file is missing.
        """
        index_path = self._index_dir / f"{name}{_INDEX_SUFFIX}"
        chunks_path = self._index_dir / f"{name}{_CHUNKS_SUFFIX}"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {chunks_path}")

        index = faiss.read_index(str(index_path))
        logger.info("FAISS index loaded from %s (ntotal=%d)", index_path, index.ntotal)

        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(c) for c in raw_chunks]
        logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
        return index, chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        index: faiss.Index,
        chunks: list[Chunk],
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the *k* most similar chunks for a query string.

        Parameters
        ----------
        query:
            Free-text search query.
        index:
            A FAISS index previously built by :meth:`build_index`.
        chunks:
            Chunk list in the same order as the index rows.
        k:
            Number of results to return.  Capped to ``len(chunks)``
            automatically.

        Returns
        -------
        list[tuple[Chunk, float]]
            Pairs of ``(chunk, score)`` ordered by descending similarity.
            Score is the inner-product (cosine similarity when normalised).

        Raises
        ------
        ValueError
            If *query* is empty or *k* < 1.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        effective_k = min(k, len(chunks))
        query_vec = self.embed([query], normalize=True)  # shape [1, D]

        distances: np.ndarray
        indices: np.ndarray
        distances, indices = index.search(query_vec, effective_k)  # type: ignore[attr-defined]

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS sentinel for "no result"
                continue
            results.append((chunks[int(idx)], float(score)))

        return results
