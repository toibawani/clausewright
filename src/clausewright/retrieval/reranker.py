"""Cross-encoder reranker: re-scores (query, chunk) pairs for precision retrieval.

Why reranking matters
---------------------
Bi-encoder retrieval (dense embeddings) encodes the query and documents
*independently*, so the scores are an approximation of relevance.  A
**cross-encoder** receives the concatenated ``[query; document]`` pair and
attends across both — producing far more accurate relevance scores at the
cost of O(N) inference rather than O(1) ANN lookup.

The standard RAG pattern is:

1. Retrieve top-50 candidates via cheap bi-encoder / hybrid search.
2. Rerank those 50 with the cross-encoder.
3. Pass the top-3 to 5 hyper-relevant results to the LLM.

Architecture
------------
``CrossEncoderReranker`` wraps ``sentence_transformers.CrossEncoder`` and
provides a single :meth:`rerank` method that accepts a list of ``(Chunk,
score)`` tuples (as returned by :class:`~clausewright.retrieval.hybrid_search.
HybridRetriever`) and returns them re-sorted by cross-encoder relevance.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
- Very fast (MiniLM architecture, 6 layers).
- Trained on MS-MARCO passage ranking; generalises well to legal Q&A.
- Swap for ``BAAI/bge-reranker-large`` for maximum accuracy when latency
  budget allows.

Persistence note
----------------
Cross-encoders are stateless (no index), so there is no save/load concern.
One ``CrossEncoderReranker`` instance per process is the intended usage.

Usage
-----
::

    from clausewright.retrieval.reranker import CrossEncoderReranker
    from clausewright.retrieval.hybrid_search import HybridRetriever

    retriever = HybridRetriever(...)
    reranker  = CrossEncoderReranker()          # loads model once

    candidates = retriever.search(query, top_k=50)
    top_chunks = reranker.rerank(query, candidates, top_n=5)
    # Returns list[tuple[Chunk, float]] — cross-encoder relevance scores.
"""

from __future__ import annotations

import logging
from typing import Final

from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# Default cross-encoder model — fast MiniLM trained on MS-MARCO.
_DEFAULT_CE_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Maximum number of pairs to score in a single cross-encoder forward pass.
_DEFAULT_BATCH_SIZE: Final[int] = 64


class CrossEncoderReranker:
    """Re-score retrieval candidates using a cross-encoder model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for ``sentence_transformers.CrossEncoder``.
        Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Defaults to ``"cpu"`` for portability.
    batch_size:
        Number of ``(query, text)`` pairs scored per forward pass.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_CE_MODEL,
        device: str = "cpu",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        logger.info("Loading cross-encoder model '%s' on device '%s'", model_name, device)
        self._model: CrossEncoder = CrossEncoder(model_name, device=device)

    @property
    def model_name(self) -> str:
        """Name of the underlying cross-encoder model."""
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        *,
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Re-score *candidates* and return them sorted by cross-encoder relevance.

        Parameters
        ----------
        query:
            The original search query string.
        candidates:
            List of ``(chunk, prior_score)`` pairs, typically from
            :meth:`~clausewright.retrieval.hybrid_search.HybridRetriever.search`.
            The prior scores are *replaced* by cross-encoder scores.
        top_n:
            If given, return only the *top_n* highest-scoring chunks.
            Defaults to returning all candidates re-sorted.

        Returns
        -------
        list[tuple[Chunk, float]]
            ``(chunk, cross_encoder_score)`` pairs sorted descending.
            Scores are raw logits from the cross-encoder (not normalised to [0,1]).

        Raises
        ------
        ValueError
            If *query* is blank, *candidates* is empty, or *top_n* < 1.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        chunks = [c for c, _ in candidates]
        pairs: list[tuple[str, str]] = [(query, c.text) for c in chunks]

        logger.debug(
            "Reranking %d candidates with cross-encoder '%s'",
            len(pairs),
            self._model_name,
        )

        predictions = self._model.predict(  # type: ignore[no-untyped-call]
            pairs,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        raw_scores: list[float] = [float(s) for s in predictions]

        # Pair up and sort descending
        scored: list[tuple[Chunk, float]] = [
            (chunk, float(score)) for chunk, score in zip(chunks, raw_scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            scored = scored[:top_n]

        return scored
