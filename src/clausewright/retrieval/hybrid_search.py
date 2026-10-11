"""Hybrid retriever: BM25 sparse search + dense FAISS vector search fused via RRF.

Architecture
------------
``HybridRetriever`` owns two retrieval channels:

1. **Sparse (BM25)** — ``rank_bm25.BM25Okapi`` over tokenised chunk texts.
   Exact keyword matching; excellent for legal terms, defined phrases, section
   references (e.g., "Section 12.3(b)"), and acronyms.

2. **Dense (FAISS)** — inner-product search on L2-normalised sentence embeddings
   produced by the existing :class:`~clausewright.retrieval.embedder.Embedder`.
   Captures semantic similarity across paraphrases.

3. **Reciprocal Rank Fusion (RRF)** — the two ranked lists are fused using::

       score_rrf(d) = Σ  1 / (k + rank_i(d))

   where *k = 60* (the standard RRF constant that smooths rank differences).
   Documents that appear highly ranked in *both* channels receive the highest
   combined scores.

Usage
-----
::

    from clausewright.retrieval.hybrid_search import HybridRetriever
    from clausewright.retrieval.embedder import Embedder

    embedder = Embedder()
    index, chunks = embedder.build_index(my_chunks)

    retriever = HybridRetriever(embedder=embedder, index=index, chunks=chunks)
    results = retriever.search("payment terms indemnification", top_k=10)
    # Returns list[tuple[Chunk, float]] sorted by descending RRF score.

Design decisions
----------------
* BM25 index is built eagerly in ``__init__`` from the chunk texts that are
  already available — O(N) and cheap.
* ``dense_candidates`` and ``sparse_candidates`` control how many results each
  channel fetches *before* fusion (default 50).  More candidates → better fusion
  recall at the cost of slightly more FAISS + BM25 work.
* The RRF constant ``rrf_k`` is exposed as a constructor parameter so callers
  can tune it without changing source code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import faiss  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from clausewright.retrieval.embedder import Embedder
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# Default RRF smoothing constant (Cormack et al. 2009 recommend 60).
_DEFAULT_RRF_K: Final[int] = 60
# Number of candidates to fetch from each channel before fusion.
_DEFAULT_CANDIDATES: Final[int] = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokeniser — mirrors what BM25Okapi expects."""
    return text.lower().split()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = _DEFAULT_RRF_K,
) -> dict[str, float]:
    """Merge multiple ranked lists into a single RRF score dict.

    Parameters
    ----------
    ranked_lists:
        Each inner list is an ordered sequence of document IDs (best first).
    rrf_k:
        Smoothing constant; the standard value is 60.

    Returns
    -------
    dict[str, float]
        Mapping from document ID to cumulative RRF score (higher is better).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class HybridRetriever:
    """Fuse BM25 sparse and dense FAISS retrieval via Reciprocal Rank Fusion.

    Parameters
    ----------
    embedder:
        A loaded :class:`~clausewright.retrieval.embedder.Embedder` instance.
    index:
        A FAISS index built from the same *chunks* (via ``embedder.build_index``).
    chunks:
        The chunk list in the same order as the FAISS index rows.
    dense_candidates:
        How many results to fetch from dense search before RRF.
    sparse_candidates:
        How many results to fetch from BM25 before RRF.
    rrf_k:
        RRF smoothing constant.  Standard value is 60.
    """

    embedder: Embedder
    index: faiss.Index
    chunks: list[Chunk]
    dense_candidates: int = _DEFAULT_CANDIDATES
    sparse_candidates: int = _DEFAULT_CANDIDATES
    rrf_k: int = _DEFAULT_RRF_K

    # BM25 index built automatically from chunk texts.
    _bm25: BM25Okapi = field(init=False, repr=False)
    _chunk_map: dict[str, Chunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("chunks must be non-empty")
        tokenised = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenised)
        self._chunk_map = {c.chunk_id: c for c in self.chunks}
        logger.info(
            "HybridRetriever ready: %d chunks, BM25 vocab=%d",
            len(self.chunks),
            len(self._bm25.idf),
        )

    # ------------------------------------------------------------------
    # Internal retrieval channels
    # ------------------------------------------------------------------

    def _dense_ranked(self, query: str, n: int) -> list[str]:
        """Return up to *n* chunk IDs ranked by dense cosine similarity."""
        results = self.embedder.search(query, self.index, self.chunks, k=n)
        return [chunk.chunk_id for chunk, _ in results]

    def _sparse_ranked(self, query: str, n: int) -> list[str]:
        """Return up to *n* chunk IDs ranked by BM25 score."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (chunk_id, score) and sort descending, take top n
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n]
        return [self.chunks[i].chunk_id for i, _score in indexed]

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Retrieve the *top_k* most relevant chunks via hybrid RRF search.

        Parameters
        ----------
        query:
            Free-text search query (natural language or keyword phrase).
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            Pairs of ``(chunk, rrf_score)`` sorted by descending RRF score.

        Raises
        ------
        ValueError
            If *query* is blank or *top_k* < 1.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        n_dense = min(self.dense_candidates, len(self.chunks))
        n_sparse = min(self.sparse_candidates, len(self.chunks))

        dense_ids = self._dense_ranked(query, n_dense)
        sparse_ids = self._sparse_ranked(query, n_sparse)

        logger.debug(
            "Dense retrieved %d, sparse retrieved %d",
            len(dense_ids),
            len(sparse_ids),
        )

        rrf_scores = _reciprocal_rank_fusion([dense_ids, sparse_ids], rrf_k=self.rrf_k)

        # Sort by descending RRF score and resolve back to Chunk objects
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        effective_top_k = min(top_k, len(sorted_ids))

        results: list[tuple[Chunk, float]] = []
        for cid in sorted_ids[:effective_top_k]:
            chunk = self._chunk_map[cid]
            results.append((chunk, rrf_scores[cid]))

        return results
