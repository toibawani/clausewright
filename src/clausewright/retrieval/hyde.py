"""HyDE: Hypothetical Document Embeddings for dense retrieval (closes #27).

What HyDE does
--------------
Instead of embedding the raw user *question* for dense retrieval, HyDE first
asks the LLM to generate a *hypothetical answer* — a plausible contract clause
that would answer the question — then embeds **that** as the dense query
vector.  Because the hypothetical clause is written in the same language and
style as real clauses, it lands closer to relevant passages in embedding space.

Experimental status
-------------------
This feature is **disabled by default** (``enable_hyde=False``).  It is
enabled only if eval numbers show recall@k improvement on the dev split.  The
acceptance criteria require documenting even negative results (see
``docs/eval.md``).

Integration
-----------
``HyDERetriever`` satisfies the same ``SupportsSearch`` protocol as
``HybridRetriever``, so it can be dropped in as the retriever passed to
``AnswerEngine`` without any engine changes.

Usage
-----
::

    from clausewright.retrieval.hyde import HyDERetriever

    retriever = HyDERetriever(
        base_retriever=hybrid_retriever,
        embedder=embedder,
        index=faiss_index,
        chunks=chunks,
        llm=llm,
        enabled=True,
    )
    results = retriever.search("What is the renewal notice period?", top_k=10)
"""

from __future__ import annotations

import logging
from typing import Protocol

import faiss  # type: ignore[import-untyped]

from clausewright.retrieval.embedder import Embedder
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are a contract drafting assistant.  Given a question about a contract, "
    "write a SHORT hypothetical contract clause (2–5 sentences) that would "
    "directly answer the question.  Write only the clause text — no preamble, "
    "no explanation."
)

_HYDE_USER = "Question: {question}"


class _SupportsSearch(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


class _SupportsComplete(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


class HyDERetriever:
    """Dense retrieval with Hypothetical Document Embeddings (HyDE).

    Parameters
    ----------
    base_retriever:
        Used for BM25 / hybrid retrieval when HyDE is disabled or as the
        sparse channel when HyDE is enabled (optional fusion is not
        implemented yet — HyDE currently replaces the dense query only).
    embedder:
        A loaded :class:`~clausewright.retrieval.embedder.Embedder` instance.
    index:
        FAISS index built from the same *chunks*.
    chunks:
        Chunk list in the same row order as the FAISS index.
    llm:
        Any ``LLMClient`` (real or replay) for hypothetical generation.
    enabled:
        Master switch.  When ``False``, delegates to ``base_retriever``
        without any LLM call — zero overhead in the off state.
    """

    def __init__(
        self,
        base_retriever: _SupportsSearch,
        embedder: Embedder,
        index: faiss.Index,
        chunks: list[Chunk],
        llm: _SupportsComplete,
        *,
        enabled: bool = False,
    ) -> None:
        self._base = base_retriever
        self._embedder = embedder
        self._index = index
        self._chunks = chunks
        self._llm = llm
        self._enabled = enabled

    # ------------------------------------------------------------------
    # SupportsSearch interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Retrieve *top_k* chunks, optionally using a hypothetical document.

        When enabled, generates a hypothetical clause and uses its embedding
        as the dense query vector.  Falls back to ``base_retriever`` on error.
        """
        if not self._enabled:
            return self._base.search(query, top_k=top_k)

        hypothetical = self._generate_hypothesis(query)
        if not hypothetical:
            logger.warning("HyDE: empty hypothesis — falling back to base retriever")
            return self._base.search(query, top_k=top_k)

        logger.debug("HyDE hypothesis: %s", hypothetical[:120])
        return self._embedder.search(hypothetical, self._index, self._chunks, k=top_k)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_hypothesis(self, question: str) -> str:
        """Ask the LLM to draft a hypothetical clause for *question*."""
        try:
            return self._llm.complete(
                system=_HYDE_SYSTEM,
                user=_HYDE_USER.format(question=question),
            ).strip()
        except Exception:  # pragma: no cover
            logger.warning("HyDE: LLM call failed", exc_info=True)
            return ""
