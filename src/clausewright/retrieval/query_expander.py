"""Query rewriting and multi-query expansion (closes #26).

Rationale
---------
A single legal question phrased in one way may miss clauses phrased
differently in the contract ("termination notice" vs "right to cancel" vs
"notice period").  Multi-query expansion generates N paraphrases via the LLM,
retrieves for each independently, then deduplicates and passes the merged pool
to the cross-encoder reranker.

Design choices
--------------
* The expander is a thin wrapper that satisfies ``SupportsSearch`` (the same
  duck-typed protocol used by ``AnswerEngine``), so callers need only swap in
  the expander — no engine changes required.
* Paraphrase generation is gated behind ``enable_query_expansion: false`` by
  default so the feature can be disabled with zero overhead.
* The LLM call for expansion uses the same ``LLMClient`` protocol as the rest
  of the stack, which means the ``ReplayLLMClient`` from the eval harness
  provides zero-cost replay during testing.
* Deduplication is by ``chunk_id`` — the first occurrence keeps the RRF score
  from the query that retrieved it first; this preserves the ordering signal.

Usage
-----
::

    from clausewright.retrieval.query_expander import QueryExpander
    from clausewright.qa.llm import AnthropicClient

    expander = QueryExpander(
        base_retriever=retriever,
        llm=llm,
        n_paraphrases=3,
    )
    results = expander.search("What is the termination notice period?", top_k=10)
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

_EXPANSION_SYSTEM = (
    "You are a legal search assistant.  Given a question about a contract, "
    "produce exactly {n} paraphrased versions of the question that would match "
    "different phrasings found in contracts.  Output ONLY the paraphrases, one "
    "per line, numbered 1. 2. 3. … with no preamble or explanation."
)

_EXPANSION_USER = "Original question: {question}"


class _SupportsSearch(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


class _SupportsComplete(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


def _parse_paraphrases(raw: str) -> list[str]:
    """Extract numbered lines from LLM paraphrase output."""
    lines: list[str] = []
    for line in raw.splitlines():
        # Strip leading "1. ", "2. " etc.
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


class QueryExpander:
    """Multi-query expansion wrapper that satisfies ``SupportsSearch``.

    Parameters
    ----------
    base_retriever:
        Any object with ``search(query, top_k) -> list[(Chunk, score)]``.
    llm:
        Any ``LLMClient`` (real or replay) used to generate paraphrases.
    n_paraphrases:
        Number of paraphrases to generate.  Set to 0 to disable expansion
        (identical to using ``base_retriever`` directly).
    enabled:
        Master switch.  If ``False``, passes through to ``base_retriever``
        without any LLM call.
    """

    def __init__(
        self,
        base_retriever: _SupportsSearch,
        llm: _SupportsComplete,
        *,
        n_paraphrases: int = 3,
        enabled: bool = True,
    ) -> None:
        if n_paraphrases < 0:
            raise ValueError(f"n_paraphrases must be >= 0, got {n_paraphrases}")
        self._retriever = base_retriever
        self._llm = llm
        self._n = n_paraphrases
        self._enabled = enabled

    # ------------------------------------------------------------------
    # SupportsSearch interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Retrieve *top_k* chunks via expanded multi-query search.

        When expansion is disabled (``enabled=False`` or ``n_paraphrases=0``),
        delegates directly to the base retriever.

        Parameters
        ----------
        query:
            The original user question.
        top_k:
            Maximum number of deduplicated results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            Deduplicated ``(chunk, score)`` pairs from the merged candidate
            pool, preserving the score of the first occurrence.
        """
        if not self._enabled or self._n == 0:
            return self._retriever.search(query, top_k=top_k)

        queries = self._generate_queries(query)
        return self._merge(queries, top_k=top_k)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_queries(self, question: str) -> list[str]:
        """Return the original question plus LLM-generated paraphrases."""
        system = _EXPANSION_SYSTEM.format(n=self._n)
        user = _EXPANSION_USER.format(question=question)
        try:
            raw = self._llm.complete(system=system, user=user)
            paraphrases = _parse_paraphrases(raw)
        except Exception:  # pragma: no cover — network/model errors
            logger.warning(
                "Query expansion LLM call failed; falling back to original query",
                exc_info=True,
            )
            paraphrases = []

        # Always include the original; cap at n paraphrases.
        all_queries = [question] + paraphrases[: self._n]
        logger.debug(
            "QueryExpander: %d queries for '%s'", len(all_queries), question[:60]
        )
        return all_queries

    def _merge(self, queries: list[str], top_k: int) -> list[tuple[Chunk, float]]:
        """Retrieve for every query and merge by chunk_id (first-wins dedup)."""
        seen: dict[str, tuple[Chunk, float]] = {}
        # Fetch more candidates per sub-query so the merged pool is large enough.
        per_query_k = max(top_k, 20)
        for q in queries:
            for chunk, score in self._retriever.search(q, top_k=per_query_k):
                if chunk.chunk_id not in seen:
                    seen[chunk.chunk_id] = (chunk, score)

        # Sort by descending score and return top_k.
        merged = sorted(seen.values(), key=lambda t: t[1], reverse=True)
        return merged[:top_k]
