"""Metadata-filtered retrieval: section type, contract type, effective date (closes #28).

Problem
-------
Retrieving across all document sections ignores structural signals.  A question
about termination notice should weight termination-section chunks more heavily.
Metadata filtering lets callers restrict results to matching section types,
contract types, or document versions before RRF scores are compared.

Design
------
``MetadataFilter`` is a predicate-based post-filter applied to candidates
returned by any ``SupportsSearch`` retriever.  It does **not** change the
retrieval scores — it simply drops candidates that don't match the filter.
This keeps the implementation retriever-agnostic and easy to test.

Chunk metadata is extended at index build time via ``ChunkMeta``.  The
``Chunk`` schema's ``section_number`` field is always present; the new
optional metadata fields (``section_type``, ``contract_type``,
``effective_date``, ``doc_version``) are passed via a parallel mapping
``chunk_id -> ChunkMeta`` that the filter holds in memory.

Usage
-----
::

    from clausewright.retrieval.metadata_filter import MetadataFilter, ChunkMeta

    meta = {
        "doc1-1.1-0": ChunkMeta(section_type="termination"),
        "doc1-1.2-0": ChunkMeta(section_type="payment"),
    }
    filtered = MetadataFilter(base_retriever=retriever, chunk_meta=meta)
    results = filtered.search(
        "termination notice",
        top_k=5,
        section_type="termination",
    )
    # Only chunks with section_type == "termination" are returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkMeta:
    """Optional metadata attached to a chunk for filtering.

    All fields are optional — chunks without metadata pass every filter.
    """

    section_type: str | None = None
    """Semantic section type: 'termination', 'payment', 'liability', …"""
    contract_type: str | None = None
    """Contract category: 'NDA', 'SaaS', 'employment', …"""
    effective_date: str | None = None
    """ISO-8601 date string: '2024-01-01'."""
    doc_version: str | None = None
    """Version or amendment identifier: 'v1.0', 'amendment-1', …"""


class _SupportsSearch(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


@dataclass
class MetadataFilter:
    """Post-filter wrapper that applies metadata predicates to retrieval results.

    Parameters
    ----------
    base_retriever:
        Any ``SupportsSearch`` retriever (hybrid, HyDE, etc.).
    chunk_meta:
        Mapping from ``chunk_id`` to :class:`ChunkMeta`.  Chunks absent from
        this mapping are treated as having all-``None`` metadata (pass all
        filters).
    oversample:
        Multiplier on ``top_k`` when calling the base retriever to ensure
        that enough unfiltered candidates exist.  Default is 4×.
    """

    base_retriever: _SupportsSearch
    chunk_meta: dict[str, ChunkMeta] = field(default_factory=dict)
    oversample: int = 4

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        section_type: str | None = None,
        contract_type: str | None = None,
        effective_date: str | None = None,
        doc_version: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search with optional metadata constraints.

        Parameters
        ----------
        query:
            The search query.
        top_k:
            Maximum results to return after filtering.
        section_type:
            If given, keep only chunks whose ``section_type`` matches exactly.
        contract_type:
            If given, keep only chunks whose ``contract_type`` matches.
        effective_date:
            If given, keep only chunks whose ``effective_date`` matches.
        doc_version:
            If given, keep only chunks whose ``doc_version`` matches.

        Returns
        -------
        list[tuple[Chunk, float]]
            Filtered results, at most *top_k* items.
        """
        # Fetch more candidates so filtering doesn't leave us empty.
        n_fetch = min(top_k * self.oversample, top_k + 200)
        candidates = self.base_retriever.search(query, top_k=n_fetch)

        results: list[tuple[Chunk, float]] = []
        for chunk, score in candidates:
            if chunk.chunk_id in self.chunk_meta:
                meta = self.chunk_meta[chunk.chunk_id]
                if not self._matches(meta, section_type, contract_type, effective_date, doc_version):
                    continue
            results.append((chunk, score))
            if len(results) >= top_k:
                break

        n_dropped = len(candidates) - len(results)
        if n_dropped and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MetadataFilter: kept %d/%d candidates (dropped %d)",
                len(results),
                len(candidates),
                n_dropped,
            )
        return results

    @staticmethod
    def _matches(
        meta: ChunkMeta,
        section_type: str | None,
        contract_type: str | None,
        effective_date: str | None,
        doc_version: str | None,
    ) -> bool:
        """Return True if *meta* satisfies all non-None filter predicates."""
        if section_type is not None and meta.section_type != section_type:
            return False
        if contract_type is not None and meta.contract_type != contract_type:
            return False
        if effective_date is not None and meta.effective_date != effective_date:
            return False
        return doc_version is None or meta.doc_version == doc_version
