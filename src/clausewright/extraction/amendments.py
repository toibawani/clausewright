"""Amendment and version linker: detect overridden clauses (closes #31).

Problem
-------
A contract with amendments can have contradictory clauses across the base
document and subsequent amendments.  The system must:

1. Accept an ordered list of documents (base, amendment1, amendment2, …).
2. Link amendment chunks to the base clauses they modify.
3. Flag overridden base clauses in retrieval results.

Design
------
``AmendmentLinker`` operates on already-chunked documents.  It detects
*override candidates* via two signals:

* **Section overlap** — an amendment chunk's ``section_number`` matches (or
  closely resembles) a base chunk's ``section_number``.  These are always
  considered override candidates.
* **Text similarity** — token-level Jaccard similarity between an amendment
  chunk's text and a base chunk's text exceeding a configurable threshold.

Override resolution rule (later-wins):
  If document D_i amends D_j (i > j), every chunk from D_i that links to a
  base chunk from D_j causes that base chunk to be marked ``is_overridden=True``.

The output is a ``VersionedCorpus`` containing:
* All chunks across all document versions, each extended with version metadata.
* A mapping from base ``chunk_id`` → override ``chunk_id`` (the amendment that
  supersedes it).

Integration with retrieval
--------------------------
``VersionedCorpus.active_chunks`` returns only non-overridden chunks, so the
retriever never surfaces a clause that has been superseded.
``VersionedCorpus.flagged_chunks`` returns all overridden chunks with a
``is_overridden`` flag for UI display.

Usage
-----
::

    from clausewright.extraction.amendments import AmendmentLinker, DocVersion

    linker = AmendmentLinker()
    corpus = linker.link([
        DocVersion(doc_id="base", version=0, chunks=base_chunks),
        DocVersion(doc_id="amend1", version=1, chunks=amend1_chunks),
    ])
    active = corpus.active_chunks   # no overridden clauses
    flagged = corpus.flagged_chunks  # (chunk, override_chunk_id) pairs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_SIMILARITY_THRESHOLD: float = 0.35


# ---------------------------------------------------------------------------
# Extended chunk wrapper
# ---------------------------------------------------------------------------


@dataclass
class VersionedChunk:
    """A ``Chunk`` annotated with version and override metadata."""

    chunk: Chunk
    doc_version: int
    """0 = base, 1 = first amendment, 2 = second, …"""
    is_overridden: bool = False
    """True when a later amendment has superseded this chunk."""
    overridden_by: str | None = None
    """``chunk_id`` of the amendment chunk that supersedes this one."""


# ---------------------------------------------------------------------------
# Input descriptor
# ---------------------------------------------------------------------------


@dataclass
class DocVersion:
    """One document version in the amendment chain.

    Parameters
    ----------
    doc_id:
        Unique document identifier (matches ``Chunk.doc_id``).
    version:
        Ordinal position in the chain (0 = base, ascending).
    chunks:
        All chunks belonging to this document version.
    """

    doc_id: str
    version: int
    chunks: list[Chunk]


# ---------------------------------------------------------------------------
# Output corpus
# ---------------------------------------------------------------------------


@dataclass
class VersionedCorpus:
    """The result of amendment linking: all chunks with override flags."""

    _versioned: list[VersionedChunk] = field(default_factory=list)
    _override_map: dict[str, str] = field(default_factory=dict)
    """Maps overridden chunk_id → overriding chunk_id."""

    @property
    def active_chunks(self) -> list[Chunk]:
        """Return only non-overridden chunks (safe for retrieval)."""
        return [vc.chunk for vc in self._versioned if not vc.is_overridden]

    @property
    def flagged_chunks(self) -> list[tuple[Chunk, str]]:
        """Return ``(overridden_chunk, overriding_chunk_id)`` pairs."""
        return [
            (vc.chunk, vc.overridden_by or "")
            for vc in self._versioned
            if vc.is_overridden
        ]

    @property
    def all_versioned(self) -> list[VersionedChunk]:
        """All chunks in version order, including overridden ones."""
        return list(self._versioned)

    def override_map(self) -> dict[str, str]:
        """Copy of the base-chunk → amendment-chunk override mapping."""
        return dict(self._override_map)

    def summary(self) -> str:
        """Human-readable summary of the corpus state."""
        total = len(self._versioned)
        overridden = sum(1 for vc in self._versioned if vc.is_overridden)
        active = total - overridden
        return (
            f"VersionedCorpus: {total} chunks total, "
            f"{active} active, {overridden} overridden"
        )


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def _section_prefix(section_number: str) -> str:
    """Return the first two dot-separated components for coarse matching.

    e.g. '12.3.b' → '12.3'
    """
    parts = section_number.split(".")
    return ".".join(parts[:2])


class AmendmentLinker:
    """Link amendment chunks to base clauses and flag overridden provisions.

    Parameters
    ----------
    similarity_threshold:
        Jaccard similarity threshold above which two chunks in different
        document versions are considered to address the same provision.
        The default (0.35) is intentionally conservative — legal text is
        verbose and a lower threshold produces too many false positives.
    """

    def __init__(self, similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        self._threshold = similarity_threshold

    def link(self, versions: list[DocVersion]) -> VersionedCorpus:
        """Process an ordered list of document versions and return a corpus.

        Parameters
        ----------
        versions:
            Ordered from oldest (base, version=0) to most recent amendment.

        Returns
        -------
        VersionedCorpus
            All chunks annotated with version and override metadata.

        Raises
        ------
        ValueError
            If *versions* is empty or versions are not in ascending order.
        """
        if not versions:
            raise ValueError("versions must be non-empty")
        sorted_versions = sorted(versions, key=lambda v: v.version)
        if sorted_versions[0].version != 0:
            raise ValueError("versions must include a base document with version=0")

        # Build corpus starting with base.
        corpus = VersionedCorpus()
        base_versioned: dict[str, VersionedChunk] = {}

        for dv in sorted_versions:
            new_vchunks: list[VersionedChunk] = []

            for chunk in dv.chunks:
                vc = VersionedChunk(chunk=chunk, doc_version=dv.version)
                new_vchunks.append(vc)

                if dv.version > 0:
                    # Find base chunks this amendment chunk overrides.
                    self._mark_overrides(vc, base_versioned, corpus)

            corpus._versioned.extend(new_vchunks)

            if dv.version == 0:
                base_versioned = {vc.chunk.chunk_id: vc for vc in new_vchunks}

        logger.info(corpus.summary())
        return corpus

    def _mark_overrides(
        self,
        amendment_vc: VersionedChunk,
        base_versioned: dict[str, VersionedChunk],
        corpus: VersionedCorpus,
    ) -> None:
        """Find base chunks overridden by *amendment_vc* and mark them."""
        amend_chunk = amendment_vc.chunk
        amend_prefix = _section_prefix(amend_chunk.section_number)

        for base_cid, base_vc in base_versioned.items():
            if base_vc.is_overridden:
                continue  # Already marked by an earlier amendment.

            base_chunk = base_vc.chunk
            base_prefix = _section_prefix(base_chunk.section_number)

            # Signal 1: section prefix match.
            section_match = amend_prefix == base_prefix

            # Signal 2: text similarity exceeds threshold.
            sim = _jaccard_similarity(amend_chunk.text, base_chunk.text)
            similarity_match = sim >= self._threshold

            if section_match or similarity_match:
                base_vc.is_overridden = True
                base_vc.overridden_by = amend_chunk.chunk_id
                corpus._override_map[base_cid] = amend_chunk.chunk_id
                logger.debug(
                    "Override detected: base '%s' → amendment '%s' "
                    "(section_match=%s, jaccard=%.3f)",
                    base_cid,
                    amend_chunk.chunk_id,
                    section_match,
                    sim,
                )
