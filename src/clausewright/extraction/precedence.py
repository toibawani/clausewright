"""Order of Precedence Resolver for multi-document contract families.

In commercial transactions, contracts are rarely a single static document.
A contract family commonly comprises:
1. Amendments (amendment N > amendment N-1 > base agreement)
2. Statements of Work (SOW) or Order Forms
3. Master Services Agreement (MSA) / Base Agreement
4. Exhibits, Schedules, and Appendices

When terms conflict (e.g. payment terms in SOW 1: Net 15 vs. MSA: Net 30),
the Order of Precedence rule governs which clause controls.

This module provides:
1. DocumentTier classification based on title and filename.
2. OrderOfPrecedenceResolver to prioritize and resolve governing clauses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


class DocumentTier(IntEnum):
    """Hierarchy ranking for contract documents (higher number = higher legal priority)."""

    EXHIBIT = 10
    SCHEDULE = 20
    BASE_AGREEMENT = 30
    STATEMENT_OF_WORK = 40
    ORDER_FORM = 50
    AMENDMENT = 60


@dataclass(frozen=True)
class PrecedencePolicy:
    """Configurable precedence rules for a contract family."""

    tier_order: tuple[DocumentTier, ...] = (
        DocumentTier.AMENDMENT,
        DocumentTier.ORDER_FORM,
        DocumentTier.STATEMENT_OF_WORK,
        DocumentTier.BASE_AGREEMENT,
        DocumentTier.SCHEDULE,
        DocumentTier.EXHIBIT,
    )
    later_version_wins: bool = True


@dataclass(frozen=True)
class ContractDocumentMeta:
    """Metadata describing a document within a contract family."""

    doc_id: str
    filename: str
    tier: DocumentTier
    version: int = 1
    effective_date: str | None = None


@dataclass(frozen=True)
class ConflictResolution:
    """Record explaining which clause governed when a conflict or overlap occurred."""

    governing_chunk_id: str
    superseded_chunk_id: str
    governing_doc_id: str
    superseded_doc_id: str
    reason: str


def classify_document_tier(filename: str, content_preview: str = "") -> DocumentTier:
    """Infer the DocumentTier from filename and header text."""
    combined = f"{filename} {content_preview}".lower().replace("_", " ").replace("-", " ")

    if re.search(r"\bamendment\b|\baddendum\b|\bvariation\b", combined):
        return DocumentTier.AMENDMENT
    if re.search(r"\border\s+form\b|\bpurchase\s+order\b", combined):
        return DocumentTier.ORDER_FORM
    if re.search(r"\bstatement\s+of\s+work\b|\bsow\b", combined):
        return DocumentTier.STATEMENT_OF_WORK
    if re.search(r"\bschedule\b", combined):
        return DocumentTier.SCHEDULE
    if re.search(r"\bexhibit\b|\bappendix\b|\battachment\b", combined):
        return DocumentTier.EXHIBIT

    return DocumentTier.BASE_AGREEMENT


class OrderOfPrecedenceResolver:
    """Resolves precedence across multiple contract documents in a family."""

    def __init__(self, policy: PrecedencePolicy | None = None) -> None:
        self.policy = policy or PrecedencePolicy()
        self._doc_meta: dict[str, ContractDocumentMeta] = {}

    def register_document(
        self,
        doc_id: str,
        filename: str,
        *,
        tier: DocumentTier | None = None,
        version: int = 1,
        effective_date: str | None = None,
    ) -> ContractDocumentMeta:
        """Register a document with its tier and version."""
        inferred_tier = tier or classify_document_tier(filename)
        meta = ContractDocumentMeta(
            doc_id=doc_id,
            filename=filename,
            tier=inferred_tier,
            version=version,
            effective_date=effective_date,
        )
        self._doc_meta[doc_id] = meta
        return meta

    def get_document_rank(self, doc_id: str) -> tuple[int, int]:
        """Return (tier_rank, version) for sorting candidates by precedence.

        Higher tuple values indicate higher legal precedence.
        """
        meta = self._doc_meta.get(doc_id)
        if meta is None:
            return (int(DocumentTier.BASE_AGREEMENT), 1)
        return (int(meta.tier), meta.version)

    def prioritize_candidates(
        self,
        candidates: Sequence[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Sort retrieved candidates by document precedence while preserving retrieval quality.

        Applies a precedence multiplier based on document tier to ensure that
        valid amendments and SOWs naturally take priority over superseded base clauses.
        """
        adjusted: list[tuple[Chunk, float]] = []

        for chunk, score in candidates:
            tier_rank, version = self.get_document_rank(chunk.doc_id)
            # Boost score slightly according to tier hierarchy
            # Tier ranks range from 10 to 60; scale factor: 1.0 + (tier_rank / 100.0) * 0.1
            precedence_boost = 1.0 + (tier_rank / 60.0) * 0.15 + (version * 0.02)
            adjusted_score = score * precedence_boost
            adjusted.append((chunk, adjusted_score))

        adjusted.sort(key=lambda pair: pair[1], reverse=True)
        return adjusted
