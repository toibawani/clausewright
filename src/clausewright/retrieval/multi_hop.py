"""Multi-Hop Query Router and Decomposition for compound legal questions (Pillar 3).

Complex legal questions frequently require synthesizing multiple distinct contract clauses:
1. General Rule vs. Exception/Carve-out (e.g. Liability Cap + Data Breach Carve-out)
2. Right vs. Procedure (e.g. Right to Terminate + Notice Period / Cure Window)
3. Indemnification Scope vs. Defense Procedure

A standard single-step retrieval often retrieves only the general rule (e.g. the liability cap clause)
while omitting the critical exception clause located 10 pages later.

This module provides:
- Automated decomposition of compound legal questions into targeted sub-queries.
- Multi-hop retrieval execution with reciprocal rank fusion across sub-queries.
- Sub-intent tagging on candidate chunks for explainability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubQuery:
    """A decomposed sub-query addressing one specific aspect of a compound question."""

    intent_label: str
    query_text: str
    priority_weight: float = 1.0


@dataclass(frozen=True)
class MultiHopCandidate:
    """A retrieved chunk annotated with the sub-queries that discovered it."""

    chunk: Chunk
    score: float
    matched_intents: tuple[str, ...]


class SupportsSearch(Protocol):
    """Protocol for any retriever component."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


# Patterns to detect compound legal question constructs
_COMPOUND_PATTERNS = [
    (
        r"\b(?:liability|damages?)\b.*\b(?:cap|limit|maximum)\b.*\b(?:except|carve|exclu|apply|breach)\b",
        [
            ("liability_cap", "limitation of liability aggregate cap maximum liability"),
            (
                "liability_carveouts",
                "exclusions carve-outs exceptions to limitation of liability gross negligence breach",
            ),
        ],
    ),
    (
        r"\b(?:terminate|termination)\b.*\b(?:right|ground|cause|convenience)\b.*\b(?:notice|day|period|cure)\b",
        [
            ("termination_rights", "termination for cause convenience breach insolvency rights"),
            (
                "termination_procedure",
                "prior written notice period days cure period termination process",
            ),
        ],
    ),
    (
        r"\b(?:indemnif\w+|hold\s+harmless)\b.*\b(?:scope|defense|settle|procedure)\b",
        [
            ("indemnity_scope", "indemnification obligations third party claims losses damages"),
            (
                "indemnity_procedure",
                "indemnification defense control notice settlement consent procedure",
            ),
        ],
    ),
]


class MultiHopQueryRouter:
    """Decomposes compound legal queries and orchestrates multi-stage retrieval."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def decompose(self, query: str) -> list[SubQuery]:
        """Decompose a complex question into sub-queries if compound patterns are detected."""
        clean = query.strip()

        for pattern, sub_specs in _COMPOUND_PATTERNS:
            if re.search(pattern, clean, re.IGNORECASE):
                logger.info("Detected compound query structure for: %s", clean[:60])
                return [
                    SubQuery(
                        intent_label=label,
                        query_text=f"{clean} {extra_terms}".strip(),
                    )
                    for label, extra_terms in sub_specs
                ]

        # Check for generic "and" conjunctions split
        if " and " in clean.lower():
            parts = [
                p.strip()
                for p in re.split(r"\band\b", clean, flags=re.IGNORECASE)
                if len(p.strip()) > 8
            ]
            if len(parts) >= 2:
                return [
                    SubQuery(intent_label=f"sub_part_{i + 1}", query_text=part)
                    for i, part in enumerate(parts[:3])
                ]

        # Single subquery fallback
        return [SubQuery(intent_label="primary", query_text=clean)]

    def retrieve_multi_hop(
        self,
        retriever: SupportsSearch,
        query: str,
        *,
        top_k_per_hop: int = 10,
        total_limit: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Execute multi-hop retrieval and merge candidates via Reciprocal Rank Fusion."""
        sub_queries = self.decompose(query)

        # Map from chunk_id -> (Chunk, rrf_score, set_of_intents)
        chunk_map: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = {}
        intent_map: dict[str, set[str]] = {}

        for sub_q in sub_queries:
            results = retriever.search(sub_q.query_text, top_k=top_k_per_hop)
            for rank, (chunk, _score) in enumerate(results, start=1):
                cid = chunk.chunk_id
                chunk_map[cid] = chunk

                # RRF calculation: weight / (rrf_k + rank)
                contribution = sub_q.priority_weight / (self.rrf_k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

                if cid not in intent_map:
                    intent_map[cid] = set()
                intent_map[cid].add(sub_q.intent_label)

        # Sort by total fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        final_results = [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_cids[:total_limit]]

        logger.debug(
            "MultiHopQueryRouter fused %d sub-queries into %d candidate chunks",
            len(sub_queries),
            len(final_results),
        )
        return final_results
