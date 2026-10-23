"""Defined terms resolution and definition chunk attachment (closes #25).

Legal contracts use capitalized defined terms (e.g. "Material Adverse Effect",
"Confidential Information", "Change of Control") whose substantive scope is
governed by a dedicated definitions section or inline definition clause.

Without attaching the definition chunk, questions touching defined terms risk
misinterpreting terms of art or missing essential carve-outs and thresholds.

This module provides:
1. Defined term detection: identifies candidate defined terms in contract clauses.
2. DefinitionsIndex: indexes definition clauses across contract chunks.
3. DefinedTermsExpander: finds defined terms in retrieved chunks and appends
   their definition chunks to the candidate evidence set.
4. DefinedTermsRetriever: wraps any SupportsSearch retriever with defined terms expansion.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# Common English and contract words that appear capitalized but are NOT specific defined terms
_COMMON_CAPITALIZED_STOPWORDS = frozenset({
    "A",
    "Agreement",
    "All",
    "An",
    "Appendix",
    "Article",
    "As",
    "At",
    "Attachment",
    "Clause",
    "Contract",
    "Date",
    "Each",
    "Either",
    "Exhibit",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "If",
    "In",
    "It",
    "Its",
    "No",
    "Not",
    "Neither",
    "On",
    "Or",
    "Parties",
    "Party",
    "Schedule",
    "Section",
    "She",
    "So",
    "Such",
    "The",
    "Their",
    "There",
    "They",
    "This",
    "To",
    "Upon",
    "We",
    "When",
    "Where",
    "Whereas",
    "Which",
    "Who",
    "Will",
    "With",
    "You",
    "Your",
})

# Patterns used to detect definitions in definition sections or inline text
_DEFINITION_PATTERNS = [
    # "Term" means / "Term" shall mean / "Term" has the meaning / 'Term' excludes
    re.compile(
        r'["“](?P<term>[^"”\n\r]{2,60})["”]\s+(?:means|shall mean|has the meaning|refers to|includes|excludes)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r"['‘](?P<term>[^'’\n\r]{2,60})['’]\s+(?:means|shall mean|has the meaning|refers to|includes|excludes)\b",
        re.IGNORECASE,
    ),
    # Section 1.1: Term means ... / Term: means ...
    re.compile(
        r"(?:Section\s+\d+(?:\.\d+)?[:\.\s]+)?(?P<term>[A-Z][A-Za-z0-9\s]{2,50})\s+(?:means|shall mean|excludes)\b",
    ),
    # (the "Term") / (each, a "Term")
    re.compile(
        r'\((?:each,?\s+a|the|collectively,?\s+the)\s+["“](?P<term>[^"”\n\r]{2,60})["”]\)',
        re.IGNORECASE,
    ),
]

# Pattern for multi-word or single capitalized terms in clause body: e.g. "Material Adverse Effect", "Change of Control"
_CAPITALIZED_PHRASE_PATTERN = re.compile(
    r"\b[A-Z][a-z0-9]+(?:\s+(?:of|and|the|for|in|to)\s+[A-Z][a-z0-9]+|\s+[A-Z][a-z0-9]+){1,5}\b"
)


@dataclass(frozen=True)
class DefinedTerm:
    """A detected or indexed defined term."""

    term: str
    chunk_id: str
    section_number: str


def extract_candidate_terms(text: str) -> list[str]:
    """Extract capitalized terms and multi-word legal phrases from *text*."""
    terms: list[str] = []
    seen: set[str] = set()

    # 1. Quoted terms: "Confidential Information"
    for m in re.finditer(r'["“]([^"”\n]{2,50})["”]', text):
        raw = m.group(1).strip()
        if raw and raw not in seen and raw not in _COMMON_CAPITALIZED_STOPWORDS:
            seen.add(raw)
            terms.append(raw)

    # 2. Multi-word Title Cased phrases: Material Adverse Effect, Change of Control
    for m in _CAPITALIZED_PHRASE_PATTERN.finditer(text):
        phrase = m.group(0).strip()
        words = phrase.split()
        if not words:
            continue
        # Skip if entirely composed of stopwords
        if all(w in _COMMON_CAPITALIZED_STOPWORDS for w in words):
            continue
        if phrase not in seen:
            seen.add(phrase)
            terms.append(phrase)

    return terms


class DefinitionsIndex:
    """Indexes definition clauses across all chunks of a contract."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._term_to_chunk: dict[str, Chunk] = {}
        self._build_index(chunks)

    def _build_index(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            text = chunk.text
            # Extract definitions matching definition patterns
            for pat in _DEFINITION_PATTERNS:
                for match in pat.finditer(text):
                    term = match.group("term").strip()
                    if term and term not in _COMMON_CAPITALIZED_STOPWORDS and len(term) >= 3:
                        norm = term.lower()
                        if norm not in self._term_to_chunk:
                            self._term_to_chunk[norm] = chunk

            # Also check section 1 or definitions heading
            sec_num = chunk.section_number.lower()
            if sec_num in ("1", "1.0", "definitions") or "definitions" in text[:60].lower():
                # Scan for any defined term markers
                for quote_match in re.finditer(r'["“]([^"”]{2,50})["”]', text):
                    t = quote_match.group(1).strip()
                    if t and t not in _COMMON_CAPITALIZED_STOPWORDS:
                        norm = t.lower()
                        if norm not in self._term_to_chunk:
                            self._term_to_chunk[norm] = chunk

    def lookup(self, term: str) -> Chunk | None:
        """Find the definition chunk for a given term, or None."""
        norm = term.strip().lower()
        if norm in self._term_to_chunk:
            return self._term_to_chunk[norm]

        # Partial token match if exact normalized match not found
        for indexed_term, chunk in self._term_to_chunk.items():
            if norm == indexed_term or norm in indexed_term:
                return chunk
        return None

    @property
    def defined_terms(self) -> list[str]:
        """List all indexed defined terms."""
        return list(self._term_to_chunk.keys())


class SupportsSearch(Protocol):
    """Retrieval protocol compatible with HybridRetriever."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


class DefinedTermsExpander:
    """Finds defined terms in retrieved chunks and appends their definition chunks.

    Parameters
    ----------
    chunks:
        All available chunks in the document or corpus.
    enabled:
        Feature flag; defaults to True (config flag enable_defined_terms).
    max_definitions_per_chunk:
        Maximum definition chunks to attach per candidate chunk.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        enabled: bool = True,
        max_definitions_per_chunk: int = 3,
    ) -> None:
        self.enabled = enabled
        self.max_definitions = max_definitions_per_chunk
        self.index = DefinitionsIndex(chunks)

    def expand(
        self,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Append definition chunks for any defined terms in *candidates*."""
        if not self.enabled or not candidates:
            return list(candidates)

        existing_ids = {c.chunk_id for c, _ in candidates}
        definition_chunks: list[tuple[Chunk, float]] = []

        for chunk, score in candidates:
            candidate_terms = extract_candidate_terms(chunk.text)
            attached_for_this_chunk = 0

            for term in candidate_terms:
                if attached_for_this_chunk >= self.max_definitions:
                    break
                def_chunk = self.index.lookup(term)
                if def_chunk is not None and def_chunk.chunk_id not in existing_ids:
                    # Give definition chunk 95% of host chunk score
                    def_score = score * 0.95
                    definition_chunks.append((def_chunk, def_score))
                    existing_ids.add(def_chunk.chunk_id)
                    attached_for_this_chunk += 1

        # Append definition chunks at the end of context
        return list(candidates) + definition_chunks


class DefinedTermsRetriever:
    """Retriever wrapper that expands defined terms in search results.

    Implements the SupportsSearch protocol.
    """

    def __init__(
        self,
        base_retriever: SupportsSearch,
        chunks: Sequence[Chunk],
        *,
        enabled: bool = True,
        max_definitions_per_chunk: int = 3,
    ) -> None:
        self.base_retriever = base_retriever
        self.expander = DefinedTermsExpander(
            chunks=chunks,
            enabled=enabled,
            max_definitions_per_chunk=max_definitions_per_chunk,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search via base_retriever, then attach definition chunks."""
        candidates = self.base_retriever.search(query, top_k=top_k)
        expanded = self.expander.expand(candidates)
        return expanded[:top_k]
