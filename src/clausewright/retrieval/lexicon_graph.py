"""Contract Lexicon and Scoped Defined Term Graph (Pillar 2).

Legal contracts rely heavily on capitalized defined terms (e.g. "Affiliate",
"Confidential Information", "Indemnified Liabilities"). Crucially, definitions
can be:
1. GLOBAL: Defined in Article 1 (Definitions) and applying across the entire agreement.
2. SECTION_SPECIFIC: Defined inline (e.g., "For purposes of this Section 12, 'Losses' means...").
3. SCHEDULE_SPECIFIC: Defined only within a particular Schedule or Statement of Work.

This module tracks definition scopes and enables dynamic query and chunk enrichment
so the LLM receives the authoritative meaning of terms without misinterpreting terms of art.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


class DefinitionScope(str, Enum):
    """The jurisdictional scope of a defined term within a contract."""

    GLOBAL = "global"
    SECTION_SPECIFIC = "section_specific"
    SCHEDULE_SPECIFIC = "schedule_specific"


@dataclass(frozen=True)
class ScopedDefinedTerm:
    """A defined term with its authoritative text and scope."""

    term: str
    definition_text: str
    chunk_id: str
    section_number: str
    scope: DefinitionScope = DefinitionScope.GLOBAL
    scoped_section: str | None = None


# Pattern to identify inline definition phrasing
_INLINE_SCOPE_PATTERN = re.compile(
    r"\b(?:for\s+(?:the\s+)?purposes\s+of\s+this\s+Section\s+([A-Za-z0-9\.\-_]+)|in\s+this\s+Section\s+([A-Za-z0-9\.\-_]+))\b",
    re.IGNORECASE,
)

# Pattern to extract quoted definitions: "Term" means / shall mean
_DEFINITION_PATTERN = re.compile(
    r'["“\'](?P<term>[A-Z][A-Za-z0-9\s\-_]{1,60})["”\']\s+(?:means|shall\s+mean|refers\s+to|has\s+the\s+meaning)\s+(?P<definition>[^;\.\n]{10,500})',
    re.IGNORECASE,
)


class LexiconGraph:
    """Manages contract terminology definitions, active scopes, and enrichment."""

    def __init__(self) -> None:
        self._global_terms: dict[str, ScopedDefinedTerm] = {}
        # Key: (term.lower(), section_number)
        self._scoped_terms: dict[tuple[str, str], ScopedDefinedTerm] = {}
        self._all_terms: dict[str, list[ScopedDefinedTerm]] = {}

    @property
    def total_terms(self) -> int:
        return sum(len(v) for v in self._all_terms.values())

    def register_term(self, item: ScopedDefinedTerm) -> None:
        """Register a defined term with its associated scope."""
        key = item.term.strip().lower()
        if key not in self._all_terms:
            self._all_terms[key] = []
        self._all_terms[key].append(item)

        if item.scope == DefinitionScope.GLOBAL:
            self._global_terms[key] = item
        elif item.scoped_section:
            self._scoped_terms[(key, item.scoped_section)] = item

    def resolve_definition(
        self,
        term: str,
        active_section: str | None = None,
    ) -> ScopedDefinedTerm | None:
        """Find the most specific active definition for a term in context.

        If a section-scoped definition exists for ``active_section``, it takes
        precedence over the global definition.
        """
        key = term.strip().lower()

        # Check section-specific override
        if active_section:
            specific = self._scoped_terms.get((key, active_section))
            if specific:
                return specific

        # Fall back to global definition
        return self._global_terms.get(key)

    def extract_from_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Scan chunks to automatically populate the lexicon graph."""
        count = 0
        for chunk in chunks:
            text = chunk.text
            # Detect if chunk has inline section scoping
            scope = DefinitionScope.GLOBAL
            scoped_sec: str | None = None

            scope_match = _INLINE_SCOPE_PATTERN.search(text)
            if scope_match:
                scope = DefinitionScope.SECTION_SPECIFIC
                scoped_sec = scope_match.group(1) or scope_match.group(2) or chunk.section_number
            elif (
                "schedule" in chunk.section_number.lower()
                or "exhibit" in chunk.section_number.lower()
            ):
                scope = DefinitionScope.SCHEDULE_SPECIFIC
                scoped_sec = chunk.section_number

            for match in _DEFINITION_PATTERN.finditer(text):
                raw_term = match.group("term").strip()
                raw_def = match.group("definition").strip()
                if len(raw_term) < 2 or len(raw_term.split()) > 7:
                    continue

                item = ScopedDefinedTerm(
                    term=raw_term,
                    definition_text=raw_def,
                    chunk_id=chunk.chunk_id,
                    section_number=chunk.section_number,
                    scope=scope,
                    scoped_section=scoped_sec,
                )
                self.register_term(item)
                count += 1

        logger.info("LexiconGraph indexed %d defined terms across %d chunks", count, len(chunks))
        return count

    def enrich_query(self, query: str, *, max_terms: int = 3) -> str:
        """Enrich user query with authoritative definitions of mentioned defined terms."""
        terms_found: list[ScopedDefinedTerm] = []

        for term_key, term_list in self._all_terms.items():
            # Check word boundary match in query
            pattern = rf"\b{re.escape(term_key)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                terms_found.append(term_list[0])
                if len(terms_found) >= max_terms:
                    break

        if not terms_found:
            return query

        glossary_items = [f'"{t.term}": {t.definition_text}' for t in terms_found]
        return f"{query} [Defined Terms: {'; '.join(glossary_items)}]"
