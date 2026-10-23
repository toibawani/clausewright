"""Cross-reference normalization and expansion (closes #24).

Contracts frequently reference other clauses, exhibits, or schedules
("subject to Section 12.3(b)", "in accordance with Schedule B").
Without resolving and retrieving the referenced clause, questions touching
cross-referenced terms cannot be answered accurately.

This module provides:
1. Reference extraction and normalization for:
   - Section / Sec. (e.g. "Section 12.3(b)", "Sec. 4.1")
   - Clause (e.g. "Clause 5.2")
   - Article (e.g. "Article 3", "Article IV")
   - Exhibit / Schedule / Attachment / Appendix (e.g. "Exhibit A", "Schedule B")
2. CrossReferenceExpander: attaches referenced chunks to candidate sets.
3. CrossReferenceRetriever: wraps any SupportsSearch retriever to expand cross-references.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for contract cross-references
# ---------------------------------------------------------------------------

# Matches patterns like:
# - Section 12.3(b), Sec. 12.3, Section 8.4, Section 9.1(b)
# - Clause 4, Clause 4.1
# - Article 5, Article IV, Article 5.2
# - Exhibit A, Schedule B, Attachment 1, Appendix C
_XREF_PATTERN = re.compile(
    r"\b(?P<type>Section|Sec\.|Clause|Article|Exhibit|Schedule|Attachment|Appendix)\s+"
    r"(?P<id>[A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*(?:\([A-Za-z0-9]+\))*)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedReference:
    """A normalized cross-reference target."""

    raw: str
    ref_type: str
    identifier: str
    canonical: str


def normalize_reference(raw: str) -> str:
    """Normalize a raw cross-reference string to standard form.

    Examples
    --------
    >>> normalize_reference("sec. 12.3(b)")
    'Section 12.3(b)'
    >>> normalize_reference("clause  4.1")
    'Clause 4.1'
    >>> normalize_reference("exhibit b")
    'Exhibit B'
    """
    raw = raw.strip()
    match = _XREF_PATTERN.search(raw)
    if not match:
        return raw

    ref_type_raw = match.group("type").lower()
    ident = match.group("id")

    if ref_type_raw.startswith("sec"):
        norm_type = "Section"
    elif ref_type_raw == "clause":
        norm_type = "Clause"
    elif ref_type_raw == "article":
        norm_type = "Article"
    elif ref_type_raw == "exhibit":
        norm_type = "Exhibit"
        ident = ident.upper()
    elif ref_type_raw == "schedule":
        norm_type = "Schedule"
        ident = ident.upper()
    elif ref_type_raw == "attachment":
        norm_type = "Attachment"
    elif ref_type_raw == "appendix":
        norm_type = "Appendix"
        ident = ident.upper()
    else:
        norm_type = match.group("type").capitalize()

    return f"{norm_type} {ident}"


def extract_cross_references(text: str) -> list[NormalizedReference]:
    """Find and normalize all cross-references in *text*."""
    results: list[NormalizedReference] = []
    seen: set[str] = set()

    for match in _XREF_PATTERN.finditer(text):
        raw = match.group(0)
        canonical = normalize_reference(raw)
        if canonical in seen:
            continue
        seen.add(canonical)

        ref_type = canonical.split()[0]
        ident = canonical.split(maxsplit=1)[1] if " " in canonical else ""

        results.append(
            NormalizedReference(
                raw=raw,
                ref_type=ref_type,
                identifier=ident,
                canonical=canonical,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Expander
# ---------------------------------------------------------------------------


class SupportsSearch(Protocol):
    """Retrieval protocol compatible with HybridRetriever."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]: ...


class CrossReferenceExpander:
    """Resolves cross-references from chunks against a document chunk index.

    Parameters
    ----------
    chunks:
        All available chunks in the document or corpus.
    enabled:
        Feature flag; if False, returns candidates unchanged.
    max_expansions_per_chunk:
        Maximum cross-referenced chunks to pull per candidate chunk.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        enabled: bool = True,
        max_expansions_per_chunk: int = 3,
    ) -> None:
        self.enabled = enabled
        self.max_expansions = max_expansions_per_chunk
        self._index: dict[str, Chunk] = {}
        self._build_index(chunks)

    def _build_index(self, chunks: Sequence[Chunk]) -> None:
        """Map identifiers and normalized references to chunks."""
        for c in chunks:
            # 1. By chunk_id
            self._index[c.chunk_id.lower()] = c

            # 2. By section_number (e.g. "8.4", "14.2")
            sec_num = c.section_number.strip().lower()
            if sec_num:
                self._index[sec_num] = c
                self._index[f"section {sec_num}"] = c
                self._index[f"clause {sec_num}"] = c
                self._index[f"article {sec_num}"] = c

            # 3. By section_title or text prefix if schedule/exhibit
            text_head = c.text[:120].lower()
            for prefix in ("schedule", "exhibit", "attachment", "appendix"):
                if prefix in text_head:
                    for match in _XREF_PATTERN.finditer(c.text[:120]):
                        canon = normalize_reference(match.group(0)).lower()
                        self._index[canon] = c

    def lookup(self, ref: NormalizedReference | str) -> Chunk | None:
        """Look up a referenced chunk in the document index."""
        if isinstance(ref, str):
            key = ref.strip().lower()
            norm = normalize_reference(ref).lower()
        else:
            key = ref.identifier.lower()
            norm = ref.canonical.lower()

        # Check canonical full form first ("section 8.4", "schedule b")
        if norm in self._index:
            return self._index[norm]

        # Check identifier alone ("8.4", "b")
        if key in self._index:
            return self._index[key]

        # Check stripped identifier without parens: "9.1(b)" -> "9.1"
        stripped = re.sub(r"\(.*?\)", "", key).strip()
        if stripped and stripped in self._index:
            return self._index[stripped]

        return None

    def expand(
        self,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Expand *candidates* by resolving references in their text.

        Referenced chunks are prepended immediately before the referencing chunk,
        inheriting 95% of its relevance score.
        """
        if not self.enabled or not candidates:
            return list(candidates)

        existing_ids = {c.chunk_id for c, _ in candidates}
        expanded: list[tuple[Chunk, float]] = []

        for chunk, score in candidates:
            refs = extract_cross_references(chunk.text)
            resolved_for_chunk: list[Chunk] = []

            for ref in refs:
                if len(resolved_for_chunk) >= self.max_expansions:
                    break
                target = self.lookup(ref)
                if target is not None and target.chunk_id not in existing_ids:
                    resolved_for_chunk.append(target)
                    existing_ids.add(target.chunk_id)

            # Prepend referenced chunks before the host chunk
            ref_score = score * 0.95
            for ref_chunk in resolved_for_chunk:
                expanded.append((ref_chunk, ref_score))

            expanded.append((chunk, score))

        return expanded


class CrossReferenceRetriever:
    """Retriever wrapper that expands cross-references in search results.

    Implements the SupportsSearch protocol.
    """

    def __init__(
        self,
        base_retriever: SupportsSearch,
        chunks: Sequence[Chunk],
        *,
        enabled: bool = True,
        max_expansions_per_chunk: int = 3,
    ) -> None:
        self.base_retriever = base_retriever
        self.expander = CrossReferenceExpander(
            chunks=chunks,
            enabled=enabled,
            max_expansions_per_chunk=max_expansions_per_chunk,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search via base_retriever, then expand cross-references."""
        candidates = self.base_retriever.search(query, top_k=top_k)
        expanded = self.expander.expand(candidates)
        return expanded[:top_k]
