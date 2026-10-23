"""Unit tests for LexiconGraph and scoped defined terms (Pillar 2)."""

from __future__ import annotations

from clausewright.retrieval.lexicon_graph import (
    DefinitionScope,
    LexiconGraph,
    ScopedDefinedTerm,
)
from clausewright.schemas import Chunk


def _make_chunk(chunk_id: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-terms",
        section_number=section,
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


def test_lexicon_graph_registration_and_scope() -> None:
    lexicon = LexiconGraph()

    # Global term
    lexicon.register_term(
        ScopedDefinedTerm(
            term="Confidential Information",
            definition_text="All non-public trade secrets and proprietary data",
            chunk_id="c1",
            section_number="1.1",
            scope=DefinitionScope.GLOBAL,
        )
    )

    # Section-specific override
    lexicon.register_term(
        ScopedDefinedTerm(
            term="Confidential Information",
            definition_text="Solely source code and encryption keys",
            chunk_id="c2",
            section_number="8.2",
            scope=DefinitionScope.SECTION_SPECIFIC,
            scoped_section="8.2",
        )
    )

    # Resolving in Section 8.2 gets the specific definition
    res_local = lexicon.resolve_definition("Confidential Information", active_section="8.2")
    assert res_local is not None
    assert "source code" in res_local.definition_text

    # Resolving elsewhere gets global definition
    res_global = lexicon.resolve_definition("Confidential Information", active_section="14.1")
    assert res_global is not None
    assert "trade secrets" in res_global.definition_text


def test_lexicon_graph_extract_from_chunks() -> None:
    chunks = [
        _make_chunk(
            "c1",
            "1.1",
            '"Affiliate" means any entity that directly or indirectly controls or is controlled by a party.',
        ),
        _make_chunk(
            "c2",
            "12.4",
            'For purposes of this Section 12.4, "Losses" shall mean all third-party judgments and legal costs.',
        ),
    ]

    lexicon = LexiconGraph()
    count = lexicon.extract_from_chunks(chunks)
    assert count >= 2

    aff = lexicon.resolve_definition("Affiliate")
    assert aff is not None
    assert aff.scope == DefinitionScope.GLOBAL

    losses = lexicon.resolve_definition("Losses", active_section="12.4")
    assert losses is not None
    assert losses.scope == DefinitionScope.SECTION_SPECIFIC


def test_lexicon_graph_enrich_query() -> None:
    lexicon = LexiconGraph()
    lexicon.register_term(
        ScopedDefinedTerm(
            term="Force Majeure",
            definition_text="Acts of God, war, pandemic or government action",
            chunk_id="c3",
            section_number="18",
            scope=DefinitionScope.GLOBAL,
        )
    )

    query = "Does a hurricane qualify as Force Majeure?"
    enriched = lexicon.enrich_query(query)
    assert "Acts of God" in enriched
    assert "[Defined Terms:" in enriched
