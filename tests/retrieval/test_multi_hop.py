"""Unit tests for MultiHopQueryRouter (Pillar 3)."""

from __future__ import annotations

from clausewright.retrieval.multi_hop import (
    MultiHopQueryRouter,
)
from clausewright.schemas import Chunk


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-hop",
        section_number="12",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


class FakeRetriever:
    """Mock retriever returning predetermined candidates per query keyword."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        q = query.lower()
        results: list[tuple[Chunk, float]] = []
        if "cap" in q or "liability" in q:
            results.append(
                (_make_chunk("chunk-cap", "Total liability is capped at $1,000,000."), 0.9)
            )
        if "carve" in q or "exception" in q or "breach" in q:
            results.append(
                (
                    _make_chunk(
                        "chunk-excl", "Confidentiality breaches are excluded from the cap."
                    ),
                    0.88,
                )
            )
        return results[:top_k]


def test_decompose_compound_liability_query() -> None:
    router = MultiHopQueryRouter()
    query = "What is the liability cap and what exceptions or carve-outs apply?"
    sub_queries = router.decompose(query)

    assert len(sub_queries) == 2
    labels = {sq.intent_label for sq in sub_queries}
    assert "liability_cap" in labels
    assert "liability_carveouts" in labels


def test_decompose_simple_query() -> None:
    router = MultiHopQueryRouter()
    query = "Who are the parties to this agreement?"
    sub_queries = router.decompose(query)
    assert len(sub_queries) == 1
    assert sub_queries[0].intent_label == "primary"


def test_retrieve_multi_hop_fuses_candidates() -> None:
    router = MultiHopQueryRouter()
    retriever = FakeRetriever()
    query = "What is the liability cap and what exceptions or carve-outs apply?"

    fused = router.retrieve_multi_hop(retriever, query, top_k_per_hop=5, total_limit=10)
    assert len(fused) == 2
    chunk_ids = {c.chunk_id for c, _ in fused}
    assert "chunk-cap" in chunk_ids
    assert "chunk-excl" in chunk_ids
