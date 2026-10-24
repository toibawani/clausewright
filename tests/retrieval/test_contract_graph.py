"""Unit tests for ContractGraph and neighborhood expansion (Pillar 1)."""

from __future__ import annotations

from clausewright.retrieval.contract_graph import (
    ContractGraph,
    DependencyType,
)
from clausewright.schemas import Chunk


def _make_chunk(chunk_id: str, section_number: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="test-doc",
        section_number=section_number,
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


def test_contract_graph_manual_edges() -> None:
    graph = ContractGraph()
    graph.add_edge("12.1", "12.2", DependencyType.EXCEPT_TO, "subject to Section 12.2")
    graph.add_edge("14.1", "8.2", DependencyType.MODIFIES, "notwithstanding Section 8.2")

    assert graph.node_count == 4
    assert graph.edge_count == 2

    deps = graph.get_dependencies("12.1")
    assert len(deps) == 1
    assert deps[0].target_id == "12.2"
    assert deps[0].edge_type == DependencyType.EXCEPT_TO

    incoming = graph.get_dependents("8.2")
    assert len(incoming) == 1
    assert incoming[0].source_id == "14.1"


def test_contract_graph_expand_neighborhood() -> None:
    graph = ContractGraph()
    graph.add_edge("1.1", "2.1", DependencyType.REFERENCES)
    graph.add_edge("2.1", "3.1", DependencyType.REFERENCES)
    graph.add_edge("4.1", "1.1", DependencyType.MODIFIES)

    # 1 hop from 1.1: out to 2.1, in from 4.1
    expanded = graph.expand_neighborhood(["1.1"], max_depth=1)
    assert "1.1" in expanded
    assert "2.1" in expanded
    assert "4.1" in expanded
    assert "3.1" not in expanded

    # 2 hops from 1.1: reaches 3.1
    expanded_2 = graph.expand_neighborhood(["1.1"], max_depth=2)
    assert "3.1" in expanded_2


def test_contract_graph_from_chunks() -> None:
    chunks = [
        _make_chunk(
            "c1",
            "12.1",
            "Except as provided in Section 12.3, aggregate liability is capped.",
        ),
        _make_chunk(
            "c2",
            "12.3",
            "Section 12.3: Liabilities for breach of confidentiality are uncapped.",
        ),
        _make_chunk(
            "c3",
            "14.1",
            "Notwithstanding Section 12.1, statutory remedies apply.",
        ),
    ]

    graph = ContractGraph.from_chunks(chunks)
    assert graph.node_count >= 2

    deps_12_1 = graph.get_dependencies("12.1")
    assert any(e.target_id == "12.3" and e.edge_type == DependencyType.EXCEPT_TO for e in deps_12_1)

    deps_14_1 = graph.get_dependencies("14.1")
    assert any(e.target_id == "12.1" and e.edge_type == DependencyType.MODIFIES for e in deps_14_1)


def test_contract_graph_to_dict() -> None:
    graph = ContractGraph()
    graph.add_edge("A", "B", DependencyType.REFERENCES)
    data = graph.to_dict()
    assert data["node_count"] == 2
    assert data["edge_count"] == 1
    assert len(data["edges"]) == 1
