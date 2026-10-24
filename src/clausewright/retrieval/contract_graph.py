"""Explicit directed dependency graph for contract sections and clauses.

Contracts are not isolated text passages; they form a dense directed graph of
cross-references, exceptions, modifications, and definitions:
- "Subject to Section 12.3(b)" -> EXCEPT_TO
- "Notwithstanding Section 8.1" -> MODIFIES / SUPERSEDES
- "In accordance with Schedule B" -> REFERENCES
- "As defined in Section 1.1" -> DEFINES

This module constructs an in-memory directed graph of contract sections to allow
1-hop and multi-hop neighborhood expansion during retrieval so that queries
touching a governing clause automatically pull its referenced exceptions and conditions.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from clausewright.retrieval.cross_reference import extract_cross_references
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Semantic relationship between two contract clauses."""

    REFERENCES = "references"
    MODIFIES = "modifies"
    EXCEPT_TO = "except_to"
    SUPERSEDES = "supersedes"
    DEFINES = "defines"
    PARENT_OF = "parent_of"


# Regex to detect legal qualifiers surrounding cross references
_EXCEPT_PATTERN = re.compile(
    r"\b(subject\s+to|except\s+as\s+(?:otherwise\s+)?provided\s+in|other\s+than\s+as\s+set\s+forth\s+in)\b",
    re.IGNORECASE,
)
_MODIFIES_PATTERN = re.compile(
    r"\b(notwithstanding(?:\s+anything\s+to\s+the\s+contrary\s+in)?|without\s+limiting|amends|supersedes)\b",
    re.IGNORECASE,
)
_DEFINES_PATTERN = re.compile(
    r"\b(as\s+defined\s+in|meaning\s+(?:assigned|given)\s+in|set\s+forth\s+in\s+Section\s+1)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContractEdge:
    """A directed relationship from source section to target section."""

    source_id: str
    target_id: str
    edge_type: DependencyType
    evidence_text: str = ""


@dataclass
class ContractGraphNode:
    """Metadata for a section node in the contract graph."""

    section_id: str
    doc_id: str = ""
    title: str | None = None
    chunk_ids: list[str] = field(default_factory=list)


class ContractGraph:
    """Directed dependency graph over contract sections."""

    def __init__(self) -> None:
        self._nodes: dict[str, ContractGraphNode] = {}
        self._out_edges: dict[str, list[ContractEdge]] = defaultdict(list)
        self._in_edges: dict[str, list[ContractEdge]] = defaultdict(list)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(e) for e in self._out_edges.values())

    def add_node(
        self,
        section_id: str,
        *,
        doc_id: str = "",
        title: str | None = None,
        chunk_id: str | None = None,
    ) -> ContractGraphNode:
        """Register a section node in the graph."""
        norm_id = section_id.strip()
        if norm_id not in self._nodes:
            self._nodes[norm_id] = ContractGraphNode(
                section_id=norm_id,
                doc_id=doc_id,
                title=title,
            )
        node = self._nodes[norm_id]
        if chunk_id and chunk_id not in node.chunk_ids:
            node.chunk_ids.append(chunk_id)
        if title and not node.title:
            node.title = title
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: DependencyType = DependencyType.REFERENCES,
        evidence_text: str = "",
    ) -> None:
        """Add a directed edge from source_id to target_id."""
        s = source_id.strip()
        t = target_id.strip()
        if not s or not t or s == t:
            return

        self.add_node(s)
        self.add_node(t)

        edge = ContractEdge(
            source_id=s,
            target_id=t,
            edge_type=edge_type,
            evidence_text=evidence_text.strip(),
        )
        if edge not in self._out_edges[s]:
            self._out_edges[s].append(edge)
            self._in_edges[t].append(edge)

    def get_dependencies(
        self,
        section_id: str,
        edge_types: Sequence[DependencyType] | None = None,
    ) -> list[ContractEdge]:
        """Return all outgoing dependency edges from section_id."""
        edges = self._out_edges.get(section_id.strip(), [])
        if edge_types is None:
            return list(edges)
        type_set = set(edge_types)
        return [e for e in edges if e.edge_type in type_set]

    def get_dependents(
        self,
        section_id: str,
        edge_types: Sequence[DependencyType] | None = None,
    ) -> list[ContractEdge]:
        """Return all incoming edges pointing to section_id."""
        edges = self._in_edges.get(section_id.strip(), [])
        if edge_types is None:
            return list(edges)
        type_set = set(edge_types)
        return [e for e in edges if e.edge_type in type_set]

    def expand_neighborhood(
        self,
        seed_section_ids: Sequence[str],
        *,
        max_depth: int = 1,
        include_incoming: bool = True,
    ) -> set[str]:
        """Breadth-first expansion to discover all related contract sections.

        Parameters
        ----------
        seed_section_ids:
            Initial section IDs (e.g. from top retrieval candidates).
        max_depth:
            Max graph hops to traverse (default: 1).
        include_incoming:
            Whether to also include clauses that reference the seed (e.g. carve-outs).
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((s.strip(), 0) for s in seed_section_ids if s.strip())

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if depth >= max_depth:
                continue

            # Outgoing neighbors (what current depends on)
            for edge in self._out_edges.get(current, []):
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))

            # Incoming neighbors (what depends on or modifies current)
            if include_incoming:
                for edge in self._in_edges.get(current, []):
                    if edge.source_id not in visited:
                        queue.append((edge.source_id, depth + 1))

        return visited

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> ContractGraph:
        """Construct a contract graph by analyzing cross-references within chunks."""
        graph = cls()

        for chunk in chunks:
            sec_id = chunk.section_number
            graph.add_node(sec_id, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)

            # Analyze text for cross-references
            text = chunk.text
            norm_refs = extract_cross_references(text)
            for ref in norm_refs:
                raw_ref = ref.raw
                norm_target = ref.canonical
                # Strip prefix like 'Section ' to match section_number
                target_sec = re.sub(
                    r"^(Section|Sec\.|Clause|Article)\s+", "", norm_target, flags=re.IGNORECASE
                )

                # Determine edge type based on local window context
                window_start = max(0, text.find(raw_ref) - 50)
                window_end = min(len(text), text.find(raw_ref) + len(raw_ref) + 50)
                local_window = text[window_start:window_end]

                edge_type = DependencyType.REFERENCES
                if _EXCEPT_PATTERN.search(local_window):
                    edge_type = DependencyType.EXCEPT_TO
                elif _MODIFIES_PATTERN.search(local_window):
                    edge_type = DependencyType.MODIFIES
                elif _DEFINES_PATTERN.search(local_window):
                    edge_type = DependencyType.DEFINES

                graph.add_edge(
                    source_id=sec_id,
                    target_id=target_sec,
                    edge_type=edge_type,
                    evidence_text=local_window,
                )

        return graph

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph representation for debugging or API inspection."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": {
                k: {
                    "doc_id": v.doc_id,
                    "title": v.title,
                    "chunk_count": len(v.chunk_ids),
                }
                for k, v in self._nodes.items()
            },
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                }
                for edges in self._out_edges.values()
                for e in edges
            ],
        }
