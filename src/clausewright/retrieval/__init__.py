"""Retrieval components: dense embedding, hybrid BM25+FAISS search, reranking, and query expansion."""

from clausewright.retrieval.embedder import Embedder
from clausewright.retrieval.hybrid_search import HybridRetriever
from clausewright.retrieval.hyde import HyDERetriever
from clausewright.retrieval.metadata_filter import ChunkMeta, MetadataFilter
from clausewright.retrieval.query_expander import QueryExpander
from clausewright.retrieval.reranker import CrossEncoderReranker
from clausewright.retrieval.reranker_comparison import (
    RerankerComparisonReport,
    run_reranker_comparison,
)

__all__ = [
    "ChunkMeta",
    "CrossEncoderReranker",
    "Embedder",
    "HyDERetriever",
    "HybridRetriever",
    "MetadataFilter",
    "QueryExpander",
    "RerankerComparisonReport",
    "run_reranker_comparison",
]
