# Clausewright

Retrieval and grounded question answering over contracts, built from scratch in Python.

Contracts are not flat text. They have a section hierarchy, cross-references
("subject to Section 12.3(b)"), and defined terms. Clausewright chunks along that
structure and retrieves with a **superpowered agentic RAG pipeline** that refuses
to hallucinate when evidence is weak.

## Architecture

```
[ User Query ]
     │
     ▼
[ QueryOptimizer ]         ← HyDE + Multi-Query Expansion (M8)
     │  3-5 variants + hypothetical document vector
     ▼
[ HybridRetriever ]        ← BM25 sparse + FAISS dense fused via RRF (M6)
     │  top-50 candidates
     ▼
[ CrossEncoderReranker ]   ← Cross-encoder re-scores (query, chunk) pairs (M7)
     │  top-5 hyper-relevant chunks
     ▼
[ Relevance Grader ]       ← LLM grades each chunk (CRAG loop)
     │
     ├──(relevant)────────► [ Generator (Claude Sonnet) ] → answer + citations
     │
     └──(low relevance)───► [ query rewrite + retry, max 2 loops ]
                                   │
                         (if still poor after retries)
                                   │
                                   ▼
                           ABSTAIN with explanation
```

## Milestone Status

| # | Branch | Description | Tests |
|---|--------|-------------|-------|
| M1 | `main` | Project skeleton, CI, PR template | — |
| M2 | `feat/ingestion-schemas` | Pydantic domain models (Block, Page, Section, Chunk) | 13 |
| M3 | `feat/pdf-loader` | PyMuPDF text extraction with layout hints | 7 |
| M4 | `feat/section-parser` | Section parser: regex + font heuristics | 22 |
| M5 | `feat/chunker` | Sliding-window chunker with page provenance | 22 |
| M6 | `feat/embedder` | BGE dense embedder + FAISS IndexFlatIP | 20 |
| M7 | `feat/hybrid-search` | BM25 + Dense hybrid search via RRF | 24 |
| M8 | `feat/reranker` | Cross-encoder reranker (MiniLM-L-6-v2) | 17 |
| M9 | `feat/query-optimizer` | HyDE + multi-query expansion (Claude Haiku) | 23 |
| M10 | `feat/agentic-rag` | Agentic CRAG pipeline + RAGResponse | 20 |

**Total: 182 tests, 0 failures**

## Tech Stack

| Component | Implementation |
|-----------|----------------|
| **Dense retrieval** | `BAAI/bge-small-en-v1.5` via `sentence-transformers` + FAISS |
| **Sparse retrieval** | `rank-bm25` (BM25Okapi) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Query optimizer** | Claude Haiku (HyDE + multi-query) |
| **Generator** | Claude Sonnet 4.5 |
| **Grader** | Claude Haiku (CRAG relevance grading) |
| **Schemas** | Pydantic v2 (frozen, strict) |

## Scope

- Public contracts only (see `data/manifest.csv`).
- Answers must cite section numbers actually retrieved.
- System abstains when evidence is insufficient — never hallucinates.

## Non-goals

- Legal advice. Engineering project only.
- Multi-tenant auth, deployment, or scale beyond a single machine.
- General-purpose "chat with any PDF".

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest          # 182 tests, ~7s
```

## Environment

```bash
export ANTHROPIC_API_KEY=your_key_here   # required for AgenticRAG at runtime
```
Tests mock all LLM calls — no API key needed for `pytest`.
