# ADR 0001: Storage and Index Architecture for Multi-Tenant Contract QA

## Status
Accepted

## Context
Clausewright requires a robust, secure, and deployable persistence layer to support the Service Layer (Milestone 3). The storage architecture must satisfy the following constraints:

1. **Multi-Tenant Isolation**: Queries and indexing must be strictly partitioned by tenant ID. Under no circumstances may Tenant A's documents, chunks, or audit logs be accessible to Tenant B.
2. **Incremental Indexing & Hard Deletion**: Duplicate uploads must be detected by content hash (SHA-256) to prevent redundant embedding. When a document is deleted, all associated text, chunks, embeddings, and cache entries must be permanently purged.
3. **Hybrid Search Support**: Both keyword (BM25) and dense vector (embeddings) search must operate with low latency and support metadata filtering (`section_type`, `contract_type`, `effective_date`, `doc_version`).
4. **Auditability & Metering**: Audit logs (who asked what, which clauses were returned, abstention status) and query cache must be stored without logging raw contract text or prompts.
5. **No Paid Services**: All components must be free, open-source, deployable locally, and runnable in CI without external cloud subscriptions.

### Options Considered

#### Option 1: SQLite (Relational Metadata + FTS5) with Local FAISS
- **Overview**: Use SQLite (WAL mode) for documents, chunks, metadata, audit trails, and cache. Use SQLite FTS5 for BM25 sparse search and local FAISS indexes (or in-memory FAISS partitioned by tenant) for vector search.
- **Pros**:
  - Zero external daemons or services required; embeds directly into the Python process.
  - ACID transactions guarantee clean document insertion and atomic hard-deletion.
  - Extremely fast in-process reads (< 1ms).
  - Trivial to run in GitHub Actions CI and local development.
  - SQLite FTS5 handles full-text keyword indexing and tenant filtering natively.
  - 100% free and open-source.
- **Cons**:
  - Multi-writer concurrency is bounded by SQLite write locking (mitigated by WAL mode and separate read connections).
  - Horizontal scaling across multiple hosts requires migrating to a client-server database.

#### Option 2: PostgreSQL + pgvector
- **Overview**: Single client-server relational database storing all metadata, documents, full-text search (tsvector/pg_trgm), and vector embeddings (HNSW via pgvector).
- **Pros**:
  - Unified storage engine for relational, full-text, and vector data.
  - Native Row-Level Security (RLS) for tenant isolation.
  - Scales to large concurrent workloads and distributed environments.
- **Cons**:
  - Requires running a separate PostgreSQL service daemon.
  - Higher setup overhead in developer environments and CI.
  - Heavyweight for small law firms or local deployments.

#### Option 3: Chroma DB
- **Overview**: Open-source embedding database with local persistence (DuckDB/ClickHouse/SQLite backend).
- **Pros**:
  - Simple Python API, handles chunk embedding and nearest-neighbor search.
- **Cons**:
  - Less mature relational transactional integrity for document upload/deletion cascades.
  - Limited sparse BM25 capabilities compared to SQLite FTS5 / rank-bm25.
  - Additional dependency footprint.

#### Option 4: Qdrant
- **Overview**: High-performance Rust-based vector search engine with payload filtering.
- **Pros**:
  - Excellent vector search speed and flexible metadata filtering.
- **Cons**:
  - Requires running a dedicated server daemon (Docker container).
  - Overkill for single-node / small-firm deployments.

#### Option 5: Flat Files + Pickle + FAISS
- **Overview**: Store document chunks as JSON/pickle files on disk and serialize FAISS index files per tenant.
- **Pros**:
  - Minimal code for read-only workflows.
- **Cons**:
  - No transactional safety; prone to corrupted state on concurrent writes.
  - Hard-deletion requires complex manual index compaction.
  - No indexed querying for audit logs or metadata filtering.

---

## Decision

We adopt **Option 1: Embedded SQLite (WAL mode) with FTS5 and Tenant-Partitioned FAISS**.

1. **Relational & Metadata Database (`clausewright.db`)**:
   - Manages `tenants`, `users`, `documents`, `chunks`, `audit_logs`, and `query_cache`.
   - SQLite `PRAGMA journal_mode = WAL` and `PRAGMA foreign_keys = ON`.
   - All queries enforce a `tenant_id` predicate at the database level.
2. **Sparse Search (BM25)**:
   - Powered by SQLite **FTS5** (or tenant-partitioned `rank-bm25`), providing native keyword search and exact tenant filtering.
3. **Dense Vector Search**:
   - In-process **FAISS CPU index** per tenant, persisted alongside document storage and rebuilt/updated incrementally upon document addition or hard deletion.
4. **Document Hard Deletion**:
   - `DELETE FROM documents WHERE id = :doc_id AND tenant_id = :tenant_id` cascades to all chunks, FTS entries, and cache records atomically. The tenant's vector index is refreshed to guarantee no residual vector remnants remain.

This design requires **zero paid services**, **zero external daemons**, executes cleanly in CI, and delivers sub-millisecond metadata operations.

---

## Consequences

### Positive
- **Simplicity & Zero Cost**: Self-contained Python application runnable via `uvicorn` without spinning up Docker or external databases.
- **Deterministic CI**: Tests execute instantly in isolated in-memory (`:memory:`) or temporary SQLite databases.
- **Tenant Security**: Database-enforced tenant filtering guarantees complete data isolation between firms.
- **Compliance & Privacy**: Hard-deletes are immediate and irreversible. No raw contract text is logged in application logs or audit tables.

### Negative / Trade-offs
- Single-node architecture: Suitable for single-instance deployments (standard for small law firms or local legal teams). If multi-instance horizontal scaling is needed in the future, the storage interface is abstracted so the repository layer can be swapped to PostgreSQL + pgvector without changing business logic.
