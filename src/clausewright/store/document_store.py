"""Tenant-isolated document storage layer.

Architecture
------------
Each tenant's data lives in a dedicated SQLite database file::

    data/store/<tenant_id>/documents.db

The database contains:

* ``documents`` table — one row per ingested document with content hash,
  upload timestamp, file name, and page count.
* ``chunks`` virtual FTS5 table — all chunks in plaintext for BM25
  full-text search, partitioned by ``tenant_id`` column for cross-tenant
  safety checks.
* ``chunks_meta`` table — chunk provenance (doc_id, section_number,
  chunk_index, page_range, word_count); joined with FTS5 at query time.

A per-tenant FAISS ``IndexFlatIP`` is rebuilt from chunks_meta whenever
documents are added or deleted.  The FAISS index is kept in-process only
(it lives under ``data/store/<tenant_id>/faiss.index`` on disk for
restart survival).

Security invariants
-------------------
* Every SQL query that touches ``chunks`` or ``chunks_meta`` always
  includes ``WHERE tenant_id = ?``.  There is no cross-tenant query path.
* Content hash deduplication prevents the same file from being indexed
  twice by the same tenant.
* Hard-delete removes both DB rows **and** the FAISS vectors (via full
  rebuild — FAISS does not support in-place deletion on FlatIP).

Thread safety
-------------
``DocumentStore`` is designed to be used inside a single-threaded async
event loop (FastAPI).  SQLite WAL mode permits concurrent readers; all
writes are serialised through Python's GIL + the WAL lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]

from clausewright.retrieval.embedder import Embedder
from clausewright.schemas import Chunk

logger = logging.getLogger(__name__)

# SQL DDL ------------------------------------------------------------------

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    filename    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at REAL NOT NULL,
    UNIQUE (tenant_id, content_hash)
);
"""

_DDL_CHUNKS_META = """
CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,
    section_number  TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    page_start      INTEGER NOT NULL,
    page_end        INTEGER NOT NULL,
    word_count      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_meta_tenant
    ON chunks_meta (tenant_id, doc_id);
"""

_DDL_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    tenant_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);
"""

_DDL_PRAGMAS = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA synchronous = NORMAL;",
]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentRecord:
    """Lightweight representation of a stored document."""

    doc_id: str
    tenant_id: str
    filename: str
    content_hash: str
    page_count: int
    chunk_count: int
    uploaded_at: float  # Unix epoch seconds


# ---------------------------------------------------------------------------
# Store implementation
# ---------------------------------------------------------------------------


class DocumentStore:
    """Manage tenant-isolated document metadata, FTS5 search, and FAISS index.

    Parameters
    ----------
    store_root:
        Base directory under which per-tenant subdirectories are created.
        Default: ``data/store`` relative to CWD.
    embedder:
        An initialised :class:`~clausewright.retrieval.embedder.Embedder`.
        If ``None``, dense search is unavailable (FTS5-only mode).
    """

    def __init__(
        self,
        store_root: Path,
        embedder: Embedder | None = None,
    ) -> None:
        self._root = store_root
        self._embedder = embedder
        # tenant_id -> sqlite3.Connection
        self._connections: dict[str, sqlite3.Connection] = {}
        # tenant_id -> (faiss.Index, list[Chunk])
        self._faiss_cache: dict[str, tuple[faiss.Index, list[Chunk]]] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _db_path(self, tenant_id: str) -> Path:
        p = self._root / tenant_id
        p.mkdir(parents=True, exist_ok=True)
        return p / "documents.db"

    def _faiss_path(self, tenant_id: str) -> Path:
        return self._root / tenant_id / "faiss.index"

    def _chunks_path(self, tenant_id: str) -> Path:
        return self._root / tenant_id / "chunks.json"

    def _conn(self, tenant_id: str) -> sqlite3.Connection:
        if tenant_id not in self._connections:
            db = self._db_path(tenant_id)
            conn = sqlite3.connect(str(db), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            for pragma in _DDL_PRAGMAS:
                conn.execute(pragma)
            conn.execute(_DDL_DOCUMENTS)
            for stmt in _DDL_CHUNKS_META.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(s)
            conn.execute(_DDL_CHUNKS_FTS)
            conn.commit()
            self._connections[tenant_id] = conn
            logger.info("Opened SQLite store for tenant=%s at %s", tenant_id, db)
        return self._connections[tenant_id]

    # ------------------------------------------------------------------
    # Public CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(raw_bytes: bytes) -> str:
        """Return SHA-256 hex digest of *raw_bytes*."""
        return hashlib.sha256(raw_bytes).hexdigest()

    def exists_by_hash(self, tenant_id: str, content_hash: str) -> str | None:
        """Return the existing ``doc_id`` if this content hash is already stored, else ``None``."""
        conn = self._conn(tenant_id)
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE tenant_id = ? AND content_hash = ?",
            (tenant_id, content_hash),
        ).fetchone()
        return str(row["doc_id"]) if row else None

    def add_document(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        filename: str,
        content_hash: str,
        chunks: list[Chunk],
        page_count: int,
    ) -> DocumentRecord:
        """Persist a document and its chunks; rebuild the FAISS index.

        Parameters
        ----------
        doc_id:
            Unique document identifier (caller-supplied, e.g. UUID).
        tenant_id:
            Owning tenant.
        filename:
            Original file name (for display only; not a path).
        content_hash:
            SHA-256 hex digest of the raw PDF bytes.
        chunks:
            Ordered list of ``Chunk`` objects produced by the chunker.
        page_count:
            Number of pages in the document.

        Returns
        -------
        DocumentRecord
            The newly created record.

        Raises
        ------
        ValueError
            If a document with the same ``content_hash`` already exists for
            this tenant (call :meth:`exists_by_hash` first to get the id).
        """
        conn = self._conn(tenant_id)
        now = time.time()

        try:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, tenant_id, filename, content_hash, page_count, chunk_count, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, tenant_id, filename, content_hash, page_count, len(chunks), now),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Document with hash {content_hash!r} already exists for tenant {tenant_id!r}"
            ) from exc

        # Insert chunk metadata and FTS rows
        for chunk in chunks:
            conn.execute(
                """
                INSERT OR IGNORE INTO chunks_meta
                    (chunk_id, doc_id, tenant_id, section_number, chunk_index,
                     page_start, page_end, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    doc_id,
                    tenant_id,
                    chunk.section_number,
                    chunk.chunk_index,
                    chunk.page_range[0],
                    chunk.page_range[1],
                    chunk.word_count,
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, tenant_id, text) VALUES (?, ?, ?)",
                (chunk.chunk_id, tenant_id, chunk.text),
            )

        conn.commit()
        logger.info(
            "Stored doc_id=%s for tenant=%s (%d chunks)",
            doc_id,
            tenant_id,
            len(chunks),
        )

        # Rebuild FAISS index for this tenant
        self._rebuild_faiss(tenant_id)

        return DocumentRecord(
            doc_id=doc_id,
            tenant_id=tenant_id,
            filename=filename,
            content_hash=content_hash,
            page_count=page_count,
            chunk_count=len(chunks),
            uploaded_at=now,
        )

    def list_documents(self, tenant_id: str) -> list[DocumentRecord]:
        """Return all documents owned by *tenant_id*, newest first."""
        conn = self._conn(tenant_id)
        rows = conn.execute(
            """
            SELECT doc_id, tenant_id, filename, content_hash, page_count, chunk_count, uploaded_at
            FROM documents
            WHERE tenant_id = ?
            ORDER BY uploaded_at DESC
            """,
            (tenant_id,),
        ).fetchall()
        return [
            DocumentRecord(
                doc_id=r["doc_id"],
                tenant_id=r["tenant_id"],
                filename=r["filename"],
                content_hash=r["content_hash"],
                page_count=r["page_count"],
                chunk_count=r["chunk_count"],
                uploaded_at=r["uploaded_at"],
            )
            for r in rows
        ]

    def get_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        """Return the record for *doc_id* if it belongs to *tenant_id*."""
        conn = self._conn(tenant_id)
        row = conn.execute(
            """
            SELECT doc_id, tenant_id, filename, content_hash, page_count, chunk_count, uploaded_at
            FROM documents
            WHERE tenant_id = ? AND doc_id = ?
            """,
            (tenant_id, doc_id),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            doc_id=row["doc_id"],
            tenant_id=row["tenant_id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            uploaded_at=row["uploaded_at"],
        )

    def delete_document(self, tenant_id: str, doc_id: str) -> bool:
        """Hard-delete a document and all its chunks; rebuild FAISS index.

        Returns ``True`` if the document existed and was deleted, ``False``
        if no matching document was found for this tenant.
        """
        conn = self._conn(tenant_id)

        # Verify ownership before deletion
        existing = self.get_document(tenant_id, doc_id)
        if existing is None:
            return False

        # Delete FTS entries for this doc's chunks
        chunk_ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks_meta WHERE tenant_id = ? AND doc_id = ?",
                (tenant_id, doc_id),
            ).fetchall()
        ]
        for cid in chunk_ids:
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))

        # ON DELETE CASCADE removes chunks_meta rows when document is deleted
        conn.execute(
            "DELETE FROM documents WHERE tenant_id = ? AND doc_id = ?",
            (tenant_id, doc_id),
        )
        conn.commit()

        logger.info("Hard-deleted doc_id=%s for tenant=%s", doc_id, tenant_id)

        # Invalidate FAISS cache and rebuild
        self._faiss_cache.pop(tenant_id, None)
        self._rebuild_faiss(tenant_id)
        return True

    # ------------------------------------------------------------------
    # BM25 / FTS5 search
    # ------------------------------------------------------------------

    def search_fts(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 10,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Full-text (BM25) search via SQLite FTS5.

        Parameters
        ----------
        tenant_id:
            Tenant scope — only documents belonging to this tenant are searched.
        query:
            FTS5 query string (supports phrase and prefix queries).
        top_k:
            Maximum number of results.
        doc_id:
            Optional: restrict search to a single document.

        Returns
        -------
        list[tuple[Chunk, float]]
            ``(chunk, bm25_score)`` pairs sorted by descending relevance.
            BM25 scores from FTS5 are *negative* (lower = less relevant);
            we return the negated value so higher = better.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")

        conn = self._conn(tenant_id)

        # FTS5's bm25() function returns negative scores; negate for consistency
        if doc_id:
            rows = conn.execute(
                """
                SELECT f.chunk_id, -bm25(chunks_fts) AS score,
                       m.doc_id, m.section_number, m.chunk_index,
                       m.page_start, m.page_end, m.word_count, f.text
                FROM chunks_fts f
                JOIN chunks_meta m ON m.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                  AND m.tenant_id = ?
                  AND m.doc_id = ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (query, tenant_id, doc_id, top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT f.chunk_id, -bm25(chunks_fts) AS score,
                       m.doc_id, m.section_number, m.chunk_index,
                       m.page_start, m.page_end, m.word_count, f.text
                FROM chunks_fts f
                JOIN chunks_meta m ON m.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                  AND m.tenant_id = ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (query, tenant_id, top_k),
            ).fetchall()

        results: list[tuple[Chunk, float]] = []
        for r in rows:
            chunk = Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                section_number=r["section_number"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                page_range=(r["page_start"], r["page_end"]),
                word_count=r["word_count"],
            )
            results.append((chunk, float(r["score"])))
        return results

    # ------------------------------------------------------------------
    # Dense (FAISS) search
    # ------------------------------------------------------------------

    def _load_chunks_for_tenant(self, tenant_id: str) -> list[Chunk]:
        """Load all chunks for tenant from the DB (for FAISS rebuild)."""
        conn = self._conn(tenant_id)
        rows = conn.execute(
            """
            SELECT m.chunk_id, m.doc_id, m.section_number, m.chunk_index,
                   m.page_start, m.page_end, m.word_count, f.text
            FROM chunks_meta m
            JOIN chunks_fts f ON f.chunk_id = m.chunk_id
            WHERE m.tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
        return [
            Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                section_number=r["section_number"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                page_range=(r["page_start"], r["page_end"]),
                word_count=r["word_count"],
            )
            for r in rows
        ]

    def _rebuild_faiss(self, tenant_id: str) -> None:
        """(Re)build the FAISS index from all current chunks for *tenant_id*."""
        if self._embedder is None:
            logger.debug("No embedder configured; skipping FAISS rebuild for %s", tenant_id)
            return

        chunks = self._load_chunks_for_tenant(tenant_id)
        if not chunks:
            # Empty tenant — create a zero-vector index placeholder
            dim = self._embedder.embedding_dim
            idx: faiss.Index = faiss.IndexFlatIP(dim)
            self._faiss_cache[tenant_id] = (idx, [])
            logger.info("FAISS index reset (empty) for tenant=%s", tenant_id)
            return

        logger.info("Rebuilding FAISS index for tenant=%s (%d chunks)…", tenant_id, len(chunks))
        index, ordered = self._embedder.build_index(chunks)
        self._faiss_cache[tenant_id] = (index, ordered)

        # Persist to disk for restart survival
        faiss_path = self._faiss_path(tenant_id)
        chunks_path = self._chunks_path(tenant_id)
        faiss.write_index(index, str(faiss_path))
        chunks_path.write_text(
            json.dumps([c.model_dump() for c in ordered], indent=2), encoding="utf-8"
        )
        logger.info("FAISS index persisted for tenant=%s at %s", tenant_id, faiss_path)

    def _get_faiss(self, tenant_id: str) -> tuple[faiss.Index, list[Chunk]] | None:
        """Return the in-process FAISS cache, loading from disk if necessary."""
        if tenant_id in self._faiss_cache:
            return self._faiss_cache[tenant_id]

        faiss_path = self._faiss_path(tenant_id)
        chunks_path = self._chunks_path(tenant_id)
        if faiss_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(faiss_path))
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [Chunk.model_validate(c) for c in raw]
            self._faiss_cache[tenant_id] = (index, chunks)
            logger.info(
                "FAISS index loaded from disk for tenant=%s (ntotal=%d)",
                tenant_id,
                index.ntotal,
            )
            return self._faiss_cache[tenant_id]
        return None

    def search_dense(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 10,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Dense vector search within a tenant's FAISS index.

        Parameters
        ----------
        tenant_id:
            Owning tenant scope.
        query:
            Natural-language query string.
        top_k:
            Maximum results.
        doc_id:
            Optional: post-filter to a single document.

        Returns
        -------
        list[tuple[Chunk, float]]
            Pairs sorted by descending cosine similarity.

        Raises
        ------
        RuntimeError
            If no embedder is configured.
        ValueError
            If the query is blank.
        """
        if self._embedder is None:
            raise RuntimeError("Dense search requires an embedder; none configured")
        if not query.strip():
            raise ValueError("query must be non-empty")

        cached = self._get_faiss(tenant_id)
        if cached is None or len(cached[1]) == 0:
            return []

        index, chunks = cached

        # Optionally restrict to a specific document after retrieval
        if doc_id:
            doc_chunks = [c for c in chunks if c.doc_id == doc_id]
            if not doc_chunks:
                return []
            # Build a temporary sub-index for this document's chunks
            sub_index, sub_chunks = self._embedder.build_index(doc_chunks)
            return self._embedder.search(query, sub_index, sub_chunks, k=top_k)

        return self._embedder.search(query, index, chunks, k=top_k)

    def search_hybrid(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 10,
        doc_id: str | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Fuse BM25 (FTS5) and dense FAISS results via RRF.

        Parameters
        ----------
        tenant_id:
            Owning tenant.
        query:
            Search query.
        top_k:
            Maximum results to return after fusion.
        doc_id:
            Optional document scope.
        rrf_k:
            RRF smoothing constant (default 60).
        """
        candidates = top_k * 5  # fetch more candidates before fusion
        sparse_results = self.search_fts(tenant_id, query, top_k=candidates, doc_id=doc_id)
        dense_results = (
            self.search_dense(tenant_id, query, top_k=candidates, doc_id=doc_id)
            if self._embedder is not None
            else []
        )

        sparse_ids = [c.chunk_id for c, _ in sparse_results]
        dense_ids = [c.chunk_id for c, _ in dense_results]

        # RRF fusion
        rrf: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(sparse_ids, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        # Build chunk lookup from both result sets
        chunk_map: dict[str, Chunk] = {}
        for c, _ in sparse_results:
            chunk_map[c.chunk_id] = c
        for c, _ in dense_results:
            chunk_map[c.chunk_id] = c

        sorted_ids = sorted(rrf, key=lambda cid: rrf[cid], reverse=True)[:top_k]
        return [(chunk_map[cid], rrf[cid]) for cid in sorted_ids if cid in chunk_map]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all SQLite connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        logger.info("DocumentStore closed all connections")

    def _get_all_chunks_meta(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return raw chunk metadata rows for *tenant_id* (for tests)."""
        conn = self._conn(tenant_id)
        rows = conn.execute(
            "SELECT * FROM chunks_meta WHERE tenant_id = ?", (tenant_id,)
        ).fetchall()
        return [dict(r) for r in rows]
