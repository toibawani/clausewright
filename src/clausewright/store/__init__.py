"""Tenant-isolated document storage: SQLite (WAL + FTS5) + per-tenant FAISS."""

from clausewright.store.document_store import DocumentRecord, DocumentStore

__all__ = ["DocumentRecord", "DocumentStore"]
