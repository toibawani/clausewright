"""Unit tests for DocumentStore: CRUD, FTS5 search, FAISS, and tenant isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from clausewright.schemas import Chunk
from clausewright.store.document_store import DocumentRecord, DocumentStore


def _make_chunk(doc_id: str, chunk_id: str, text: str, section: str = "1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_number=section,
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    """DocumentStore instance backed by a temp directory (no embedder)."""
    return DocumentStore(store_root=tmp_path / "store")


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def test_content_hash_deterministic() -> None:
    raw = b"hello world"
    assert DocumentStore.content_hash(raw) == DocumentStore.content_hash(raw)


def test_content_hash_differs_for_different_bytes() -> None:
    assert DocumentStore.content_hash(b"a") != DocumentStore.content_hash(b"b")


# ---------------------------------------------------------------------------
# add_document / list / get
# ---------------------------------------------------------------------------


def test_add_and_list_documents(store: DocumentStore) -> None:
    chunks = [_make_chunk("doc1", "doc1-1-0", "The indemnification clause covers all losses.")]
    record = store.add_document(
        doc_id="doc1",
        tenant_id="tenant-a",
        filename="contract.pdf",
        content_hash="abc123",
        chunks=chunks,
        page_count=5,
    )
    assert isinstance(record, DocumentRecord)
    assert record.doc_id == "doc1"
    assert record.tenant_id == "tenant-a"
    assert record.chunk_count == 1
    assert record.page_count == 5

    listed = store.list_documents("tenant-a")
    assert len(listed) == 1
    assert listed[0].doc_id == "doc1"


def test_get_document_returns_correct_record(store: DocumentStore) -> None:
    chunks = [_make_chunk("d2", "d2-1-0", "Payment terms thirty days net.")]
    store.add_document(
        doc_id="d2",
        tenant_id="tenant-b",
        filename="lease.pdf",
        content_hash="xyz999",
        chunks=chunks,
        page_count=2,
    )
    result = store.get_document("tenant-b", "d2")
    assert result is not None
    assert result.filename == "lease.pdf"


def test_get_document_wrong_tenant_returns_none(store: DocumentStore) -> None:
    chunks = [_make_chunk("d3", "d3-1-0", "Governing law shall be New York.")]
    store.add_document(
        doc_id="d3",
        tenant_id="tenant-c",
        filename="gov.pdf",
        content_hash="hash111",
        chunks=chunks,
        page_count=1,
    )
    # Cross-tenant lookup must return None
    result = store.get_document("tenant-other", "d3")
    assert result is None


def test_tenant_isolation_in_list(store: DocumentStore) -> None:
    chunks_a = [_make_chunk("da", "da-1-0", "Tenant A document.")]
    chunks_b = [_make_chunk("db", "db-1-0", "Tenant B document.")]
    store.add_document(
        doc_id="da",
        tenant_id="ta",
        filename="a.pdf",
        content_hash="ha",
        chunks=chunks_a,
        page_count=1,
    )
    store.add_document(
        doc_id="db",
        tenant_id="tb",
        filename="b.pdf",
        content_hash="hb",
        chunks=chunks_b,
        page_count=1,
    )
    assert [r.doc_id for r in store.list_documents("ta")] == ["da"]
    assert [r.doc_id for r in store.list_documents("tb")] == ["db"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_exists_by_hash_returns_none_for_unknown(store: DocumentStore) -> None:
    assert store.exists_by_hash("tenant-a", "nonexistent") is None


def test_exists_by_hash_returns_doc_id(store: DocumentStore) -> None:
    chunks = [_make_chunk("d4", "d4-1-0", "Duplicate detection test.")]
    store.add_document(
        doc_id="d4",
        tenant_id="tenant-dup",
        filename="dup.pdf",
        content_hash="duphash",
        chunks=chunks,
        page_count=1,
    )
    found = store.exists_by_hash("tenant-dup", "duphash")
    assert found == "d4"


def test_duplicate_hash_raises_value_error(store: DocumentStore) -> None:
    chunks = [_make_chunk("d5", "d5-1-0", "Same content twice.")]
    store.add_document(
        doc_id="d5",
        tenant_id="tenant-dup2",
        filename="dup2.pdf",
        content_hash="dupdup",
        chunks=chunks,
        page_count=1,
    )
    with pytest.raises(ValueError, match="already exists"):
        store.add_document(
            doc_id="d5-again",
            tenant_id="tenant-dup2",
            filename="dup2.pdf",
            content_hash="dupdup",
            chunks=chunks,
            page_count=1,
        )


# ---------------------------------------------------------------------------
# Hard delete
# ---------------------------------------------------------------------------


def test_delete_document_removes_record(store: DocumentStore) -> None:
    chunks = [_make_chunk("d6", "d6-1-0", "Termination clause.")]
    store.add_document(
        doc_id="d6",
        tenant_id="tenant-del",
        filename="del.pdf",
        content_hash="delhash",
        chunks=chunks,
        page_count=1,
    )
    deleted = store.delete_document("tenant-del", "d6")
    assert deleted is True
    assert store.get_document("tenant-del", "d6") is None
    assert store.list_documents("tenant-del") == []


def test_delete_nonexistent_returns_false(store: DocumentStore) -> None:
    result = store.delete_document("tenant-del", "ghost-doc")
    assert result is False


def test_delete_removes_chunks_meta(store: DocumentStore) -> None:
    chunks = [
        _make_chunk("d7", "d7-1-0", "Force majeure provision."),
        _make_chunk("d7", "d7-2-0", "Warranty disclaimer.", "2"),
    ]
    store.add_document(
        doc_id="d7",
        tenant_id="tenant-del2",
        filename="doc7.pdf",
        content_hash="delhash2",
        chunks=chunks,
        page_count=1,
    )
    assert len(store._get_all_chunks_meta("tenant-del2")) == 2
    store.delete_document("tenant-del2", "d7")
    assert len(store._get_all_chunks_meta("tenant-del2")) == 0


# ---------------------------------------------------------------------------
# FTS5 search
# ---------------------------------------------------------------------------


def test_search_fts_basic(store: DocumentStore) -> None:
    chunks = [
        _make_chunk("d8", "d8-1-0", "The indemnification clause limits liability."),
        _make_chunk("d8", "d8-2-0", "Payment is due within thirty days.", "2"),
    ]
    store.add_document(
        doc_id="d8",
        tenant_id="tenant-fts",
        filename="c.pdf",
        content_hash="ftshash",
        chunks=chunks,
        page_count=1,
    )
    results = store.search_fts("tenant-fts", "indemnification")
    assert len(results) >= 1
    top_chunk, top_score = results[0]
    assert "indemnification" in top_chunk.text.lower()
    assert top_score > 0


def test_search_fts_tenant_isolation(store: DocumentStore) -> None:
    # Store chunk for tenant-x
    store.add_document(
        doc_id="dx",
        tenant_id="tenant-x",
        filename="x.pdf",
        content_hash="hashx",
        chunks=[_make_chunk("dx", "dx-1-0", "Confidential trade secret.")],
        page_count=1,
    )
    # Searching from tenant-y must return zero results
    results = store.search_fts("tenant-y", "confidential")
    assert results == []


def test_search_fts_doc_filter(store: DocumentStore) -> None:
    chunks_a = [_make_chunk("da2", "da2-1-0", "Arbitration clause applies here.")]
    chunks_b = [_make_chunk("db2", "db2-1-0", "Arbitration and dispute resolution.")]
    store.add_document(
        doc_id="da2",
        tenant_id="tenant-fts2",
        filename="a.pdf",
        content_hash="ha2",
        chunks=chunks_a,
        page_count=1,
    )
    store.add_document(
        doc_id="db2",
        tenant_id="tenant-fts2",
        filename="b.pdf",
        content_hash="hb2",
        chunks=chunks_b,
        page_count=1,
    )
    results = store.search_fts("tenant-fts2", "arbitration", doc_id="da2")
    assert all(c.doc_id == "da2" for c, _ in results)


def test_search_fts_empty_query_raises(store: DocumentStore) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        store.search_fts("t", "  ")


# ---------------------------------------------------------------------------
# Dense search (no embedder → empty result)
# ---------------------------------------------------------------------------


def test_search_dense_without_embedder_raises(store: DocumentStore) -> None:
    with pytest.raises(RuntimeError, match="embedder"):
        store.search_dense("t", "anything")


# ---------------------------------------------------------------------------
# Hybrid search (FTS5-only path without embedder)
# ---------------------------------------------------------------------------


def test_search_hybrid_fts_only(store: DocumentStore) -> None:
    chunks = [_make_chunk("d9", "d9-1-0", "Governing law New York applies.")]
    store.add_document(
        doc_id="d9",
        tenant_id="tenant-hybrid",
        filename="h.pdf",
        content_hash="hh",
        chunks=chunks,
        page_count=1,
    )
    results = store.search_hybrid("tenant-hybrid", "governing law")
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


def test_close_is_idempotent(store: DocumentStore) -> None:
    store.close()
    store.close()  # second call must not raise
