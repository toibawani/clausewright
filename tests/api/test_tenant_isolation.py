"""Security regression tests for tenant-isolated storage and retrieval (closes #35).

Ensures that User A never sees User B's documents or chunks, even when queries
and contract text are identical. Verifies mandatory tenant_id filtering across:
1. BM25 / FTS5 full-text search
2. FAISS dense vector search
3. Reciprocal Rank Fusion (hybrid) search
4. DocumentStore operations (get, list, delete)
5. HTTP API endpoints (cross-tenant 403 enforcement and scoping)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clausewright.api.app import create_app
from clausewright.api.auth import create_access_token
from clausewright.api.settings import Settings
from clausewright.schemas import Chunk
from clausewright.store.document_store import DocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        jwt_secret="isolation-test-secret-key-35-super-secure-32chars",
        jwt_algorithm="HS256",
        app_version="0.1.0-test",
        environment="test",
    )


@pytest.fixture()
def token_for(settings: Settings):  # type: ignore[no-untyped-def]
    """Factory that returns a valid JWT for a given tenant."""

    def _make(tenant_id: str, user_id: str = "test-user") -> str:
        return create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            email=f"{tenant_id}@example.com",
            role="analyst",
            secret=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )

    return _make


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    """Isolated DocumentStore instance for direct unit testing."""
    s = DocumentStore(store_root=tmp_path / "store")
    yield s
    s.close()


@pytest.fixture()
def client(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient wired to a fresh DocumentStore."""
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: settings))
    app = create_app()
    app_store = DocumentStore(store_root=tmp_path / "api_store")

    with TestClient(app) as tc:
        tc.app.state.document_store = app_store  # type: ignore[union-attr]
        yield tc

    app_store.close()


def _make_chunk(doc_id: str, chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_number="1",
        chunk_index=0,
        text=text,
        page_range=(1, 1),
        word_count=len(text.split()),
    )


def _minimal_pdf_bytes() -> bytes:
    """Return the bytes of a minimal valid PDF with text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 55 >>\nstream\n"
        b"BT /F1 12 Tf 72 720 Td (Indemnification and liability terms.) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000371 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n452\n%%EOF\n"
    )


# ---------------------------------------------------------------------------
# Store-Level Isolation Tests
# ---------------------------------------------------------------------------


def test_bm25_search_tenant_isolation(store: DocumentStore) -> None:
    """Tenant A's search query must NEVER return chunks belonging to Tenant B."""
    identical_text = "The Supplier shall indemnify Customer against third party claims."

    # Ingest into Tenant Alpha
    chunk_a = _make_chunk("doc-alpha", "chunk-alpha-1", identical_text)
    store.add_document(
        doc_id="doc-alpha",
        tenant_id="tenant-alpha",
        filename="contract_alpha.pdf",
        content_hash="hash-alpha",
        chunks=[chunk_a],
        page_count=1,
    )

    # Ingest identical text into Tenant Beta
    chunk_b = _make_chunk("doc-beta", "chunk-beta-1", identical_text)
    store.add_document(
        doc_id="doc-beta",
        tenant_id="tenant-beta",
        filename="contract_beta.pdf",
        content_hash="hash-beta",
        chunks=[chunk_b],
        page_count=1,
    )

    # Search as Tenant Alpha
    results_a = store.search_fts("tenant-alpha", "indemnify")
    assert len(results_a) == 1
    assert results_a[0][0].chunk_id == "chunk-alpha-1"
    assert results_a[0][0].doc_id == "doc-alpha"

    # Search as Tenant Beta
    results_b = store.search_fts("tenant-beta", "indemnify")
    assert len(results_b) == 1
    assert results_b[0][0].chunk_id == "chunk-beta-1"
    assert results_b[0][0].doc_id == "doc-beta"

    # Search as Tenant Gamma (empty tenant)
    results_gamma = store.search_fts("tenant-gamma", "indemnify")
    assert results_gamma == []


def test_hybrid_search_tenant_isolation(store: DocumentStore) -> None:
    """Hybrid search for Tenant A must never return Tenant B's chunks."""
    text_shared = "Confidential Information shall remain the property of Disclosing Party."

    store.add_document(
        doc_id="doc-a1",
        tenant_id="tenant-1",
        filename="nda_1.pdf",
        content_hash="hash-nda-1",
        chunks=[_make_chunk("doc-a1", "c-1", text_shared)],
        page_count=1,
    )
    store.add_document(
        doc_id="doc-b1",
        tenant_id="tenant-2",
        filename="nda_2.pdf",
        content_hash="hash-nda-2",
        chunks=[_make_chunk("doc-b1", "c-2", text_shared)],
        page_count=1,
    )

    results = store.search_hybrid("tenant-1", "Confidential Information")
    assert len(results) == 1
    chunk, _ = results[0]
    assert chunk.chunk_id == "c-1"
    assert chunk.doc_id == "doc-a1"


def test_store_crud_tenant_isolation(store: DocumentStore) -> None:
    """Documents of Tenant B are inaccessible to Tenant A via get, list, or delete."""
    store.add_document(
        doc_id="doc-secret",
        tenant_id="tenant-secret",
        filename="secret.pdf",
        content_hash="hash-secret",
        chunks=[_make_chunk("doc-secret", "c-sec", "Secret terms.")],
        page_count=1,
    )

    # Tenant other cannot get
    assert store.get_document("tenant-other", "doc-secret") is None

    # Tenant other list is empty
    assert store.list_documents("tenant-other") == []

    # Tenant other cannot delete
    assert store.delete_document("tenant-other", "doc-secret") is False

    # Document still intact for rightful tenant
    assert store.get_document("tenant-secret", "doc-secret") is not None


# ---------------------------------------------------------------------------
# API-Level Isolation Tests
# ---------------------------------------------------------------------------


def test_api_list_and_get_tenant_isolation(client: TestClient, token_for: Any) -> None:
    """An authenticated user cannot read another tenant's documents via API."""
    token_a = token_for("tenant-A")
    token_b = token_for("tenant-B")
    pdf_bytes = _minimal_pdf_bytes()

    # Upload document for Tenant B
    resp_upload = client.post(
        "/tenants/tenant-B/documents",
        headers={"Authorization": f"Bearer {token_b}"},
        files={"file": ("agreement_b.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp_upload.status_code == 201
    doc_b_id = resp_upload.json()["doc_id"]

    # Tenant A attempts to list Tenant B's documents -> 403 Forbidden
    resp_list = client.get(
        "/tenants/tenant-B/documents",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_list.status_code == 403

    # Tenant A attempts to get Tenant B's document via Tenant B path -> 403 Forbidden
    resp_get_cross = client.get(
        f"/tenants/tenant-B/documents/{doc_b_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_get_cross.status_code == 403

    # Tenant A attempts to get Tenant B's document via Tenant A path -> 404 Not Found (no leak)
    resp_get_own_path = client.get(
        f"/tenants/tenant-A/documents/{doc_b_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_get_own_path.status_code == 404

    # Tenant A lists their own documents -> empty
    resp_list_own = client.get(
        "/tenants/tenant-A/documents",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_list_own.status_code == 200
    assert resp_list_own.json()["total"] == 0


def test_api_delete_tenant_isolation(client: TestClient, token_for: Any) -> None:
    """An authenticated user cannot delete another tenant's documents via API."""
    token_a = token_for("tenant-A")
    token_b = token_for("tenant-B")
    pdf_bytes = _minimal_pdf_bytes()

    # Upload for Tenant B
    resp_upload = client.post(
        "/tenants/tenant-B/documents",
        headers={"Authorization": f"Bearer {token_b}"},
        files={"file": ("agreement_b.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp_upload.status_code == 201
    doc_b_id = resp_upload.json()["doc_id"]

    # Tenant A tries to delete via Tenant B path -> 403 Forbidden
    resp_del_cross = client.delete(
        f"/tenants/tenant-B/documents/{doc_b_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_del_cross.status_code == 403

    # Tenant A tries to delete Tenant B's doc_id in Tenant A's namespace -> 404 Not Found
    resp_del_spoof = client.delete(
        f"/tenants/tenant-A/documents/{doc_b_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_del_spoof.status_code == 404

    # Verify Tenant B's document still exists
    resp_check = client.get(
        f"/tenants/tenant-B/documents/{doc_b_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_check.status_code == 200
