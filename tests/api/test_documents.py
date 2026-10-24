"""API tests for document upload, list, get, and delete endpoints (closes #36)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clausewright.api.app import create_app
from clausewright.api.auth import create_access_token
from clausewright.api.settings import Settings
from clausewright.store.document_store import DocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        jwt_secret="test-secret-key-for-documents-api",
        jwt_algorithm="HS256",
        app_version="0.1.0-test",
        environment="test",
    )


@pytest.fixture()
def token_for(settings: Settings):  # type: ignore[no-untyped-def]
    """Factory that returns a valid JWT for a given tenant."""

    def _make(tenant_id: str, user_id: str = "user-1") -> str:
        return create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            email="user@example.com",
            role="analyst",
            secret=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )

    return _make


@pytest.fixture()
def client(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with a real DocumentStore backed by a temp directory."""
    # Patch Settings.load to return test settings
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: settings))

    app = create_app()

    # Override the document_store in app.state with a temp-dir store
    store = DocumentStore(store_root=tmp_path / "store")

    # Replace the lifespan store with ours via app.state override during testing
    with TestClient(app) as tc:
        tc.app.state.document_store = store  # type: ignore[union-attr]
        yield tc  # type: ignore[misc]

    store.close()


# ---------------------------------------------------------------------------
# Minimal synthetic PDF bytes (1-page, valid structure)
# ---------------------------------------------------------------------------


def _minimal_pdf_bytes() -> bytes:
    """Return the bytes of a minimal but valid PDF with one text line."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 72 720 Td (Payment terms.) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n441\n%%EOF\n"
    )


# ---------------------------------------------------------------------------
# Helper: inject a document directly via the store (bypasses PDF parsing)
# ---------------------------------------------------------------------------


def _inject_doc(client: TestClient, tenant_id: str, doc_id: str = "doc-a") -> dict[str, Any]:
    """Directly add a document to the store without PDF parsing."""
    from clausewright.schemas import Chunk

    chunk = Chunk(
        chunk_id=f"{doc_id}-1-0",
        doc_id=doc_id,
        section_number="1",
        chunk_index=0,
        text="Governing law shall be New York.",
        page_range=(1, 1),
        word_count=6,
    )
    store: DocumentStore = client.app.state.document_store  # type: ignore[union-attr]
    rec = store.add_document(
        doc_id=doc_id,
        tenant_id=tenant_id,
        filename="contract.pdf",
        content_hash=f"hash-{doc_id}",
        chunks=[chunk],
        page_count=1,
    )
    return {
        "doc_id": rec.doc_id,
        "filename": rec.filename,
        "content_hash": rec.content_hash,
        "page_count": rec.page_count,
        "chunk_count": rec.chunk_count,
        "uploaded_at": rec.uploaded_at,
    }


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------


def test_list_documents_empty(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-1")
    resp = client.get(
        "/tenants/tenant-1/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "tenant-1"
    assert body["total"] == 0
    assert body["documents"] == []


def test_list_documents_after_inject(client: TestClient, token_for: Any) -> None:
    _inject_doc(client, "tenant-2", "doc-b")
    token = token_for("tenant-2")
    resp = client.get(
        "/tenants/tenant-2/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["documents"][0]["doc_id"] == "doc-b"


def test_list_documents_cross_tenant_rejected(client: TestClient, token_for: Any) -> None:
    _inject_doc(client, "tenant-3", "doc-c")
    # Token is for tenant-4 but path is /tenants/tenant-3/...
    token = token_for("tenant-4")
    resp = client.get(
        "/tenants/tenant-3/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_list_documents_unauthenticated(client: TestClient) -> None:
    resp = client.get("/tenants/tenant-1/documents")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Get document
# ---------------------------------------------------------------------------


def test_get_document_returns_record(client: TestClient, token_for: Any) -> None:
    injected = _inject_doc(client, "tenant-5", "doc-d")
    token = token_for("tenant-5")
    resp = client.get(
        f"/tenants/tenant-5/documents/{injected['doc_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == injected["doc_id"]
    assert body["filename"] == "contract.pdf"


def test_get_document_not_found(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-5")
    resp = client.get(
        "/tenants/tenant-5/documents/does-not-exist",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------


def test_delete_document(client: TestClient, token_for: Any) -> None:
    injected = _inject_doc(client, "tenant-6", "doc-e")
    token = token_for("tenant-6")
    resp = client.delete(
        f"/tenants/tenant-6/documents/{injected['doc_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True

    # Verify it's gone
    resp2 = client.get(
        f"/tenants/tenant-6/documents/{injected['doc_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 404


def test_delete_nonexistent_returns_404(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-6")
    resp = client.delete(
        "/tenants/tenant-6/documents/ghost",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_delete_cross_tenant_rejected(client: TestClient, token_for: Any) -> None:
    injected = _inject_doc(client, "tenant-7", "doc-f")
    # Attacker uses their own token but targets tenant-7's doc_id in tenant-8's namespace
    token = token_for("tenant-8")
    resp = client.delete(
        f"/tenants/tenant-7/documents/{injected['doc_id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Upload document (POST)
# ---------------------------------------------------------------------------


def test_upload_pdf_success(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload")
    pdf_bytes = _minimal_pdf_bytes()
    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test_contract.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "test_contract.pdf"
    assert body["page_count"] == 1
    assert body["chunk_count"] >= 1
    assert body["doc_id"] is not None


def test_upload_pdf_deduplication(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload-dedup")
    pdf_bytes = _minimal_pdf_bytes()

    # First upload -> 201
    resp1 = client.post(
        "/tenants/tenant-upload-dedup/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("contract_v1.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp1.status_code == 201
    doc1 = resp1.json()

    # Second upload of same content -> 200 (idempotent)
    resp2 = client.post(
        "/tenants/tenant-upload-dedup/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("contract_duplicate.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp2.status_code == 200
    doc2 = resp2.json()
    assert doc2["doc_id"] == doc1["doc_id"]
    assert doc2["content_hash"] == doc1["content_hash"]


def test_upload_unsupported_mime_type(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload")
    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("contract.pdf", io.BytesIO(b"dummy text"), "text/plain")},
    )
    assert resp.status_code == 415


def test_upload_invalid_extension(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload")
    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("contract.docx", io.BytesIO(b"dummy"), "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_upload_empty_file(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload")
    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 422


def test_upload_malformed_pdf(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-upload")
    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("corrupt.pdf", io.BytesIO(b"not a valid pdf content"), "application/pdf")},
    )
    assert resp.status_code == 422


def test_upload_exceeds_max_size(
    client: TestClient, token_for: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = token_for("tenant-upload")
    pdf_bytes = _minimal_pdf_bytes()
    # Temporarily set max size to smaller than minimal pdf
    tiny_settings = Settings(
        jwt_secret="test-secret-key-for-documents-api",
        max_upload_size_bytes=10,
    )
    client.app.state.settings = tiny_settings  # type: ignore[union-attr]

    resp = client.post(
        "/tenants/tenant-upload/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("toolarge.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 413


def test_upload_cross_tenant_forbidden(client: TestClient, token_for: Any) -> None:
    token = token_for("tenant-a")
    resp = client.post(
        "/tenants/tenant-b/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("contract.pdf", io.BytesIO(_minimal_pdf_bytes()), "application/pdf")},
    )
    assert resp.status_code == 403


def test_upload_unauthenticated(client: TestClient) -> None:
    resp = client.post(
        "/tenants/tenant-1/documents",
        files={"file": ("contract.pdf", io.BytesIO(_minimal_pdf_bytes()), "application/pdf")},
    )
    assert resp.status_code == 401
