"""Document management endpoints: upload, list, and hard-delete (closes #36).

Endpoints
---------
POST   /tenants/{tenant_id}/documents          – Upload a PDF; incremental index.
GET    /tenants/{tenant_id}/documents          – List all documents for tenant.
GET    /tenants/{tenant_id}/documents/{doc_id} – Get a single document record.
DELETE /tenants/{tenant_id}/documents/{doc_id} – Hard-delete document + chunks.

Security
--------
All routes require a valid JWT bearer token (via ``CurrentUser``) and enforce
that the calling user belongs to the target tenant (via ``RequireTenant``).
Content-hash deduplication prevents double-ingestion of the same file.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from clausewright.api.auth import AuthenticatedUser, require_tenant
from clausewright.api.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from clausewright.api.settings import Settings
from clausewright.store.document_store import DocumentRecord, DocumentStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants/{tenant_id}/documents", tags=["Documents"])

# ---------------------------------------------------------------------------
# Store dependency — resolved from app.state set during lifespan
# ---------------------------------------------------------------------------


def _get_store(request: Request) -> DocumentStore:
    store: DocumentStore = request.app.state.document_store
    return store


StoreDep = Annotated[DocumentStore, Depends(_get_store)]

# Type alias for the combined tenant-enforcement dependency
RequireTenant = Annotated[AuthenticatedUser, Depends(require_tenant("tenant_id"))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(rec: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        doc_id=rec.doc_id,
        filename=rec.filename,
        content_hash=rec.content_hash,
        page_count=rec.page_count,
        chunk_count=rec.chunk_count,
        uploaded_at=rec.uploaded_at,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF contract for indexing",
)
async def upload_document(
    tenant_id: str,
    file: UploadFile,
    store: StoreDep,
    request: Request,
    response: Response,
    _user: RequireTenant,
) -> DocumentResponse:
    """Upload a PDF and add it to the tenant's index.

    * Accepted content types: ``application/pdf`` or ``application/x-pdf``.
    * Duplicate detection: if the same content hash already exists the
      existing document record is returned with HTTP 200 (idempotent).
    * The document is parsed, chunked, embedded, and added to both the
      FTS5 table and the per-tenant FAISS index within this request.
    """
    if file.filename is None or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="filename must be provided",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid file extension for '{file.filename}'. Only .pdf files are accepted.",
        )

    if file.content_type and file.content_type not in (
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: '{file.content_type}'. Expected 'application/pdf'.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty",
        )

    settings: Settings = getattr(request.app.state, "settings", Settings.load())
    if len(raw) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_upload_size_bytes} bytes",
        )

    content_hash = DocumentStore.content_hash(raw)

    # Idempotency: if already stored, return existing record
    existing_id = store.exists_by_hash(tenant_id, content_hash)
    if existing_id is not None:
        existing = store.get_document(tenant_id, existing_id)
        if existing is not None:
            logger.info(
                "Duplicate upload detected for tenant=%s hash=%s doc_id=%s",
                tenant_id,
                content_hash[:16],
                existing_id,
            )
            response.status_code = status.HTTP_200_OK
            return _to_response(existing)

    # Parse and chunk the document
    from clausewright.ingestion.chunker import chunk_sections
    from clausewright.ingestion.pdf_loader import load_pdf_from_bytes
    from clausewright.ingestion.section_parser import parse_sections

    doc_id = str(uuid.uuid4())

    try:
        parsed = load_pdf_from_bytes(raw, doc_id=doc_id, filename=file.filename)
    except Exception as exc:
        logger.error(
            "PDF parse failed for tenant=%s filename=%s: %s",
            tenant_id,
            file.filename,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to parse PDF file. Ensure it is a valid, non-encrypted PDF.",
        ) from exc

    sections = parse_sections(parsed)
    chunks = chunk_sections(doc_id, sections)
    page_count = parsed.page_count

    record = store.add_document(
        doc_id=doc_id,
        tenant_id=tenant_id,
        filename=file.filename,
        content_hash=content_hash,
        chunks=chunks,
        page_count=page_count,
    )
    logger.info(
        "Indexed doc_id=%s for tenant=%s (%d pages, %d chunks)",
        doc_id,
        tenant_id,
        page_count,
        len(chunks),
    )
    return _to_response(record)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents for a tenant",
)
async def list_documents(
    tenant_id: str,
    store: StoreDep,
    _user: RequireTenant,
) -> DocumentListResponse:
    """Return all documents owned by the tenant, newest first."""
    records = store.list_documents(tenant_id)
    return DocumentListResponse(
        tenant_id=tenant_id,
        total=len(records),
        documents=[_to_response(r) for r in records],
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Get a single document record",
)
async def get_document(
    tenant_id: str,
    doc_id: str,
    store: StoreDep,
    _user: RequireTenant,
) -> DocumentResponse:
    """Return metadata for a single document. 404 if not found."""
    record = store.get_document(tenant_id, doc_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id!r} not found for tenant {tenant_id!r}",
        )
    return _to_response(record)


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Hard-delete a document and all its indexed chunks",
)
async def delete_document(
    tenant_id: str,
    doc_id: str,
    store: StoreDep,
    _user: RequireTenant,
) -> DocumentDeleteResponse:
    """Permanently remove a document, its chunks from FTS5, and rebuild the FAISS index."""
    deleted = store.delete_document(tenant_id, doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id!r} not found for tenant {tenant_id!r}",
        )
    return DocumentDeleteResponse(
        doc_id=doc_id,
        deleted=True,
        message="Document and all associated chunks have been permanently removed.",
    )
