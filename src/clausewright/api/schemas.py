"""Pydantic v2 request and response schemas for Clausewright API (closes #33)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(default="ok", description="Overall health status")
    version: str = Field(..., description="Service semantic version")
    git_commit: str = Field(..., description="Active git commit SHA")
    uptime_seconds: float = Field(..., ge=0.0, description="Uptime in seconds since startup")


class ErrorResponse(BaseModel):
    """Sanitized API error schema; no stack traces leaked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: str = Field(..., description="High-level error code or category")
    message: str = Field(..., description="Human-readable description")
    request_id: str | None = Field(
        default=None, description="Correlation ID for audit and debugging"
    )


class DocumentResponse(BaseModel):
    """Representation of a stored document returned by the API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., description="Unique document identifier (UUID)")
    filename: str = Field(..., description="Original uploaded file name")
    content_hash: str = Field(..., description="SHA-256 hex digest of raw PDF bytes")
    page_count: int = Field(..., ge=0, description="Number of pages in the document")
    chunk_count: int = Field(..., ge=0, description="Number of indexed chunks")
    uploaded_at: float = Field(..., description="Unix epoch timestamp of upload")


class DocumentListResponse(BaseModel):
    """Paginated list of documents for a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(..., description="Tenant this list belongs to")
    total: int = Field(..., ge=0, description="Total number of documents")
    documents: list[DocumentResponse] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """Confirmation of a document hard-delete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    deleted: bool
    message: str
