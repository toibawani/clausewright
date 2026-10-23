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
    request_id: str | None = Field(default=None, description="Correlation ID for audit and debugging")
