"""FastAPI application skeleton with lifespan, error handlers, and health check (closes #33)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from clausewright.api.auth import CurrentUser
from clausewright.api.schemas import ErrorResponse, HealthResponse
from clausewright.api.settings import Settings, get_git_commit

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan manager: initialize runtime state and dependencies."""
    app.state.start_time = time.monotonic()
    app.state.git_commit = get_git_commit()
    app.state.settings = Settings.load()
    logger.info(
        "Clausewright API starting (version=%s, commit=%s, env=%s)",
        app.state.settings.app_version,
        app.state.git_commit,
        app.state.settings.environment,
    )
    yield
    logger.info("Clausewright API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Clausewright API",
        description="Structure-aware legal contract question answering backend.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # ---------------------------------------------------------------------------
    # Sanitized Exception Handlers (no stack traces or contract leaks)
    # ---------------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException | StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        msg = "; ".join(f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg')}" for err in errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="VALIDATION_ERROR",
                message=msg or "Request validation failed",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(uuid.uuid4())
        # Log sanitized exception with request_id for debugging
        logger.exception("Unhandled server error [request_id=%s]: %s", request_id, type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred. Please reference the request ID when reporting.",
                request_id=request_id,
            ).model_dump(),
        )

    # ---------------------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service health check",
        tags=["System"],
    )
    async def health(request: Request) -> HealthResponse:
        """Return service health status, version, git commit, and uptime."""
        start_time: float = getattr(request.app.state, "start_time", time.monotonic())
        uptime = max(0.0, time.monotonic() - start_time)
        version: str = getattr(request.app.state.settings, "app_version", "0.1.0")
        git_commit: str = getattr(request.app.state, "git_commit", "unknown")

        return HealthResponse(
            status="ok",
            version=version,
            git_commit=git_commit,
            uptime_seconds=round(uptime, 3),
        )

    @app.get(
        "/me",
        summary="Current user profile and tenant scope",
        tags=["Auth"],
    )
    async def get_me(
        current_user: CurrentUser,
    ) -> dict[str, str | None]:
        """Return the calling user's identity and tenant isolation boundary."""
        return {
            "user_id": current_user.user_id,
            "tenant_id": current_user.tenant_id,
            "email": current_user.email,
            "role": current_user.role,
        }

    @app.get(
        "/tenants/{target_tenant_id}/verify",
        summary="Verify tenant membership",
        tags=["Auth"],
    )
    async def verify_tenant_membership(
        target_tenant_id: str,
        current_user: CurrentUser,
    ) -> dict[str, str]:
        """Verify the caller belongs to target_tenant_id, rejecting cross-tenant calls."""
        if current_user.tenant_id != target_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: cross-tenant access denied (caller belongs to '{current_user.tenant_id}')",
            )
        return {"status": "authorized", "tenant_id": target_tenant_id, "user_id": current_user.user_id}

    return app


app = create_app()
