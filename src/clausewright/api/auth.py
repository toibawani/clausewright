"""Authentication and tenant model using standard PyJWT (closes #34).

Enforces:
1. Standard JWT validation (no custom cryptography).
2. Strict tenant model: every authenticated user belongs to exactly one tenant.
3. Fast dependency injection for route handlers via get_current_user.
4. Tenant boundary enforcement via require_tenant.
"""

from __future__ import annotations

import collections.abc
import datetime
import logging
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from clausewright.api.settings import Settings

logger = logging.getLogger(__name__)

security_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """The identity and tenant scope of an authenticated request."""

    user_id: str
    tenant_id: str
    email: str | None = None
    role: str = "member"


def create_access_token(
    user_id: str,
    tenant_id: str,
    *,
    secret: str,
    algorithm: str = "HS256",
    expires_in_minutes: int = 1440,
    email: str | None = None,
    role: str = "member",
) -> str:
    """Create a signed JWT token carrying user and tenant identity."""
    now = datetime.datetime.now(datetime.UTC)
    exp = now + datetime.timedelta(minutes=expires_in_minutes)

    payload: dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "role": role,
    }
    if email:
        payload["email"] = email

    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str,
    *,
    secret: str,
    algorithm: str = "HS256",
) -> AuthenticatedUser:
    """Decode and verify a signed JWT token.

    Raises
    ------
    HTTPException(401)
        If token is expired, invalid, or missing required tenant claims.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err

    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")

    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required user or tenant identity",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(
        user_id=str(user_id),
        tenant_id=str(tenant_id),
        email=payload.get("email"),
        role=str(payload.get("role", "member")),
    )


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security_bearer)],
) -> AuthenticatedUser:
    """FastAPI dependency: extract and verify the current authenticated user."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings: Settings = getattr(request.app.state, "settings", None) or Settings.load()
    return decode_access_token(
        credentials.credentials,
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


# Annotated type alias for route parameters
CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]


def require_tenant(
    path_param: str = "tenant_id",
) -> collections.abc.Callable[..., collections.abc.Coroutine[Any, Any, AuthenticatedUser]]:
    """Dependency factory: enforce caller belongs to the tenant named in the URL path.

    Parameters
    ----------
    path_param:
        The name of the path parameter that carries the target tenant ID.
        Defaults to ``"tenant_id"``.

    Usage::

        RequireTenant = Annotated[AuthenticatedUser, Depends(require_tenant("tenant_id"))]

        @router.get("/tenants/{tenant_id}/documents")
        async def list_docs(tenant_id: str, _user: RequireTenant) -> ...:
            ...
    """

    async def _verify_tenant(
        request: Request,
        current_user: CurrentUser,
    ) -> AuthenticatedUser:
        target_tenant_id: str = request.path_params.get(path_param, "")
        if not target_tenant_id or current_user.tenant_id != target_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Forbidden: cross-tenant access denied "
                    f"(caller belongs to tenant '{current_user.tenant_id}')"
                ),
            )
        return current_user

    return _verify_tenant
