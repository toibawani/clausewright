"""Unit tests for JWT authentication and tenant boundary enforcement (closes #34)."""

from __future__ import annotations

import jwt
import pytest
from starlette.testclient import TestClient

from clausewright.api.app import create_app
from clausewright.api.auth import create_access_token


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_valid_jwt_passes(client: TestClient) -> None:
    secret = "clausewright-insecure-secret-key-change-in-prod"
    token = create_access_token(
        user_id="user_123",
        tenant_id="tenant_alpha",
        secret=secret,
        email="lawyer@alpha.com",
        role="associate",
    )

    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_123"
    assert data["tenant_id"] == "tenant_alpha"
    assert data["email"] == "lawyer@alpha.com"
    assert data["role"] == "associate"


def test_missing_token_returns_401(client: TestClient) -> None:
    response = client.get("/me")
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "HTTP_401"
    assert "Missing" in data["message"]


def test_invalid_token_returns_401(client: TestClient) -> None:
    response = client.get("/me", headers={"Authorization": "Bearer bad-token-format"})
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "HTTP_401"
    assert "Invalid" in data["message"]


def test_expired_token_returns_401(client: TestClient) -> None:
    secret = "clausewright-insecure-secret-key-change-in-prod"
    token = create_access_token(
        user_id="user_123",
        tenant_id="tenant_alpha",
        secret=secret,
        expires_in_minutes=-15,  # 15 minutes in the past
    )

    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "HTTP_401"
    assert "expired" in data["message"].lower()


def test_token_missing_tenant_id_returns_401(client: TestClient) -> None:
    secret = "clausewright-insecure-secret-key-change-in-prod"
    # Token without tenant_id
    payload = {"sub": "user_no_tenant", "exp": 9999999999}
    token = jwt.encode(payload, secret, algorithm="HS256")

    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "HTTP_401"


def test_same_tenant_passes_verification(client: TestClient) -> None:
    secret = "clausewright-insecure-secret-key-change-in-prod"
    token = create_access_token(user_id="user_1", tenant_id="firm_law", secret=secret)

    response = client.get(
        "/tenants/firm_law/verify",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authorized"
    assert data["tenant_id"] == "firm_law"


def test_wrong_tenant_returns_403_forbidden(client: TestClient) -> None:
    secret = "clausewright-insecure-secret-key-change-in-prod"
    # User belongs to tenant "firm_law", tries to access "rival_firm"
    token = create_access_token(user_id="user_1", tenant_id="firm_law", secret=secret)

    response = client.get(
        "/tenants/rival_firm/verify",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    data = response.json()
    assert data["error"] == "HTTP_403"
    assert "cross-tenant access denied" in data["message"].lower()
