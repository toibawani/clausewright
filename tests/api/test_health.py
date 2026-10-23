"""Tests for FastAPI skeleton and health endpoint (closes #33)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from clausewright.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "git_commit" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0


def test_http_exception_sanitized(client: TestClient) -> None:
    # Trigger 404
    response = client.get("/non-existent-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "HTTP_404"
    assert "message" in data


def test_unhandled_exception_sanitized() -> None:
    app = create_app()

    # Register an intentionally crashing route to test catch-all
    @app.get("/test-crash")
    async def crash():
        raise RuntimeError("Secret internal database password: do-not-leak")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/test-crash")
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "INTERNAL_SERVER_ERROR"
        # Secret message must never leak to client
        assert "Secret" not in data["message"]
        assert "password" not in data["message"]
        assert "Traceback" not in data["message"]
        # Must provide correlation request_id
        assert data.get("request_id") is not None
