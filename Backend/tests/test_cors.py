"""CORS hardening tests.

The Vite dev server (default port 5173) is the only intended browser
origin during development. These tests pin that configuration so it
cannot silently regress to open or broken CORS.
"""

from __future__ import annotations

import pytest

from main import CORS_DEV_ORIGINS

ALLOWED_ORIGIN = "http://localhost:5173"
FOREIGN_ORIGIN = "https://evil.example.com"


class TestCorsConfiguration:
    def test_allowed_origin_is_listed(self):
        assert ALLOWED_ORIGIN in CORS_DEV_ORIGINS
        # Both loopback spellings must be covered: browsers send the
        # exact origin string they were served from.
        assert "http://127.0.0.1:5173" in CORS_DEV_ORIGINS

    def test_no_wildcard_origins(self):
        assert "*" not in CORS_DEV_ORIGINS

    def test_simple_get_request_gets_cors_header(self, client):
        response = client.get(
            "/health", headers={"Origin": ALLOWED_ORIGIN}
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == (
            ALLOWED_ORIGIN
        )

    def test_preflight_for_post_is_accepted(self, client):
        response = client.options(
            "/ask",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == ALLOWED_ORIGIN
        )
        assert "POST" in response.headers.get(
            "access-control-allow-methods", ""
        )

    def test_preflight_from_foreign_origin_is_rejected(self, client):
        response = client.options(
            "/ask",
            headers={
                "Origin": FOREIGN_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        # Starlette CORSMiddleware rejects disallowed preflights.
        assert response.status_code == 400
        assert response.headers.get("access-control-allow-origin") is None

    def test_requests_without_origin_are_unaffected(self, client):
        # Server-to-server / curl usage needs no CORS headers at all.
        response = client.get("/health")
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") is None
