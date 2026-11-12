"""Helpers for talking to the Kinship ASGI app in-process (no sockets)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette

from kinship.config import Settings

# A loopback Host header, which the server's DNS-rebinding protection accepts.
BASE_URL = "http://127.0.0.1:8000"
TOKEN_A = "token-a"
TOKEN_B = "token-b"


def settings_for(tokens: dict[str, int]) -> Settings:
    """`Settings` with exactly these token -> owner_id pairs, ignoring `.env`."""
    pairs = ",".join(f"{token}:{owner_id}" for token, owner_id in tokens.items())
    return Settings(kinship_tokens=pairs, _env_file=None)


def http_client(app: Starlette, token: str | None = None) -> httpx2.AsyncClient:
    """An HTTP client for `app`, optionally sending `Authorization: Bearer <token>`."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url=BASE_URL, headers=headers
    )


@asynccontextmanager
async def connect(app: Starlette, token: str, *, mode: str = "auto") -> AsyncIterator[Client]:
    """A connected SDK `Client` for `app`. `mode="legacy"` forces the initialize handshake."""
    async with (
        http_client(app, token) as http,
        Client(streamable_http_client(f"{BASE_URL}/mcp", http_client=http), mode=mode) as client,
    ):
        yield client
