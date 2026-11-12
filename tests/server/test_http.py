"""HTTP surface: `/healthz`, bearer-token auth, and raw JSON-RPC over `/mcp`."""

from __future__ import annotations

import httpx2
import pytest
from mcp_testing import TOKEN_A, TOKEN_B, http_client, settings_for
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from kinship.server.app import create_app
from kinship.server.auth import BearerAuthMiddleware

pytestmark = pytest.mark.anyio

MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def rpc(method: str, params: dict | None = None, id: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        body["params"] = params
    return body


INITIALIZE = rpc(
    "initialize",
    {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
)


def bearer(token: str) -> dict[str, str]:
    return {**MCP_HEADERS, "Authorization": f"Bearer {token}"}


# -- /healthz -------------------------------------------------------------------


async def test_healthz_needs_no_token(http: httpx2.AsyncClient) -> None:
    response = await http.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_is_503_when_the_database_is_down() -> None:
    dead = create_engine(
        "postgresql+psycopg://kinship:kinship@127.0.0.1:1/kinship",
        connect_args={"connect_timeout": 2},
    )
    app = create_app(settings_for({"t": 1}), sessionmaker(bind=dead))
    async with app.router.lifespan_context(app), http_client(app) as client:
        response = await client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


# -- auth -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer"},
        {"Authorization": f"Basic {TOKEN_A}"},
        {"Authorization": TOKEN_A},
        {"Authorization": f"Bearer {TOKEN_A}x"},
        {"Authorization": f"Bearer {TOKEN_A[:-1]}"},
    ],
    ids=["none", "wrong", "empty", "scheme-only", "basic", "no-scheme", "longer", "shorter"],
)
@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
async def test_mcp_without_a_valid_token_is_401(
    http: httpx2.AsyncClient, headers: dict[str, str], method: str
) -> None:
    response = await http.request(
        method, "/mcp", headers={**MCP_HEADERS, **headers}, json=INITIALIZE
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert response.json()["error"] == "unauthorized"


@pytest.mark.parametrize("path", ["/mcp/", "/mcp/anything", "/", "/other"])
async def test_everything_but_healthz_requires_a_token(http: httpx2.AsyncClient, path: str) -> None:
    assert (await http.post(path, json=INITIALIZE)).status_code == 401


async def test_no_configured_tokens_fails_closed(make_app) -> None:
    async with make_app(settings_for({})) as app, http_client(app) as client:
        assert (await client.get("/healthz")).status_code == 200
        for headers in ({}, bearer(""), bearer(TOKEN_A), bearer("None")):
            response = await client.post("/mcp", headers=headers, json=INITIALIZE)
            assert response.status_code == 401


async def test_bearer_scheme_is_case_insensitive(http: httpx2.AsyncClient) -> None:
    headers = {**MCP_HEADERS, "Authorization": f"bearer {TOKEN_A}"}
    assert (await http.post("/mcp", headers=headers, json=INITIALIZE)).status_code == 200


def test_owner_for_maps_tokens_to_owners() -> None:
    auth = BearerAuthMiddleware(None, {"one": 1, "two": 2})  # type: ignore[arg-type]
    assert auth.owner_for(b"Bearer one") == 1
    assert auth.owner_for(b"Bearer two") == 2
    assert auth.owner_for(b"Bearer three") is None
    assert auth.owner_for(b"Bearer one two") is None
    assert auth.owner_for(None) is None
    assert auth.owner_for("Bearer é".encode()) is None


# -- raw JSON-RPC ---------------------------------------------------------------


async def test_initialize_list_and_call_over_plain_http(http: httpx2.AsyncClient, networks) -> None:
    response = await http.post("/mcp", headers=bearer(TOKEN_A), json=INITIALIZE)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    result = response.json()["result"]
    assert result["serverInfo"]["name"] == "kinship"
    assert "mcp-session-id" not in response.headers  # stateless

    response = await http.post("/mcp", headers=bearer(TOKEN_A), json=rpc("tools/list", id=2))
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    assert "network_summary" in tools
    assert tools["network_summary"]["annotations"]["readOnlyHint"] is True

    call = rpc("tools/call", {"name": "network_summary", "arguments": {}}, id=3)
    response = await http.post("/mcp", headers=bearer(TOKEN_A), json=call)
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["counts"]["connections"] == 3
    assert result["content"][0]["text"].startswith("3 connections, 2 companies and 3 messages")

    response = await http.post("/mcp", headers=bearer(TOKEN_B), json=call)
    assert response.json()["result"]["structuredContent"]["counts"]["connections"] == 5
