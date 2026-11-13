"""End-to-end MCP over HTTP with the SDK client, and owner isolation between tokens."""

from __future__ import annotations

import threading

import anyio
import pytest
from mcp import Client
from mcp_testing import TOKEN_A, TOKEN_B, connect, settings_for

from kinship.repo import owned_models
from kinship.server import context
from kinship.server.app import build_mcp
from kinship.server.context import NoOwnerError, bind_owner, current_owner_id, repo_session

pytestmark = pytest.mark.anyio


def nonzero(counts: dict[str, int]) -> dict[str, int]:
    return {table: n for table, n in counts.items() if n}


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_initialize_list_tools_and_call_network_summary(mcp_client, networks, mode) -> None:
    async with mcp_client(TOKEN_A, mode=mode) as client:
        tools = await client.list_tools()
        assert "network_summary" in {tool.name for tool in tools.tools}

        result = await client.call_tool("network_summary", {})

    assert not result.is_error
    data = result.structured_content
    assert set(data["counts"]) == {m.__tablename__ for m in owned_models()}
    assert nonzero(data["counts"]) == networks.a.expected_counts
    assert data["latest_import"]["id"] == networks.a.import_id
    assert data["latest_import"]["source_sha256"] == "sha-alpha"
    text = result.content[0].text
    assert text.startswith("3 connections, 2 companies and 3 messages; latest import on ")


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_each_token_sees_only_its_own_network(mcp_client, networks, mode) -> None:
    async with mcp_client(TOKEN_A, mode=mode) as a, mcp_client(TOKEN_B, mode=mode) as b:
        # Interleave the two clients on purpose.
        first_a = (await a.call_tool("network_summary", {})).structured_content
        first_b = (await b.call_tool("network_summary", {})).structured_content
        second_a = (await a.call_tool("network_summary", {})).structured_content

    assert nonzero(first_a["counts"]) == networks.a.expected_counts
    assert nonzero(first_b["counts"]) == networks.b.expected_counts
    assert first_a["latest_import"]["id"] == networks.a.import_id
    assert first_b["latest_import"]["id"] == networks.b.import_id
    assert first_b["latest_import"]["source_sha256"] == "sha-beta"
    assert second_a == first_a


async def test_another_token_with_a_small_network_sees_only_that(make_app, seed_owner) -> None:
    small = seed_owner("Small", connections=0, tag="gamma")
    async with (
        make_app(settings_for({"token-c": small.owner_id})) as app,
        connect(app, "token-c") as client,
    ):
        data = (await client.call_tool("network_summary", {})).structured_content
    assert nonzero(data["counts"]) == {"imports": 1, "companies": 2, "my_positions": 1}


async def test_concurrent_requests_each_see_their_own_owner(
    app, mcp_client, networks, monkeypatch
) -> None:
    """Hold many requests for both owners open at once; each must resolve its own owner."""
    in_flight = threading.Barrier(2, timeout=10)
    seen: list[tuple[int, int]] = []
    real_open = context.open_repository

    def slow_open(session_factory, owner_id):
        # Both owners' tool calls are inside their handlers at the same moment.
        in_flight.wait()
        seen.append((owner_id, current_owner_id()))
        return real_open(session_factory, owner_id)

    monkeypatch.setattr(context, "open_repository", slow_open)

    results: dict[str, list[int]] = {TOKEN_A: [], TOKEN_B: []}

    async def call(token: str) -> None:
        async with mcp_client(token) as client:
            for _ in range(3):
                data = (await client.call_tool("network_summary", {})).structured_content
                results[token].append(data["counts"]["connections"])

    async with anyio.create_task_group() as tg:
        tg.start_soon(call, TOKEN_A)
        tg.start_soon(call, TOKEN_B)

    assert results == {TOKEN_A: [3, 3, 3], TOKEN_B: [5, 5, 5]}
    assert len(seen) == 6
    assert all(bound == current for bound, current in seen)
    assert {owner for owner, _ in seen} == {networks.a.owner_id, networks.b.owner_id}


# -- context --------------------------------------------------------------------


async def test_no_owner_outside_a_request(session_factory, networks) -> None:
    with pytest.raises(NoOwnerError):
        current_owner_id()
    with pytest.raises(NoOwnerError), repo_session():
        pass

    with bind_owner(networks.a.owner_id, session_factory):
        assert current_owner_id() == networks.a.owner_id
        with repo_session() as repo:
            assert repo.owner_id == networks.a.owner_id
            assert repo.counts()["connections"] == 3

    with pytest.raises(NoOwnerError):
        current_owner_id()


async def test_tool_without_an_authenticated_request_fails(session_factory, networks) -> None:
    """Calling the server in-process (no HTTP request, so no token) must not reach any data."""
    mcp = build_mcp(session_factory)
    async with Client(mcp) as client:
        result = await client.call_tool("network_summary", {})
    assert result.is_error
    assert "counts" not in (result.structured_content or {})
