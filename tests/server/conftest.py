"""Fixtures for MCP server tests.

The server opens its own database sessions, so it cannot see rows inside the
rolled-back `session` fixture. These fixtures therefore *commit* data to the
throwaway test database and delete it again afterwards:

- `anyio_backend`: async tests (`pytestmark = pytest.mark.anyio`) run on asyncio.
- `session_factory`: a `sessionmaker` bound to the test `engine`.
- `seed_owner`: factory for committed owners with a small network:
  `seed_owner("Name", connections=2, tag="gamma") -> SeededNetwork`.
- `networks`: two committed owners with the same data shape: `networks.a`
  (3 connections, token `token-a`) and `networks.b` (5 connections, `token-b`).
- `app`: the ASGI app for `networks`, with its lifespan running.
- `make_app`: `async with make_app(settings) as app` for other configurations.
- `http`: an `httpx2.AsyncClient` talking to `app` in-process, no token.
- `mcp_client`: `async with mcp_client("token-a") as client` gives a connected
  SDK `Client` for `app`; pass `mode="legacy"` to force the initialize handshake.

Plain helpers (`http_client`, `connect`, `settings_for`, tokens) are in
`mcp_testing.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from functools import partial

import httpx2
import pytest
import sqlalchemy as sa
from mcp import Client
from mcp_testing import TOKEN_A, TOKEN_B, connect, http_client, settings_for
from seeding import SeededNetwork, seed_network
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.applications import Starlette

from kinship.config import Settings
from kinship.db.models import Owner
from kinship.server.app import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seed_owner(session_factory: sessionmaker[Session]) -> Iterator[Callable[..., SeededNetwork]]:
    """Factory for committed owners with a seeded network; all are deleted afterwards."""
    owner_ids: list[int] = []

    def _seed(display_name: str, *, connections: int, tag: str) -> SeededNetwork:
        with session_factory() as s:
            owner = Owner(display_name=display_name)
            s.add(owner)
            s.flush()
            owner_ids.append(owner.id)
            network = seed_network(s, owner, connections=connections, tag=tag)
            s.commit()
        return network

    try:
        yield _seed
    finally:
        with session_factory() as s:
            # ON DELETE CASCADE removes everything the owners own.
            s.execute(sa.delete(Owner).where(Owner.id.in_(owner_ids)))
            s.commit()


@dataclass(frozen=True)
class Networks:
    a: SeededNetwork
    b: SeededNetwork

    @property
    def settings(self) -> Settings:
        return settings_for({TOKEN_A: self.a.owner_id, TOKEN_B: self.b.owner_id})


@pytest.fixture
def networks(seed_owner: Callable[..., SeededNetwork]) -> Networks:
    return Networks(
        a=seed_owner("Owner A", connections=3, tag="alpha"),
        b=seed_owner("Owner B", connections=5, tag="beta"),
    )


MakeApp = Callable[[Settings], AbstractAsyncContextManager[Starlette]]


@pytest.fixture
def make_app(session_factory: sessionmaker[Session]) -> MakeApp:
    @asynccontextmanager
    async def _make_app(settings: Settings) -> AsyncIterator[Starlette]:
        app = create_app(settings, session_factory)
        async with app.router.lifespan_context(app):
            yield app

    return _make_app


@pytest.fixture
async def app(networks: Networks, make_app: MakeApp) -> AsyncIterator[Starlette]:
    async with make_app(networks.settings) as app:
        yield app


@pytest.fixture
async def http(app: Starlette) -> AsyncIterator[httpx2.AsyncClient]:
    async with http_client(app) as client:
        yield client


@pytest.fixture
def mcp_client(app: Starlette) -> Callable[..., AbstractAsyncContextManager[Client]]:
    return partial(connect, app)
