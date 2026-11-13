"""Fixtures for `match_job` tests (data and constants are in `match_job_seeding.py`).

- `job_world`: owners A and B with both synthetic exports and fake-embedded
  chunks, inside the rolled-back `session` fixture. `repo_a` / `repo_b` are
  repositories for them.
- `committed_job_world` (module scope): the same, committed, for MCP round
  trips over HTTP. The owners are deleted afterwards (cascade).
- `make_job_app`: `async with make_job_app() as app` serves the committed
  world with the fake embedder injected; pass `embedder=None` for a server
  without vector search.
- `job_client`: `async with job_client(app, TOKEN_A) as client`.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from ingest.conftest import V1_AT, V1_DIR, V2_AT, V2_DIR
from match_job_seeding import JobWorld, build_job_world
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.applications import Starlette

from kinship.db.models import Owner
from kinship.parser import ParsedExport, parse_export
from kinship.rag import FakeEmbedder
from kinship.repo import Repository
from kinship.server.app import create_app

# `mcp_testing` lives next to the server tests, which are not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from mcp_testing import TOKEN_A, TOKEN_B, connect, settings_for


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def job_exports() -> tuple[ParsedExport, ParsedExport]:
    return parse_export(V1_DIR, exported_at=V1_AT), parse_export(V2_DIR, exported_at=V2_AT)


@pytest.fixture
def job_world(session: Session, job_exports: tuple[ParsedExport, ParsedExport]) -> JobWorld:
    return build_job_world(session, *job_exports)


@pytest.fixture
def repo_a(session: Session, job_world: JobWorld) -> Repository:
    return Repository(session, job_world.a)


@pytest.fixture
def repo_b(session: Session, job_world: JobWorld) -> Repository:
    return Repository(session, job_world.b)


@pytest.fixture(scope="module")
def job_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture(scope="module")
def committed_job_world(
    job_session_factory: sessionmaker[Session], job_exports: tuple[ParsedExport, ParsedExport]
) -> Iterator[JobWorld]:
    with job_session_factory() as s:
        world = build_job_world(s, *job_exports)
        s.commit()
    try:
        yield world
    finally:
        with job_session_factory() as s:
            # ON DELETE CASCADE removes everything the owners own.
            s.execute(sa.delete(Owner).where(Owner.id.in_([world.a, world.b])))
            s.commit()


MakeJobApp = Callable[..., AbstractAsyncContextManager[Starlette]]


@pytest.fixture
def make_job_app(
    committed_job_world: JobWorld, job_session_factory: sessionmaker[Session]
) -> MakeJobApp:
    @asynccontextmanager
    async def _make(**kwargs: Any) -> AsyncIterator[Starlette]:
        kwargs.setdefault("embedder", FakeEmbedder())
        settings = settings_for({TOKEN_A: committed_job_world.a, TOKEN_B: committed_job_world.b})
        app = create_app(settings, job_session_factory, **kwargs)
        async with app.router.lifespan_context(app):
            yield app

    return _make


@pytest.fixture
async def job_app(make_job_app: MakeJobApp) -> AsyncIterator[Starlette]:
    async with make_job_app() as app:
        yield app


@pytest.fixture
def job_client() -> Callable[..., AbstractAsyncContextManager]:
    return connect


__all__ = ["TOKEN_A", "TOKEN_B"]
