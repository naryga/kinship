"""Fixtures for retrieval tests (data and helpers are in `retrieval_seeding.py`).

- `corpus`: owners A and B with the synthetic v2 export and fake-embedded
  chunks, inside the rolled-back `session` fixture.
- `committed_corpus`: the same, committed to the test database, for MCP round
  trips over HTTP (the server opens its own sessions). The owners are deleted
  afterwards.
- `make_search_app`: `async with make_search_app() as app` serves
  `committed_corpus` with the fake embedder injected; pass `embedder=None` (or
  another embedder) to change that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import pytest
import sqlalchemy as sa
from ingest.conftest import V2_AT, V2_DIR
from retrieval_seeding import Corpus, build_corpus, settings_for
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.applications import Starlette

from kinship.db.models import Owner
from kinship.parser import ParsedExport, parse_export
from kinship.rag import FakeEmbedder
from kinship.server.app import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def v2_export() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


@pytest.fixture
def corpus(session: Session, v2_export: ParsedExport) -> Corpus:
    return build_corpus(session, v2_export)


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def committed_corpus(
    session_factory: sessionmaker[Session], v2_export: ParsedExport
) -> Iterator[Corpus]:
    with session_factory() as s:
        corpus = build_corpus(s, v2_export)
        s.commit()
    try:
        yield corpus
    finally:
        with session_factory() as s:
            # ON DELETE CASCADE removes everything the owners own.
            s.execute(sa.delete(Owner).where(Owner.id.in_([corpus.a, corpus.b])))
            s.commit()


MakeSearchApp = Callable[..., AbstractAsyncContextManager[Starlette]]


@pytest.fixture
def make_search_app(
    committed_corpus: Corpus, session_factory: sessionmaker[Session]
) -> MakeSearchApp:
    @asynccontextmanager
    async def _make(**kwargs: Any) -> AsyncIterator[Starlette]:
        kwargs.setdefault("embedder", FakeEmbedder())
        app = create_app(settings_for(committed_corpus), session_factory, **kwargs)
        async with app.router.lifespan_context(app):
            yield app

    return _make


@pytest.fixture
async def search_app(make_search_app: MakeSearchApp) -> AsyncIterator[Starlette]:
    async with make_search_app() as app:
        yield app
