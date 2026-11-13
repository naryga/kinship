"""Fixtures for the lookup tools: the synthetic exports for two owners.

Owner A and owner B both import synthetic export v1 then v2, so their data has
the same shape. Owner B then imports a third export, `B_EXTRA`, with people and
a company only B has. Anything of B's that shows up for A is a leak.

- `two_owners` lives in the rolled-back `session` (repository tests).
- `committed` commits the same data for the MCP round-trip tests, which run the
  server on its own sessions, and deletes both owners afterwards.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from functools import partial

import pytest
import sqlalchemy as sa
from companies.conftest import URL
from ingest.conftest import V1_AT, V1_DIR, V2_AT, V2_DIR
from mcp import Client
from server.mcp_testing import TOKEN_A, TOKEN_B, connect, settings_for
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.applications import Starlette

from kinship.db.models import Owner
from kinship.ingest import import_export
from kinship.parser import ParsedExport, parse_export
from kinship.parser import records as r
from kinship.repo import Repository
from kinship.server.app import create_app
from kinship.synth import build_world

WORLD = build_world()
B_EXTRA_AT = datetime(2026, 12, 1, 12, tzinfo=UTC)
B_ONLY_COMPANY = "Bumblethwaite Tidewater"

# Only owner B has these three people. Two of them are at Gifthealth, which both owners have.
B_ONLY_PEOPLE = [
    r.Connection(
        first_name=first,
        last_name="Onlybee",
        profile_url=URL + f"{first.lower()}-onlybee",
        email=None,
        company=company,
        position=position,
        connected_on=date(2026, 11, 1),
    )
    for first, company, position in (
        ("Beatrix", "Gifthealth", "Pharmacist"),
        ("Bertram", "Gift Health, Inc.", "Engineer"),
        ("Bryony", B_ONLY_COMPANY, "Founder"),
    )
]


@pytest.fixture(scope="session")
def v1() -> ParsedExport:
    return parse_export(V1_DIR, exported_at=V1_AT)


@pytest.fixture(scope="session")
def v2() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


@pytest.fixture(scope="session")
def b_extra(v2: ParsedExport) -> ParsedExport:
    """Export v2 plus `B_ONLY_PEOPLE`, a month or more later."""
    return replace(
        v2,
        source_sha256="b-extra",
        exported_at=B_EXTRA_AT,
        connections=[*v2.connections, *B_ONLY_PEOPLE],
    )


def _import_both(
    session: Session, v1: ParsedExport, v2: ParsedExport, b_extra: ParsedExport
) -> tuple[int, int]:
    owners = [Owner(display_name="Owner A"), Owner(display_name="Owner B")]
    session.add_all(owners)
    session.flush()
    for owner in owners:
        import_export(session, owner.id, v1)
        import_export(session, owner.id, v2)
    import_export(session, owners[1].id, b_extra)
    session.flush()
    return owners[0].id, owners[1].id


@dataclass(frozen=True)
class TwoOwners:
    a: Repository
    b: Repository


@pytest.fixture
def two_owners(
    session: Session, v1: ParsedExport, v2: ParsedExport, b_extra: ParsedExport
) -> TwoOwners:
    a, b = _import_both(session, v1, v2, b_extra)
    return TwoOwners(Repository(session, a), Repository(session, b))


@pytest.fixture
def repo(two_owners: TwoOwners) -> Repository:
    """Owner A: exactly the synthetic world after v1 then v2."""
    return two_owners.a


# -- MCP round trips ------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def committed(
    engine: Engine, v1: ParsedExport, v2: ParsedExport, b_extra: ParsedExport
) -> Iterator[tuple[int, int]]:
    """Committed owners A and B (ids), deleted again after the test."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        ids = _import_both(session, v1, v2, b_extra)
        session.commit()
    try:
        yield ids
    finally:
        with factory() as session:
            session.execute(sa.delete(Owner).where(Owner.id.in_(ids)))
            session.commit()


@pytest.fixture
async def app(engine: Engine, committed: tuple[int, int]) -> AsyncIterator[Starlette]:
    a, b = committed
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    app = create_app(settings_for({TOKEN_A: a, TOKEN_B: b}), factory)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
def mcp_client(app: Starlette) -> Callable[..., AbstractAsyncContextManager[Client]]:
    """`async with mcp_client(TOKEN_A) as client` gives a connected SDK client."""
    return partial(connect, app)
