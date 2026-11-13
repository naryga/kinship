"""Fixtures for the network tools: the synthetic exports for one or two owners.

- `repo`: one owner with exports v1 then v2, inside the rolled-back `session`.
- `without_messages`: exports v1 and v2 generated with `include_messages=False`.
- `two_networks` (module scope, committed): owners A and B with the same two
  exports, plus data only B has, and owner C with only export v1 and no
  messages. The MCP server opens its own sessions, so this data is committed
  and the owners are deleted afterwards (cascade).
- `app` / `mcp_client`: the ASGI app for `two_networks` and SDK clients for it.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path

import pytest
import sqlalchemy as sa
from ingest.conftest import V1_AT, V1_DIR, V2_AT, V2_DIR
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from starlette.applications import Starlette

from kinship.db.models import (
    Company,
    Connection,
    ConnectionHistory,
    Import,
    JobApplication,
    Message,
    MyPosition,
    Owner,
    Recommendation,
)
from kinship.ingest import import_export
from kinship.parser import ParsedExport, parse_export
from kinship.repo import Repository
from kinship.server.app import create_app
from kinship.synth import generate_export

# `mcp_testing` lives next to the server tests, which are not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from mcp_testing import TOKEN_A, TOKEN_B, connect, settings_for

TOKEN_C = "token-c"
B_ONLY_COMPANY = "Borealis Kiteworks"
B_ONLY_URL = "https://www.linkedin.com/in/only-b-person-00000001"
B_ONLY_AT = datetime(2026, 12, 1, 12, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def v1() -> ParsedExport:
    return parse_export(V1_DIR, exported_at=V1_AT)


@pytest.fixture(scope="session")
def v2() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


@pytest.fixture(scope="session")
def without_messages(tmp_path_factory: pytest.TempPathFactory) -> list[ParsedExport]:
    """Exports v1 and v2 as LinkedIn delivers them when messages were not requested."""
    root = tmp_path_factory.mktemp("no-messages")
    return [
        parse_export(
            generate_export(root / f"v{version}", version=version, include_messages=False),
            exported_at=at,
        )
        for version, at in ((1, V1_AT), (2, V2_AT))
    ]


@pytest.fixture
def repo(session: Session, owner: Owner, v1: ParsedExport, v2: ParsedExport) -> Repository:
    """An owner with both synthetic exports imported."""
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    return Repository(session, owner.id)


def add_b_only_data(session: Session, owner_id: int) -> tuple[int, int]:
    """Rows only owner B has; returns (company id, connection id).

    A third, later import in which one connection (first seen in export v2 at
    Acme) has moved to a company nobody else has. That person wrote B a
    recommendation and exchanged a message; B also applied to the company and
    lists it as a past employer.
    """
    acme_id = session.scalars(
        sa.select(Company.id).where(Company.owner_id == owner_id, Company.name_normalized == "acme")
    ).one()
    previous = session.scalars(
        sa.select(Import.id).where(Import.owner_id == owner_id, Import.exported_at == V2_AT)
    ).one()
    later = Import(owner_id=owner_id, source_sha256="sha-b-only", exported_at=B_ONLY_AT)
    company = Company(
        owner_id=owner_id, name_normalized="borealiskiteworks", display_name=B_ONLY_COMPANY
    )
    session.add_all([later, company])
    session.flush()
    person = Connection(
        owner_id=owner_id,
        profile_url=B_ONLY_URL,
        first_name="Onlyb",
        last_name="Person",
        company_raw=B_ONLY_COMPANY,
        company_id=company.id,
        position="Kite Engineer",
        connected_on=date(2026, 6, 1),
        first_seen_import_id=previous,
        last_seen_import_id=later.id,
    )
    session.add(person)
    session.flush()
    session.add_all(
        [
            ConnectionHistory(
                owner_id=owner_id,
                connection_id=person.id,
                import_id=previous,
                company_raw="Acme",
                company_id=acme_id,
                position="Kite Engineer",
            ),
            ConnectionHistory(
                owner_id=owner_id,
                connection_id=person.id,
                import_id=later.id,
                company_raw=B_ONLY_COMPANY,
                company_id=company.id,
                position="Kite Engineer",
            ),
            Message(
                owner_id=owner_id,
                conversation_id="b-only",
                from_name="Onlyb Person",
                sent_at=datetime(2026, 11, 20, tzinfo=UTC),
                body="hello from B's network",
                connection_id=person.id,
            ),
            Recommendation(
                owner_id=owner_id,
                direction="received",
                first_name="Onlyb",
                last_name="Person",
                text="Great to work with.",
                created_at=datetime(2026, 11, 1, tzinfo=UTC),
                connection_id=person.id,
            ),
            JobApplication(
                owner_id=owner_id,
                applied_at=datetime(2026, 12, 1, 9, tzinfo=UTC),
                company_name=B_ONLY_COMPANY,
                job_title="Kite Lead",
                company_id=company.id,
            ),
            MyPosition(
                owner_id=owner_id, company=B_ONLY_COMPANY, company_id=company.id, title="Advisor"
            ),
        ]
    )
    session.flush()
    return company.id, person.id


@dataclass(frozen=True)
class TwoNetworks:
    owner_a: int
    owner_b: int
    owner_c: int
    b_company_id: int
    b_connection_id: int
    session_factory: sessionmaker[Session]


@pytest.fixture(scope="module")
def two_networks(
    engine: Engine, v1: ParsedExport, v2: ParsedExport, without_messages: list[ParsedExport]
) -> Iterator[TwoNetworks]:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        owners = [Owner(display_name=f"Owner {letter}") for letter in "ABC"]
        s.add_all(owners)
        s.flush()
        owner_ids = [owner.id for owner in owners]
        for owner_id in owner_ids[:2]:
            import_export(s, owner_id, v1)
            import_export(s, owner_id, v2)
        b_company_id, b_connection_id = add_b_only_data(s, owner_ids[1])
        import_export(s, owner_ids[2], without_messages[0])
        s.commit()
    try:
        yield TwoNetworks(*owner_ids, b_company_id, b_connection_id, factory)
    finally:
        with factory() as s:
            s.execute(sa.delete(Owner).where(Owner.id.in_(owner_ids)))
            s.commit()


@pytest.fixture
async def app(two_networks: TwoNetworks) -> AsyncIterator[Starlette]:
    settings = settings_for(
        {
            TOKEN_A: two_networks.owner_a,
            TOKEN_B: two_networks.owner_b,
            TOKEN_C: two_networks.owner_c,
        }
    )
    app = create_app(settings, two_networks.session_factory)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
def mcp_client(app: Starlette) -> Callable[..., AbstractAsyncContextManager]:
    return partial(connect, app)
