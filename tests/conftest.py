"""Shared database fixtures.

- `engine` (session scope): creates a uniquely named throwaway database on the
  server that `DATABASE_URL` points at, runs `alembic upgrade head` against it,
  and drops it at the end. Concurrent pytest runs never share a database, and
  the schema under test is always the one the migrations produce.
- `alembic_cfg` (session scope): the Alembic `Config` for that database.
- `session` (function scope): a SQLAlchemy `Session` inside a transaction that
  is rolled back after the test. Code under test may call `session.commit()`
  freely; nothing leaks into the next test.
- `make_owner` / `owner`: create `Owner` rows in that session.

Tests need a reachable Postgres with the pgvector extension available (see
`docker-compose.yml`); the `kinship` role must be allowed to create databases.
"""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from alembic import command
from kinship.config import Settings
from kinship.db.models import Owner

REPO_ROOT = Path(__file__).resolve().parent.parent


def alembic_config(url: URL | str) -> Config:
    """An Alembic `Config` for this repo that targets `url` instead of `DATABASE_URL`."""
    config = Config(str(REPO_ROOT / "alembic.ini"))
    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture(scope="session")
def database_url() -> Iterator[URL]:
    """URL of a fresh, empty throwaway database; the database is dropped afterwards."""
    server_url = make_url(Settings().database_url)
    name = f"kinship_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(server_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield server_url.set(database=name)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def alembic_cfg(database_url: URL) -> Config:
    """Alembic `Config` pointed at the throwaway database."""
    return alembic_config(database_url)


@pytest.fixture(scope="session")
def engine(database_url: URL, alembic_cfg: Config) -> Iterator[Engine]:
    """Engine for the throwaway database, migrated to `head` by Alembic."""
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session whose work, including commits, is rolled back when the test ends."""
    with engine.connect() as connection:
        transaction = connection.begin()
        session = Session(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            session.close()
            transaction.rollback()


@pytest.fixture
def make_owner(session: Session) -> Callable[..., Owner]:
    """Factory: `make_owner("Ada")` adds and flushes an `Owner`, returning it with its id."""

    def _make_owner(display_name: str = "Test Owner") -> Owner:
        owner = Owner(display_name=display_name)
        session.add(owner)
        session.flush()
        return owner

    return _make_owner


@pytest.fixture
def owner(make_owner: Callable[..., Owner]) -> Owner:
    """A single owner for tests that need just one."""
    return make_owner()
