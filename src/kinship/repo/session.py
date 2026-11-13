"""Opening repositories and checking the database, for callers outside `kinship.repo`.

The server never handles a `Session` itself: it asks for a `Repository` here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Session

from kinship.db.engine import get_sessionmaker

if TYPE_CHECKING:
    from kinship.repo import Repository

SessionFactory = Callable[[], Session]
"""Anything that returns a new `Session` when called, e.g. a `sessionmaker`."""


def default_session_factory() -> SessionFactory:
    """The session factory for the configured `DATABASE_URL`."""
    return get_sessionmaker()


@contextmanager
def open_repository(session_factory: SessionFactory, owner_id: int) -> Iterator[Repository]:
    """Yield a `Repository` for `owner_id` on a fresh, read-only transaction.

    The transaction is `READ ONLY` at the database level, so nothing reached
    through the repository can write. It is rolled back and closed on exit.
    """
    from kinship.repo import Repository

    session = session_factory()
    try:
        repo = Repository(session, owner_id)  # validates owner_id before any SQL runs
        session.execute(sa.text("SET TRANSACTION READ ONLY"))
        yield repo
    finally:
        session.rollback()
        session.close()


def database_is_up(session_factory: SessionFactory) -> bool:
    """True if `SELECT 1` succeeds on a new session."""
    try:
        session = session_factory()
        try:
            session.execute(sa.text("SELECT 1"))
        finally:
            session.close()
    except Exception:
        return False
    return True
