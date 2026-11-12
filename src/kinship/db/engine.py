"""Engine and session helpers.

`make_engine` builds an engine for an explicit URL (tests, scripts). `get_engine`
and `get_sessionmaker` are process-wide singletons for the configured
`DATABASE_URL`. `session_scope` is the usual commit-or-rollback unit of work.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from kinship.config import get_settings


def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """Create a new engine for `url` (defaults to the configured `DATABASE_URL`)."""
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, **kwargs)


@lru_cache
def get_engine() -> Engine:
    """The shared engine for the configured `DATABASE_URL`."""
    return make_engine()


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    """The shared session factory bound to `get_engine()`."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Yield a session; commit on success, roll back on error, always close."""
    session = (factory or get_sessionmaker())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
