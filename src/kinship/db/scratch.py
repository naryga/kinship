"""Throwaway databases: create one, migrate it to `head`, drop it afterwards.

    with scratch_database(prefix="kinship_eval") as url:
        engine = create_engine(url)
        ...

Used by `kinship eval`, which needs an empty, fully migrated database for a
moment and must never touch real data. The migrations are read from the
repository checkout (`alembic.ini` and `alembic/` are not part of the installed
package), so this only works from a source checkout.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from alembic import command
from kinship.config import get_settings

_PREFIX = re.compile(r"[a-z][a-z0-9_]{0,40}")


class ScratchDatabaseError(RuntimeError):
    """A throwaway database could not be set up."""


def repo_root() -> Path:
    """The source checkout that holds `alembic.ini`, or raise `ScratchDatabaseError`."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "alembic.ini").is_file() or not (root / "alembic").is_dir():
        raise ScratchDatabaseError(
            f"alembic.ini and alembic/ were not found in {root}. This command needs a source "
            "checkout of the kinship repository (run it with `uv run` from the checkout)."
        )
    return root


def alembic_config(url: URL | str) -> Config:
    """An Alembic `Config` for this checkout that targets `url`."""
    config = Config(str(repo_root() / "alembic.ini"))
    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    config.attributes["configure_logger"] = False
    return config


def new_database_name(prefix: str) -> str:
    """`<prefix>_<8 random hex digits>`."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@contextmanager
def scratch_database(
    server_url: URL | str | None = None, *, prefix: str = "kinship_scratch"
) -> Iterator[URL]:
    """Yield the URL of a new, uniquely named database migrated to `head`.

    The database is created on the server of `server_url` (default: the
    configured `DATABASE_URL`), whose role must be allowed to create databases,
    and is dropped on exit whatever happened inside the block. The database
    that `server_url` itself names is only connected to, never changed.
    """
    if not _PREFIX.fullmatch(prefix):
        raise ValueError("prefix must be lower-case letters, digits and underscores")
    repo_root()  # fail before creating anything if this is not a checkout
    server = make_url(server_url or get_settings().database_url)
    name = new_database_name(prefix)
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
        try:
            url = server.set(database=name)
            command.upgrade(alembic_config(url), "head")
            yield url
        finally:
            with admin.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()
