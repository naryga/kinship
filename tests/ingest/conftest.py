"""Helpers for importer tests: the two synthetic exports and a normalized dump."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from kinship.db.models import Base, Connection, Import
from kinship.parser import ParsedExport, parse_export
from kinship.synth import V1_EXPORTED_AT, V2_EXPORTED_AT

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"
V1_DIR = FIXTURES / "export_v1"
V2_DIR = FIXTURES / "export_v2"


def at_noon(day: date) -> datetime:
    return datetime.combine(day, time(12), tzinfo=UTC)


V1_AT = at_noon(V1_EXPORTED_AT)
V2_AT = at_noon(V2_EXPORTED_AT)


@pytest.fixture(scope="session")
def v1() -> ParsedExport:
    """Committed fixture export #1. Its mtimes are checkout times, so the date is explicit."""
    return parse_export(V1_DIR, exported_at=V1_AT)


@pytest.fixture(scope="session")
def v2() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


_SKIP_COLUMNS = {"id", "owner_id", "imported_at"}
_IMPORT_COLUMNS = {"import_id", "first_seen_import_id", "last_seen_import_id"}


def dump(session: Session, owner_id: int) -> dict[str, list[tuple[Any, ...]]]:
    """Every owned row of `owner_id`, without surrogate ids or load times.

    Import ids become the import's SHA-256 and connection ids the connection's
    natural key, so dumps of different owners (or import orders) compare equal
    exactly when the data is the same.
    """
    session.flush()
    session.expire_all()
    imports = {
        row.id: row.source_sha256
        for row in session.scalars(select(Import).where(Import.owner_id == owner_id))
    }
    connections = {
        row.id: row.profile_url or (row.first_name, row.last_name, row.connected_on)
        for row in session.scalars(select(Connection).where(Connection.owner_id == owner_id))
    }
    result: dict[str, list[tuple[Any, ...]]] = {}
    for table in Base.metadata.sorted_tables:
        if "owner_id" not in table.c:
            continue
        rows = []
        for row in session.execute(select(table).where(table.c.owner_id == owner_id)).mappings():
            values = []
            for name, value in row.items():
                if name in _SKIP_COLUMNS:
                    continue
                if name in _IMPORT_COLUMNS and value is not None:
                    value = imports[value]
                elif name == "connection_id" and value is not None:
                    value = connections[value]
                values.append((name, value))
            rows.append(tuple(values))
        result[table.name] = sorted(rows, key=repr)
    return result
