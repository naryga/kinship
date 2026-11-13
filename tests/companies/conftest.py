"""Fixtures for company tests: the synthetic exports and small hand-built ones."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from ingest.conftest import V1_AT, V1_DIR, V2_AT, V2_DIR

from kinship.parser import ParsedExport, parse_export
from kinship.parser import records as r

URL = "https://www.linkedin.com/in/"
T1, T2, T3 = (datetime(2026, month, 1, tzinfo=UTC) for month in (1, 2, 3))


@pytest.fixture(scope="session")
def v1() -> ParsedExport:
    return parse_export(V1_DIR, exported_at=V1_AT)


@pytest.fixture(scope="session")
def v2() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


def person(slug: str, company: str | None, position: str | None = "Engineer") -> r.Connection:
    return r.Connection(
        first_name=slug.capitalize(),
        last_name="Test",
        profile_url=URL + slug,
        email=None,
        company=company,
        position=position,
        connected_on=date(2020, 1, 1),
    )


def export(sha: str, when: datetime, connections: list[r.Connection], **parts: Any) -> ParsedExport:
    counts = {key: len(value) for key, value in parts.items()}
    counts["connections"] = len(connections)
    return ParsedExport(
        source_sha256=sha,
        exported_at=when,
        file_counts=counts,
        warnings=[],
        connections=connections,
        **parts,
    )
