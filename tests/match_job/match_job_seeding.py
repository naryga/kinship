"""Data for `match_job` tests: two owners with the same synthetic exports.

`build_job_world` imports exports v1 and v2 for owners A and B and embeds every
chunk with `FakeEmbedder`. B alone also has:

- a company nobody else has (`B_ONLY_COMPANY`) with one person whose title is
  exactly the sample job's title;
- a person at Tessellate Health (the sample job's company) with a data title;
- a person who used to be at Tessellate Health and is now at B's own company;
- a recommendation full of the sample job's words, plus `B_ONLY_WORD`;
- a skill containing `B_ONLY_WORD`, which `B_ONLY_JOB` names.

None of that may ever show up for A.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import sqlalchemy as sa
from ingest.conftest import V2_AT
from sqlalchemy.orm import Session

from kinship.db.models import (
    Company,
    Connection,
    ConnectionHistory,
    Import,
    Owner,
    Recommendation,
    Skill,
)
from kinship.ingest import import_export
from kinship.parser import ParsedExport
from kinship.rag import FakeEmbedder, sync_chunks

SAMPLE_JOB = (Path(__file__).parent / "sample_job.txt").read_text(encoding="utf-8")
SAMPLE_TITLE = "Senior Data Engineer"
SAMPLE_COMPANY = "Tessellate Health"
SAMPLE_COMPANY_KEY = "tessellate"

B_ONLY_COMPANY = "Borealis Kiteworks"
B_ONLY_WORD = "zeppelin"
B_ONLY_SKILL = "Zeppelin Piloting"
B_ONLY_NAMES = ("Onlyb Kiteperson", "Onlyb Tessellator", "Onlyb Leaver")
B_ONLY_URL = "https://www.linkedin.com/in/onlyb-{}-00000001"
B_ONLY_AT = datetime(2026, 12, 1, 12, tzinfo=UTC)
B_ONLY_TEXT = (
    f"Sam ran the {B_ONLY_WORD} telemetry data pipelines in Airflow, modelled pharmacy claims "
    "and healthcare billing data in dbt, and kept the warehouse honest."
)
B_ONLY_JOB = f"""Kite Engineer
{B_ONLY_COMPANY} - Remote

We are looking for a Kite Engineer to build data pipelines for our fleet.
You have done some {B_ONLY_SKILL.lower()} and you know SQL.
"""
LEAK_MARKERS = ("Onlyb", B_ONLY_WORD, "Zeppelin", "Borealis", "Kiteworks")
"""Strings that only B's data contains (the sample job contains none of them)."""


@dataclass(frozen=True)
class JobWorld:
    a: int
    b: int
    b_company_id: int
    b_connection_ids: tuple[int, ...]
    """Kiteperson, Tessellator, Leaver."""


def add_b_only_data(session: Session, owner_id: int) -> tuple[int, tuple[int, ...]]:
    """B's extra rows (see the module docstring); returns (company id, connection ids)."""
    tessellate_id = session.scalars(
        sa.select(Company.id).where(
            Company.owner_id == owner_id, Company.name_normalized == "tessellatehealth"
        )
    ).one()
    v2_import = session.scalars(
        sa.select(Import.id).where(Import.owner_id == owner_id, Import.exported_at == V2_AT)
    ).one()
    later = Import(owner_id=owner_id, source_sha256="sha-b-only-job", exported_at=B_ONLY_AT)
    company = Company(
        owner_id=owner_id,
        name_normalized="borealiskiteworks",
        display_name=B_ONLY_COMPANY,
        aliases=[B_ONLY_COMPANY],
    )
    session.add_all([later, company])
    session.flush()

    def person(index: int, title: str, company_raw: str, company_id: int) -> Connection:
        first, last = B_ONLY_NAMES[index].split()
        return Connection(
            owner_id=owner_id,
            profile_url=B_ONLY_URL.format(last.lower()),
            first_name=first,
            last_name=last,
            company_raw=company_raw,
            company_id=company_id,
            position=title,
            connected_on=date(2026, 6, 1 + index),
            first_seen_import_id=v2_import,
            last_seen_import_id=later.id,
        )

    people = [
        person(0, SAMPLE_TITLE, B_ONLY_COMPANY, company.id),
        person(1, "Data Engineer II", SAMPLE_COMPANY, tessellate_id),
        person(2, "Kite Rigger", B_ONLY_COMPANY, company.id),
    ]
    session.add_all(people)
    session.flush()
    leaver = people[2]
    session.add_all(
        [
            ConnectionHistory(
                owner_id=owner_id,
                connection_id=leaver.id,
                import_id=v2_import,
                company_raw=SAMPLE_COMPANY,
                company_id=tessellate_id,
                position="Claims Data Analyst",
            ),
            ConnectionHistory(
                owner_id=owner_id,
                connection_id=leaver.id,
                import_id=later.id,
                company_raw=B_ONLY_COMPANY,
                company_id=company.id,
                position="Kite Rigger",
            ),
            Recommendation(
                owner_id=owner_id,
                direction="received",
                first_name="Onlyb",
                last_name="Kiteperson",
                text=B_ONLY_TEXT,
                connection_id=people[0].id,
            ),
            Skill(owner_id=owner_id, name=B_ONLY_SKILL),
        ]
    )
    session.flush()
    return company.id, tuple(p.id for p in people)


def build_job_world(session: Session, v1: ParsedExport, v2: ParsedExport) -> JobWorld:
    """Owners A and B with exports v1 and v2; B gets its extra rows. Does not commit."""
    owners = [Owner(display_name="Job A"), Owner(display_name="Job B")]
    session.add_all(owners)
    session.flush()
    for owner in owners:
        import_export(session, owner.id, v1)
        import_export(session, owner.id, v2)
    company_id, connection_ids = add_b_only_data(session, owners[1].id)
    for owner in owners:
        sync_chunks(session, owner.id, FakeEmbedder())
    session.flush()
    return JobWorld(owners[0].id, owners[1].id, company_id, connection_ids)
