"""Derive one owner's `companies` and every `company_id` link from the raw names.

`rebuild_companies` is a recompute over *all* of the owner's rows, not an
increment: the result depends only on the rows present, so it is idempotent and
independent of import order. `import_export` runs it as its last step; run it
on its own after changing the rules in `kinship.companies`.

Sources of company names (table.column), each with a `company_id` to fill:

    connections.company_raw          connection_history.company_raw
    my_positions.company             company_follows.organization
    job_applications.company_name    saved_jobs.company_name

For each distinct `company_key` among those names the owner gets one
`companies` row:

- `name_normalized`: the key;
- `aliases`: every distinct raw spelling with that key, from all six sources,
  sorted;
- `display_name`: the spelling used by the most current (not removed)
  connections; when no current connection names the company, the most common
  spelling across all six sources. Ties go to the longer spelling, then to the
  one that sorts first.

A company row keeps its id for as long as its key is in use, and is deleted
once no row names it. Blank names get `company_id` NULL. Everything goes
through `ImportStore`, so it never reads or writes another owner's rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session

from kinship.companies import company_key
from kinship.db.models import (
    Company,
    CompanyFollow,
    Connection,
    ConnectionHistory,
    JobApplication,
    MyPosition,
    OwnedMixin,
    SavedJob,
)
from kinship.ingest.store import ImportStore

COMPANY_SOURCES: tuple[tuple[type[OwnedMixin], str], ...] = (
    (Connection, "company_raw"),
    (ConnectionHistory, "company_raw"),
    (MyPosition, "company"),
    (CompanyFollow, "organization"),
    (JobApplication, "company_name"),
    (SavedJob, "company_name"),
)
"""Every (model, raw company name column) that carries a `company_id`."""


@dataclass(frozen=True)
class CompaniesRebuilt:
    """What one `rebuild_companies` run did."""

    total: int
    created: int
    deleted: int
    relinked: int
    """Rows whose `company_id` changed."""


def _spelling(raw: str) -> str:
    return " ".join(raw.split())


def _display_name(spellings: Counter[str]) -> str:
    return min(spellings, key=lambda name: (-spellings[name], -len(name), name))


def rebuild_companies(session: Session, owner_id: int) -> CompaniesRebuilt:
    """Recompute `owner_id`'s companies and company links (no commit)."""
    store = ImportStore(session, owner_id)
    store.flush()

    current: dict[str, Counter[str]] = defaultdict(Counter)  # by current connections
    overall: dict[str, Counter[str]] = defaultdict(Counter)
    # Per source: (row id, company key or None, stored company_id).
    rows: list[tuple[type[OwnedMixin], list[tuple[int, str | None, int | None]]]] = []
    for model, column in COMPANY_SOURCES:
        is_connection = model is Connection
        extra = ("removed_at",) if is_connection else ()
        linked: list[tuple[int, str | None, int | None]] = []
        for row_id, raw, company_id, *rest in store.values(model, column, "company_id", *extra):
            key = company_key(raw)
            linked.append((row_id, key, company_id))
            if key is None:
                continue
            overall[key][_spelling(raw)] += 1
            if is_connection and rest[0] is None:
                current[key][_spelling(raw)] += 1
        rows.append((model, linked))

    companies = {row.name_normalized: row for row in store.rows(Company)}
    created = 0
    for key, spellings in overall.items():
        display_name = _display_name(current.get(key) or spellings)
        aliases = sorted(spellings)
        company = companies.get(key)
        if company is None:
            companies[key] = store.add(
                Company(name_normalized=key, display_name=display_name, aliases=aliases)
            )
            created += 1
        else:
            if company.display_name != display_name:
                company.display_name = display_name
            if list(company.aliases or []) != aliases:
                company.aliases = aliases
    store.flush()  # assigns ids to the new companies

    relinked = 0
    for model, linked in rows:
        changes = {
            row_id: (companies[key].id if key is not None else None)
            for row_id, key, company_id in linked
            if company_id != (companies[key].id if key is not None else None)
        }
        store.set_values(model, "company_id", changes)
        relinked += len(changes)

    unused = [row.id for key, row in companies.items() if key not in overall]
    store.delete_ids(Company, unused)
    return CompaniesRebuilt(
        total=len(overall), created=created, deleted=len(unused), relinked=relinked
    )
