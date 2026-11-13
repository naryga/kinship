"""Network queries: who is at a company, how well the owner knows them, what changed.

Two groups of methods, both set-based (a fixed number of statements however
many people are involved):

- For `warm_paths`: `connections_at_companies`, `former_connections_of_companies`
  and `warmth_signals`, which gathers a `kinship.warmth.Signals` per connection.
- For `network_changes`: `imports_in_order`, `job_changes`, `new_connections`,
  `removed_connections` and `applied_companies`.

"Newer" always means a later `imports.exported_at`, with the SHA-256 breaking
ties: the same order the importer uses (see `kinship.ingest.importer`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Literal

import sqlalchemy as sa

from kinship.db.models import (
    Company,
    Connection,
    ConnectionHistory,
    ConnectionNote,
    Endorsement,
    Import,
    Invitation,
    JobApplication,
    Message,
    MyPosition,
    Recommendation,
)
from kinship.repo.base import BaseRepository
from kinship.repo.imports import ImportInfo
from kinship.warmth import Signals

MAX_NAMES_PER_COMPANY = 3
"""How many connection names `applied_companies` returns for each company."""


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], ())}


@dataclass(frozen=True)
class NetworkPerson:
    """A connection as the network tools show it, detached from the session."""

    id: int
    name: str
    title: str | None
    company: str | None
    company_id: int | None
    profile_url: str | None
    connected_on: date | None

    def as_dict(self) -> dict[str, Any]:
        """The compact, JSON-ready record: no ids, empty fields left out."""
        return _compact(
            {
                "name": self.name,
                "title": self.title,
                "company": self.company,
                "profile_url": self.profile_url,
                "connected_on": self.connected_on.isoformat() if self.connected_on else None,
            }
        )


@dataclass(frozen=True)
class JobChange:
    """One step in a connection's role timeline, first shown by the export of `as_of`.

    `company_changed` is about the employer, not the spelling: it compares the
    normalized company (`company_id`), falling back to the raw text when either
    side has none. A change that is neither a company nor a title change is a
    new spelling of the same company (`is_spelling_only`).
    """

    person: NetworkPerson
    from_company: str | None
    from_title: str | None
    to_company: str | None
    to_title: str | None
    as_of: datetime | None
    company_changed: bool
    title_changed: bool

    @property
    def is_spelling_only(self) -> bool:
        return not (self.company_changed or self.title_changed)


@dataclass(frozen=True)
class ConnectionEvent:
    """A connection that appeared in, or disappeared by, the export of `at`."""

    person: NetworkPerson
    at: datetime | None


@dataclass(frozen=True)
class AppliedCompany:
    """A job the owner applied to at a company where they have current connections."""

    company: str
    job_title: str | None
    applied_at: datetime | None
    connection_count: int
    connection_names: tuple[str, ...]


def _full_name(first: str | None, last: str | None) -> str:
    return " ".join(part for part in (first, last) if part)


_PERSON_COLUMNS = (
    Connection.id,
    Connection.first_name,
    Connection.last_name,
    Connection.position,
    Connection.company_raw,
    Connection.company_id,
    Connection.profile_url,
    Connection.connected_on,
)
_BY_NAME = (Connection.last_name, Connection.first_name, Connection.id)


def _person(row: Any) -> NetworkPerson:
    connection_id, first, last, position, company_raw, company_id, url, connected_on = row[:8]
    return NetworkPerson(
        id=connection_id,
        name=_full_name(first, last),
        title=position,
        company=company_raw,
        company_id=company_id,
        profile_url=url,
        connected_on=connected_on,
    )


class NetworkRepository(BaseRepository):
    # -- imports ------------------------------------------------------------

    def imports_in_order(self) -> list[ImportInfo]:
        """All of the owner's imports, oldest export first (undated ones before the rest)."""
        rows = self.scalars(
            self.select(Import).order_by(
                Import.exported_at.asc().nulls_first(), Import.source_sha256, Import.id
            )
        )
        return [
            ImportInfo(
                id=row.id,
                source_sha256=row.source_sha256,
                exported_at=row.exported_at,
                imported_at=row.imported_at,
                file_counts=dict(row.file_counts or {}),
            )
            for row in rows
        ]

    # -- warm paths ---------------------------------------------------------

    def has_messages(self) -> bool:
        """Whether the owner has any messages at all (exports can omit them)."""
        return self.exists(Message)

    def connections_at_companies(self, company_ids: Collection[int]) -> list[NetworkPerson]:
        """Current (not removed) connections whose company is one of `company_ids`."""
        if not company_ids:
            return []
        stmt = (
            self.select(Connection, *_PERSON_COLUMNS)
            .where(Connection.removed_at.is_(None), Connection.company_id.in_(company_ids))
            .order_by(*_BY_NAME)
        )
        return [_person(row) for row in self.rows(stmt)]

    def former_connections_of_companies(self, company_ids: Collection[int]) -> list[NetworkPerson]:
        """Current connections who used to be at one of `company_ids` and are now elsewhere.

        "Used to be" means their `connection_history` has a row at the company;
        "elsewhere" that their current company is a different one (or blank).
        """
        if not company_ids:
            return []
        was_there = self.select(ConnectionHistory, ConnectionHistory.connection_id).where(
            ConnectionHistory.company_id.in_(company_ids)
        )
        stmt = (
            self.select(Connection, *_PERSON_COLUMNS)
            .where(
                Connection.removed_at.is_(None),
                Connection.id.in_(was_there),
                sa.or_(Connection.company_id.is_(None), Connection.company_id.not_in(company_ids)),
            )
            .order_by(*_BY_NAME)
        )
        return [_person(row) for row in self.rows(stmt)]

    def warmth_signals(
        self,
        connection_ids: Collection[int],
        *,
        exclude_company_ids: Collection[int] = (),
    ) -> dict[int, Signals]:
        """The `Signals` for each of the owner's connections in `connection_ids`.

        Ids that are missing or belong to another owner are left out. Companies
        in `exclude_company_ids` (the company being asked about) do not count
        as shared employers. Runs seven statements regardless of how many ids.
        """
        ids = sorted(set(connection_ids))
        if not ids:
            return {}
        signals = {
            connection_id: Signals(connected_on=connected_on)
            for connection_id, connected_on in self.rows(
                self.select(Connection, Connection.id, Connection.connected_on).where(
                    Connection.id.in_(ids)
                )
            )
        }

        def update(connection_id: int, **changes: Any) -> None:
            if connection_id in signals:
                signals[connection_id] = replace(signals[connection_id], **changes)

        for connection_id, count, last_at in self.rows(
            self.select(
                Message, Message.connection_id, sa.func.count(), sa.func.max(Message.sent_at)
            )
            .where(Message.connection_id.in_(ids))
            .group_by(Message.connection_id)
        ):
            update(connection_id, message_count=count, last_message_at=last_at)

        recommendations: dict[tuple[int, str], list[date | None]] = defaultdict(list)
        for connection_id, direction, created_at in self.rows(
            self.select(
                Recommendation,
                Recommendation.connection_id,
                Recommendation.direction,
                Recommendation.created_at,
            )
            .where(Recommendation.connection_id.in_(ids))
            .order_by(Recommendation.created_at, Recommendation.id)
        ):
            recommendations[connection_id, direction].append(
                created_at.date() if created_at else None
            )
        for (connection_id, direction), dates in recommendations.items():
            update(connection_id, **{f"recommendations_{direction}": tuple(dates)})

        endorsements: dict[tuple[int, str], list[str | None]] = defaultdict(list)
        for connection_id, direction, skill in self.rows(
            self.select(
                Endorsement,
                Endorsement.connection_id,
                Endorsement.direction,
                Endorsement.skill_name,
            )
            .where(Endorsement.connection_id.in_(ids))
            .order_by(Endorsement.skill_name, Endorsement.id)
        ):
            endorsements[connection_id, direction].append(skill)
        for (connection_id, direction), skills in endorsements.items():
            update(connection_id, **{f"endorsements_{direction}": tuple(skills)})

        for connection_id, count in self.rows(
            self.select(ConnectionNote, ConnectionNote.connection_id, sa.func.count())
            .where(ConnectionNote.connection_id.in_(ids))
            .group_by(ConnectionNote.connection_id)
        ):
            update(connection_id, note_count=count)

        # Oldest first, so the invitation that made the connection wins.
        invited: set[int] = set()
        for connection_id, direction, message in self.rows(
            self.select(
                Invitation, Invitation.connection_id, Invitation.direction, Invitation.message
            )
            .where(Invitation.connection_id.in_(ids))
            .order_by(Invitation.sent_at.asc().nulls_last(), Invitation.id)
        ):
            if connection_id not in invited:
                invited.add(connection_id)
                update(
                    connection_id,
                    invitation=direction,
                    invitation_had_message=bool(message and message.strip()),
                )

        mine = self.select(MyPosition, MyPosition.company_id).where(
            MyPosition.company_id.is_not(None)
        )
        shared_stmt = (
            self.select(ConnectionHistory, ConnectionHistory.connection_id, Company.display_name)
            .join(Company, Company.id == ConnectionHistory.company_id)
            .where(
                self.owned(Company),
                ConnectionHistory.connection_id.in_(ids),
                ConnectionHistory.company_id.in_(mine),
            )
            .distinct()
            .order_by(ConnectionHistory.connection_id, Company.display_name)
        )
        if exclude_company_ids:
            shared_stmt = shared_stmt.where(
                ConnectionHistory.company_id.not_in(exclude_company_ids)
            )
        shared: dict[int, list[str]] = defaultdict(list)
        for connection_id, company_name in self.rows(shared_stmt):
            shared[connection_id].append(company_name)
        for connection_id, names in shared.items():
            update(connection_id, shared_employers=tuple(names))

        return signals

    # -- changes ------------------------------------------------------------

    def job_changes(self, since: datetime) -> list[JobChange]:
        """Role changes first shown by an export dated after `since`, newest first.

        A change is a `connection_history` row that is not the connection's
        first: the previous row is the state it changed from. Spelling-only
        changes are included; see `JobChange.is_spelling_only`.
        """
        timeline = {
            "partition_by": ConnectionHistory.connection_id,
            "order_by": (
                Import.exported_at.asc().nulls_first(),
                Import.source_sha256,
                ConnectionHistory.id,
            ),
        }
        history = (
            self.select(
                ConnectionHistory,
                ConnectionHistory.connection_id.label("connection_id"),
                ConnectionHistory.company_raw.label("to_company"),
                ConnectionHistory.company_id.label("to_company_id"),
                ConnectionHistory.position.label("to_title"),
                Import.exported_at.label("as_of"),
                sa.func.lag(ConnectionHistory.company_raw).over(**timeline).label("from_company"),
                sa.func.lag(ConnectionHistory.company_id).over(**timeline).label("from_company_id"),
                sa.func.lag(ConnectionHistory.position).over(**timeline).label("from_title"),
                sa.func.row_number().over(**timeline).label("step"),
            )
            .join(Import, Import.id == ConnectionHistory.import_id)
            .where(self.owned(Import))
            .subquery()
        )
        stmt = (
            self.select(
                Connection,
                *_PERSON_COLUMNS,
                history.c.from_company,
                history.c.from_company_id,
                history.c.from_title,
                history.c.to_company,
                history.c.to_company_id,
                history.c.to_title,
                history.c.as_of,
            )
            .join(history, history.c.connection_id == Connection.id)
            .where(history.c.step > 1, history.c.as_of > since)
            .order_by(history.c.as_of.desc(), *_BY_NAME)
        )
        changes = []
        for row in self.rows(stmt):
            from_company, from_id, from_title, to_company, to_id, to_title, as_of = row[8:]
            if from_id is not None and to_id is not None:
                company_changed = from_id != to_id
            else:
                company_changed = (from_company or "") != (to_company or "")
            changes.append(
                JobChange(
                    person=_person(row),
                    from_company=from_company,
                    from_title=from_title,
                    to_company=to_company,
                    to_title=to_title,
                    as_of=as_of,
                    company_changed=company_changed,
                    title_changed=(from_title or "") != (to_title or ""),
                )
            )
        return changes

    def _connection_events(
        self, which: Literal["new", "removed"], since: datetime
    ) -> list[ConnectionEvent]:
        if which == "removed":
            at = Connection.removed_at
            stmt = self.select(Connection, *_PERSON_COLUMNS, at).where(at > since)
        else:
            at = Import.exported_at
            # The owner's earliest import is the baseline: nothing in it is "new".
            baseline = (
                self.select(Import, Import.id)
                .order_by(Import.exported_at.asc().nulls_first(), Import.source_sha256, Import.id)
                .limit(1)
                .scalar_subquery()
            )
            stmt = (
                self.select(Connection, *_PERSON_COLUMNS, at)
                .join(Import, Import.id == Connection.first_seen_import_id)
                .where(self.owned(Import), at > since, Import.id != baseline)
            )
        stmt = stmt.order_by(at.desc(), *_BY_NAME)
        return [ConnectionEvent(person=_person(row), at=row[8]) for row in self.rows(stmt)]

    def new_connections(self, since: datetime) -> list[ConnectionEvent]:
        """Connections first seen in an export dated after `since`, newest first.

        Connections of the owner's earliest import are never "new": there is
        nothing earlier to compare with. `at` is the date of the first export
        that contains the connection.
        """
        return self._connection_events("new", since)

    def removed_connections(self, since: datetime) -> list[ConnectionEvent]:
        """Connections whose `removed_at` is after `since`, newest first.

        `at` is the date of the first export known not to contain them.
        """
        return self._connection_events("removed", since)

    def applied_companies(self, since: datetime | None = None) -> list[AppliedCompany]:
        """Job applications at companies where the owner has current connections.

        Newest application first; only those applied to after `since` when it
        is given. Each carries the number of current connections at the company
        and the names of up to `MAX_NAMES_PER_COMPANY` of them (by name).
        """
        ranked = (
            self.select(
                Connection,
                Connection.company_id.label("company_id"),
                Connection.first_name.label("first_name"),
                Connection.last_name.label("last_name"),
                sa.func.row_number()
                .over(partition_by=Connection.company_id, order_by=_BY_NAME)
                .label("place"),
                sa.func.count().over(partition_by=Connection.company_id).label("total"),
            )
            .where(Connection.removed_at.is_(None), Connection.company_id.is_not(None))
            .subquery()
        )
        stmt = (
            self.select(
                JobApplication,
                JobApplication.id,
                JobApplication.company_name,
                JobApplication.job_title,
                JobApplication.applied_at,
                ranked.c.total,
                ranked.c.first_name,
                ranked.c.last_name,
            )
            .join(ranked, ranked.c.company_id == JobApplication.company_id)
            .where(ranked.c.place <= MAX_NAMES_PER_COMPANY)
            .order_by(
                JobApplication.applied_at.desc().nulls_last(), JobApplication.id, ranked.c.place
            )
        )
        if since is not None:
            stmt = stmt.where(JobApplication.applied_at > since)

        found: dict[int, AppliedCompany] = {}
        for application_id, company, job_title, applied_at, total, first, last in self.rows(stmt):
            name = _full_name(first, last)
            current = found.get(application_id)
            if current is None:
                found[application_id] = AppliedCompany(
                    company=company or "",
                    job_title=job_title,
                    applied_at=applied_at,
                    connection_count=total,
                    connection_names=(name,),
                )
            else:
                found[application_id] = replace(
                    current, connection_names=(*current.connection_names, name)
                )
        return list(found.values())
