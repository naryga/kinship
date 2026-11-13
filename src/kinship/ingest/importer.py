"""Idempotent, order-tolerant import of a parsed LinkedIn export.

The invariant: **the database ends up the same whatever order a set of exports
is imported in**, and importing the same export twice changes nothing.

"Newer" always means a later `imports.exported_at` (ties broken by SHA-256),
never "imported later".

Connections
    Identity is the canonical `profile_url`; a row without a URL is matched on
    (first_name, last_name, connected_on) among the owner's URL-less rows.

    Every import writes one `connection_sightings` row per connection it
    contains, holding that import's (company_raw, position). Everything else
    about a connection is derived from its sightings, in chronological order:

    - name, email, company_raw, position: from the newest import containing it;
    - `first_seen_import_id` / `last_seen_import_id`: the chronologically
      earliest / latest imports containing it;
    - `removed_at`: NULL if the newest import overall contains it, otherwise
      the `exported_at` of the first import after its last sighting (the first
      export known not to contain it). Rows are never deleted. An old export
      imported late inserts the connections only it has, already removed.
    - `connection_history`: one row for the first import containing the
      connection, plus one row for every later import whose (company_raw,
      position) differs from the previous import that contained it. Each row
      holds the state the connection changed *to*, and `import_id` is the
      import that first showed it. Ordered by the import's `exported_at`, the
      rows are the connection's full role timeline: the state as of any export
      is the latest row at or before it, and every row after the first is a
      job change dated by its import. A spelling-only change of the company
      name counts as a change (company normalization is a later step).

Owner snapshot tables (`my_positions`, `profiles`, `skills`, `education`)
    Mirror the newest import that has any rows for that table. Rows are matched
    on a natural key so ids stay stable; rows absent from that export are
    deleted. `Profile Summary.csv` fills `profiles.summary` when Profile.csv
    has none.

Event tables (endorsements, recommendations, invitations, notes, follows, job
applications, saved jobs, messages)
    The union of all exports, deduplicated on a natural key (see the `_KEY`
    functions and `_message_key`); never deleted. Fields outside the key (a
    status, an edited note) are overwritten only by the newest import overall.

Links
    `connection_id` is recomputed for the owner's event rows on every import,
    so rows imported before their connection existed get linked later: by
    canonical profile URL where the row has one, otherwise by a unique full
    name among the owner's connections. `company_id` is left NULL.

Known limits, all outside the tested invariant: a mutable event field is only
order-independent while the newest import still contains the row; message
participant URLs are not stored, so a message can only be re-linked by URL
while an export containing it is being imported (afterwards by name only).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kinship.db.models import (
    INVITATION_DIRECTIONS,
    CompanyFollow,
    Connection,
    ConnectionHistory,
    ConnectionNote,
    ConnectionSighting,
    Education,
    Endorsement,
    Import,
    Invitation,
    JobApplication,
    Message,
    MyPosition,
    Owner,
    Profile,
    Recommendation,
    SavedJob,
    Skill,
)
from kinship.ingest.store import ImportStore
from kinship.parser import ExportError, ParsedExport, parse_export
from kinship.parser import records as parsed_records

_EPOCH = datetime.min.replace(tzinfo=UTC)

State = tuple[str | None, str | None]
"""A connection's (company_raw, position) in one import."""


class EmptyExportError(ExportError):
    """The export has no connections; importing it would mark every connection removed."""


class OwnerNotFoundError(LookupError):
    """No owner has the given id."""


@dataclass
class ImportSummary:
    """What one import did. Connection counts describe the change to current data."""

    owner_id: int
    import_id: int
    source_sha256: str
    exported_at: datetime | None
    already_imported: bool = False
    is_newest: bool = True
    owner_created: bool = False
    new: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    restored: int = 0
    inserted: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def format(self) -> str:
        """A short human-readable report for the CLI."""
        exported = self.exported_at.date().isoformat() if self.exported_at else "unknown date"
        owner = f"owner {self.owner_id}" + (" (created)" if self.owner_created else "")
        lines: list[str] = []
        if self.already_imported:
            lines.append(
                f"Export {self.source_sha256[:12]} ({exported}) was already imported for "
                f"{owner} as import #{self.import_id}; nothing to do."
            )
        else:
            lines.append(
                f"Imported export {self.source_sha256[:12]} ({exported}) for {owner} "
                f"as import #{self.import_id}."
            )
            lines.append(
                f"Connections: {self.new} new, {self.changed} changed, {self.removed} removed, "
                f"{self.restored} restored, {self.unchanged} unchanged."
            )
            added = ", ".join(f"{table} {count}" for table, count in self.inserted.items() if count)
            lines.append(f"Rows added: {added or 'none'}.")
            if not self.is_newest:
                lines.append(
                    "This export is older than the newest import: it filled in history, "
                    "and current data was kept."
                )
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)


# -- chronology ---------------------------------------------------------------


def _chronological_key(record: Import) -> tuple[datetime, str]:
    return (record.exported_at or _EPOCH, record.source_sha256)


class _Timeline:
    """The owner's imports in export order."""

    def __init__(self, imports: Iterable[Import]) -> None:
        self.imports = sorted(imports, key=_chronological_key)
        self.rank = {record.id: position for position, record in enumerate(self.imports)}

    def is_newest(self, import_id: int) -> bool:
        return self.imports[-1].id == import_id

    def removed_at(self, last_seen_import_id: int) -> datetime | None:
        """When a connection last seen in that import was first known to be gone."""
        following = self.rank[last_seen_import_id] + 1
        return self.imports[following].exported_at if following < len(self.imports) else None

    def is_newest_with(self, import_id: int, *file_keys: str) -> bool:
        """Is this the newest import that has rows for any of `file_keys`?"""
        for record in reversed(self.imports):
            if any((record.file_counts or {}).get(key, 0) for key in file_keys):
                return record.id == import_id
        return False


# -- names and links ------------------------------------------------------------


def _full_name(first: str | None, last: str | None) -> str | None:
    name = " ".join(part for part in (first, last) if part)
    return name or None


def _name_key(name: str | None) -> str | None:
    key = " ".join((name or "").split()).casefold()
    return key or None


class _ConnectionIndex:
    """Finds one owner's connection ids by profile URL or by unique full name."""

    def __init__(self, connections: Iterable[Connection]) -> None:
        self._by_url: dict[str, int] = {}
        self._by_name: dict[str, int | None] = {}
        for row in connections:
            if row.profile_url:
                self._by_url[row.profile_url] = row.id
            key = _name_key(_full_name(row.first_name, row.last_name))
            if key:
                # None marks a name shared by several connections: never guess.
                self._by_name[key] = None if key in self._by_name else row.id

    def by_url(self, url: str | None) -> int | None:
        return self._by_url.get(url) if url else None

    def by_name(self, name: str | None) -> int | None:
        key = _name_key(name)
        return self._by_name.get(key) if key else None

    def by_names(self, names: Iterable[str | None]) -> int | None:
        """The single connection among `names`, or None if there are none or several."""
        found = {self.by_name(name) for name in names} - {None}
        return found.pop() if len(found) == 1 else None

    def person(self, url: str | None, name: str | None) -> int | None:
        """By URL when the row has one, else by unique name."""
        return self.by_url(url) if url else self.by_name(name)

    def message(self, urls: Iterable[str | None], from_name: str | None, to_names: list[str]):
        """The other participant: by URL if any participant is a connection, else by name."""
        by_url = {self.by_url(url) for url in urls} - {None}
        if by_url:
            return by_url.pop() if len(by_url) == 1 else None
        return self.by_name(from_name) or self.by_names(to_names)


# -- connections ----------------------------------------------------------------


def _connection_key(row: Any) -> tuple[Any, ...]:
    """Identity of a `Connection` row or a parsed connection record."""
    if row.profile_url:
        return ("url", row.profile_url)
    return ("name", row.first_name, row.last_name, row.connected_on)


def _import_connections(
    store: ImportStore,
    parsed: ParsedExport,
    record: Import,
    timeline: _Timeline,
    summary: ImportSummary,
) -> list[Connection]:
    existing = store.rows(Connection)
    by_key = {_connection_key(row): row for row in existing}
    before = {row.id: (row.company_raw, row.position, row.removed_at) for row in existing}

    incoming: dict[tuple[Any, ...], parsed_records.Connection] = {}
    for item in parsed.connections:
        incoming.setdefault(_connection_key(item), item)
    if len(incoming) < len(parsed.connections):
        summary.warnings.append(
            f"Connections.csv: {len(parsed.connections) - len(incoming)} duplicate row(s) ignored"
        )

    seen: list[tuple[Connection, parsed_records.Connection]] = []
    created: list[Connection] = []
    for key, item in incoming.items():
        row = by_key.get(key)
        if row is None:
            row = store.add(
                Connection(
                    profile_url=item.profile_url,
                    first_name=item.first_name,
                    last_name=item.last_name,
                    email=item.email,
                    company_raw=item.company,
                    position=item.position,
                    connected_on=item.connected_on,
                )
            )
            created.append(row)
        seen.append((row, item))
    store.flush()  # assigns ids to the new connections

    store.insert_many(
        ConnectionSighting,
        (
            {
                "connection_id": row.id,
                "import_id": record.id,
                "company_raw": item.company,
                "position": item.position,
            }
            for row, item in seen
        ),
    )
    sightings: dict[int, dict[int, State]] = defaultdict(dict)
    for sighting in store.sightings():
        sightings[sighting.connection_id][sighting.import_id] = (
            sighting.company_raw,
            sighting.position,
        )

    # Re-derive every connection from its sightings, in export order.
    in_this_export = {row.id: item for row, item in seen}
    history: dict[tuple[int, int], State] = {}
    connections = existing + created
    for row in connections:
        observed = sorted(sightings[row.id].items(), key=lambda pair: timeline.rank[pair[0]])
        if not observed:  # not written by this importer; leave it alone
            continue
        row.first_seen_import_id = observed[0][0]
        row.last_seen_import_id = observed[-1][0]
        row.removed_at = timeline.removed_at(row.last_seen_import_id)
        if row.last_seen_import_id == record.id:
            item = in_this_export[row.id]
            row.first_name = item.first_name
            row.last_name = item.last_name
            row.email = item.email
            row.company_raw = item.company
            row.position = item.position
            row.connected_on = item.connected_on
        previous: State | None = None
        for import_id, state in observed:
            if state != previous:
                history[(row.id, import_id)] = state
            previous = state

    summary.new = len(created)
    for row in existing:
        company, position, removed_at = before[row.id]
        summary.removed += removed_at is None and row.removed_at is not None
        summary.restored += removed_at is not None and row.removed_at is None
        if row.id in in_this_export:
            if (company, position) != (row.company_raw, row.position):
                summary.changed += 1
            else:
                summary.unchanged += 1

    summary.inserted["connections"] = len(created)
    summary.inserted["connection_history"] = _sync_history(store, history)
    return connections


def _sync_history(store: ImportStore, wanted: dict[tuple[int, int], State]) -> int:
    """Make `connection_history` equal `wanted`, touching only rows that differ.

    Returns the net number of rows added (an out-of-order import moves rows).
    """
    stale: list[int] = []
    current: set[tuple[int, int]] = set()
    for row in store.history():
        key = (row.connection_id, row.import_id)
        if wanted.get(key, _ABSENT) == (row.company_raw, row.position) and key not in current:
            current.add(key)
        else:
            stale.append(row.id)
    store.delete_ids(ConnectionHistory, stale)
    return -len(stale) + store.insert_many(
        ConnectionHistory,
        (
            {
                "connection_id": connection_id,
                "import_id": import_id,
                "company_raw": state[0],
                "position": state[1],
            }
            for (connection_id, import_id), state in wanted.items()
            if (connection_id, import_id) not in current
        ),
    )


_ABSENT: Any = object()


# -- owner snapshot tables --------------------------------------------------------


def _sync_snapshot(
    store: ImportStore, model: type[Any], incoming: list[dict[str, Any]], key: tuple[str, ...]
) -> int:
    """Make the owner's `model` rows equal `incoming`, keeping ids of rows that persist."""
    existing: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in store.rows(model):
        existing[tuple(getattr(row, name) for name in key)].append(row)
    inserted = 0
    for fields in incoming:
        bucket = existing.get(tuple(fields[name] for name in key))
        if bucket:
            row = bucket.pop(0)
            for name, value in fields.items():
                setattr(row, name, value)
        else:
            store.add(model(**fields))
            inserted += 1
    store.delete_ids(model, [row.id for bucket in existing.values() for row in bucket])
    return inserted


def _import_snapshots(
    store: ImportStore,
    parsed: ParsedExport,
    record: Import,
    timeline: _Timeline,
    summary: ImportSummary,
) -> None:
    if timeline.is_newest_with(record.id, "positions"):
        summary.inserted["my_positions"] = _sync_snapshot(
            store,
            MyPosition,
            [
                {
                    "company": item.company_name,
                    "title": item.title,
                    "started_on": item.started_on,
                    "ended_on": item.finished_on,
                    "description": item.description,
                    "location": item.location,
                }
                for item in parsed.positions
            ],
            ("company", "title", "started_on"),
        )
    if timeline.is_newest_with(record.id, "education"):
        summary.inserted["education"] = _sync_snapshot(
            store,
            Education,
            [
                {
                    "school_name": item.school_name,
                    "degree_name": item.degree_name,
                    "started_on": item.started_on,
                    "ended_on": item.ended_on,
                    "notes": item.notes,
                    "activities": item.activities,
                }
                for item in parsed.education
            ],
            ("school_name", "degree_name", "started_on"),
        )
    if timeline.is_newest_with(record.id, "skills"):
        names = dict.fromkeys(item.name for item in parsed.skills if item.name)
        summary.inserted["skills"] = _sync_snapshot(
            store, Skill, [{"name": name} for name in names], ("name",)
        )
    if timeline.is_newest_with(record.id, "profile", "profile_summary"):
        profile = parsed.profile
        summary_text = (profile.summary if profile else None) or next(
            (item.text for item in parsed.profile_summary if item.text), None
        )
        summary.inserted["profiles"] = _sync_snapshot(
            store,
            Profile,
            [
                {
                    "headline": profile.headline if profile else None,
                    "summary": summary_text,
                    "industry": profile.industry if profile else None,
                }
            ],
            (),
        )


# -- event tables -------------------------------------------------------------------


def _person_key(row: Any) -> tuple[Any, ...]:
    if row.profile_url:
        return ("url", row.profile_url)
    return ("name", _name_key(_full_name(row.first_name, row.last_name)))


def _endorsement_key(row: Endorsement) -> tuple[Any, ...]:
    return (row.direction, _person_key(row), row.skill_name, row.endorsed_at)


def _recommendation_key(row: Recommendation) -> tuple[Any, ...]:
    return (row.direction, row.first_name, row.last_name, row.created_at)


def _invitation_other_name(row: Invitation) -> str | None:
    return row.from_name if row.direction == "incoming" else row.to_name


def _invitation_key(row: Invitation) -> tuple[Any, ...]:
    other = (
        ("url", row.profile_url)
        if row.profile_url
        else ("name", _name_key(_invitation_other_name(row)))
    )
    return (row.direction, other, row.sent_at)


def _note_key(row: ConnectionNote) -> tuple[Any, ...]:
    return (_person_key(row), row.created_at)


def _company_follow_key(row: CompanyFollow) -> tuple[Any, ...]:
    return (row.organization, row.followed_at)


def _job_key(row: JobApplication | SavedJob) -> tuple[Any, ...]:
    return ("url", row.job_url) if row.job_url else ("job", row.company_name, row.job_title)


def _job_application_key(row: JobApplication) -> tuple[Any, ...]:
    return (_job_key(row), row.applied_at)


def _saved_job_key(row: SavedJob) -> tuple[Any, ...]:
    return (_job_key(row), row.saved_at)


def _merge_events[T](
    store: ImportStore,
    model: type[T],
    candidates: Iterable[T],
    key: Callable[[T], tuple[Any, ...]],
    mutable: tuple[str, ...],
    overwrite: bool,
) -> tuple[int, list[T]]:
    """Add the candidates not stored yet; return (inserted count, all rows).

    A candidate whose key is already stored only refreshes the `mutable`
    columns, and only when `overwrite` is set (the newest import overall).
    """
    rows = {key(row): row for row in store.rows(model)}  # type: ignore[type-var]
    inserted = 0
    for candidate in candidates:
        identity = key(candidate)
        row = rows.get(identity)
        if row is None:
            rows[identity] = store.add(candidate)  # type: ignore[type-var]
            inserted += 1
        elif overwrite:
            for name in mutable:
                setattr(row, name, getattr(candidate, name))
    return inserted, list(rows.values())


def _invitations(parsed: ParsedExport, warnings: list[str]) -> Iterable[Invitation]:
    skipped = 0
    for item in parsed.invitations:
        direction = (item.direction or "").strip().lower()
        if direction not in INVITATION_DIRECTIONS:
            skipped += 1
            continue
        yield Invitation(
            direction=direction,
            from_name=item.from_name,
            to_name=item.to_name,
            sent_at=item.sent_at,
            message=item.message,
            inviter_profile_url=item.inviter_profile_url,
            invitee_profile_url=item.invitee_profile_url,
            profile_url=(
                item.inviter_profile_url if direction == "incoming" else item.invitee_profile_url
            ),
        )
    if skipped:
        warnings.append(f"Invitations.csv: {skipped} row(s) with an unknown Direction skipped")


def _endorsements(parsed: ParsedExport) -> Iterable[Endorsement]:
    for item in parsed.endorsements_received:
        yield Endorsement(
            direction="received",
            skill_name=item.skill_name,
            first_name=item.endorser_first_name,
            last_name=item.endorser_last_name,
            profile_url=item.endorser_profile_url,
            endorsed_at=item.endorsed_at,
            status=item.status,
        )
    for given in parsed.endorsements_given:
        yield Endorsement(
            direction="given",
            skill_name=given.skill_name,
            first_name=given.endorsee_first_name,
            last_name=given.endorsee_last_name,
            profile_url=given.endorsee_profile_url,
            endorsed_at=given.endorsed_at,
            status=given.status,
        )


def _recommendations(parsed: ParsedExport) -> Iterable[Recommendation]:
    for direction, items in (
        ("received", parsed.recommendations_received),
        ("given", parsed.recommendations_given),
    ):
        for item in items:
            yield Recommendation(
                direction=direction,
                first_name=item.first_name,
                last_name=item.last_name,
                company=item.company,
                job_title=item.job_title,
                text=item.text,
                created_at=item.created_at,
                status=item.status,
            )


def _import_events(
    store: ImportStore,
    parsed: ParsedExport,
    overwrite: bool,
    index: _ConnectionIndex,
    summary: ImportSummary,
) -> None:
    inserted = summary.inserted

    inserted["endorsements"], endorsements = _merge_events(
        store,
        Endorsement,
        _endorsements(parsed),
        _endorsement_key,
        ("status", "first_name", "last_name"),
        overwrite,
    )
    for row in endorsements:
        row.connection_id = index.person(row.profile_url, _full_name(row.first_name, row.last_name))

    inserted["recommendations"], recommendations = _merge_events(
        store,
        Recommendation,
        _recommendations(parsed),
        _recommendation_key,
        ("company", "job_title", "text", "status"),
        overwrite,
    )
    for row in recommendations:
        row.connection_id = index.by_name(_full_name(row.first_name, row.last_name))

    inserted["invitations"], invitations = _merge_events(
        store,
        Invitation,
        _invitations(parsed, summary.warnings),
        _invitation_key,
        ("message", "from_name", "to_name", "inviter_profile_url", "invitee_profile_url"),
        overwrite,
    )
    for row in invitations:
        row.connection_id = index.person(row.profile_url, _invitation_other_name(row))

    inserted["connection_notes"], notes = _merge_events(
        store,
        ConnectionNote,
        (
            ConnectionNote(
                first_name=item.first_name,
                last_name=item.last_name,
                profile_url=item.profile_url,
                note=item.note,
                created_at=item.created_at,
                edited_at=item.edited_at,
            )
            for item in parsed.notes
        ),
        _note_key,
        ("note", "edited_at", "first_name", "last_name"),
        overwrite,
    )
    for row in notes:
        row.connection_id = index.person(row.profile_url, _full_name(row.first_name, row.last_name))

    inserted["company_follows"], _ = _merge_events(
        store,
        CompanyFollow,
        (
            CompanyFollow(organization=item.organization, followed_at=item.followed_at)
            for item in parsed.company_follows
            if item.organization
        ),
        _company_follow_key,
        (),
        overwrite,
    )
    inserted["job_applications"], _ = _merge_events(
        store,
        JobApplication,
        (
            JobApplication(
                applied_at=item.applied_at,
                company_name=item.company_name,
                job_title=item.job_title,
                job_url=item.job_url,
                resume_name=item.resume_name,
                question_and_answers=item.question_and_answers,
            )
            for item in parsed.job_applications
        ),
        _job_application_key,
        ("company_name", "job_title", "resume_name", "question_and_answers"),
        overwrite,
    )
    inserted["saved_jobs"], _ = _merge_events(
        store,
        SavedJob,
        (
            SavedJob(
                saved_at=item.saved_at,
                job_url=item.job_url,
                job_title=item.job_title,
                company_name=item.company_name,
            )
            for item in parsed.saved_jobs
        ),
        _saved_job_key,
        ("company_name", "job_title"),
        overwrite,
    )
    inserted["messages"] = _import_messages(store, parsed, index)


# -- messages -------------------------------------------------------------------------


def _body_sha256(body: str | None) -> str:
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def _message_key(
    conversation_id: str | None, sent_at: datetime | None, from_name: str | None, body_hash: str
) -> tuple[Any, ...]:
    return (conversation_id, sent_at, from_name, body_hash)


def _to_names(item: parsed_records.Message) -> list[str]:
    """`TO` as a list. It is only split on commas when there are several recipient URLs."""
    if not item.to_names:
        return []
    if len(item.recipient_profile_urls) > 1:
        return [name.strip() for name in item.to_names.split(",") if name.strip()]
    return [item.to_names]


def _import_messages(store: ImportStore, parsed: ParsedExport, index: _ConnectionIndex) -> int:
    stored = {
        _message_key(row.conversation_id, row.sent_at, row.from_name, row.body_sha256): row
        for row in store.messages()
    }
    links: dict[int, int | None] = {}
    in_this_export: set[tuple[Any, ...]] = set()
    new: list[dict[str, Any]] = []
    for item in parsed.messages:
        key = _message_key(
            item.conversation_id, item.sent_at, item.from_name, _body_sha256(item.content)
        )
        if key in in_this_export:
            continue
        in_this_export.add(key)
        to_names = _to_names(item)
        connection_id = index.message(
            (item.sender_profile_url, *item.recipient_profile_urls), item.from_name, to_names
        )
        row = stored.get(key)
        if row is None:
            new.append(
                {
                    "conversation_id": item.conversation_id,
                    "from_name": item.from_name,
                    "to_names": to_names,
                    "sent_at": item.sent_at,
                    "subject": item.subject,
                    "body": item.content,
                    "connection_id": connection_id,
                }
            )
        elif row.connection_id != connection_id:
            links[row.id] = connection_id
    # Messages this export does not contain: their participant URLs are gone, so
    # an unlinked one can only be matched by name now.
    for key, row in stored.items():
        if key not in in_this_export and row.connection_id is None:
            connection_id = index.by_name(row.from_name) or index.by_names(row.to_names or [])
            if connection_id is not None:
                links[row.id] = connection_id
    store.set_message_connections(links)
    return store.insert_many(Message, new)


# -- entry points ----------------------------------------------------------------------


def import_export(session: Session, owner_id: int, parsed: ParsedExport) -> ImportSummary:
    """Import `parsed` for `owner_id` inside the caller's transaction (no commit).

    Raises `OwnerNotFoundError` for an unknown owner and `EmptyExportError` when
    the export has no connections.
    """
    store = ImportStore(session, owner_id)
    if not store.lock_owner():
        raise OwnerNotFoundError(f"owner {owner_id} does not exist")
    if not parsed.connections:
        raise EmptyExportError(
            "the export has no connections (is Connections.csv empty or in an unknown "
            "format?); refusing to import it, because that would mark every connection removed"
        )

    duplicate = store.find_import(parsed.source_sha256)
    if duplicate is not None:
        return ImportSummary(
            owner_id=owner_id,
            import_id=duplicate.id,
            source_sha256=duplicate.source_sha256,
            exported_at=duplicate.exported_at,
            already_imported=True,
            is_newest=_Timeline(store.imports()).is_newest(duplicate.id),
            warnings=list(parsed.warnings),
        )

    record = store.add(
        Import(
            source_sha256=parsed.source_sha256,
            exported_at=parsed.exported_at,
            file_counts=dict(parsed.file_counts),
        )
    )
    store.flush()
    timeline = _Timeline(store.imports())
    summary = ImportSummary(
        owner_id=owner_id,
        import_id=record.id,
        source_sha256=record.source_sha256,
        exported_at=record.exported_at,
        is_newest=timeline.is_newest(record.id),
        warnings=list(parsed.warnings),
    )

    connections = _import_connections(store, parsed, record, timeline, summary)
    _import_snapshots(store, parsed, record, timeline, summary)
    _import_events(store, parsed, summary.is_newest, _ConnectionIndex(connections), summary)
    store.flush()
    return summary


def resolve_owner(session: Session, owner: int | str) -> tuple[int, bool]:
    """Return `(owner_id, created)` for an owner id or an exact display name.

    An id must exist. A display name is looked up exactly and created if absent.
    """
    if isinstance(owner, int):
        if session.scalar(select(Owner.id).where(Owner.id == owner)) is None:
            raise OwnerNotFoundError(f"owner {owner} does not exist")
        return owner, False
    name = owner.strip()
    if not name:
        raise OwnerNotFoundError("owner name is empty")
    found = session.scalar(
        select(Owner.id).where(Owner.display_name == name).order_by(Owner.id).limit(1)
    )
    if found is not None:
        return found, False
    created = Owner(display_name=name)
    session.add(created)
    session.flush()
    return created.id, True


def run_import(
    path: Path,
    owner: int | str,
    *,
    exported_at: datetime | None = None,
    session_factory: Callable[[], Session] | None = None,
) -> ImportSummary:
    """Parse the export at `path` and import it for `owner` in one transaction.

    `owner` is an owner id (must exist) or a display name (created if absent).
    Anything that fails rolls the whole import back, a newly created owner
    included.
    """
    from kinship.db.engine import session_scope

    parsed = parse_export(Path(path), exported_at=exported_at)
    with session_scope(session_factory) as session:  # type: ignore[arg-type]
        owner_id, created = resolve_owner(session, owner)
        summary = import_export(session, owner_id, parsed)
        summary.owner_created = created
    return summary
