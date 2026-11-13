"""Connection lookups: who is at a company, search, and one person in detail.

Every method returns JSON-ready plain data. A connection is always the same
small record, with empty fields left out:

    {"name": ..., "title": ..., "company": ..., "profile_url": ..., "connected_on": ...}

`company` is the display name of the connection's normalized company (the raw
spelling when there is none); dates are ISO 8601 (`YYYY-MM-DD`). Methods add
their own keys to that record (`status`, `removed`, ...). Row ids and
`owner_id` are never part of a result.

"Former" employees come from `connection_history` (see `kinship.ingest`): one
row per import that showed a connection in a new role, dated by the import's
`exported_at`. Someone is a former employee of a company when a history row
places them there and their current company is a different one.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import sqlalchemy as sa

from kinship.companies import company_key
from kinship.db.models import (
    Company,
    Connection,
    ConnectionHistory,
    ConnectionNote,
    Endorsement,
    Import,
    Invitation,
    Message,
    Recommendation,
)
from kinship.parser import canonical_profile_url
from kinship.repo.companies import CompaniesRepository, CompanyMatch

MAX_MATCHED_COMPANIES = 5
"""Without an exact match, at most this many of the best-matching companies are used."""

MAX_SUGGESTIONS = 5
MAX_SEARCH_LIMIT = 100
MAX_CANDIDATES = 10
"""How many people `lookup_connection` lists when a name is ambiguous."""

RECOMMENDATION_TEXT_LIMIT = 300

_PROFILE_URL_PREFIX = "https://www.linkedin.com/in/"

Record = dict[str, Any]


@dataclass(frozen=True)
class CompanyResolution:
    """What a typed company name resolved to.

    `matched` holds the companies to use: the exact match if there is one,
    otherwise the best prefix matches, otherwise the best "contains" matches.
    `suggestions` are display names of other companies the user may have meant.
    """

    query: str
    matched: tuple[CompanyMatch, ...] = ()
    suggestions: tuple[str, ...] = ()

    @property
    def company_ids(self) -> list[int]:
        return [match.id for match in self.matched]

    def matched_companies(self) -> list[Record]:
        """JSON-ready `matched`: display name, other spellings, and how it matched."""
        return [
            _compact(
                {
                    "display_name": match.display_name,
                    "aliases": [a for a in match.aliases if a != match.display_name],
                    "match": match.match,
                }
            )
            for match in self.matched
        ]


@dataclass(frozen=True)
class ConnectionSearch:
    """One page of `search_connections`: `total` matches, the first `limit` of them."""

    total: int
    limit: int
    connections: list[Record]
    company: CompanyResolution | None = None


@dataclass(frozen=True)
class ConnectionLookup:
    """Result of `lookup_connection`.

    - "found": `connection` is the full detail record.
    - "ambiguous": `candidates` are the first `MAX_CANDIDATES` people matching;
      `more_candidates` says whether there are others.
    - "not_found": `suggestions` are people with a similar name, possibly none.
    """

    status: Literal["found", "ambiguous", "not_found"]
    connection: Record | None = None
    candidates: tuple[Record, ...] = ()
    more_candidates: bool = False
    suggestions: tuple[Record, ...] = ()


def _compact(data: Record) -> Record:
    """`data` without None, empty strings and empty collections (False and 0 stay)."""
    return {key: value for key, value in data.items() if value not in (None, "", [], {}, ())}


def _day(value: datetime | date | None) -> str | None:
    """ISO date; timestamps are read in UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.astimezone(UTC).date() if value.tzinfo else value.date()
    return value.isoformat()


def _full_name(first: str | None, last: str | None) -> str:
    return " ".join(part.strip() for part in (first, last) if part and part.strip())


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _words(text: str | None) -> list[str]:
    return (text or "").split()


def _looks_like_url(text: str) -> bool:
    lowered = text.lower()
    return "://" in lowered or "linkedin.com" in lowered or lowered.startswith("www.")


class ConnectionsRepository(CompaniesRepository):
    # -- companies ------------------------------------------------------------

    def resolve_company(self, query: str) -> CompanyResolution:
        """Resolve a typed company name to the owner's companies.

        Uses `find_companies`, so case, spacing, punctuation and legal suffixes
        are ignored. An exact match wins on its own. Otherwise up to
        `MAX_MATCHED_COMPANIES` matches of the best kind are used (prefix
        matches before "contains" matches) and the remaining candidates become
        suggestions. When nothing matches at all, suggestions are companies
        that match one word of the query or are spelled similarly.
        """
        found = self.find_companies(query, limit=MAX_MATCHED_COMPANIES + MAX_SUGGESTIONS + 1)
        if not found:
            return CompanyResolution(query, suggestions=tuple(self._suggest_companies(query)))
        best = found[0].match
        of_best_kind = [match for match in found if match.match == best]
        matched = of_best_kind[:1] if best == "exact" else of_best_kind[:MAX_MATCHED_COMPANIES]
        others = [m.display_name for m in found if m not in matched][:MAX_SUGGESTIONS]
        return CompanyResolution(query, tuple(matched), tuple(others))

    def _suggest_companies(self, query: str) -> list[str]:
        names: list[str] = []
        for word in _words(query):
            if len(company_key(word) or "") < 3:
                continue
            names += [match.display_name for match in self.find_companies(word, MAX_SUGGESTIONS)]
        key = company_key(query)
        if key is not None:
            by_key = dict(
                self.rows(self.select(Company, Company.name_normalized, Company.display_name))
            )
            close = difflib.get_close_matches(key, list(by_key), n=MAX_SUGGESTIONS, cutoff=0.75)
            names += [by_key[match] for match in close]
        return list(dict.fromkeys(names))[:MAX_SUGGESTIONS]

    # -- the connection record --------------------------------------------------

    def _connections(self) -> sa.Select[Any]:
        """Connections with their company's display name; rows for `_record`."""
        return self.select(
            Connection,
            Connection.id,
            Connection.first_name,
            Connection.last_name,
            Connection.position,
            sa.func.coalesce(Company.display_name, Connection.company_raw),
            Connection.profile_url,
            Connection.connected_on,
            Connection.removed_at,
        ).outerjoin(Company, self._company_of(Connection))

    def _company_of(self, model: Any, company: Any = Company) -> sa.ColumnElement[bool]:
        """ON clause linking `model.company_id` to the owner's company."""
        return sa.and_(model.company_id == company.id, self.owned(company))

    @staticmethod
    def _record(row: Any) -> Record:
        _id, first, last, position, company, profile_url, connected_on, removed_at = row
        return _compact(
            {
                "name": _full_name(first, last),
                "title": position,
                "company": company,
                "profile_url": profile_url,
                "connected_on": _day(connected_on),
                "removed": True if removed_at is not None else None,
            }
        )

    _BY_NAME = (
        Connection.last_name.asc().nulls_last(),
        Connection.first_name.asc().nulls_last(),
        Connection.id,
    )

    # -- who is (or was) at a company ---------------------------------------------

    def connections_at_companies(
        self, company_ids: Sequence[int], include_former: bool = False
    ) -> list[Record]:
        """Connections at any of these companies, current ones first, each by name.

        Current: the connection's company is one of `company_ids` and it has
        not been removed. Each record has `"status": "current"`.

        With `include_former`, also people whose `connection_history` places
        them at one of the companies but whose company is now a different one
        (or blank). Those records have `"status": "former"`, their current
        `company` and `title`, plus `former_company`, `former_title` (their
        last known role there) and `left_by`: the export date of the first
        import that showed them elsewhere. A former employee who is no longer
        a connection is included with `"removed": true`.

        A removed connection whose last known company is one of `company_ids`
        is in neither group.
        """
        ids = list(company_ids)
        if not ids:
            return []
        current = self.rows(
            self._connections()
            .where(Connection.company_id.in_(ids), Connection.removed_at.is_(None))
            .order_by(*self._BY_NAME)
        )
        records = [{**self._record(row), "status": "current"} for row in current]
        if include_former:
            records += self._former(ids)
        return records

    def _former(self, company_ids: list[int]) -> list[Record]:
        elsewhere = sa.or_(
            Connection.company_id.is_(None), Connection.company_id.not_in(company_ids)
        )
        was_there = (
            self.select(ConnectionHistory, ConnectionHistory.connection_id)
            .where(ConnectionHistory.company_id.in_(company_ids))
            .distinct()
        )
        people = self.rows(
            self._connections()
            .where(elsewhere, Connection.id.in_(was_there))
            .order_by(*self._BY_NAME)
        )
        if not people:
            return []
        timelines = self._timelines([row[0] for row in people])
        records = []
        for row in people:
            timeline = timelines[row[0]]
            last_there = max(i for i, role in enumerate(timeline) if role[0] in company_ids)
            _, former_company, former_title, _ = timeline[last_there]
            left_by = timeline[last_there + 1][3] if last_there + 1 < len(timeline) else None
            records.append(
                _compact(
                    {
                        **self._record(row),
                        "status": "former",
                        "former_company": former_company,
                        "former_title": former_title,
                        "left_by": _day(left_by),
                    }
                )
            )
        return records

    def _timelines(
        self, connection_ids: Iterable[int]
    ) -> dict[int, list[tuple[int | None, str | None, str | None, datetime | None]]]:
        """Per connection, its history oldest first: (company_id, company, title, exported_at).

        Ordered the way the importer orders imports: `exported_at`, then SHA-256.
        """
        stmt = (
            self.select(
                ConnectionHistory,
                ConnectionHistory.connection_id,
                ConnectionHistory.company_id,
                sa.func.coalesce(Company.display_name, ConnectionHistory.company_raw),
                ConnectionHistory.position,
                Import.exported_at,
            )
            .join(Import, ConnectionHistory.import_id == Import.id)
            .where(self.owned(Import))
            .outerjoin(Company, self._company_of(ConnectionHistory))
            .where(ConnectionHistory.connection_id.in_(list(connection_ids)))
            .order_by(
                Import.exported_at.asc().nulls_first(),
                Import.source_sha256,
                ConnectionHistory.id,
            )
        )
        timelines: dict[int, list[Any]] = defaultdict(list)
        for connection_id, *role in self.rows(stmt):
            timelines[connection_id].append(tuple(role))
        return timelines

    # -- search -------------------------------------------------------------------

    def search_connections(
        self,
        query: str = "",
        title: str | None = None,
        company: str | None = None,
        limit: int = 20,
    ) -> ConnectionSearch:
        """Search current (not removed) connections; all text matching ignores case.

        - `query`: every whitespace-separated word must appear in the person's
          first or last name, their title, or their company (any spelling).
        - `title`: substring of the title.
        - `company`: resolved with `resolve_company`; the connection must be at
          one of the matched companies. If nothing resolves, falls back to a
          substring of the raw company name.

        People whose *name* contains every query word come first; then by last
        name, first name. `limit` is clamped to 1..`MAX_SEARCH_LIMIT`; `total`
        counts all matches.
        """
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        criteria: list[sa.ColumnElement[bool]] = [Connection.removed_at.is_(None)]
        in_name: list[sa.ColumnElement[bool]] = []
        for word in _words(query):
            name = sa.or_(
                self._has(Connection.first_name, word), self._has(Connection.last_name, word)
            )
            elsewhere = [
                self._has(Connection.position, word),
                self._has(Connection.company_raw, word),
                self._has(Company.display_name, word),
            ]
            key = company_key(word)
            if key:
                elsewhere.append(Company.name_normalized.contains(key, autoescape=True))
            in_name.append(name)
            criteria.append(sa.or_(name, *elsewhere))
        if title and title.strip():
            criteria.append(self._has(Connection.position, title.strip()))
        resolution = None
        if company and company.strip():
            resolution = self.resolve_company(company)
            if resolution.matched:
                criteria.append(Connection.company_id.in_(resolution.company_ids))
            else:
                criteria.append(self._has(Connection.company_raw, company.strip()))

        total = self.scalar(
            self.select(Connection, sa.func.count())
            .outerjoin(Company, self._company_of(Connection))
            .where(*criteria)
        )
        name_first = sa.case((sa.and_(*in_name), 0), else_=1) if in_name else sa.literal(0)
        rows = self.rows(
            self._connections().where(*criteria).order_by(name_first, *self._BY_NAME).limit(limit)
        )
        return ConnectionSearch(
            total=total or 0,
            limit=limit,
            connections=[self._record(row) for row in rows],
            company=resolution,
        )

    @staticmethod
    def _has(column: Any, text: str) -> sa.ColumnElement[bool]:
        """Case-insensitive substring match with LIKE wildcards in `text` escaped."""
        return column.icontains(text, autoescape=True)

    # -- one connection -------------------------------------------------------------

    def lookup_connection(self, name_or_url: str) -> ConnectionLookup:
        """Find one connection by profile URL or name, removed connections included.

        A LinkedIn profile URL (any form `canonical_profile_url` understands)
        matches at most one connection. A name matches, in order: the full
        name exactly (ignoring case); else every word somewhere in the first or
        last name; else, for a single word, the profile URL's slug. One match
        gives "found" with `connection_detail`; several give "ambiguous" with
        the first `MAX_CANDIDATES`; none gives "not_found" with up to
        `MAX_SUGGESTIONS` people whose name contains any of the words.
        """
        text = " ".join(_words(name_or_url))
        if not text:
            return ConnectionLookup("not_found")
        if _looks_like_url(text):
            url = canonical_profile_url(text)
            rows = self._lookup_rows(Connection.profile_url == url) if url else []
            return self._lookup_result(rows, suggest_for="")

        full_name = sa.func.lower(
            sa.func.concat_ws(
                " ", sa.func.btrim(Connection.first_name), sa.func.btrim(Connection.last_name)
            )
        )
        rows = self._lookup_rows(full_name == text.lower())
        if not rows:
            rows = self._lookup_rows(*(self._name_has(word) for word in text.split()))
        if not rows and " " not in text:
            rows = self._lookup_rows(Connection.profile_url == _PROFILE_URL_PREFIX + text.lower())
        return self._lookup_result(rows, suggest_for=text)

    def _name_has(self, word: str) -> sa.ColumnElement[bool]:
        return sa.or_(self._has(Connection.first_name, word), self._has(Connection.last_name, word))

    def _lookup_rows(self, *criteria: sa.ColumnElement[bool]) -> Sequence[Any]:
        """Up to `MAX_CANDIDATES + 1` matches, current connections before removed ones."""
        return self.rows(
            self._connections()
            .where(*criteria)
            .order_by(Connection.removed_at.is_not(None), *self._BY_NAME)
            .limit(MAX_CANDIDATES + 1)
        )

    def _lookup_result(self, rows: Sequence[Any], *, suggest_for: str) -> ConnectionLookup:
        if len(rows) == 1:
            return ConnectionLookup("found", connection=self.connection_detail(rows[0][0]))
        if rows:
            return ConnectionLookup(
                "ambiguous",
                candidates=tuple(self._record(row) for row in rows[:MAX_CANDIDATES]),
                more_candidates=len(rows) > MAX_CANDIDATES,
            )
        suggestions: Sequence[Any] = []
        words = suggest_for.split()
        if words:
            suggestions = self.rows(
                self._connections()
                .where(sa.or_(*(self._name_has(word) for word in words)))
                .order_by(Connection.removed_at.is_not(None), *self._BY_NAME)
                .limit(MAX_SUGGESTIONS)
            )
        return ConnectionLookup(
            "not_found", suggestions=tuple(self._record(row) for row in suggestions)
        )

    def connection_detail(self, connection_id: int) -> Record | None:
        """Everything known about one connection, or None if it is not the owner's.

        The connection record plus, when present: `email`, `removed_on`,
        `role_history` (company, title, `as_of` export date; oldest first,
        spelling-only changes folded), `message_count` and `last_contact`,
        `endorsements` and `recommendations` (each split into `received` from
        them and `given` to them), `notes` and the latest `invitation`.
        """
        found = self.rows(self._connections().where(Connection.id == connection_id).limit(1))
        if not found:
            return None
        row = found[0]
        email = self.scalar(
            self.select(Connection, Connection.email).where(Connection.id == connection_id)
        )

        roles: list[Record] = []
        for _, company, title, exported_at in self._timelines([connection_id])[connection_id]:
            if roles and (roles[-1].get("company"), roles[-1].get("title")) == (company, title):
                continue
            roles.append(_compact({"company": company, "title": title, "as_of": _day(exported_at)}))

        message_count, last_contact = self.rows(
            self.select(Message, sa.func.count(), sa.func.max(Message.sent_at)).where(
                Message.connection_id == connection_id
            )
        )[0]

        endorsements: dict[str, list[Record]] = {}
        for item in self.scalars(
            self.select(Endorsement)
            .where(Endorsement.connection_id == connection_id)
            .order_by(Endorsement.endorsed_at.asc().nulls_last(), Endorsement.id)
        ):
            endorsements.setdefault(item.direction, []).append(
                _compact({"skill": item.skill_name, "date": _day(item.endorsed_at)})
            )

        recommendations: dict[str, list[Record]] = {}
        for item in self.scalars(
            self.select(Recommendation)
            .where(Recommendation.connection_id == connection_id)
            .order_by(Recommendation.created_at.asc().nulls_last(), Recommendation.id)
        ):
            recommendations.setdefault(item.direction, []).append(
                _compact(
                    {
                        "text": _truncate(item.text, RECOMMENDATION_TEXT_LIMIT),
                        "date": _day(item.created_at),
                    }
                )
            )

        notes = [
            _compact({"note": item.note, "date": _day(item.edited_at or item.created_at)})
            for item in self.scalars(
                self.select(ConnectionNote)
                .where(ConnectionNote.connection_id == connection_id)
                .order_by(ConnectionNote.created_at.asc().nulls_last(), ConnectionNote.id)
            )
        ]

        invitation = self.scalar(
            self.select(Invitation)
            .where(Invitation.connection_id == connection_id)
            .order_by(Invitation.sent_at.desc().nulls_last(), Invitation.id.desc())
            .limit(1)
        )

        removed_at = row[7]
        return _compact(
            {
                **self._record(row),
                "email": email,
                "removed_on": _day(removed_at),
                "role_history": roles,
                "message_count": message_count,
                "last_contact": _day(last_contact),
                "endorsements": endorsements,
                "recommendations": recommendations,
                "notes": [note for note in notes if note],
                "invitation": _compact(
                    {
                        "direction": invitation.direction,
                        "date": _day(invitation.sent_at),
                        "message": invitation.message,
                    }
                )
                if invitation is not None
                else None,
            }
        )
