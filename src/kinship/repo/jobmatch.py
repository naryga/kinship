"""Queries behind `match_job`: what in the owner's data is close to a job description.

- `companies_named_in(text)`: which of the owner's companies the text names.
  Only the owner's own `companies` rows are candidates, never a global list.
- `skill_names()`: the owner's skills.
- `terms_in_chunks(terms)`: which words occur somewhere in the owner's chunks.
- `connections_with_title_words(words)`: current connections whose title
  contains one of the words (the caller scores them).
- `company_departures(company_ids, connection_ids)`: when a connection was
  last seen at a company and when they first showed up elsewhere.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import REGCONFIG

from kinship.companies import normalize_company_name
from kinship.db.models import Chunk, Company, Connection, ConnectionHistory, Import, Skill
from kinship.repo.base import BaseRepository
from kinship.repo.network import _BY_NAME, _PERSON_COLUMNS, NetworkPerson, _person
from kinship.repo.retrieval import FTS_CONFIG

MIN_COMPANY_NAME_CHARS = 3
"""Company names shorter than this are never looked for in free text."""
MAX_TEXT_CHARS = 20_000
MAX_TERMS = 40
MAX_TITLE_WORDS = 12
MAX_TITLE_CANDIDATES = 2000

NOT_COMPANY_NAMES = frozenset(
    {
        "confidential",
        "freelance",
        "freelancer",
        "independent",
        "remote",
        "retired",
        "self employed",
        "selfemployed",
        "stealth",
        "stealth startup",
        "student",
        "unemployed",
        "various",
    }
)
"""Things people type into LinkedIn's company field that a job description
uses as ordinary words; they are never inferred as the hiring company."""


@dataclass(frozen=True)
class NamedCompany:
    """One of the owner's companies whose name occurs in a text.

    `mentions` counts occurrences (any spelling, ignoring case and punctuation);
    `exact_spelling` says one of the stored spellings occurs letter for letter.
    """

    id: int
    display_name: str
    connection_count: int
    mentions: int
    exact_spelling: bool


@dataclass(frozen=True)
class Departure:
    """A connection's time at a company, as far as the exports show it.

    `title` is their last title there, `last_seen_at` the date of the last
    export that showed them there and `moved_by` the date of the first later
    export that showed them somewhere else (None if no export did).
    """

    connection_id: int
    company_id: int
    title: str | None
    last_seen_at: datetime | None
    moved_by: datetime | None


def _spaced(text: str | None) -> str:
    """`text` normalized like a company name, with a space at both ends."""
    return f" {normalize_company_name(text) or ''} "


class JobMatchRepository(BaseRepository):
    def companies_named_in(self, text: str, limit: int = 5) -> list[NamedCompany]:
        """The owner's companies whose name or one of whose spellings occurs in `text`.

        Matching is on whole words, ignoring case, accents, punctuation and
        legal suffixes, for names of at least `MIN_COMPANY_NAME_CHARS`
        characters. Best first: companies spelled exactly as stored, then by
        number of mentions, then longer names, then more current connections.
        """
        text = (text or "")[:MAX_TEXT_CHARS]
        haystack = _spaced(text)
        if not haystack.strip() or limit <= 0:
            return []
        counts = (
            self.select(Connection, Connection.company_id, sa.func.count().label("n"))
            .where(Connection.removed_at.is_(None), Connection.company_id.is_not(None))
            .group_by(Connection.company_id)
            .subquery()
        )
        stmt = self.select(
            Company,
            Company.id,
            Company.display_name,
            Company.aliases,
            sa.func.coalesce(counts.c.n, 0),
        ).outerjoin(counts, counts.c.company_id == Company.id)

        found: list[tuple[tuple[int, int, int, int], NamedCompany]] = []
        for company_id, display_name, aliases, connection_count in self.rows(stmt):
            spellings = {display_name, *(aliases or ())} - {None, ""}
            names = {_spaced(spelling).strip() for spelling in spellings}
            names = {
                name
                for name in names
                if len(name) >= MIN_COMPANY_NAME_CHARS and name not in NOT_COMPANY_NAMES
            }
            # One company's names differ only in spacing, so no mention counts twice.
            mentions = sum(
                len(re.findall(rf"(?<= ){re.escape(name)}(?= )", haystack))
                for name in names
                if f" {name} " in haystack  # cheap test first: most companies are not named
            )
            if not mentions:
                continue
            longest = max(names, key=len)
            exact = any(
                re.search(rf"(?<!\w){re.escape(spelling)}(?!\w)", text) for spelling in spellings
            )
            company = NamedCompany(company_id, display_name, connection_count, mentions, exact)
            key = (-int(exact), -mentions, -len(longest), -connection_count)
            found.append((key, company))
        found.sort(key=lambda item: (item[0], item[1].display_name, item[1].id))
        return [company for _, company in found[:limit]]

    def skill_names(self) -> list[str]:
        """The owner's skills, by name (ignoring case, whatever the database collation)."""
        names = self.scalars(self.select(Skill, Skill.name).order_by(Skill.id))
        distinct = dict.fromkeys(name.strip() for name in names if name and name.strip())
        return sorted(distinct, key=lambda name: (name.casefold(), name))

    def terms_in_chunks(
        self, terms: Sequence[str], *, sources: Sequence[str] | None = None
    ) -> set[str]:
        """Which of `terms` occur (as English word stems) in any of the owner's chunks.

        Each term is a bound parameter to `plainto_tsquery`; at most `MAX_TERMS`
        are checked and the rest are reported as absent.
        """
        terms = list(dict.fromkeys(terms))[:MAX_TERMS]
        if not terms:
            return set()
        config = sa.cast(FTS_CONFIG, REGCONFIG)
        present = [
            sa.func.coalesce(
                sa.func.bool_or(Chunk.tsv.bool_op("@@")(sa.func.plainto_tsquery(config, term))),
                False,
            )
            for term in terms
        ]
        stmt = self.select(Chunk, *present)
        if sources is not None:
            stmt = stmt.where(Chunk.source_table.in_(list(sources)))
        row = self.rows(stmt)[0]
        return {term for term, found in zip(terms, row, strict=True) if found}

    def connections_with_title_words(
        self, words: Collection[str], limit: int = MAX_TITLE_CANDIDATES
    ) -> list[NetworkPerson]:
        """Current connections whose title contains any of `words` (case-insensitive).

        A substring match, to be scored by the caller; `words` are bound
        parameters with LIKE wildcards escaped.
        """
        wanted = sorted({word.strip() for word in words if word and word.strip()})
        wanted = wanted[:MAX_TITLE_WORDS]
        if not wanted or limit <= 0:
            return []
        stmt = (
            self.select(Connection, *_PERSON_COLUMNS)
            .where(
                Connection.removed_at.is_(None),
                Connection.position.is_not(None),
                sa.or_(*(Connection.position.icontains(w, autoescape=True) for w in wanted)),
            )
            .order_by(*_BY_NAME)
            .limit(limit)
        )
        return [_person(row) for row in self.rows(stmt)]

    def company_departures(
        self, company_ids: Collection[int], connection_ids: Collection[int]
    ) -> dict[int, Departure]:
        """For each of `connection_ids` that was once at one of `company_ids`: when.

        Reads the connection's role timeline in export order. Ids that are
        missing, another owner's or never at the companies are left out.
        """
        if not company_ids or not connection_ids:
            return {}
        stmt = (
            self.select(
                ConnectionHistory,
                ConnectionHistory.connection_id,
                ConnectionHistory.company_id,
                ConnectionHistory.position,
                Import.exported_at,
            )
            .join(Import, Import.id == ConnectionHistory.import_id)
            .where(self.owned(Import), ConnectionHistory.connection_id.in_(connection_ids))
            .order_by(
                ConnectionHistory.connection_id,
                Import.exported_at.asc().nulls_first(),
                Import.source_sha256,
                ConnectionHistory.id,
            )
        )
        wanted = set(company_ids)
        departures: dict[int, Departure] = {}
        for connection_id, company_id, position, exported_at in self.rows(stmt):
            if company_id in wanted:
                departures[connection_id] = Departure(
                    connection_id, company_id, position, exported_at, None
                )
            elif connection_id in departures and departures[connection_id].moved_by is None:
                there = departures[connection_id]
                departures[connection_id] = Departure(
                    connection_id, there.company_id, there.title, there.last_seen_at, exported_at
                )
        return departures
