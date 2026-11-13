"""Company lookup: resolve what a user typed to the owner's normalized companies.

Matching is on `kinship.companies.company_key`, the same key the importer
groups spellings by, so "gift health" finds the company whose spellings are
"Gifthealth" and "Gift Health, Inc.".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import sqlalchemy as sa

from kinship.companies import company_key
from kinship.db.models import Company, Connection
from kinship.repo.base import BaseRepository

MatchKind = Literal["exact", "prefix", "contains"]

MIN_CONTAINS_KEY_LENGTH = 3
"""Shorter queries only match as an exact key or a prefix."""

_MATCH_KINDS: tuple[MatchKind, ...] = ("exact", "prefix", "contains")


@dataclass(frozen=True)
class CompanyMatch:
    """One of the owner's companies, detached from the session.

    `connection_count` counts current connections (not removed) at the company.
    `match` says how the query matched; it is "exact" for `get_company`.
    """

    id: int
    display_name: str
    name_normalized: str
    aliases: tuple[str, ...]
    connection_count: int
    match: MatchKind

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data


class CompaniesRepository(BaseRepository):
    def _companies(self, rank: sa.ColumnElement[int]) -> tuple[sa.Select[Any], Any]:
        """Companies with their current-connection count and a match rank."""
        counts = (
            self.select(Connection, Connection.company_id, sa.func.count().label("n"))
            .where(Connection.removed_at.is_(None), Connection.company_id.is_not(None))
            .group_by(Connection.company_id)
            .subquery()
        )
        connection_count = sa.func.coalesce(counts.c.n, 0)
        stmt = self.select(
            Company,
            Company.id,
            Company.display_name,
            Company.name_normalized,
            Company.aliases,
            connection_count,
            rank,
        ).outerjoin(counts, counts.c.company_id == Company.id)
        return stmt, connection_count

    @staticmethod
    def _match(row: Any) -> CompanyMatch:
        company_id, display_name, name_normalized, aliases, connection_count, rank = row
        return CompanyMatch(
            id=company_id,
            display_name=display_name,
            name_normalized=name_normalized,
            aliases=tuple(aliases or ()),
            connection_count=connection_count,
            match=_MATCH_KINDS[rank],
        )

    def find_companies(self, query: str, limit: int = 10) -> list[CompanyMatch]:
        """The owner's companies matching `query`, best first.

        The query is reduced to its company key (case, spacing, punctuation and
        legal suffixes ignored). A company whose key equals it is an "exact"
        match, one whose key starts with it a "prefix" match, and, for query
        keys of at least `MIN_CONTAINS_KEY_LENGTH` characters, one whose key
        contains it a "contains" match. Ordered by match quality, then by
        number of current connections, then by name.
        """
        key = company_key(query)
        if key is None or limit <= 0:
            return []
        name = Company.name_normalized
        is_prefix = name.startswith(key, autoescape=True)
        wanted = is_prefix
        if len(key) >= MIN_CONTAINS_KEY_LENGTH:
            wanted = name.contains(key, autoescape=True)
        rank = sa.case((name == key, 0), (is_prefix, 1), else_=2)
        stmt, connection_count = self._companies(rank)
        stmt = (
            stmt.where(wanted)
            .order_by(rank, connection_count.desc(), Company.display_name, Company.id)
            .limit(limit)
        )
        return [self._match(row) for row in self.rows(stmt)]

    def get_company(self, company_id: int) -> CompanyMatch | None:
        """The company with this id, or None if it is missing or another owner's."""
        stmt, _ = self._companies(sa.literal(0))
        found = self.rows(stmt.where(Company.id == company_id).limit(1))
        return self._match(found[0]) if found else None
