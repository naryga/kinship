"""Retrieval over `chunks`: full-text search, vector search and provenance.

Every statement starts from `self.select(Chunk, ...)`, so it only ever sees the
owner's chunks. Results are plain frozen dataclasses.

Full-text search
----------------
`fts_search` builds two tsqueries from the user's text, both from bound
parameters only (user text is never spliced into SQL or into tsquery syntax):

- *strict*: `websearch_to_tsquery('english', :query)`. It never raises on odd
  input, ANDs the terms and understands "quoted phrases" and `-exclusions`.
- *loose*: the query is split into words in Python and each word becomes its
  own `plainto_tsquery('english', :word)`; these are OR-ed with the tsquery
  `||` operator. Stop words become empty tsqueries and drop out.

A natural-language question rarely has every one of its words in one chunk, so
the strict query alone often finds nothing. Chunks therefore qualify when they
match the loose query, and are ordered with strict matches first, then by
`ts_rank` against the loose query (normalised by document length), then id.

Vector search
-------------
`vector_search` orders by cosine distance and only looks at chunks whose stored
`model` is the query embedder's: vectors from different models are never
compared, and chunks without a vector are skipped.

The HNSW index (`ix_chunks_embedding_hnsw`) covers every owner's vectors, and
Postgres applies the owner / model / source filters *after* the index scan. If
other rows crowd the index's candidate list, a filtered query can come back
short. Two things keep the answer right:

1. `hnsw.ef_search` is raised for the transaction, so the index hands back more
   candidates than the default 40;
2. if the result is still shorter than both `limit` and the number of rows that
   qualify, the query is run again as an exact scan of the owner's rows (the
   `+ 0` on the distance keeps the planner off the HNSW index).

For one owner and a few thousand chunks the planner usually picks the exact
plan by itself. With many owners, per-owner partial indexes or pgvector's
iterative index scans (0.8+) are the better tools.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import reduce
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import REGCONFIG

from kinship.db.models import EMBEDDING_DIM, Chunk, Message, MyPosition, Profile, Recommendation
from kinship.repo.base import BaseRepository

FTS_CONFIG = "english"
MAX_QUERY_CHARS = 2000
"""The strict full-text query reads at most this much of the user's text."""
MAX_QUERY_TERMS = 32
"""The loose full-text query ORs at most this many distinct words."""
MAX_TERM_CHARS = 64

EF_SEARCH_MIN = 100
EF_SEARCH_MAX = 1000  # pgvector's upper bound for hnsw.ef_search

_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class ChunkHit:
    """One chunk found by a retriever.

    `score` is higher-is-better: `ts_rank` for full-text search, cosine
    similarity (1 - distance) for vector search. `rank` is the 1-based position
    in that retriever's result. `source_table` / `source_id` name the row the
    chunk was built from (for messages: the conversation's first message).
    """

    chunk_id: int
    source_table: str
    source_id: int
    chunk_index: int
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class SourceInfo:
    """Where a chunk came from, in words: a title and, if known, a date."""

    source_table: str
    source_id: int
    title: str
    date: date | None = None


def query_terms(query: str) -> list[str]:
    """Distinct lower-cased words of `query`, in order, capped for the loose query."""
    words = (word.casefold() for word in _WORD.findall(query))
    distinct = dict.fromkeys(word for word in words if len(word) <= MAX_TERM_CHARS)
    return list(distinct)[:MAX_QUERY_TERMS]


def _hits(rows: Iterable[Any]) -> list[ChunkHit]:
    hits: list[ChunkHit] = []
    for chunk_id, source_table, source_id, chunk_index, text, score in rows:
        if score is None or math.isnan(score):
            continue  # e.g. cosine distance to a zero vector
        hits.append(ChunkHit(chunk_id, source_table, source_id, chunk_index, text, float(score), 0))
    return [
        ChunkHit(h.chunk_id, h.source_table, h.source_id, h.chunk_index, h.text, h.score, rank)
        for rank, h in enumerate(hits, start=1)
    ]


class RetrievalRepository(BaseRepository):
    # -- full-text ----------------------------------------------------------

    def fts_search(
        self, query: str, *, sources: Sequence[str] | None = None, limit: int = 50
    ) -> list[ChunkHit]:
        """The owner's chunks that match `query`'s words, best first.

        Chunks containing every term (the strict query) come first; chunks that
        contain only some of the words follow. See the module docstring.
        """
        terms = query_terms(query)
        if not terms or limit <= 0:
            return []
        config = sa.cast(FTS_CONFIG, REGCONFIG)
        strict = sa.func.websearch_to_tsquery(config, query[:MAX_QUERY_CHARS])
        loose = reduce(
            lambda left, right: left.op("||")(right),
            [sa.func.plainto_tsquery(config, term) for term in terms],
        )
        c = Chunk
        is_strict = c.tsv.bool_op("@@")(strict)
        rank = sa.func.ts_rank(c.tsv, loose, 1)
        stmt = self.select(c, c.id, c.source_table, c.source_id, c.chunk_index, c.text, rank).where(
            c.tsv.bool_op("@@")(loose)
        )
        if sources is not None:
            stmt = stmt.where(c.source_table.in_(list(sources)))
        return _hits(self.rows(stmt.order_by(is_strict.desc(), rank.desc(), c.id).limit(limit)))

    # -- vectors ------------------------------------------------------------

    def _embedded(self, model: str, sources: Sequence[str] | None) -> list[Any]:
        criteria = [Chunk.embedding.is_not(None), Chunk.model == model]
        if sources is not None:
            criteria.append(Chunk.source_table.in_(list(sources)))
        return criteria

    def embedded_count(self, model: str, *, sources: Sequence[str] | None = None) -> int:
        """How many of the owner's chunks have a vector from `model`."""
        return self.count(Chunk, *self._embedded(model, sources))

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        model: str,
        sources: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[ChunkHit]:
        """The owner's chunks nearest to `query_vector` by cosine distance.

        Only chunks embedded with `model` are considered. A vector of the wrong
        length matches nothing.
        """
        vector = [float(x) for x in query_vector]
        if len(vector) != EMBEDDING_DIM or limit <= 0:
            return []
        c = Chunk
        distance = c.embedding.cosine_distance(vector)
        criteria = self._embedded(model, sources)

        def nearest(order: Any) -> list[ChunkHit]:
            stmt = self.select(
                c, c.id, c.source_table, c.source_id, c.chunk_index, c.text, 1 - distance
            ).where(*criteria)
            return _hits(self.rows(stmt.order_by(order, c.id).limit(limit)))

        self.set_local("hnsw.ef_search", min(max(limit * 4, EF_SEARCH_MIN), EF_SEARCH_MAX))
        hits = nearest(distance)
        if len(hits) < limit and len(hits) < self.count(c, *criteria):
            # The index scan was starved by rows that the filters then removed.
            hits = nearest(distance + 0)
        return hits

    # -- provenance ---------------------------------------------------------

    def source_info(self, keys: Iterable[tuple[str, int]]) -> dict[tuple[str, int], SourceInfo]:
        """Titles for `(source_table, source_id)` pairs, e.g. taken from hits.

        Pairs whose row is missing, belongs to another owner or comes from an
        unknown table are left out.
        """
        wanted: dict[str, set[int]] = {}
        for table, source_id in keys:
            wanted.setdefault(table, set()).add(source_id)
        found: dict[tuple[str, int], SourceInfo] = {}

        def add(table: str, source_id: int, title: str, when: Any = None) -> None:
            day = when.date() if hasattr(when, "date") else when
            if day is not None:
                title += f" ({day.isoformat()})"
            found[(table, source_id)] = SourceInfo(table, source_id, title, day)

        r = Recommendation
        if ids := wanted.get(r.__tablename__):
            for row in self.rows(
                self.select(r, r.id, r.direction, r.first_name, r.last_name, r.created_at).where(
                    r.id.in_(ids)
                )
            ):
                given = row.direction == "given"
                title = "Recommendation given" if given else "Recommendation received"
                if name := _join(" ", row.first_name, row.last_name):
                    title += (" to " if given else " from ") + name
                add(r.__tablename__, row.id, title, row.created_at)

        p = MyPosition
        if ids := wanted.get(p.__tablename__):
            for row in self.rows(self.select(p, p.id, p.title, p.company).where(p.id.in_(ids))):
                role = _join(" at ", row.title, row.company) or "(untitled)"
                add(p.__tablename__, row.id, f"Position: {role}")

        f = Profile
        if ids := wanted.get(f.__tablename__):
            for row in self.rows(self.select(f, f.id, f.headline).where(f.id.in_(ids))):
                headline = (row.headline or "").strip()
                add(f.__tablename__, row.id, f"Profile: {headline}" if headline else "Profile")

        m = Message
        if ids := wanted.get(m.__tablename__):
            for row in self.rows(self.select(m, m.id, m.from_name, m.sent_at).where(m.id.in_(ids))):
                sender = (row.from_name or "").strip()
                title = f"Conversation started by {sender}" if sender else "Conversation"
                add(m.__tablename__, row.id, title, row.sent_at)
        return found


def _join(separator: str, *parts: str | None) -> str:
    return separator.join(part.strip() for part in parts if part and part.strip())
