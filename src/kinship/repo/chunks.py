"""Reads behind chunking and embedding: source rows, existing chunks, counts.

Everything is returned as plain frozen dataclasses, detached from the session.
Writes to `chunks` live in `kinship.rag.store`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import sqlalchemy as sa

from kinship.db.models import Chunk, Message, MyPosition, Profile, Recommendation
from kinship.repo.base import BaseRepository


@dataclass(frozen=True)
class RecommendationSource:
    id: int
    direction: str
    first_name: str | None = None
    last_name: str | None = None
    company: str | None = None
    job_title: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class PositionSource:
    id: int
    company: str | None = None
    title: str | None = None
    description: str | None = None
    location: str | None = None
    started_on: date | None = None
    ended_on: date | None = None


@dataclass(frozen=True)
class ProfileSource:
    id: int
    headline: str | None = None
    summary: str | None = None
    industry: str | None = None


@dataclass(frozen=True)
class MessageSource:
    id: int
    conversation_id: str | None = None
    from_name: str | None = None
    sent_at: datetime | None = None
    body: str | None = None


@dataclass(frozen=True)
class ChunkSources:
    """Every row of one owner that can become chunk text."""

    recommendations: list[RecommendationSource] = field(default_factory=list)
    positions: list[PositionSource] = field(default_factory=list)
    profile: ProfileSource | None = None
    messages: list[MessageSource] = field(default_factory=list)


@dataclass(frozen=True)
class ExistingChunk:
    """A stored chunk without its text or vector."""

    id: int
    source_table: str
    source_id: int
    chunk_index: int
    content_hash: str | None
    model: str | None
    dim: int | None
    has_embedding: bool

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.source_table, self.source_id, self.chunk_index)


@dataclass(frozen=True)
class PendingChunk:
    """A chunk that still needs a vector from the current embedder."""

    id: int
    source_table: str
    text: str


class ChunksRepository(BaseRepository):
    def chunk_sources(self, *, include_messages: bool = False) -> ChunkSources:
        """The owner's recommendations, positions and profile, in id order.

        Message bodies are read only when `include_messages` is true.
        """
        r = Recommendation
        recommendations = [
            RecommendationSource(*row)
            for row in self.rows(
                self.select(
                    r, r.id, r.direction, r.first_name, r.last_name, r.company, r.job_title, r.text
                ).order_by(r.id)
            )
        ]
        p = MyPosition
        positions = [
            PositionSource(*row)
            for row in self.rows(
                self.select(
                    p, p.id, p.company, p.title, p.description, p.location, p.started_on, p.ended_on
                ).order_by(p.id)
            )
        ]
        profile_rows = self.rows(
            self.select(
                Profile, Profile.id, Profile.headline, Profile.summary, Profile.industry
            ).order_by(Profile.id)
        )
        messages: list[MessageSource] = []
        if include_messages:
            m = Message
            messages = [
                MessageSource(*row)
                for row in self.rows(
                    self.select(m, m.id, m.conversation_id, m.from_name, m.sent_at, m.body)
                    .where(m.body.is_not(None))
                    .order_by(m.id)
                )
            ]
        return ChunkSources(
            recommendations=recommendations,
            positions=positions,
            profile=ProfileSource(*profile_rows[0]) if profile_rows else None,
            messages=messages,
        )

    def existing_chunks(self) -> list[ExistingChunk]:
        """Every stored chunk of this owner (keys, hash and embedding state only)."""
        c = Chunk
        return [
            ExistingChunk(*row)
            for row in self.rows(
                self.select(
                    c,
                    c.id,
                    c.source_table,
                    c.source_id,
                    c.chunk_index,
                    c.content_hash,
                    c.model,
                    c.dim,
                    c.embedding.is_not(None),
                ).order_by(c.id)
            )
        ]

    def pending_chunks(
        self,
        model: str,
        dim: int,
        *,
        include_messages: bool = False,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[PendingChunk]:
        """Chunks with no vector, or one from another model or dimension, in id order.

        Page with `after_id` (the last id of the previous page). Message chunks
        are skipped unless `include_messages` is true.
        """
        c = Chunk
        stmt = self.select(c, c.id, c.source_table, c.text).where(
            c.id > after_id,
            sa.or_(
                c.embedding.is_(None),
                c.model.is_distinct_from(model),
                c.dim.is_distinct_from(dim),
            ),
        )
        if not include_messages:
            stmt = stmt.where(c.source_table != Message.__tablename__)
        return [PendingChunk(*row) for row in self.rows(stmt.order_by(c.id).limit(limit))]

    def chunk_counts(self) -> dict[str, int]:
        """Number of this owner's chunks per `source_table`."""
        rows = self.rows(
            self.select(Chunk, Chunk.source_table, sa.func.count())
            .group_by(Chunk.source_table)
            .order_by(Chunk.source_table)
        )
        return {table: count for table, count in rows}

    def embedded_chunk_counts(self) -> dict[str, int]:
        """Number of this owner's chunks that have a vector, per `source_table`."""
        rows = self.rows(
            self.select(Chunk, Chunk.source_table, sa.func.count())
            .where(Chunk.embedding.is_not(None))
            .group_by(Chunk.source_table)
            .order_by(Chunk.source_table)
        )
        return {table: count for table, count in rows}
