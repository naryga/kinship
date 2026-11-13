"""Source rows for chunking tests, and a counting embedder.

`seed_sources` gives an owner 2 recommendations (one without text), 2 positions,
a profile and 3 messages in 2 conversations: 4 chunks without messages, 6 with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Message, MyPosition, Profile, Recommendation
from kinship.rag.embedder import FakeEmbedder

CHUNKS_WITHOUT_MESSAGES = 4
MESSAGE_CHUNKS = 2


@dataclass(frozen=True)
class SeededSources:
    owner_id: int
    recommendation_ids: list[int]
    position_ids: list[int]
    profile_id: int
    message_ids: list[int]


def seed_sources(session: Session, owner_id: int, tag: str) -> SeededSources:
    recommendations = [
        Recommendation(
            owner_id=owner_id,
            direction="received",
            first_name="Freya",
            last_name="Rookwood",
            company=f"{tag} Analytics",
            job_title="Analytics Engineer",
            text=f"{tag}: rebuilt our reporting stack with dbt and careful testing.",
        ),
        Recommendation(
            owner_id=owner_id,
            direction="given",
            first_name="Bo",
            last_name="Marsh",
            text=f"{tag}: a generous mentor who runs calm incident reviews.",
        ),
        Recommendation(owner_id=owner_id, direction="received", first_name="Nobody", text=None),
    ]
    positions = [
        MyPosition(
            owner_id=owner_id,
            company=f"{tag} Co",
            title="Data Engineer",
            description="Built streaming pipelines on Kafka.",
            started_on=date(2020, 1, 1),
            ended_on=date(2022, 6, 1),
        ),
    ]
    profile = Profile(
        owner_id=owner_id,
        headline=f"{tag} headline",
        summary="Engineer who likes data.",
        industry="Software",
    )
    messages = [
        Message(
            owner_id=owner_id,
            conversation_id=f"{tag}-conv-1",
            from_name="Freya Rookwood",
            sent_at=datetime(2024, 3, 5, 9, 0, tzinfo=UTC),
            body="Are you free for coffee next week?",
        ),
        Message(
            owner_id=owner_id,
            conversation_id=f"{tag}-conv-1",
            from_name="Me",
            sent_at=datetime(2024, 3, 5, 9, 30, tzinfo=UTC),
            body="Yes, Tuesday works.",
        ),
        Message(
            owner_id=owner_id,
            conversation_id=f"{tag}-conv-2",
            from_name="Bo Marsh",
            sent_at=datetime(2024, 4, 1, 12, 0, tzinfo=UTC),
            body="Thanks for the referral!",
        ),
    ]
    session.add_all([*recommendations, *positions, profile, *messages])
    session.flush()
    return SeededSources(
        owner_id=owner_id,
        recommendation_ids=[r.id for r in recommendations],
        position_ids=[p.id for p in positions],
        profile_id=profile.id,
        message_ids=[m.id for m in messages],
    )


class ChunkRow(NamedTuple):
    id: int
    text: str
    content_hash: str | None
    embedding: tuple[float, ...] | None
    model: str | None
    dim: int | None


def chunk_rows(session: Session, owner_id: int) -> dict[tuple[str, int, int], ChunkRow]:
    """This owner's chunks straight from the table, keyed by (table, source id, index)."""
    rows = session.execute(
        sa.select(
            Chunk.source_table,
            Chunk.source_id,
            Chunk.chunk_index,
            Chunk.id,
            Chunk.text,
            Chunk.content_hash,
            Chunk.embedding,
            Chunk.model,
            Chunk.dim,
        ).where(Chunk.owner_id == owner_id)
    ).all()
    return {
        (row.source_table, row.source_id, row.chunk_index): ChunkRow(
            row.id,
            row.text,
            row.content_hash,
            None if row.embedding is None else tuple(float(x) for x in row.embedding),
            row.model,
            row.dim,
        )
        for row in rows
    }


class CountingEmbedder(FakeEmbedder):
    """A `FakeEmbedder` that records every text it is asked to embed."""

    def __init__(self, model: str | None = None, dim: int = 1536) -> None:
        super().__init__(dim)
        if model is not None:
            self.model = model
        self.calls: list[list[str]] = []

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed(texts)
