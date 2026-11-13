"""Writes to `chunks`, bound to one owner.

This is the only place that inserts, updates or deletes chunks. Reads go
through `kinship.repo` (`ChunksRepository`). Every statement here either sets
`owner_id` (inserts) or filters on `Chunk.owner_id == owner_id` (updates and
deletes), so a chunk id or source key that belongs to another owner is never
touched, whatever the caller passes in.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import Session

from kinship.db.models import Chunk
from kinship.rag.chunking import ChunkDraft
from kinship.repo.base import OwnerScope

_TABLE = Chunk.__table__


class ChunkWriter(OwnerScope):
    """Insert, update and delete one owner's chunks. Never commits."""

    def __init__(self, session: Session, owner_id: int) -> None:
        super().__init__(owner_id)  # validates owner_id
        self._session = session

    def _mine(self) -> sa.ColumnElement[bool]:
        return _TABLE.c.owner_id == self.owner_id

    def insert(self, drafts: Sequence[ChunkDraft]) -> int:
        """Add new chunks (without vectors) for this owner."""
        if not drafts:
            return 0
        self._session.execute(
            sa.insert(_TABLE),
            [
                {
                    "owner_id": self.owner_id,
                    "source_table": draft.source_table,
                    "source_id": draft.source_id,
                    "chunk_index": draft.chunk_index,
                    "text": draft.text,
                    "content_hash": draft.content_hash,
                }
                for draft in drafts
            ],
        )
        return len(drafts)

    def update_text(self, changes: Sequence[tuple[int, ChunkDraft]]) -> int:
        """Replace the text of chunks `(chunk_id, draft)` and drop their stale vectors."""
        if not changes:
            return 0
        stmt = (
            sa.update(_TABLE)
            .where(self._mine(), _TABLE.c.id == sa.bindparam("b_id"))
            .values(
                text=sa.bindparam("b_text"),
                content_hash=sa.bindparam("b_hash"),
                embedding=None,
                model=None,
                dim=None,
            )
        )
        self._session.connection().execute(
            stmt,
            [
                {"b_id": chunk_id, "b_text": draft.text, "b_hash": draft.content_hash}
                for chunk_id, draft in changes
            ],
        )
        return len(changes)

    def delete(self, chunk_ids: Sequence[int]) -> int:
        """Delete this owner's chunks with these ids; returns the number deleted."""
        if not chunk_ids:
            return 0
        result = self._session.execute(
            sa.delete(_TABLE).where(self._mine(), _TABLE.c.id.in_(list(chunk_ids)))
        )
        return result.rowcount  # type: ignore[attr-defined]

    def delete_source(self, source_table: str) -> int:
        """Delete all of this owner's chunks from one source table."""
        result = self._session.execute(
            sa.delete(_TABLE).where(self._mine(), _TABLE.c.source_table == source_table)
        )
        return result.rowcount  # type: ignore[attr-defined]

    def set_embeddings(
        self, vectors: Sequence[tuple[int, Sequence[float]]], *, model: str, dim: int
    ) -> int:
        """Store `(chunk_id, vector)` pairs together with the model and dimension."""
        if not vectors:
            return 0
        stmt = (
            sa.update(_TABLE)
            .where(self._mine(), _TABLE.c.id == sa.bindparam("b_id"))
            .values(embedding=sa.bindparam("b_embedding"), model=model, dim=dim)
        )
        self._session.connection().execute(
            stmt, [{"b_id": chunk_id, "b_embedding": list(vector)} for chunk_id, vector in vectors]
        )
        return len(vectors)
