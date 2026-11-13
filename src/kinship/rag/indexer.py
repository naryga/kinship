"""Keep an owner's `chunks` in step with their data, and embed what is new.

`sync_chunks` is the whole job behind `kinship embed`:

1. build the chunks the owner's rows should have (`kinship.rag.chunking`);
2. compare them with the stored chunks by `(source_table, source_id,
   chunk_index)` and `content_hash`;
3. insert new chunks, update changed ones (dropping the stale vector), delete
   chunks whose source row is gone;
4. embed every chunk that has no vector, or one from another model or dimension.

Unchanged, already embedded chunks cost no embedder calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.orm import Session

from kinship.db.models import EMBEDDING_DIM, Owner
from kinship.rag.chunking import MESSAGES, SOURCE_TABLES, ChunkDraft, build_drafts
from kinship.rag.embedder import Embedder, EmbedderError
from kinship.rag.store import ChunkWriter
from kinship.repo import Repository

DEFAULT_EMBED_BATCH = 64

_COUNTERS = ("created", "updated", "deleted", "embedded", "unchanged")


@dataclass
class SourceCounts:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    embedded: int = 0
    unchanged: int = 0


@dataclass
class EmbedSummary:
    """What one `sync_chunks` run did, per source table and in total.

    `unchanged` counts chunks that needed neither a text change nor a vector.
    """

    owner_id: int
    model: str
    dim: int
    include_messages: bool = False
    sources: dict[str, SourceCounts] = field(default_factory=dict)

    def source(self, table: str) -> SourceCounts:
        return self.sources.setdefault(table, SourceCounts())

    def _total(self, name: str) -> int:
        return sum(getattr(counts, name) for counts in self.sources.values())

    @property
    def created(self) -> int:
        return self._total("created")

    @property
    def updated(self) -> int:
        return self._total("updated")

    @property
    def deleted(self) -> int:
        return self._total("deleted")

    @property
    def embedded(self) -> int:
        return self._total("embedded")

    @property
    def unchanged(self) -> int:
        return self._total("unchanged")

    def format(self) -> str:
        """A small table for the command line."""
        lines = [
            f"Embedded chunks for owner {self.owner_id} with {self.model} ({self.dim} dimensions).",
            f"{'source':<16}" + "".join(f"{name:>10}" for name in _COUNTERS),
        ]
        known = [t for t in SOURCE_TABLES if t in self.sources]
        for table in known + sorted(set(self.sources) - set(known)):
            counts = self.sources[table]
            lines.append(f"{table:<16}" + "".join(f"{getattr(counts, n):>10}" for n in _COUNTERS))
        lines.append(f"{'total':<16}" + "".join(f"{self._total(n):>10}" for n in _COUNTERS))
        if not self.include_messages:
            lines.append("Messages were not included (pass --include-messages to embed them).")
        return "\n".join(lines)


def sync_chunks(
    session: Session,
    owner_id: int,
    embedder: Embedder,
    *,
    include_messages: bool = False,
    purge_messages: bool = False,
    batch_size: int = DEFAULT_EMBED_BATCH,
    after_batch: Callable[[], None] | None = None,
) -> EmbedSummary:
    """Bring `owner_id`'s chunks up to date and embed the ones that need it.

    Does not commit; the caller owns the transaction. `after_batch`, if given,
    is called once the chunk rows are in step and again after each stored batch
    of vectors, so a caller may commit there and keep finished work if a later
    batch fails.

    Without `include_messages`, message chunks are neither created, refreshed,
    re-embedded nor deleted. `purge_messages` deletes them.
    """
    if embedder.dim != EMBEDDING_DIM:
        raise EmbedderError(
            f"Embedder {embedder.model} produces {embedder.dim}-dimensional vectors, but "
            f"chunks.embedding holds {EMBEDDING_DIM}. Use a {EMBEDDING_DIM}-dimensional embedder "
            "or migrate the column."
        )
    if include_messages and purge_messages:
        raise ValueError("include_messages and purge_messages cannot be used together")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    repo = Repository(session, owner_id)
    writer = ChunkWriter(session, owner_id)
    summary = EmbedSummary(
        owner_id=owner_id,
        model=embedder.model,
        dim=embedder.dim,
        include_messages=include_messages,
    )

    if purge_messages:
        summary.source(MESSAGES).deleted += writer.delete_source(MESSAGES)

    drafts = {d.key: d for d in build_drafts(repo.chunk_sources(include_messages=include_messages))}
    existing = {
        chunk.key: chunk
        for chunk in repo.existing_chunks()
        if include_messages or chunk.source_table != MESSAGES
    }

    to_insert: list[ChunkDraft] = []
    to_update: list[tuple[int, ChunkDraft]] = []
    for key, draft in drafts.items():
        chunk = existing.get(key)
        if chunk is None:
            to_insert.append(draft)
            summary.source(draft.source_table).created += 1
        elif chunk.content_hash != draft.content_hash:
            to_update.append((chunk.id, draft))
            summary.source(draft.source_table).updated += 1
        elif chunk.has_embedding and chunk.model == embedder.model and chunk.dim == embedder.dim:
            summary.source(draft.source_table).unchanged += 1
    stale = [chunk for key, chunk in existing.items() if key not in drafts]
    for chunk in stale:
        summary.source(chunk.source_table).deleted += 1

    # Deletes first: a row that moved to another key must free its unique slot.
    writer.delete([chunk.id for chunk in stale])
    writer.update_text(to_update)
    writer.insert(to_insert)
    if after_batch is not None:
        after_batch()

    last_id = 0
    while True:
        pending = repo.pending_chunks(
            embedder.model,
            embedder.dim,
            include_messages=include_messages,
            after_id=last_id,
            limit=batch_size,
        )
        if not pending:
            break
        vectors = embedder.embed([chunk.text for chunk in pending])
        if len(vectors) != len(pending):
            raise EmbedderError(
                f"Embedder {embedder.model} returned {len(vectors)} vectors for "
                f"{len(pending)} texts"
            )
        for vector in vectors:
            if len(vector) != embedder.dim:
                raise EmbedderError(
                    f"Embedder {embedder.model} returned a vector of length {len(vector)}, "
                    f"expected {embedder.dim}"
                )
        writer.set_embeddings(
            [(chunk.id, vector) for chunk, vector in zip(pending, vectors, strict=True)],
            model=embedder.model,
            dim=embedder.dim,
        )
        for chunk in pending:
            summary.source(chunk.source_table).embedded += 1
        last_id = pending[-1].id
        if after_batch is not None:
            after_batch()
    return summary


class OwnerNotFoundError(LookupError):
    """No owner, or more than one, matches what was asked for."""


def resolve_owner_id(session: Session, owner: str) -> int:
    """The id of the owner given as a numeric id or an exact display name."""
    owner = owner.strip()
    if owner.isdecimal():
        found = session.scalar(sa.select(Owner.id).where(Owner.id == int(owner)))
        if found is not None:
            return found
    ids = session.scalars(
        sa.select(Owner.id).where(Owner.display_name == owner).order_by(Owner.id)
    ).all()
    if not ids:
        raise OwnerNotFoundError(f"No owner with id or display name {owner!r}")
    if len(ids) > 1:
        raise OwnerNotFoundError(
            f"{len(ids)} owners are named {owner!r} (ids {', '.join(map(str, ids))}); use the id"
        )
    return ids[0]
