"""The HNSW index, and vector search staying correct when another owner crowds it.

Postgres filters by owner *after* an HNSW index scan. If another owner has many
vectors closer to the query, the scan's candidate list holds none of ours and a
plain filtered `ORDER BY embedding <=> :q LIMIT n` comes back short or empty.
`RetrievalRepository.vector_search` must still return the owner's best chunks.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Owner
from kinship.rag import FakeEmbedder
from kinship.rag.embedder import FAKE_MODEL
from kinship.repo import Repository

INDEX = "ix_chunks_embedding_hnsw"
QUERY = "postgres query planner tuning for analytical workloads"
CROWD = 400  # more than the largest hnsw.ef_search that vector_search sets for small limits


def add_chunks(
    session: Session, owner_id: int, texts: list[str], *, model: str = FAKE_MODEL, first_id: int = 1
) -> list[int]:
    embedder = FakeEmbedder()
    chunks = [
        Chunk(
            owner_id=owner_id,
            source_table="recommendations",
            source_id=first_id + index,
            text=text,
            embedding=vector,
            model=model,
            dim=embedder.dim,
        )
        for index, (text, vector) in enumerate(zip(texts, embedder.embed(texts), strict=True))
    ]
    session.add_all(chunks)
    session.flush()
    return [chunk.id for chunk in chunks]


def force_index_scans(session: Session) -> None:
    """Make the planner take the HNSW index even though the table is tiny."""
    for setting in ("enable_seqscan", "enable_sort", "enable_bitmapscan"):
        session.execute(sa.text(f"SET LOCAL {setting} = off"))


def ann_statement(owner_id: int, vector: list[float], limit: int) -> sa.Select:
    return (
        sa.select(Chunk.id)
        .where(Chunk.owner_id == owner_id, Chunk.embedding.is_not(None), Chunk.model == FAKE_MODEL)
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(limit)
    )


def test_hnsw_index_exists_with_cosine_ops(engine: Engine) -> None:
    with engine.connect() as conn:
        definition = conn.scalar(
            sa.text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": INDEX}
        )
    assert definition is not None
    assert "USING hnsw (embedding vector_cosine_ops)" in definition
    assert "ON public.chunks" in definition


def test_owner_gets_own_best_hits_when_another_owner_crowds_the_index(
    session: Session, make_owner
) -> None:
    a, b = make_owner("Few chunks"), make_owner("Many closer chunks")
    # A: three chunks, in a known order of similarity to the query.
    ids_a = add_chunks(
        session,
        a.id,
        [
            f"{QUERY} explained",
            "postgres query planner notes",
            "tuning a guitar for analytical listening",
        ],
    )
    # B: hundreds of chunks, every one closer to the query than A's best.
    ids_b = add_chunks(session, b.id, [f"{QUERY} {i}" for i in range(CROWD)])
    query = FakeEmbedder().embed_query(QUERY)
    force_index_scans(session)

    # The situation is real: the planner uses the HNSW index and the plain
    # filtered query is starved of A's rows.
    stmt = ann_statement(a.id, query, 3)
    compiled = stmt.compile(session.get_bind(), compile_kwargs={"literal_binds": True})
    plan = "\n".join(session.scalars(sa.text(f"EXPLAIN {compiled}")))
    assert INDEX in plan
    assert len(session.scalars(stmt).all()) < 3

    hits = Repository(session, a.id).vector_search(query, model=FAKE_MODEL, limit=3)

    assert [hit.chunk_id for hit in hits] == ids_a
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score > hits[2].score > 0

    # A larger limit than A has chunks: still exactly A's chunks.
    hits = Repository(session, a.id).vector_search(query, model=FAKE_MODEL, limit=50)
    assert [hit.chunk_id for hit in hits] == ids_a

    # B is unaffected and never sees A's chunks.
    hits_b = Repository(session, b.id).vector_search(query, model=FAKE_MODEL, limit=50)
    assert len(hits_b) == 50
    assert {hit.chunk_id for hit in hits_b} <= set(ids_b)
    assert session.scalar(sa.select(sa.func.count()).select_from(Owner)) >= 2


def test_model_filter_holds_when_other_model_vectors_crowd_the_index(
    session: Session, owner: Owner
) -> None:
    """Same starvation, one owner: old-model vectors must not hide current ones."""
    current = add_chunks(session, owner.id, ["postgres query planner notes"])
    add_chunks(
        session,
        owner.id,
        [f"{QUERY} {i}" for i in range(CROWD)],
        model="old-model",
        first_id=100,
    )
    query = FakeEmbedder().embed_query(QUERY)
    force_index_scans(session)

    hits = Repository(session, owner.id).vector_search(query, model=FAKE_MODEL, limit=5)

    assert [hit.chunk_id for hit in hits] == current
