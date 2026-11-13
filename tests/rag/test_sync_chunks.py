"""`sync_chunks`: diffing, embedding only what changed, and owner isolation."""

from collections.abc import Callable

import pytest
import sqlalchemy as sa
from rag_seeding import (
    CHUNKS_WITHOUT_MESSAGES,
    MESSAGE_CHUNKS,
    CountingEmbedder,
    chunk_rows,
    seed_sources,
)
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Message, MyPosition, Owner, Profile, Recommendation
from kinship.rag import EmbedderError, FakeEmbedder, content_hash, sync_chunks
from kinship.rag.chunking import ChunkDraft
from kinship.rag.store import ChunkWriter
from kinship.repo import Repository

MakeOwner = Callable[..., Owner]


def test_first_sync_creates_and_embeds_every_chunk(session: Session, owner: Owner) -> None:
    seeded = seed_sources(session, owner.id, "alpha")
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert (summary.created, summary.embedded) == (CHUNKS_WITHOUT_MESSAGES, CHUNKS_WITHOUT_MESSAGES)
    assert (summary.updated, summary.deleted, summary.unchanged) == (0, 0, 0)
    assert {t: c.created for t, c in summary.sources.items()} == {
        "recommendations": 2,
        "my_positions": 1,
        "profiles": 1,
    }
    rows = chunk_rows(session, owner.id)
    assert set(rows) == {
        ("recommendations", seeded.recommendation_ids[0], 0),
        ("recommendations", seeded.recommendation_ids[1], 0),
        ("my_positions", seeded.position_ids[0], 0),
        ("profiles", seeded.profile_id, 0),
    }
    for row in rows.values():
        assert row.content_hash == content_hash(row.text)
        assert (row.model, row.dim) == ("fake-hash-v1", 1536)
        assert list(row.embedding) == pytest.approx(FakeEmbedder().embed([row.text])[0], abs=1e-6)
    received = rows[("recommendations", seeded.recommendation_ids[0], 0)]
    assert received.text.startswith(
        "Recommendation received from Freya Rookwood, Analytics Engineer at alpha Analytics:\n"
    )
    assert sorted(embedder.texts) == sorted(row.text for row in rows.values())


def test_full_text_search_column_follows_the_text(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    hits = session.scalars(
        sa.select(Chunk.source_table).where(
            Chunk.owner_id == owner.id,
            Chunk.tsv.bool_op("@@")(sa.func.plainto_tsquery("english", "kafka pipelines")),
        )
    ).all()
    assert hits == ["my_positions"]


def test_second_sync_embeds_nothing(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    before = chunk_rows(session, owner.id)
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert embedder.calls == []
    assert summary.unchanged == CHUNKS_WITHOUT_MESSAGES
    assert (summary.created, summary.updated, summary.deleted, summary.embedded) == (0, 0, 0, 0)
    assert chunk_rows(session, owner.id) == before


def test_changed_text_re_embeds_only_that_chunk(session: Session, owner: Owner) -> None:
    seeded = seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    before = chunk_rows(session, owner.id)
    position = session.get(MyPosition, seeded.position_ids[0])
    assert position is not None
    position.description = "Now leads the search relevance team."
    session.flush()
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert (summary.updated, summary.embedded, summary.unchanged) == (1, 1, 3)
    assert summary.sources["my_positions"].updated == 1
    assert len(embedder.texts) == 1 and "search relevance" in embedder.texts[0]
    after = chunk_rows(session, owner.id)
    key = ("my_positions", seeded.position_ids[0], 0)
    assert after[key].id == before[key].id
    assert after[key].content_hash == content_hash(after[key].text) != before[key].content_hash
    assert list(after[key].embedding) != list(before[key].embedding)
    assert {k: v for k, v in after.items() if k != key} == {
        k: v for k, v in before.items() if k != key
    }


def test_deleted_or_emptied_source_removes_its_chunk(session: Session, owner: Owner) -> None:
    seeded = seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    session.execute(sa.delete(MyPosition).where(MyPosition.id == seeded.position_ids[0]))
    session.execute(
        sa.update(Recommendation)
        .where(Recommendation.id == seeded.recommendation_ids[1])
        .values(text="  ")
    )
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert (summary.deleted, summary.unchanged, summary.embedded) == (2, 2, 0)
    assert embedder.calls == []
    assert set(chunk_rows(session, owner.id)) == {
        ("recommendations", seeded.recommendation_ids[0], 0),
        ("profiles", seeded.profile_id, 0),
    }


def test_model_change_re_embeds_everything(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    embedder = CountingEmbedder(model="fake-hash-v2")

    summary = sync_chunks(session, owner.id, embedder)

    assert (summary.embedded, summary.unchanged) == (CHUNKS_WITHOUT_MESSAGES, 0)
    assert (summary.created, summary.updated, summary.deleted) == (0, 0, 0)
    assert len(embedder.texts) == CHUNKS_WITHOUT_MESSAGES
    assert {row.model for row in chunk_rows(session, owner.id).values()} == {"fake-hash-v2"}


def test_chunks_without_a_vector_are_embedded_on_the_next_run(
    session: Session, owner: Owner
) -> None:
    seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder())
    session.execute(
        sa.update(Chunk)
        .where(Chunk.owner_id == owner.id, Chunk.source_table == "profiles")
        .values(embedding=None)
    )
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert summary.embedded == 1 and summary.sources["profiles"].embedded == 1
    assert all(row.embedding is not None for row in chunk_rows(session, owner.id).values())


def test_embedding_is_batched(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    embedder = CountingEmbedder()
    commits: list[int] = []

    sync_chunks(
        session,
        owner.id,
        embedder,
        include_messages=True,
        batch_size=4,
        after_batch=lambda: commits.append(1),
    )

    assert [len(call) for call in embedder.calls] == [4, 2]
    assert len(commits) == 3  # once after the chunk rows, once per batch


def test_wrong_dimension_fails_before_any_work(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    embedder = CountingEmbedder(dim=8)
    with pytest.raises(EmbedderError, match="1536"):
        sync_chunks(session, owner.id, embedder)
    assert embedder.calls == [] and chunk_rows(session, owner.id) == {}


def test_embedder_returning_too_few_vectors_is_an_error(session: Session, owner: Owner) -> None:
    class Short(FakeEmbedder):
        def embed(self, texts):  # type: ignore[no-untyped-def]
            return super().embed(texts)[:-1]

    seed_sources(session, owner.id, "alpha")
    with pytest.raises(EmbedderError, match="returned 3 vectors for 4 texts"):
        sync_chunks(session, owner.id, Short())


# -- messages ------------------------------------------------------------------------


def test_messages_are_excluded_by_default(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder)

    assert "messages" not in summary.sources
    assert not any(key[0] == "messages" for key in chunk_rows(session, owner.id))
    assert not any("coffee" in text for text in embedder.texts)
    assert "Messages were not included" in summary.format()


def test_include_messages_groups_conversations(session: Session, owner: Owner) -> None:
    seeded = seed_sources(session, owner.id, "alpha")

    summary = sync_chunks(session, owner.id, FakeEmbedder(), include_messages=True)

    assert summary.sources["messages"].created == MESSAGE_CHUNKS
    assert summary.embedded == CHUNKS_WITHOUT_MESSAGES + MESSAGE_CHUNKS
    rows = chunk_rows(session, owner.id)
    first = rows[("messages", seeded.message_ids[0], 0)]
    assert first.text == (
        "Freya Rookwood (2024-03-05): Are you free for coffee next week?\n"
        "Me (2024-03-05): Yes, Tuesday works."
    )
    assert rows[("messages", seeded.message_ids[2], 0)].text == (
        "Bo Marsh (2024-04-01): Thanks for the referral!"
    )
    assert first.embedding is not None


def test_message_chunks_are_left_alone_without_the_flag(session: Session, owner: Owner) -> None:
    seeded = seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder(), include_messages=True)
    before = chunk_rows(session, owner.id)
    session.execute(
        sa.update(Message).where(Message.id == seeded.message_ids[2]).values(body="Edited body")
    )
    embedder = CountingEmbedder(model="fake-hash-v2")

    summary = sync_chunks(session, owner.id, embedder)

    assert "messages" not in summary.sources
    assert summary.embedded == CHUNKS_WITHOUT_MESSAGES
    after = chunk_rows(session, owner.id)
    message_keys = [key for key in before if key[0] == "messages"]
    assert len(message_keys) == MESSAGE_CHUNKS
    assert all(after[key] == before[key] for key in message_keys)  # same text, vector, model

    # With the flag, the edited conversation is refreshed and everything is on the new model.
    summary = sync_chunks(session, owner.id, embedder, include_messages=True)
    assert summary.sources["messages"].updated == 1
    assert summary.sources["messages"].embedded == MESSAGE_CHUNKS
    assert {row.model for row in chunk_rows(session, owner.id).values()} == {"fake-hash-v2"}


def test_purge_messages_deletes_only_message_chunks(session: Session, owner: Owner) -> None:
    seed_sources(session, owner.id, "alpha")
    sync_chunks(session, owner.id, FakeEmbedder(), include_messages=True)
    embedder = CountingEmbedder()

    summary = sync_chunks(session, owner.id, embedder, purge_messages=True)

    assert summary.sources["messages"].deleted == MESSAGE_CHUNKS
    assert summary.deleted == MESSAGE_CHUNKS and embedder.calls == []
    assert Repository(session, owner.id).chunk_counts() == {
        "my_positions": 1,
        "profiles": 1,
        "recommendations": 2,
    }
    with pytest.raises(ValueError):
        sync_chunks(session, owner.id, embedder, include_messages=True, purge_messages=True)


# -- owner isolation -------------------------------------------------------------------


def test_sync_never_touches_another_owners_chunks(session: Session, make_owner: MakeOwner) -> None:
    a, b = make_owner("Owner A"), make_owner("Owner B")
    seeded_a = seed_sources(session, a.id, "alpha")
    seed_sources(session, b.id, "beta")

    summary_a = sync_chunks(session, a.id, FakeEmbedder(), include_messages=True)

    assert summary_a.created == CHUNKS_WITHOUT_MESSAGES + MESSAGE_CHUNKS
    assert chunk_rows(session, b.id) == {}
    assert Repository(session, b.id).chunk_counts() == {}

    # Give B chunks with the *same* (source_table, source_id, chunk_index) keys as A's,
    # plus one that matches nothing. From B's side they are all stale; from A's, invisible.
    a_keys = list(chunk_rows(session, a.id))
    session.add_all(
        Chunk(
            owner_id=b.id,
            source_table=table,
            source_id=source_id,
            chunk_index=index,
            text=f"beta private {table} {source_id}",
            content_hash="not-a-real-hash",
        )
        for table, source_id, index in [*a_keys, ("recommendations", 987654321, 0)]
    )
    session.flush()
    b_before = chunk_rows(session, b.id)
    a_before = chunk_rows(session, a.id)

    # Change and delete things on A's side, then sync A again with a new model.
    session.execute(
        sa.update(Profile).where(Profile.id == seeded_a.profile_id).values(headline="New headline")
    )
    session.execute(sa.delete(MyPosition).where(MyPosition.id == seeded_a.position_ids[0]))
    embedder = CountingEmbedder(model="fake-hash-v2")
    summary_a = sync_chunks(session, a.id, embedder, include_messages=True)

    assert (summary_a.updated, summary_a.deleted) == (1, 1)
    assert summary_a.embedded == len(a_before) - 1
    assert chunk_rows(session, b.id) == b_before  # not updated, not deleted, not embedded
    assert not any("beta" in text for text in embedder.texts)

    # And purging A's messages leaves B's message chunks in place.
    sync_chunks(session, a.id, embedder, purge_messages=True)
    assert Repository(session, a.id).chunk_counts().get("messages") is None
    assert chunk_rows(session, b.id) == b_before

    # B's own sync then repairs B's chunks from B's rows only.
    summary_b = sync_chunks(session, b.id, CountingEmbedder())
    assert summary_b.created == CHUNKS_WITHOUT_MESSAGES
    b_after = chunk_rows(session, b.id)
    assert all("alpha" not in row.text for row in b_after.values())
    assert sum(1 for key in b_after if key[0] != "messages") == CHUNKS_WITHOUT_MESSAGES
    # B's stale non-message chunks are gone; its message chunks were left alone (no flag).
    assert {k: v for k, v in b_after.items() if k[0] == "messages"} == {
        k: v for k, v in b_before.items() if k[0] == "messages"
    }


def test_each_owner_gets_their_own_counts(session: Session, make_owner: MakeOwner) -> None:
    a, b = make_owner("Owner A"), make_owner("Owner B")
    seed_sources(session, a.id, "alpha")
    seed_sources(session, b.id, "beta")
    session.add(MyPosition(owner_id=b.id, title="Extra role", company="Beta Co"))
    session.flush()

    sync_chunks(session, a.id, FakeEmbedder(), include_messages=True)
    sync_chunks(session, b.id, FakeEmbedder())

    repo_a, repo_b = Repository(session, a.id), Repository(session, b.id)
    assert repo_a.chunk_counts() == {
        "messages": 2,
        "my_positions": 1,
        "profiles": 1,
        "recommendations": 2,
    }
    assert repo_b.chunk_counts() == {"my_positions": 2, "profiles": 1, "recommendations": 2}
    assert repo_a.embedded_chunk_counts() == repo_a.chunk_counts()
    assert repo_a.count(Chunk) == 6 and repo_b.count(Chunk) == 5
    assert {c.id for c in repo_a.existing_chunks()}.isdisjoint(
        c.id for c in repo_b.existing_chunks()
    )
    assert not any("beta" in row.text for row in chunk_rows(session, a.id).values())
    assert not any("alpha" in row.text for row in chunk_rows(session, b.id).values())


def test_repository_reads_are_owner_scoped(session: Session, make_owner: MakeOwner) -> None:
    a, b = make_owner("Owner A"), make_owner("Owner B")
    seeded_a = seed_sources(session, a.id, "alpha")
    seeded_b = seed_sources(session, b.id, "beta")
    repo_a = Repository(session, a.id)

    sources = repo_a.chunk_sources(include_messages=True)
    assert [r.id for r in sources.recommendations] == seeded_a.recommendation_ids
    assert [p.id for p in sources.positions] == seeded_a.position_ids
    assert sources.profile is not None and sources.profile.id == seeded_a.profile_id
    assert [m.id for m in sources.messages] == seeded_a.message_ids
    assert repo_a.chunk_sources().messages == []

    sync_chunks(session, b.id, FakeEmbedder(), include_messages=True)
    assert repo_a.existing_chunks() == []
    assert repo_a.pending_chunks("another-model", 1536, include_messages=True) == []
    assert repo_a.chunk_counts() == {} and repo_a.embedded_chunk_counts() == {}
    pending_b = Repository(session, b.id).pending_chunks("another-model", 1536)
    assert len(pending_b) == CHUNKS_WITHOUT_MESSAGES
    assert seeded_b.owner_id == b.id


def test_writer_ignores_ids_that_belong_to_another_owner(
    session: Session, make_owner: MakeOwner
) -> None:
    a, b = make_owner("Owner A"), make_owner("Owner B")
    seed_sources(session, b.id, "beta")
    sync_chunks(session, b.id, FakeEmbedder(), include_messages=True)
    before = chunk_rows(session, b.id)
    b_ids = [row.id for row in before.values()]
    writer_a = ChunkWriter(session, a.id)

    assert writer_a.delete(b_ids) == 0
    assert writer_a.delete_source("messages") == 0
    writer_a.set_embeddings([(i, [0.5] * 1536) for i in b_ids], model="evil", dim=1536)
    writer_a.update_text([(i, ChunkDraft("profiles", 1, 0, "overwritten")) for i in b_ids])

    assert chunk_rows(session, b.id) == before
    with pytest.raises((TypeError, ValueError)):
        ChunkWriter(session, 0)
    with pytest.raises(TypeError):
        ChunkWriter(session, None)  # type: ignore[arg-type]
