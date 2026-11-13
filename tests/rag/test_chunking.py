"""Chunk text formats and message packing (no database)."""

from datetime import UTC, date, datetime, timedelta

from kinship.rag.chunking import (
    MAX_CHUNK_TOKENS,
    approx_tokens,
    build_drafts,
    content_hash,
    message_chunks,
    position_chunk,
    profile_chunk,
    recommendation_chunk,
    split_text,
)
from kinship.repo.chunks import (
    ChunkSources,
    MessageSource,
    PositionSource,
    ProfileSource,
    RecommendationSource,
)

T0 = datetime(2024, 3, 5, 9, 0, tzinfo=UTC)


def msg(id: int, conv: str | None, body: str | None, *, minutes: int = 0, who: str = "Freya"):
    return MessageSource(
        id=id,
        conversation_id=conv,
        from_name=who,
        sent_at=T0 + timedelta(minutes=minutes),
        body=body,
    )


# -- recommendations, positions, profile -----------------------------------------


def test_recommendation_received_has_a_context_line() -> None:
    draft = recommendation_chunk(
        RecommendationSource(
            id=7,
            direction="received",
            first_name="Freya",
            last_name="Rookwood",
            company="Acme",
            job_title="Analytics Engineer",
            text="  Great with dbt.  ",
        )
    )
    assert draft is not None
    assert draft.key == ("recommendations", 7, 0)
    assert draft.text == (
        "Recommendation received from Freya Rookwood, Analytics Engineer at Acme:\nGreat with dbt."
    )
    assert draft.content_hash == content_hash(draft.text) and len(draft.content_hash) == 64


def test_recommendation_given_and_sparse_fields() -> None:
    given = recommendation_chunk(
        RecommendationSource(id=1, direction="given", first_name="Bo", company="Acme", text="x")
    )
    assert given is not None and given.text == "Recommendation given to Bo, Acme:\nx"
    bare = recommendation_chunk(RecommendationSource(id=2, direction="received", text="x"))
    assert bare is not None and bare.text == "Recommendation received:\nx"
    no_name = recommendation_chunk(
        RecommendationSource(id=3, direction="received", job_title="CTO", text="x")
    )
    assert no_name is not None and no_name.text == "Recommendation received (CTO):\nx"


def test_recommendation_without_text_is_skipped() -> None:
    assert recommendation_chunk(RecommendationSource(id=1, direction="received", text=" ")) is None
    assert recommendation_chunk(RecommendationSource(id=1, direction="received")) is None


def test_position_chunk_format() -> None:
    draft = position_chunk(
        PositionSource(
            id=3,
            company="Acme",
            title="Data Engineer",
            description="Built the warehouse.",
            location="Leeds",
            started_on=date(2020, 1, 15),
            ended_on=date(2022, 6, 1),
        )
    )
    assert draft is not None
    assert draft.key == ("my_positions", 3, 0)
    assert draft.text == (
        "Position: Data Engineer at Acme (Jan 2020 \N{EN DASH} Jun 2022), Leeds\n"
        "Built the warehouse."
    )


def test_position_current_and_minimal() -> None:
    current = position_chunk(PositionSource(id=1, title="CTO", started_on=date(2023, 12, 1)))
    assert current is not None and current.text == "Position: CTO (Dec 2023 \N{EN DASH} present)"
    company_only = position_chunk(PositionSource(id=2, company="Acme"))
    assert company_only is not None and company_only.text == "Position: Acme"
    assert position_chunk(PositionSource(id=3, location="Leeds")) is None


def test_profile_chunk_format() -> None:
    draft = profile_chunk(
        ProfileSource(id=9, headline="Data person", summary="I like data.", industry="Software")
    )
    assert draft is not None
    assert draft.key == ("profiles", 9, 0)
    assert draft.text == "Profile: Data person\nIndustry: Software\nI like data."
    assert profile_chunk(ProfileSource(id=9, headline=" ")) is None


# -- messages ----------------------------------------------------------------------


def test_messages_are_grouped_by_conversation_in_time_order() -> None:
    drafts = message_chunks(
        [
            msg(12, "b", "other thread", minutes=1, who="Bo"),
            msg(11, "a", "second", minutes=5, who="Me"),
            msg(10, "a", "first", minutes=0),
        ]
    )
    assert [d.key for d in drafts] == [("messages", 10, 0), ("messages", 12, 0)]
    assert drafts[0].text == "Freya (2024-03-05): first\nMe (2024-03-05): second"
    assert drafts[1].text == "Bo (2024-03-05): other thread"


def test_messages_without_text_or_conversation() -> None:
    drafts = message_chunks(
        [
            msg(1, None, "solo one"),
            msg(2, None, "solo two"),
            msg(3, "c", "   "),
            MessageSource(id=4, conversation_id="d", body="no sender or date"),
        ]
    )
    assert [(d.source_id, d.text) for d in drafts] == [
        (1, "Freya (2024-03-05): solo one"),
        (2, "Freya (2024-03-05): solo two"),
        (4, "Unknown: no sender or date"),
    ]


def test_conversation_is_split_near_500_tokens_on_message_boundaries() -> None:
    body = "word " * 60  # ~300 characters per message
    messages = [msg(i, "long", body, minutes=i) for i in range(1, 41)]
    drafts = message_chunks(messages)

    assert len(drafts) > 1
    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))
    assert {d.source_id for d in drafts} == {1}
    for draft in drafts:
        assert approx_tokens(draft.text) <= MAX_CHUNK_TOKENS
        # Whole messages only: every line is a complete "<name> (<date>): <body>".
        assert all(line.endswith("word") for line in draft.text.split("\n"))
    # Chunks are well filled, not one message each.
    assert approx_tokens(drafts[0].text) > MAX_CHUNK_TOKENS * 0.8
    # Nothing lost or reordered.
    lines = [line for d in drafts for line in d.text.split("\n")]
    assert len(lines) == 40


def test_one_oversized_message_is_split_on_sentences() -> None:
    sentence = "This sentence is about data pipelines and it ends here."
    drafts = message_chunks([msg(1, "c", " ".join([sentence] * 120))])

    assert len(drafts) > 1
    for draft in drafts:
        assert approx_tokens(draft.text) <= MAX_CHUNK_TOKENS
        for line in draft.text.split("\n"):
            assert line.startswith("Freya (2024-03-05): This sentence")
            assert line.endswith("ends here.")


def test_new_messages_only_change_the_last_chunk() -> None:
    body = "word " * 60
    before = message_chunks([msg(i, "long", body, minutes=i) for i in range(1, 21)])
    after = message_chunks([msg(i, "long", body, minutes=i) for i in range(1, 23)])
    assert [d.text for d in after[: len(before) - 1]] == [d.text for d in before[:-1]]
    assert {d.source_id for d in after} == {1}


def test_split_text() -> None:
    assert split_text("short", 10) == ["short"]
    pieces = split_text("One two. Three four! Five six? Seven", 12)
    assert pieces == ["One two.", "Three four!", "Five six?", "Seven"]
    assert split_text("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]
    assert all(len(p) <= 20 for p in split_text("lorem ipsum dolor sit amet " * 20, 20))


# -- everything together -------------------------------------------------------------


def test_build_drafts_covers_every_source() -> None:
    sources = ChunkSources(
        recommendations=[RecommendationSource(id=1, direction="received", text="Great.")],
        positions=[PositionSource(id=1, title="CTO")],
        profile=ProfileSource(id=1, headline="Hi"),
        messages=[msg(1, "c", "hello")],
    )
    drafts = build_drafts(sources)
    assert [d.source_table for d in drafts] == [
        "recommendations",
        "my_positions",
        "profiles",
        "messages",
    ]
    assert len({d.key for d in drafts}) == 4
    assert build_drafts(ChunkSources()) == []
