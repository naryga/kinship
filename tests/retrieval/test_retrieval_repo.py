"""`RetrievalRepository`: full-text search, vector search, provenance, owner isolation."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from retrieval_seeding import B_ONLY_TEXT, B_ONLY_WORD, Corpus, chunk_ids, vec
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Message, MyPosition, Recommendation
from kinship.rag.embedder import FAKE_MODEL
from kinship.repo import ChunkHit, Repository
from kinship.repo.retrieval import MAX_QUERY_TERMS, query_terms

ODD_QUERIES = [
    "'; drop table chunks; --",
    "a & | !",
    '"unbalanced',
    "(((",
    ":* <-> !!",
    "\\",
    "\U0001f600 \U0001f680",
    "-",
    "the of and",
    "pipelines " * 2000,  # > 10k characters
    "x" * 10_000,
    "%s %(name)s :q $1",
]

QUERIES = ["data pipelines", "security audit", "healthcare billing", "Sam", B_ONLY_WORD]


def titles(repo: Repository, hits: list[ChunkHit]) -> list[str]:
    info = repo.source_info((h.source_table, h.source_id) for h in hits)
    return [info[(h.source_table, h.source_id)].title for h in hits]


# -- full-text ------------------------------------------------------------------


def test_fts_finds_a_recommendation_by_a_distinctive_phrase(
    session: Session, corpus: Corpus
) -> None:
    repo = Repository(session, corpus.a)

    hits = repo.fts_search("penetration test findings")

    top = hits[0]
    assert top.source_table == "recommendations"
    assert "penetration test findings" in top.text
    assert (top.rank, top.chunk_index) == (1, 0)
    assert top.score > 0
    assert titles(repo, hits[:1]) == ["Recommendation received from Casey Foxglove (2023-08-21)"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_fts_stems_words(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    assert "Kubernetes" in repo.fts_search("migrating service")[0].text


def test_fts_question_that_no_chunk_fully_contains_still_finds_partial_matches(
    session: Session, corpus: Corpus
) -> None:
    repo = Repository(session, corpus.a)
    question = "who praised my Kubernetes migration last winter?"

    # No chunk has every word, so the strict (AND) query alone finds nothing ...
    strict = session.scalar(
        sa.select(sa.func.count())
        .select_from(Chunk)
        .where(
            Chunk.owner_id == corpus.a,
            Chunk.tsv.bool_op("@@")(sa.func.websearch_to_tsquery("english", question)),
        )
    )
    assert strict == 0
    # ... and the OR query over its words finds the right chunk.
    assert "Kubernetes" in repo.fts_search(question)[0].text


def test_fts_puts_chunks_with_every_term_before_partial_matches(
    session: Session, corpus: Corpus
) -> None:
    repo = Repository(session, corpus.a)

    query = "data pipelines Airflow"
    with_every_term = set(
        session.scalars(
            sa.select(Chunk.id).where(
                Chunk.owner_id == corpus.a,
                Chunk.tsv.bool_op("@@")(sa.func.websearch_to_tsquery("english", query)),
            )
        )
    )

    hits = repo.fts_search(query)

    assert with_every_term
    assert len(hits) > len(with_every_term)
    assert {h.chunk_id for h in hits[: len(with_every_term)]} == with_every_term
    assert "Airflow" in hits[0].text


def test_fts_limit_and_no_words(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    assert len(repo.fts_search("Sam", limit=3)) == 3
    assert repo.fts_search("Sam", limit=0) == []
    assert repo.fts_search("   ") == []
    assert repo.fts_search("?!") == []


def test_query_terms_are_distinct_lowercase_and_capped() -> None:
    assert query_terms("Data data PIPELINES, pipelines!") == ["data", "pipelines"]
    assert len(query_terms(" ".join(f"w{i}" for i in range(500)))) == MAX_QUERY_TERMS
    assert query_terms("x" * 500) == []


@pytest.mark.parametrize("query", ODD_QUERIES, ids=range(len(ODD_QUERIES)))
def test_fts_survives_hostile_and_malformed_queries(
    session: Session, corpus: Corpus, query: str
) -> None:
    repo = Repository(session, corpus.a)
    before = chunk_ids(session, corpus.a)

    hits = repo.fts_search(query)

    assert {h.chunk_id for h in hits} <= before
    assert chunk_ids(session, corpus.a) == before


# -- vectors --------------------------------------------------------------------


def test_vector_search_finds_a_recommendation_by_overlapping_words(
    session: Session, corpus: Corpus
) -> None:
    repo = Repository(session, corpus.a)

    hits = repo.vector_search(vec("security audit and penetration testing"), model=FAKE_MODEL)

    assert "SOC 2 security audit" in hits[0].text
    assert hits[0].source_table == "recommendations"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert 0 < hits[0].score <= 1
    # Nearest-neighbour search returns every embedded chunk when there is room.
    assert len(hits) == repo.embedded_count(FAKE_MODEL) == len(chunk_ids(session, corpus.a))
    assert len(repo.vector_search(vec("security audit"), model=FAKE_MODEL, limit=2)) == 2


def test_sources_filter(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    query = "data pipelines healthcare billing"

    for sources in (["my_positions"], ["recommendations", "profiles"]):
        fts = repo.fts_search(query, sources=sources)
        vector = repo.vector_search(vec(query), model=FAKE_MODEL, sources=sources)
        assert fts and vector
        assert {h.source_table for h in fts} <= set(sources)
        assert {h.source_table for h in vector} == set(sources)

    assert repo.fts_search(query, sources=["messages"]) == []
    assert repo.vector_search(vec(query), model=FAKE_MODEL, sources=["messages"]) == []
    assert repo.embedded_count(FAKE_MODEL, sources=["my_positions"]) == 5
    assert repo.embedded_count(FAKE_MODEL, sources=["messages"]) == 0


def test_vector_search_ignores_other_models_and_missing_vectors(
    session: Session, corpus: Corpus
) -> None:
    repo = Repository(session, corpus.a)
    query = vec("SOC 2 security audit penetration test")
    best, second, *_ = repo.vector_search(query, model=FAKE_MODEL)
    total = repo.embedded_count(FAKE_MODEL)

    # The best chunk now carries a vector from another model; the second has none.
    session.execute(sa.update(Chunk).where(Chunk.id == best.chunk_id).values(model="other-model"))
    session.execute(
        sa.update(Chunk)
        .where(Chunk.id == second.chunk_id)
        .values(embedding=None, model=None, dim=None)
    )

    hits = repo.vector_search(query, model=FAKE_MODEL)
    assert {best.chunk_id, second.chunk_id}.isdisjoint(h.chunk_id for h in hits)
    assert len(hits) == repo.embedded_count(FAKE_MODEL) == total - 2

    other = repo.vector_search(query, model="other-model")
    assert [h.chunk_id for h in other] == [best.chunk_id]
    assert repo.embedded_count("other-model") == 1
    assert repo.vector_search(query, model="unknown-model") == []
    # Full-text search does not depend on vectors.
    assert best.chunk_id in {h.chunk_id for h in repo.fts_search("SOC 2 security audit")}


def test_vector_of_the_wrong_length_matches_nothing(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    assert repo.vector_search([1.0, 0.0, 0.0], model=FAKE_MODEL) == []
    assert repo.vector_search([], model=FAKE_MODEL) == []


def test_set_local_only_accepts_known_settings(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    repo.set_local("hnsw.ef_search", 200)
    assert session.scalar(sa.select(sa.func.current_setting("hnsw.ef_search"))) == "200"
    with pytest.raises(ValueError):
        repo.set_local("search_path", "public")
    with pytest.raises(ValueError):
        repo.set_local("row_security", "off")


# -- provenance -----------------------------------------------------------------


def test_source_info_describes_each_kind_of_source(session: Session, corpus: Corpus) -> None:
    repo = Repository(session, corpus.a)
    hits = repo.fts_search("Sam data engineer pipelines", limit=50)
    info = repo.source_info((h.source_table, h.source_id) for h in hits)

    by_table: dict[str, list[str]] = {}
    for item in info.values():
        by_table.setdefault(item.source_table, []).append(item.title)
    assert "Recommendation received from Freya Rookwood (2022-12-09)" in by_table["recommendations"]
    assert "Position: Senior Data Engineer at Gift Health, Inc." in by_table["my_positions"]
    assert by_table["profiles"] == [
        "Profile: Senior Data Engineer at Gifthealth | Data pipelines, healthcare billing"
    ]
    given = [t for t in by_table["recommendations"] if t.startswith("Recommendation given to ")]
    assert given

    assert repo.source_info([]) == {}
    assert repo.source_info([("recommendations", 0), ("no_such_table", 1)]) == {}


def test_source_info_for_a_conversation(session: Session, corpus: Corpus) -> None:
    first = session.scalars(
        sa.select(Message).where(Message.owner_id == corpus.a).order_by(Message.id).limit(1)
    ).one()
    info = Repository(session, corpus.a).source_info([("messages", first.id)])
    title = info[("messages", first.id)].title
    assert title.startswith(f"Conversation started by {first.from_name}")
    assert Repository(session, corpus.b).source_info([("messages", first.id)]) == {}


# -- owner isolation ------------------------------------------------------------


def test_owners_have_identical_texts_but_see_only_their_own_chunks(
    session: Session, corpus: Corpus
) -> None:
    a, b = Repository(session, corpus.a), Repository(session, corpus.b)
    ids_a, ids_b = chunk_ids(session, corpus.a), chunk_ids(session, corpus.b)
    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)
    assert len(ids_b) == len(ids_a) + 1

    for query in QUERIES:
        for repo, mine in ((a, ids_a), (b, ids_b)):
            fts = repo.fts_search(query)
            vector = repo.vector_search(vec(query), model=FAKE_MODEL)
            assert {h.chunk_id for h in fts} <= mine
            assert {h.chunk_id for h in vector} == mine  # every own chunk, nothing else

    # The same question gives both owners the same texts from different rows.
    hits_a, hits_b = a.fts_search("security audit"), b.fts_search("security audit")
    assert [h.text for h in hits_a] == [h.text for h in hits_b]
    assert {h.chunk_id for h in hits_a}.isdisjoint(h.chunk_id for h in hits_b)


def test_text_that_only_b_has_is_invisible_to_a(session: Session, corpus: Corpus) -> None:
    a, b = Repository(session, corpus.a), Repository(session, corpus.b)

    assert a.fts_search(B_ONLY_WORD) == []
    assert a.fts_search(B_ONLY_TEXT) != []  # common words match A's own chunks
    for hits in (
        a.fts_search(B_ONLY_TEXT),
        a.vector_search(vec(B_ONLY_TEXT), model=FAKE_MODEL),
    ):
        assert all(B_ONLY_WORD not in h.text for h in hits)

    assert B_ONLY_WORD in b.fts_search(B_ONLY_WORD)[0].text
    top = b.vector_search(vec(B_ONLY_TEXT), model=FAKE_MODEL)[0]
    assert B_ONLY_WORD in top.text
    assert top.score == pytest.approx(1.0, abs=0.2)


def test_source_info_does_not_resolve_another_owners_rows(session: Session, corpus: Corpus) -> None:
    b_recommendation = session.scalars(
        sa.select(Recommendation.id).where(
            Recommendation.owner_id == corpus.b, Recommendation.first_name == "Zed"
        )
    ).one()
    b_position = session.scalars(
        sa.select(MyPosition.id).where(MyPosition.owner_id == corpus.b).limit(1)
    ).one()
    keys = [("recommendations", b_recommendation), ("my_positions", b_position)]

    assert Repository(session, corpus.a).source_info(keys) == {}
    assert set(Repository(session, corpus.b).source_info(keys)) == set(keys)
