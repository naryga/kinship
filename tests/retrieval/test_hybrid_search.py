"""`kinship.rag.search.search`: fusion, modes, degradation, validation, owner isolation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import sqlalchemy as sa
from retrieval_seeding import B_ONLY_TEXT, B_ONLY_WORD, Corpus, chunk_ids
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Recommendation
from kinship.rag import EmbedderError, FakeEmbedder
from kinship.rag.chunking import SOURCE_TABLES
from kinship.rag.search import SearchResult, search
from kinship.repo import Repository

ODD_QUERIES = [
    "'; drop table chunks; --",
    "a & | !",
    '"unbalanced',
    "\U0001f600",
    "data pipelines " * 700,  # > 10k characters
    "%s %(name)s :q $1",
]


class RecordingEmbedder(FakeEmbedder):
    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        if model is not None:
            self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed(texts)


class BrokenEmbedder(FakeEmbedder):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbedderError("the embedding service is down")


def run(session: Session, owner_id: int, query: str, **kwargs: object) -> SearchResult:
    kwargs.setdefault("embedder", FakeEmbedder())
    embedder = kwargs.pop("embedder")
    return search(Repository(session, owner_id), embedder, query, **kwargs)  # type: ignore[arg-type]


# -- hybrid ---------------------------------------------------------------------


def test_hybrid_ranks_chunks_found_by_both_retrievers_first(
    session: Session, corpus: Corpus
) -> None:
    result = run(session, corpus.a, "Kubernetes Terraform migration", k=10)

    assert (result.mode, result.mode_used, result.note) == ("hybrid", "hybrid", None)
    top = result.hits[0]
    assert top.title == "Recommendation received from Oakley Nettlefold (2019-03-26)"
    assert (top.fts_rank, top.vector_rank) == (1, 1)
    assert top.matched == ("fts", "vector")
    assert top.source_table == "recommendations"
    assert "Kubernetes with Terraform" in top.text

    both = [len(hit.matched) == 2 for hit in result.hits]
    assert True in both and False in both  # found by both, and found by one only
    assert both == sorted(both, reverse=True)  # ... and every "both" comes first
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert len({hit.chunk_id for hit in result.hits}) == len(result.hits)


def test_hybrid_returns_k_hits_with_provenance(session: Session, corpus: Corpus) -> None:
    result = run(session, corpus.a, "healthcare billing claims", k=3)

    assert [hit.title for hit in result.hits] == [
        "Recommendation received from Vesper Lindenmoor (2021-12-22)",
        "Position: Senior Data Engineer at Gift Health, Inc.",
        "Profile: Senior Data Engineer at Gifthealth | Data pipelines, healthcare billing",
    ]
    assert [hit.source_table for hit in result.hits] == [
        "recommendations",
        "my_positions",
        "profiles",
    ]
    # source_table + source_id lead back to the row the chunk was built from.
    first = result.hits[0]
    row = session.get(Recommendation, first.source_id)
    assert row is not None and row.owner_id == corpus.a
    assert (row.first_name, row.last_name) == ("Vesper", "Lindenmoor")
    assert row.text in first.text


def test_fts_and_vector_modes(session: Session, corpus: Corpus) -> None:
    fts = run(session, corpus.a, "SOC 2 security audit", mode="fts")
    vector = run(session, corpus.a, "SOC 2 security audit", mode="vector")

    assert (fts.mode_used, vector.mode_used) == ("fts", "vector")
    assert [hit.fts_rank for hit in fts.hits] == list(range(1, len(fts.hits) + 1))
    assert all(hit.vector_rank is None and hit.matched == ("fts",) for hit in fts.hits)
    assert [hit.vector_rank for hit in vector.hits] == [1, 2, 3, 4, 5]
    assert all(hit.fts_rank is None and hit.matched == ("vector",) for hit in vector.hits)
    assert "SOC 2 security audit" in fts.hits[0].text
    assert fts.hits[0].chunk_id == vector.hits[0].chunk_id
    assert 0 < vector.hits[0].score <= 1


def test_fts_mode_does_not_use_the_embedder(session: Session, corpus: Corpus) -> None:
    embedder = RecordingEmbedder()
    run(session, corpus.a, "security audit", mode="fts", embedder=embedder)
    assert embedder.calls == []
    run(session, corpus.a, "security audit", embedder=embedder)
    assert embedder.calls == [["security audit"]]


def test_sources_filter(session: Session, corpus: Corpus) -> None:
    for mode in ("hybrid", "fts", "vector"):
        result = run(session, corpus.a, "data pipelines", sources=["my_positions"], mode=mode, k=20)
        assert result.hits
        assert {hit.source_table for hit in result.hits} == {"my_positions"}
        assert all(hit.title.startswith("Position: ") for hit in result.hits)

    everything = run(session, corpus.a, "data pipelines", k=20)
    assert run(session, corpus.a, "data pipelines", sources=[], k=20) == everything
    assert run(session, corpus.a, "data pipelines", sources=list(SOURCE_TABLES), k=20).hits == (
        everything.hits
    )


def test_chunks_unrelated_to_the_query_are_not_returned(session: Session, corpus: Corpus) -> None:
    """Nearest-neighbour search alone would return k chunks for any query."""
    assert run(session, corpus.a, "\U0001f600").hits == []
    assert run(session, corpus.a, "\U0001f600", mode="vector").hits == []


# -- degradation ----------------------------------------------------------------


def test_hybrid_without_an_embedder_is_full_text_only(session: Session, corpus: Corpus) -> None:
    result = run(session, corpus.a, "SOC 2 security audit", embedder=None)

    assert (result.mode, result.mode_used) == ("hybrid", "fts")
    assert result.note == "No embedder is configured; results are from full-text search only."
    assert result.hits == run(session, corpus.a, "SOC 2 security audit", mode="fts").hits
    assert all(hit.vector_rank is None for hit in result.hits)


def test_hybrid_with_no_vectors_from_this_model_is_full_text_only(
    session: Session, corpus: Corpus
) -> None:
    embedder = RecordingEmbedder(model="another-model")

    result = run(session, corpus.a, "SOC 2 security audit", embedder=embedder)

    assert result.mode_used == "fts"
    assert "another-model" in (result.note or "")
    assert "full-text search only" in (result.note or "")
    assert "SOC 2 security audit" in result.hits[0].text
    assert embedder.calls == []  # no point embedding a query nothing can be compared with

    # The same when the owner's chunks have no vectors at all ...
    session.execute(
        sa.update(Chunk)
        .where(Chunk.owner_id == corpus.a)
        .values(embedding=None, model=None, dim=None)
    )
    result = run(session, corpus.a, "SOC 2 security audit")
    assert (result.mode_used, bool(result.hits)) == ("fts", True)
    # ... even though the other owner still has them.
    assert run(session, corpus.b, "SOC 2 security audit").mode_used == "hybrid"

    vector_only = run(session, corpus.a, "SOC 2 security audit", mode="vector")
    assert (vector_only.mode_used, vector_only.hits) == ("vector", [])
    assert vector_only.note and "full-text" not in vector_only.note


def test_hybrid_degrades_per_source(session: Session, corpus: Corpus) -> None:
    session.execute(
        sa.update(Chunk)
        .where(Chunk.owner_id == corpus.a, Chunk.source_table == "my_positions")
        .values(embedding=None, model=None, dim=None)
    )
    assert run(session, corpus.a, "data", sources=["my_positions"]).mode_used == "fts"
    assert run(session, corpus.a, "data", sources=["recommendations"]).mode_used == "hybrid"
    assert run(session, corpus.a, "data").mode_used == "hybrid"


def test_a_failing_embedder_degrades_hybrid_but_not_vector_mode(
    session: Session, corpus: Corpus
) -> None:
    result = run(session, corpus.a, "SOC 2 security audit", embedder=BrokenEmbedder())
    assert result.mode_used == "fts"
    assert result.note == "Embedding the query failed; results are from full-text search only."
    assert result.hits

    with pytest.raises(EmbedderError):
        run(session, corpus.a, "SOC 2 security audit", mode="vector", embedder=BrokenEmbedder())
    with pytest.raises(ValueError, match="needs an embedder"):
        run(session, corpus.a, "SOC 2 security audit", mode="vector", embedder=None)


# -- validation and odd input ---------------------------------------------------


def test_unknown_sources_are_rejected_with_the_valid_ones(session: Session, corpus: Corpus) -> None:
    with pytest.raises(ValueError) as error:
        run(session, corpus.a, "data", sources=["recommendations", "connections"])
    message = str(error.value)
    assert "'connections'" in message
    for source in SOURCE_TABLES:
        assert source in message
    with pytest.raises(ValueError):
        run(session, corpus.a, "data", sources="recommendations")


def test_bad_arguments(session: Session, corpus: Corpus) -> None:
    for query in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="empty"):
            run(session, corpus.a, query)
    with pytest.raises(ValueError, match="mode"):
        run(session, corpus.a, "data", mode="bm25")
    with pytest.raises(ValueError, match="k must"):
        run(session, corpus.a, "data", k=0)


def test_candidates_never_limit_below_k(session: Session, corpus: Corpus) -> None:
    result = run(session, corpus.a, "Sam", k=8, candidates=2)
    assert len(result.hits) == 8


@pytest.mark.parametrize("mode", ["hybrid", "fts", "vector"])
@pytest.mark.parametrize("query", ODD_QUERIES, ids=range(len(ODD_QUERIES)))
def test_hostile_and_malformed_queries_do_not_error(
    session: Session, corpus: Corpus, query: str, mode: str
) -> None:
    before = chunk_ids(session, corpus.a)

    result = run(session, corpus.a, query, mode=mode, k=20)

    assert {hit.chunk_id for hit in result.hits} <= before
    assert chunk_ids(session, corpus.a) == before
    assert result.query == query.strip()


# -- owner isolation ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "fts", "vector"])
def test_search_returns_only_the_owners_chunks(session: Session, corpus: Corpus, mode: str) -> None:
    ids_a, ids_b = chunk_ids(session, corpus.a), chunk_ids(session, corpus.b)

    for query in ("data pipelines", "SOC 2 security audit", "Sam", B_ONLY_TEXT, B_ONLY_WORD):
        a = run(session, corpus.a, query, mode=mode, k=50)
        b = run(session, corpus.b, query, mode=mode, k=50)
        assert {hit.chunk_id for hit in a.hits} <= ids_a
        assert {hit.chunk_id for hit in b.hits} <= ids_b
        assert all(B_ONLY_WORD not in hit.text for hit in a.hits)
        assert all("Onlybee" not in hit.title for hit in a.hits)
        # Source ids point at the owner's own rows too.
        owners = set(
            session.scalars(
                sa.select(Recommendation.owner_id).where(
                    Recommendation.id.in_(
                        [h.source_id for h in a.hits if h.source_table == "recommendations"]
                    )
                )
            )
        )
        assert owners <= {corpus.a}

    b = run(session, corpus.b, B_ONLY_TEXT, mode=mode)
    assert B_ONLY_WORD in b.hits[0].text
    assert b.hits[0].title == "Recommendation received from Zed Onlybee"


def test_identical_texts_resolve_to_each_owners_own_rows(session: Session, corpus: Corpus) -> None:
    a = run(session, corpus.a, "SOC 2 security audit", k=3)
    b = run(session, corpus.b, "SOC 2 security audit", k=3)

    assert len(a.hits) == 3
    assert [hit.text for hit in a.hits] == [hit.text for hit in b.hits]
    assert [hit.title for hit in a.hits] == [hit.title for hit in b.hits]
    assert {hit.chunk_id for hit in a.hits}.isdisjoint(hit.chunk_id for hit in b.hits)
    keys_a = {(hit.source_table, hit.source_id) for hit in a.hits}
    assert keys_a.isdisjoint((hit.source_table, hit.source_id) for hit in b.hits)
