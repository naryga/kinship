"""The metric functions and the report rendering, on hand-made rankings. No database."""

import json

import pytest

from kinship.rag.evaluate import (
    EvalReport,
    Metrics,
    QueryOutcome,
    first_relevant_rank,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(["a", "b", "c"], {"a"}) == 1
    assert first_relevant_rank(["a", "b", "c"], {"c", "b"}) == 2
    assert first_relevant_rank(["a", "b", "c"], {"z"}) is None
    assert first_relevant_rank([], {"a"}) is None


def test_recall_at_k() -> None:
    ranking = ["a", "b", "c", "d"]
    assert recall_at_k(ranking, {"a"}, 1) == 1.0
    assert recall_at_k(ranking, {"d"}, 3) == 0.0
    assert recall_at_k(ranking, {"d"}, 4) == 1.0
    assert recall_at_k(ranking, {"a", "d"}, 3) == 0.5
    assert recall_at_k(ranking, {"a", "d", "z"}, 10) == pytest.approx(2 / 3)
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_recall_at_k_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError, match="k"):
        recall_at_k(["a"], {"a"}, 0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def outcome(id_: str, kind: str, rank: int | None, recall: float) -> QueryOutcome:
    return QueryOutcome(
        id=id_,
        kind=kind,
        query=f"query {id_}",
        expected=2,
        ranks={"fts": rank},
        recall={"fts": recall},
    )


def test_summarize() -> None:
    outcomes = [
        outcome("a", "keyword", 1, 1.0),  # hit at 1
        outcome("b", "keyword", 2, 0.5),  # hit at 2, one of two expected items found
        outcome("c", "question", 8, 0.0),  # outside the top 5, still counts for MRR
        outcome("d", "paraphrase", None, 0.0),  # miss
    ]

    metrics = summarize(outcomes, "fts", k=5)

    assert metrics.queries == 4
    assert metrics.recall_at_k == 0.5  # a and b
    assert metrics.mean_recall == pytest.approx((1.0 + 0.5) / 4)
    assert metrics.mrr == pytest.approx((1 + 1 / 2 + 1 / 8 + 0) / 4)


def test_summarize_counts_a_hit_exactly_at_k() -> None:
    assert summarize([outcome("a", "keyword", 5, 1.0)], "fts", k=5).recall_at_k == 1.0
    assert summarize([outcome("a", "keyword", 6, 0.0)], "fts", k=5).recall_at_k == 0.0


def test_summarize_of_nothing_is_zero() -> None:
    assert summarize([], "fts", k=5) == Metrics(
        queries=0, recall_at_k=0.0, mean_recall=0.0, mrr=0.0
    )


def report() -> EvalReport:
    outcomes = [
        QueryOutcome(
            "k1", "keyword", "alpha", 1, {"fts": 1, "hybrid": 1}, {"fts": 1.0, "hybrid": 1.0}
        ),
        QueryOutcome(
            "p1", "paraphrase", "beta", 1, {"fts": None, "hybrid": 4}, {"fts": 0.0, "hybrid": 1.0}
        ),
    ]
    modes = ("fts", "hybrid")
    return EvalReport(
        embedder="some-model",
        k=5,
        mrr_depth=20,
        modes=modes,
        overall={mode: summarize(outcomes, mode, 5) for mode in modes},
        by_kind={
            kind: {
                mode: summarize([o for o in outcomes if o.kind == kind], mode, 5) for mode in modes
            }
            for kind in ("keyword", "paraphrase")
        },
        queries=outcomes,
        notes=["p1 (hybrid): something to know"],
    )


def test_to_markdown() -> None:
    assert report().to_markdown().splitlines() == [
        "Embedder: `some-model` · 2 queries · recall@5 · MRR over the top 20",
        "",
        "| Queries | n | Full-text recall@5 | Full-text MRR | Hybrid recall@5 | Hybrid MRR |",
        "|---|---:|---:|---:|---:|---:|",
        "| all | 2 | 0.50 | 0.50 | 1.00 | 0.62 |",
        "| keyword | 1 | 1.00 | 1.00 | 1.00 | 1.00 |",
        "| paraphrase | 1 | 0.00 | 0.00 | 1.00 | 0.25 |",
    ]


def test_to_text_lists_every_query_and_note() -> None:
    text = report().to_text()

    assert "some-model" in text
    lines = text.splitlines()
    assert any(line.split()[:4] == ["k1", "keyword", "1", "1"] for line in lines)
    assert any(line.split()[:4] == ["p1", "paraphrase", "-", "4"] for line in lines)
    assert lines[-1] == "Note: p1 (hybrid): something to know"


def test_to_json_round_trips() -> None:
    data = json.loads(report().to_json())

    assert data["embedder"] == "some-model"
    assert data["k"] == 5
    assert data["modes"] == ["fts", "hybrid"]
    assert data["overall"]["hybrid"] == {
        "queries": 2,
        "recall_at_k": 1.0,
        "mean_recall": 1.0,
        "mrr": 0.625,
    }
    assert data["by_kind"]["paraphrase"]["fts"]["recall_at_k"] == 0.0
    assert data["queries"][1]["ranks"] == {"fts": None, "hybrid": 4}
