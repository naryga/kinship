"""`run_eval` end to end in the test database, with the fake embedder."""

import pytest
from conftest import CI_FLOOR, REPO_ROOT, EvalOwners, add_decoys, add_owner
from sqlalchemy.orm import Session

from kinship.db.models import Owner
from kinship.ingest import import_export
from kinship.parser import ParsedExport
from kinship.rag import FakeEmbedder
from kinship.rag.evaluate import MRR_DEPTH, GoldenError, GoldenQuery, parse_golden, run_eval
from kinship.repo import Repository

START = "<!-- eval-results:fake:start -->"
END = "<!-- eval-results:fake:end -->"


def test_fake_embedder_report_is_sane(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    report = run_eval(Repository(session, eval_owners.a), FakeEmbedder(), golden)

    assert report.embedder == FakeEmbedder().model
    assert report.k == 5
    assert report.mrr_depth == MRR_DEPTH
    assert report.modes == ("fts", "vector", "hybrid")
    assert report.notes == []  # hybrid really was hybrid: no fall back to full-text
    assert [outcome.id for outcome in report.queries] == [query.id for query in golden]
    assert set(report.by_kind) == {"keyword", "question", "paraphrase"}
    assert sum(m["hybrid"].queries for m in report.by_kind.values()) == len(golden)

    # Exact words must be found by every retriever.
    assert report.by_kind["keyword"]["hybrid"].recall_at_k >= 0.9
    assert report.by_kind["keyword"]["fts"].recall_at_k >= 0.9
    assert report.by_kind["keyword"]["vector"].recall_at_k >= 0.9
    assert report.overall["hybrid"].recall_at_k >= CI_FLOOR
    for metrics in report.overall.values():
        assert 0.0 <= metrics.mean_recall <= metrics.recall_at_k <= 1.0
        assert 0.0 < metrics.mrr <= 1.0


def test_readme_table_matches_the_fake_embedder_report(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    """Regenerate with `make eval ARGS="--format markdown"` if this fails on purpose."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(START) == 1 and readme.count(END) == 1
    block = readme.split(START)[1].split(END)[0].strip()

    report = run_eval(Repository(session, eval_owners.a), FakeEmbedder(), golden)

    assert block == report.to_markdown()


def test_ci_floor_is_the_one_the_workflow_uses() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert f"uv run kinship eval --embedder fake --min-recall {CI_FLOOR}\n" in workflow


def test_another_owners_data_does_not_change_the_report(
    session: Session, v2_parsed: ParsedExport, golden: list[GoldenQuery]
) -> None:
    a = add_owner(session, "Solo A", v2_parsed)
    alone = run_eval(Repository(session, a), FakeEmbedder(), golden)

    # B: the same export, plus rows that contain every query word for word.
    b = add_owner(session, "Crowding B", v2_parsed)
    add_decoys(session, b, golden)
    crowded = run_eval(Repository(session, a), FakeEmbedder(), golden)

    assert crowded == alone
    assert crowded.to_json() == alone.to_json()


def test_k_changes_recall_but_not_mrr(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    repo = Repository(session, eval_owners.a)
    at_1 = run_eval(repo, FakeEmbedder(), golden, k=1, modes=["vector"])
    at_5 = run_eval(repo, FakeEmbedder(), golden, k=5, modes=["vector"])

    assert at_1.modes == ("vector",)
    assert at_1.overall["vector"].recall_at_k <= at_5.overall["vector"].recall_at_k
    assert at_1.overall["vector"].mrr == at_5.overall["vector"].mrr
    ranks = [outcome.ranks["vector"] for outcome in at_1.queries]
    assert at_1.overall["vector"].recall_at_k == sum(rank == 1 for rank in ranks) / len(ranks)


def test_sources_filter_is_passed_to_search(session: Session, eval_owners: EvalOwners) -> None:
    """`pipelines` is in a recommendation, two positions and the profile."""
    item = '{"source": "profiles"}'
    queries = parse_golden(
        [
            f'{{"id": "any", "kind": "keyword", "query": "pipelines", "expected": [{item}]}}',
            f'{{"id": "only", "kind": "keyword", "query": "pipelines", "expected": [{item}], '
            '"sources": ["profiles"]}',
        ]
    )

    report = run_eval(Repository(session, eval_owners.a), None, queries, modes=["fts"])

    assert report.embedder == "none"
    by_id = {outcome.id: outcome for outcome in report.queries}
    assert by_id["only"].ranks["fts"] == 1
    assert by_id["any"].ranks["fts"] is not None


def test_full_text_alone_needs_no_embedder_or_vectors(
    session: Session, v2_parsed: ParsedExport, golden: list[GoldenQuery]
) -> None:
    owner = Owner(display_name="Not embedded")
    session.add(owner)
    session.flush()
    import_export(session, owner.id, v2_parsed)
    repo = Repository(session, owner.id)

    # No chunks at all: full-text finds nothing, and the vector modes refuse to run.
    report = run_eval(repo, None, golden, modes=["fts"])
    assert report.overall["fts"].recall_at_k == 0.0
    with pytest.raises(ValueError, match="sync_chunks"):
        run_eval(repo, FakeEmbedder(), golden)
    with pytest.raises(ValueError, match="need an embedder"):
        run_eval(repo, None, golden)


def test_bad_arguments(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    repo = Repository(session, eval_owners.a)
    with pytest.raises(ValueError, match="k must be"):
        run_eval(repo, FakeEmbedder(), golden, k=0)
    with pytest.raises(ValueError, match="modes must be"):
        run_eval(repo, FakeEmbedder(), golden, modes=["telepathy"])
    with pytest.raises(ValueError, match="modes must be"):
        run_eval(repo, FakeEmbedder(), golden, modes=[])


def test_unresolvable_golden_item_stops_the_run(session: Session, eval_owners: EvalOwners) -> None:
    queries = parse_golden(
        [
            '{"id": "x", "kind": "keyword", "query": "anything", "expected": '
            '[{"source": "my_positions", "company": "Nowhere", "title": "Nothing"}]}'
        ]
    )

    with pytest.raises(GoldenError, match=r"x: .*Nowhere.*no row"):
        run_eval(Repository(session, eval_owners.a), FakeEmbedder(), queries)
