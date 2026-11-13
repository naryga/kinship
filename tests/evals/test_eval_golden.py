"""The committed golden set: well-formed, and every item resolves against the fixture."""

import json
from collections import Counter

import pytest
import sqlalchemy as sa
from conftest import EvalOwners
from sqlalchemy.orm import Session

from kinship.db.models import MyPosition, Profile, Recommendation
from kinship.rag.evaluate import (
    KINDS,
    GoldenError,
    GoldenItem,
    GoldenQuery,
    parse_golden,
    resolve_golden,
)
from kinship.repo import Repository

MODELS = {"recommendations": Recommendation, "my_positions": MyPosition, "profiles": Profile}


def line(**fields: object) -> str:
    base: dict[str, object] = {
        "id": "x1",
        "kind": "keyword",
        "query": "words",
        "expected": [{"source": "profiles"}],
    }
    return json.dumps({**base, **fields})


# -- the committed file -------------------------------------------------------


def test_golden_set_shape(golden: list[GoldenQuery]) -> None:
    assert 20 <= len(golden) <= 24
    assert len({query.id for query in golden}) == len(golden)
    assert len({query.query.casefold() for query in golden}) == len(golden)
    kinds = Counter(query.kind for query in golden)
    assert set(kinds) == set(KINDS)
    assert min(kinds.values()) >= 6  # roughly a third each
    assert all(1 <= len(query.expected) <= 3 for query in golden)
    # Messages are not embedded by default, so the golden set leaves them out.
    assert all(item.source in MODELS for query in golden for item in query.expected)


def test_every_golden_item_resolves_against_the_fixture(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    resolved = resolve_golden(Repository(session, eval_owners.a), golden)

    assert set(resolved) == {query.id for query in golden}
    for query in golden:
        assert len(resolved[query.id]) == len(query.expected)
        assert {table for table, _ in resolved[query.id]} == {i.source for i in query.expected}


# -- owner isolation ----------------------------------------------------------


def test_golden_items_resolve_only_to_the_repository_owners_rows(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    """B has the same export, so every natural key exists twice in the tables."""
    resolved = resolve_golden(Repository(session, eval_owners.a), golden)

    for keys in resolved.values():
        for table, source_id in keys:
            model = MODELS[table]
            owner_id = session.scalar(sa.select(model.owner_id).where(model.id == source_id))
            assert owner_id == eval_owners.a


def test_another_owners_rows_never_resolve(session: Session, eval_owners: EvalOwners) -> None:
    b_only = GoldenQuery(
        id="b1",
        kind="keyword",
        query="zeppelin",
        expected=(GoldenItem("recommendations", direction="given", name="Onlybee Zeppelin"),),
    )

    assert resolve_golden(Repository(session, eval_owners.b), [b_only])["b1"]
    with pytest.raises(GoldenError, match=r"b1.*Onlybee Zeppelin.*no row"):
        resolve_golden(Repository(session, eval_owners.a), [b_only])


def test_a_duplicate_natural_key_is_ambiguous_only_for_its_owner(
    session: Session, eval_owners: EvalOwners, golden: list[GoldenQuery]
) -> None:
    """B has a second "Freya Rookwood" and a second Copperline position; A does not."""
    with pytest.raises(GoldenError) as error:
        resolve_golden(Repository(session, eval_owners.b), golden)

    assert "Freya Rookwood" in str(error.value)
    assert "Copperline Freight" in str(error.value)
    assert "2 rows" in str(error.value)


# -- resolving ----------------------------------------------------------------


def test_unresolved_items_are_all_reported(session: Session, eval_owners: EvalOwners) -> None:
    queries = parse_golden(
        [
            line(id="ok"),
            line(
                id="gone",
                expected=[
                    {"source": "recommendations", "direction": "received", "name": "No Body"},
                    {"source": "my_positions", "company": "Nowhere", "title": "Nothing"},
                ],
            ),
            # The right name with the wrong direction is a different item.
            line(
                id="wrong-way",
                expected=[
                    {"source": "recommendations", "direction": "given", "name": "Freya Rookwood"}
                ],
            ),
        ]
    )

    with pytest.raises(GoldenError) as error:
        resolve_golden(Repository(session, eval_owners.a), queries)

    message = str(error.value)
    assert "No Body" in message and "Nowhere" in message and "wrong-way" in message
    assert "ok:" not in message
    assert "generator" in message


def test_matching_ignores_case_and_spacing(session: Session, eval_owners: EvalOwners) -> None:
    queries = parse_golden(
        [
            line(
                expected=[
                    {"source": "recommendations", "direction": "given", "name": " zoë  KINGSMOOR "},
                    {"source": "my_positions", "company": "acme, inc.", "title": "data engineer"},
                ]
            )
        ]
    )

    assert len(resolve_golden(Repository(session, eval_owners.a), queries)["x1"]) == 2


# -- parsing ------------------------------------------------------------------


def test_parse_golden_reads_all_fields() -> None:
    [query] = parse_golden(
        [
            "",
            line(
                kind="question",
                sources=["recommendations"],
                expected=[{"source": "recommendations", "direction": "given", "name": "A B"}],
            ),
        ]
    )

    assert query == GoldenQuery(
        id="x1",
        kind="question",
        query="words",
        expected=(GoldenItem("recommendations", direction="given", name="A B"),),
        sources=("recommendations",),
    )


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "no queries"),
        (["{not json"], "line 1: not valid JSON"),
        (["[1]"], "must be a JSON object"),
        ([line(), line()], "line 2: duplicate id 'x1'"),
        ([line(id="")], "id must be a non-empty string"),
        ([line(query="  ")], "query must be a non-empty string"),
        ([line(kind="vibes")], "kind must be one of"),
        ([line(expected=[])], "expected must be a non-empty list"),
        ([line(extra=1)], "unknown field"),
        ([line(expected=[{"source": "messages"}])], "source must be one of"),
        ([line(expected=[{"source": "profiles", "id": 3}])], "exactly the fields source"),
        ([line(expected=[{"source": "my_positions", "company": "Acme"}])], "exactly the fields"),
        (
            [line(expected=[{"source": "recommendations", "direction": "sideways", "name": "A"}])],
            "direction must be",
        ),
        (
            [line(expected=[{"source": "recommendations", "direction": "given", "name": " "}])],
            "name must be a non-empty string",
        ),
        ([line(expected=[{"source": "profiles"}, {"source": "profiles"}])], "same item twice"),
        ([line(sources=["nope"])], "unknown source"),
        ([line(sources=[])], "sources must be a non-empty list"),
        ([line(sources=["recommendations"])], "expects a profiles item but searches only"),
    ],
)
def test_parse_golden_rejects(lines: list[str], message: str) -> None:
    with pytest.raises(GoldenError, match=message):
        parse_golden(lines)
