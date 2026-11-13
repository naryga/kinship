"""`Repository.find_companies` / `get_company`: matching, ordering, owner isolation."""

from __future__ import annotations

import pytest

from kinship.ingest import import_export
from kinship.repo import CompanyMatch, Repository
from kinship.synth import build_world

from .conftest import T1, T2, export, person

WORLD = build_world()


@pytest.fixture
def repo(session, owner, v1, v2) -> Repository:
    """An owner with both synthetic exports imported."""
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    return Repository(session, owner.id)


def test_gift_health_finds_every_spelling(repo: Repository) -> None:
    """The issue's acceptance example."""
    matches = repo.find_companies("gift health")
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, CompanyMatch)
    assert match.match == "exact"
    assert match.name_normalized == "gifthealth"
    assert {"Gifthealth", "Gift Health, Inc."} <= set(match.aliases)
    assert match.display_name in match.aliases
    assert match.connection_count == len(WORLD.connections_at("gifthealth", 2))
    assert match.connection_count > 0

    for query in ("Gifthealth", "GIFT HEALTH", "Gift Health, Inc.", " gift-health llc "):
        assert repo.find_companies(query) == matches, query


def test_exact_prefix_and_contains(repo: Repository) -> None:
    acme = repo.find_companies("acme")
    assert [(m.name_normalized, m.match) for m in acme] == [("acme", "exact")]
    assert set(acme[0].aliases) == {"Acme", "Acme, Inc.", "ACME Corp"}
    assert acme[0].connection_count == len(WORLD.connections_at("acme", 2))

    quill = repo.find_companies("quill")
    assert [(m.name_normalized, m.match) for m in quill] == [("quillfeathersystems", "prefix")]
    assert quill[0].connection_count == len(WORLD.connections_at("quillfeather", 2))

    feather = repo.find_companies("feather")
    assert [(m.name_normalized, m.match) for m in feather] == [("quillfeathersystems", "contains")]

    assert repo.find_companies("no such company") == []
    assert repo.find_companies("") == []
    assert repo.find_companies("   ") == []
    assert repo.find_companies("acme", limit=0) == []


def test_results_are_plain_data(repo: Repository) -> None:
    match = repo.find_companies("acme")[0]
    assert isinstance(match.aliases, tuple)
    assert match.as_dict() == {
        "id": match.id,
        "display_name": match.display_name,
        "name_normalized": "acme",
        "aliases": list(match.aliases),
        "connection_count": match.connection_count,
        "match": "exact",
    }
    with pytest.raises(AttributeError):
        match.display_name = "other"  # type: ignore[misc]

    assert repo.get_company(match.id) == match
    assert repo.get_company(10**9) is None


def test_ordering_limit_and_short_queries(session, owner) -> None:
    people = [
        person("a", "Health"),
        person("b", "Healthy Foods"),
        person("c", "Healthy Foods"),
        person("d", "Health Partners"),
        person("e", "Gift Health, Inc."),
        person("f", "Gifthealth"),
        person("g", "Gifthealth"),
        person("h", "Unified Health Group"),
        person("i", "A_B 100%"),
    ]
    import_export(session, owner.id, export("one", T1, people))
    repo = Repository(session, owner.id)

    found = repo.find_companies("health")
    assert [(m.display_name, m.match, m.connection_count) for m in found] == [
        ("Health", "exact", 1),
        ("Healthy Foods", "prefix", 2),
        ("Health Partners", "prefix", 1),
        ("Gifthealth", "contains", 3),
        ("Unified Health Group", "contains", 1),
    ]
    assert repo.find_companies("health", limit=2) == found[:2]

    # Under three characters: exact and prefix only.
    assert [m.display_name for m in repo.find_companies("he")] == [
        "Healthy Foods",
        "Health",
        "Health Partners",
    ]
    assert [m.display_name for m in repo.find_companies("un")] == ["Unified Health Group"]
    assert repo.find_companies("th") == []

    # LIKE wildcards in a query are literal.
    assert repo.find_companies("%") == []
    assert repo.find_companies("_") == []
    assert [m.display_name for m in repo.find_companies("100")] == ["A_B 100%"]


def test_connection_count_is_current_connections_only(session, owner) -> None:
    import_export(
        session,
        owner.id,
        export("one", T1, [person("a", "Acme"), person("b", "Acme"), person("c", "Initech")]),
    )
    # a is removed, c moves to Acme: Initech lives on through c's history only.
    import_export(
        session,
        owner.id,
        export("two", T2, [person("b", "Acme"), person("c", "Acme, Inc.")]),
    )
    repo = Repository(session, owner.id)
    (acme,) = repo.find_companies("acme")
    assert acme.connection_count == 2
    (initech,) = repo.find_companies("initech")
    assert initech.connection_count == 0
    assert repo.get_company(initech.id) == initech


def test_lookup_never_returns_another_owners_company(session, make_owner, v1, v2) -> None:
    ada, bob, cyd = make_owner("Ada"), make_owner("Bob"), make_owner("Cyd")
    import_export(session, ada.id, v1)
    import_export(session, bob.id, v1)
    import_export(session, bob.id, v2)
    import_export(session, bob.id, export("extra", T1, [person("z", "Only Bob Has This")]))
    repo_a, repo_b, repo_c = (Repository(session, o.id) for o in (ada, bob, cyd))

    (gift_a,) = repo_a.find_companies("gift health")
    (gift_b,) = repo_b.find_companies("gift health")
    assert gift_a.id != gift_b.id
    assert gift_a.name_normalized == gift_b.name_normalized == "gifthealth"
    # Counts are each owner's own: Ada has v1 only, Bob's current state is v2.
    assert gift_a.connection_count == len(WORLD.connections_at("gifthealth", 1))
    assert gift_b.connection_count == len(WORLD.connections_at("gifthealth", 2))
    assert gift_a.connection_count != gift_b.connection_count

    # The other owner's id is as good as missing.
    assert repo_a.get_company(gift_b.id) is None
    assert repo_b.get_company(gift_a.id) is None
    assert repo_a.get_company(gift_a.id) == gift_a

    assert repo_a.find_companies("only bob") == []
    assert [m.display_name for m in repo_b.find_companies("only bob")] == ["Only Bob Has This"]

    # Broad queries stay inside the owner; an owner with no data sees nothing.
    ids_a = {m.id for m in repo_a.find_companies("a", limit=1000)}
    ids_b = {m.id for m in repo_b.find_companies("a", limit=1000)}
    assert ids_a and ids_b and not ids_a & ids_b
    assert repo_c.find_companies("gift health") == []
    assert repo_c.find_companies("a", limit=1000) == []
    assert repo_c.get_company(gift_a.id) is None
