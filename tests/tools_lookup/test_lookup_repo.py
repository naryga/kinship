"""`ConnectionsRepository` against the synthetic truth, edge cases, and owner isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from companies.conftest import T1, T2, T3, URL, export, person

from kinship.db.models import Connection, Message
from kinship.ingest import import_export
from kinship.parser import records as r
from kinship.repo import Repository
from kinship.repo.connections import MAX_CANDIDATES, MAX_SEARCH_LIMIT
from kinship.synth import V1_EXPORTED_AT, V2_EXPORTED_AT

from .conftest import B_ONLY_COMPANY, B_ONLY_PEOPLE, WORLD, TwoOwners

TRUTH = WORLD.truth
V1 = {c.url: c for c in WORLD.connections(1) if c.url}
V2 = {c.url: c for c in WORLD.connections(2) if c.url}
RECORD_KEYS = {"name", "title", "company", "profile_url", "connected_on"}


def identity(person) -> tuple[str, str | None]:
    """(name, url) of a synthetic connection or of a result record."""
    if isinstance(person, dict):
        return person["name"], person.get("profile_url")
    return f"{person.first_name} {person.last_name}".strip(), person.url or None


def identities(people) -> list[tuple[str, str | None]]:
    return sorted(map(identity, people), key=repr)


def at_company(repo: Repository, name: str, include_former: bool = False) -> list[dict]:
    resolution = repo.resolve_company(name)
    return repo.connections_at_companies(resolution.company_ids, include_former)


# -- find_connections_at_company ---------------------------------------------------


@pytest.mark.parametrize("query", ["gift health", "Gifthealth", "GIFT HEALTH, INC."])
def test_gift_health_returns_exactly_the_current_connections(repo: Repository, query: str) -> None:
    """The issue's acceptance example: every spelling, nobody else."""
    resolution = repo.resolve_company(query)
    assert [m.match for m in resolution.matched] == ["exact"]
    (matched,) = resolution.matched_companies()
    assert {matched["display_name"], *matched["aliases"]} == set(WORLD.company_groups["gifthealth"])
    assert matched["display_name"] not in matched["aliases"]

    people = repo.connections_at_companies(resolution.company_ids)
    expected = WORLD.connections_at("gifthealth", 2)
    assert len(expected) > 1
    assert identities(people) == identities(expected)
    assert {p["status"] for p in people} == {"current"}
    assert {p["company"] for p in people} == {matched["display_name"]}
    for record in people:
        assert RECORD_KEYS | {"status"} >= set(record)


def test_every_company_matches_the_truth_with_and_without_former(repo: Repository) -> None:
    moved_away_total = 0
    for key, spellings in WORLD.company_groups.items():
        current = identities(WORLD.connections_at(key, 2))
        moved_away = sorted(u for u in TRUTH.company_changed if V1[u].company_key == key)
        moved_away_total += len(moved_away)

        assert identities(at_company(repo, spellings[0])) == current, key

        people = at_company(repo, spellings[0], include_former=True)
        assert identities(p for p in people if p["status"] == "current") == current, key
        former = [p for p in people if p["status"] == "former"]
        assert sorted(p["profile_url"] for p in former) == moved_away, key
        assert people == [p for p in people if p["status"] == "current"] + former
        for record in former:
            now, before = V2[record["profile_url"]], V1[record["profile_url"]]
            assert record["left_by"] == V2_EXPORTED_AT.isoformat()
            assert record["title"] == now.position
            assert record["former_title"] == before.position
            assert record["former_company"] in WORLD.company_groups[key]
            assert record["company"] in WORLD.company_groups[now.company_key]
            assert "removed" not in record
    assert moved_away_total == len(TRUTH.company_changed) > 0


def test_renamed_and_removed_people_are_not_former(repo: Repository) -> None:
    """A new spelling of the same company is not a move; a removed connection is not current."""
    assert TRUTH.company_renamed and TRUTH.removed
    for url in TRUTH.company_renamed:
        key = V2[url].company_key
        people = at_company(repo, WORLD.company_groups[key][0], include_former=True)
        assert [p["status"] for p in people if p.get("profile_url") == url] == ["current"]
    for url in TRUTH.removed:
        people = at_company(repo, V1[url].company, include_former=True)
        assert url not in {p.get("profile_url") for p in people}


def test_former_and_removed_people(session, owner) -> None:
    """Hand-built history: moved away, moved back, moved twice, removed, lost their company."""
    first = [
        person("stay", "Acme"),
        person("left", "Acme", "Analyst"),
        person("back", "Acme"),
        person("twice", "Acme"),
        person("gone", "Acme"),
        person("leftgone", "Acme, Inc."),
        person("jobless", "ACME Corp"),
    ]
    second = [
        person("stay", "Acme"),
        person("left", "Beta", "Manager"),
        person("back", "Beta"),
        person("twice", "Beta"),
        person("gone", "Acme"),
        person("leftgone", "Beta"),
        person("jobless", None, None),
    ]
    third = [
        person("stay", "Acme"),
        person("left", "Beta", "Manager"),
        person("back", "Acme"),
        person("twice", "Gamma"),
        person("jobless", None, None),
    ]
    # Imported out of order on purpose: dates come from `exported_at`, not load order.
    import_export(session, owner.id, export("three", T3, third))
    import_export(session, owner.id, export("one", T1, first))
    import_export(session, owner.id, export("two", T2, second))
    repo = Repository(session, owner.id)

    people = {p["profile_url"].removeprefix(URL): p for p in at_company(repo, "acme", True)}
    assert {slug: p["status"] for slug, p in people.items()} == {
        "stay": "current",
        "back": "current",
        "left": "former",
        "twice": "former",
        "leftgone": "former",
        "jobless": "former",
    }
    assert people["left"] == {
        "name": "Left Test",
        "title": "Manager",
        "company": "Beta",
        "profile_url": URL + "left",
        "connected_on": "2020-01-01",
        "status": "former",
        "former_company": "Acme",
        "former_title": "Analyst",
        "left_by": "2026-02-01",
    }
    assert (people["twice"]["company"], people["twice"]["left_by"]) == ("Gamma", "2026-02-01")
    assert people["leftgone"]["removed"] is True
    assert people["leftgone"]["company"] == "Beta"
    assert "company" not in people["jobless"] and "title" not in people["jobless"]
    assert people["jobless"]["left_by"] == "2026-02-01"
    # "gone" was last seen at Acme and then disappeared: neither current nor known to have left.
    assert "gone" not in people
    assert [p["status"] for p in at_company(repo, "acme")] == ["current", "current"]

    beta = {p["profile_url"].removeprefix(URL): p for p in at_company(repo, "beta", True)}
    assert {slug: p["status"] for slug, p in beta.items()} == {
        "left": "current",
        "back": "former",
        "twice": "former",
    }
    assert beta["back"]["left_by"] == "2026-03-01"
    assert beta["twice"]["left_by"] == "2026-03-01"

    detail = repo.lookup_connection(URL + "back").connection
    assert detail is not None
    assert detail["role_history"] == [
        {"company": "Acme", "title": "Engineer", "as_of": "2026-01-01"},
        {"company": "Beta", "title": "Engineer", "as_of": "2026-02-01"},
        {"company": "Acme", "title": "Engineer", "as_of": "2026-03-01"},
    ]


def test_company_resolution(repo: Repository) -> None:
    quill = repo.resolve_company("quill")
    assert [m.match for m in quill.matched] == ["prefix"]
    assert len(at_company(repo, "quill")) == len(WORLD.connections_at("quillfeather", 2))

    # Without an exact match, every company of the best kind is used.
    health = repo.resolve_company("health")
    assert {m.match for m in health.matched} == {"contains"}
    assert len(health.matched) >= 2
    companies = {p["company"] for p in at_company(repo, "health")}
    assert companies == {m.display_name for m in health.matched if m.connection_count}

    typo = repo.resolve_company("Gifthelth")
    assert typo.matched == () and typo.company_ids == []
    assert typo.suggestions == (repo.resolve_company("gift health").matched[0].display_name,)
    assert repo.resolve_company("helth gift").suggestions == typo.suggestions

    for nothing in ("", "   ", "no such company anywhere", "%", "_"):
        resolution = repo.resolve_company(nothing)
        assert resolution.matched == () and resolution.suggestions == (), nothing
        assert repo.connections_at_companies(resolution.company_ids, True) == []


# -- search_connections --------------------------------------------------------------


def test_search_matches_names_titles_and_companies(repo: Repository) -> None:
    current = WORLD.connections(2)

    everyone = repo.search_connections(limit=MAX_SEARCH_LIMIT)
    assert everyone.total == len(current) == len(everyone.connections)
    assert identities(everyone.connections) == identities(current)
    assert all(set(record) <= RECORD_KEYS for record in everyone.connections)

    elio = repo.search_connections("ELIO")
    assert sorted(p["name"] for p in elio.connections) == sorted(
        f"{c.first_name} {c.last_name}" for c in current if "elio" in c.first_name.lower()
    )
    assert repo.search_connections("emberly  elio").connections == [
        p for p in elio.connections if p["name"] == "Elio Emberly"
    ]

    engineers = [c for c in current if "engineer" in c.position.lower()]
    found = repo.search_connections("engineer", limit=MAX_SEARCH_LIMIT)
    assert identities(found.connections) == identities(engineers)

    # Every word must match, each in any of name, title or company (any spelling).
    there = [c for c in engineers if c.company_key == "gifthealth"]
    assert len(there) > 1
    for query in ("engineer gift", "gifthealth ENGINEER", "health engineer gift"):
        found = repo.search_connections(query, limit=MAX_SEARCH_LIMIT)
        assert identities(found.connections) == identities(there), query
    assert repo.search_connections("engineer nosuchword").total == 0


def test_search_filters(repo: Repository) -> None:
    current = WORLD.connections(2)
    managers_at_acme = [
        c for c in current if c.company_key == "acme" and "manager" in c.position.lower()
    ]
    assert managers_at_acme
    found = repo.search_connections(title="MANAGER", company="acme corp")
    assert identities(found.connections) == identities(managers_at_acme)
    assert found.company is not None
    assert [m.name_normalized for m in found.company.matched] == ["acme"]

    by_company = repo.search_connections(company="gift health", limit=MAX_SEARCH_LIMIT)
    assert identities(by_company.connections) == identities(WORLD.connections_at("gifthealth", 2))

    # A company that resolves to nothing (here: a two-letter key that is nobody's prefix)
    # falls back to a substring of the raw name.
    inc = [c for c in WORLD.connections_at("gifthealth", 2) if "h, i" in c.company.lower()]
    odd = repo.search_connections(company="h, I")
    assert odd.company is not None and odd.company.matched == ()
    assert identities(odd.connections) == identities(inc)
    assert 0 < odd.total < by_company.total

    assert repo.search_connections("elio", title="nosuchtitle").total == 0
    assert repo.search_connections(title="  ", company=" ").total == len(current)


def test_search_excludes_removed_connections(repo: Repository) -> None:
    for url in TRUTH.removed:
        gone = V1[url]
        found = repo.search_connections(f"{gone.first_name} {gone.last_name}")
        assert url not in {p.get("profile_url") for p in found.connections}


def test_search_order_limit_and_total(repo: Repository) -> None:
    everyone = repo.search_connections(limit=MAX_SEARCH_LIMIT).connections
    by_name = {identity(c): (c.last_name, c.first_name) for c in WORLD.connections(2)}
    keys = [by_name[identity(p)] for p in everyone]
    assert keys == sorted(keys)

    page = repo.search_connections(limit=3)
    assert page.total == len(everyone) and page.limit == 3
    assert page.connections == everyone[:3]

    assert repo.search_connections(limit=0).limit == 1
    assert len(repo.search_connections(limit=-5).connections) == 1
    assert repo.search_connections(limit=10**6).limit == MAX_SEARCH_LIMIT


def test_search_puts_name_matches_first_and_escapes_wildcards(session, owner) -> None:
    people = [
        person("zed", "Miller Brewing"),
        person("amy", "Acme", "Miller"),
        replace(person("bob", "Acme"), last_name="Miller"),
        person("pct", "100% Pure_Juice", "Taster"),
    ]
    import_export(session, owner.id, export("one", T1, people))
    repo = Repository(session, owner.id)

    found = repo.search_connections("miller")
    assert [p["name"] for p in found.connections] == ["Bob Miller", "Amy Test", "Zed Test"]

    assert repo.search_connections("%").total == 1
    assert repo.search_connections("_").total == 1
    assert repo.search_connections("100%").connections[0]["name"] == "Pct Test"
    assert repo.search_connections("a_y").total == 0
    assert repo.search_connections(title="%").total == 0
    assert repo.search_connections(title="_aster").total == 0
    # Too short to resolve as a company, so these fall back to the raw-name substring.
    assert repo.search_connections(company="0% P").total == 1
    assert repo.search_connections(company="0_ P").total == 0


# -- lookup_connection ----------------------------------------------------------------


def test_lookup_by_url_in_any_form(repo: Repository) -> None:
    url = WORLD.messages[0].sender_url
    person_ = V2[url]
    slug = url.rsplit("/", 1)[1]
    forms = (url, url + "/", f"http://linkedin.com/in/{slug.upper()}?trk=x", f"  www.{url[12:]} ")
    found = repo.lookup_connection(url)
    assert found.status == "found" and found.connection is not None
    assert found.connection["profile_url"] == url
    for form in (*forms, slug, f"{person_.first_name} {person_.last_name}".upper()):
        assert repo.lookup_connection(form) == found, form

    missing = repo.lookup_connection("https://www.linkedin.com/in/nobody-at-all")
    assert (missing.status, missing.suggestions) == ("not_found", ())


def test_connection_detail_matches_the_synthetic_world(repo: Repository) -> None:
    def count(events, url: str) -> int:
        return sum(1 for e in WORLD.visible(events, 2) if e.url == url)

    checked = set()
    for url in {*TRUTH.company_changed, *TRUTH.title_changed, WORLD.messages[0].sender_url}:
        now = V2[url]
        detail = repo.lookup_connection(url).connection
        assert detail is not None
        assert detail["name"] == f"{now.first_name} {now.last_name}"
        assert detail["title"] == now.position
        assert detail["company"] in WORLD.company_groups[now.company_key]
        assert detail["connected_on"] == now.connected_on.isoformat()
        assert detail.get("email") == (now.email or None)
        assert "removed" not in detail and "removed_on" not in detail

        roles = detail["role_history"]
        if url in TRUTH.company_changed or url in TRUTH.title_changed:
            before = V1[url]
            assert [role["as_of"] for role in roles] == [
                V1_EXPORTED_AT.isoformat(),
                V2_EXPORTED_AT.isoformat(),
            ]
            assert [role["title"] for role in roles] == [before.position, now.position]
            assert roles[0]["company"] in WORLD.company_groups[before.company_key]
            assert roles[1]["company"] in WORLD.company_groups[now.company_key]
            checked.add("history")

        messages = [
            m for m in WORLD.visible(WORLD.messages, 2) if url in (m.sender_url, m.recipient_urls)
        ]
        assert detail["message_count"] == len(messages)
        if messages:
            assert detail["last_contact"] == max(m.at for m in messages).date().isoformat()
            checked.add("messages")
        else:
            assert "last_contact" not in detail

        endorsements = detail.get("endorsements", {})
        assert len(endorsements.get("received", [])) == count(WORLD.endorsements_received, url)
        assert len(endorsements.get("given", [])) == count(WORLD.endorsements_given, url)
        assert len(detail.get("notes", [])) == count(WORLD.notes, url)
    assert checked == {"history", "messages"}


def test_connection_detail_events(repo: Repository) -> None:
    """Notes, endorsements, recommendations and the invitation, for people who have them."""
    note = WORLD.notes[0]
    detail = repo.lookup_connection(note.url).connection
    assert detail is not None
    expected = {"note": note.text, "date": (note.edited_at or note.at).date().isoformat()}
    assert expected in detail["notes"]

    endorsement = WORLD.endorsements_received[0]
    detail = repo.lookup_connection(endorsement.url).connection
    assert detail is not None
    expected = {"skill": endorsement.skill, "date": endorsement.at.date().isoformat()}
    assert expected in detail["endorsements"]["received"]

    for direction, recommendations in (
        ("received", WORLD.recommendations_received),
        ("given", WORLD.recommendations_given),
    ):
        recommendation = recommendations[0]
        name = f"{recommendation.first_name} {recommendation.last_name}"
        detail = repo.lookup_connection(name).connection
        assert detail is not None, name
        (item,) = detail["recommendations"][direction]
        assert item["date"] == recommendation.at.date().isoformat()
        assert len(item["text"]) <= 300
        assert " ".join(recommendation.text.split()).startswith(item["text"].rstrip("…"))

    invitation = next(i for i in WORLD.visible(WORLD.invitations, 2) if i.message)
    outgoing = invitation.direction == "OUTGOING"
    detail = repo.lookup_connection(
        invitation.invitee_url if outgoing else invitation.inviter_url
    ).connection
    assert detail is not None
    assert detail["invitation"] == {
        "direction": invitation.direction.lower(),
        "date": invitation.at.date().isoformat(),
        "message": invitation.message,
    }


def test_long_recommendations_are_truncated(session, owner) -> None:
    recommendation = r.Recommendation(
        first_name="Amy",
        last_name="Test",
        company="Acme",
        job_title="Engineer",
        text="word " * 200,
        created_at=datetime(2025, 1, 2, 23, 30, tzinfo=UTC),
        status="VISIBLE",
    )
    import_export(
        session,
        owner.id,
        export("one", T1, [person("amy", "Acme")], recommendations_received=[recommendation]),
    )
    detail = Repository(session, owner.id).lookup_connection("amy test").connection
    assert detail is not None
    (item,) = detail["recommendations"]["received"]
    assert len(item["text"]) <= 300 and item["text"].endswith("…")
    assert item["date"] == "2025-01-02"
    assert "given" not in detail["recommendations"]
    assert detail["message_count"] == 0
    assert "last_contact" not in detail and "notes" not in detail and "invitation" not in detail


def test_lookup_of_a_removed_connection(repo: Repository) -> None:
    url = TRUTH.removed[0]
    detail = repo.lookup_connection(url).connection
    assert detail is not None
    assert detail["removed"] is True
    assert detail["removed_on"] == V2_EXPORTED_AT.isoformat()
    assert [role["as_of"] for role in detail["role_history"]] == [V1_EXPORTED_AT.isoformat()]


def test_lookup_by_name_ambiguous_and_missing(repo: Repository) -> None:
    elios = sorted(
        f"{c.first_name} {c.last_name}" for c in WORLD.connections(2) if c.first_name == "Elio"
    )
    assert len(elios) > 1

    several = repo.lookup_connection("elio")
    assert several.status == "ambiguous" and several.connection is None
    assert sorted(p["name"] for p in several.candidates) == elios
    assert not several.more_candidates
    assert all(set(p) <= RECORD_KEYS for p in several.candidates)

    # Part of a name is enough when it identifies one person.
    one = repo.lookup_connection("emberly")
    assert one.status == "found"
    assert one.connection is not None and one.connection["name"] == "Elio Emberly"

    none = repo.lookup_connection("Elio Nosuchperson")
    assert none.status == "not_found" and none.connection is None
    assert sorted(p["name"] for p in none.suggestions) == elios

    for nothing in ("", "   ", "zzzzzz", "%", "_"):
        found = repo.lookup_connection(nothing)
        assert (found.status, found.candidates, found.suggestions) == ("not_found", (), ())


def test_lookup_lists_at_most_max_candidates(session, owner) -> None:
    people = [person(f"p{i:02}", "Acme") for i in range(MAX_CANDIDATES + 2)]
    import_export(session, owner.id, export("one", T1, people))
    found = Repository(session, owner.id).lookup_connection("test")
    assert found.status == "ambiguous"
    assert len(found.candidates) == MAX_CANDIDATES and found.more_candidates


# -- owner isolation -------------------------------------------------------------------


def test_owner_a_never_sees_owner_b(two_owners: TwoOwners, session) -> None:
    a, b = two_owners.a, two_owners.b
    b_urls = {p.profile_url for p in B_ONLY_PEOPLE}
    gifthealth = identities(WORLD.connections_at("gifthealth", 2))

    # Same company name, different people.
    assert identities(at_company(a, "gift health", True)) == gifthealth
    b_there = at_company(b, "gift health", True)
    assert len(b_there) == len(gifthealth) + 2
    assert len({p.get("profile_url") for p in b_there} & b_urls) == 2

    # B's own company, people and URLs do not exist for A, by name or by id.
    assert a.resolve_company(B_ONLY_COMPANY).matched == ()
    assert a.resolve_company("bumblethwaite").suggestions == ()
    assert [p["name"] for p in at_company(b, B_ONLY_COMPANY)] == ["Bryony Onlybee"]
    assert a.connections_at_companies(b.resolve_company(B_ONLY_COMPANY).company_ids, True) == []
    assert a.connections_at_companies(b.resolve_company("gift health").company_ids, True) == []

    assert a.search_connections("onlybee").total == 0
    assert a.search_connections(company=B_ONLY_COMPANY).total == 0
    assert a.search_connections(company="bumblethwaite").total == 0
    assert b.search_connections("onlybee").total == 3
    assert a.search_connections(limit=100).total == len(WORLD.connections(2))
    assert b.search_connections(limit=100).total == len(WORLD.connections(2)) + 3

    for url in b_urls:
        assert a.lookup_connection(url).status == "not_found"
        assert b.lookup_connection(url).status == "found"
    assert a.lookup_connection("Bryony Onlybee").status == "not_found"
    assert a.lookup_connection("onlybee").suggestions == ()

    # The same person exists for both owners; each owner gets their own row's detail.
    url = WORLD.messages[0].sender_url
    b_row = session.query(Connection).filter_by(owner_id=b.owner_id, profile_url=url).one()
    session.add(Message(owner_id=b.owner_id, connection_id=b_row.id, body="only for B"))
    session.flush()
    detail_a, detail_b = a.lookup_connection(url).connection, b.lookup_connection(url).connection
    assert detail_a is not None and detail_b is not None
    assert detail_b["message_count"] == detail_a["message_count"] + 1
    assert a.connection_detail(b_row.id) is None
    assert b.connection_detail(b_row.id) == detail_b
