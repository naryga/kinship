"""`NetworkRepository` against the synthetic exports: correctness and owner isolation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from ingest.conftest import V1_AT, V2_AT
from sqlalchemy.orm import Session

from kinship.db.models import Import, Owner
from kinship.ingest import import_export
from kinship.parser import ParsedExport
from kinship.repo import Repository
from kinship.repo.network import MAX_NAMES_PER_COMPANY, NetworkPerson
from kinship.synth import build_world
from kinship.warmth import Signals, score_warmth

from .conftest import B_ONLY_AT, B_ONLY_COMPANY, B_ONLY_URL, add_b_only_data

WORLD = build_world()
TRUTH = WORLD.truth
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
OWNER_COMPANY_KEYS = {position.company_key for position in WORLD.owner.positions}
V1_KEYS = {c.url: c.company_key for c in WORLD.connections(1) if c.url}


def utc(moment: datetime) -> datetime:
    """The synthetic world's datetimes are naive UTC."""
    return moment.replace(tzinfo=UTC)


def company_id(repo: Repository, name: str) -> int:
    matches = repo.find_companies(name)
    assert matches and matches[0].match == "exact", name
    return matches[0].id


def expected_signals(url: str, name: str, target_key: str, current_key: str) -> dict:
    """What the synthetic world says the owner shares with one connection."""
    me = WORLD.owner.url
    messages = [m for m in WORLD.messages if url in (m.sender_url, m.recipient_urls)]
    invitations = [
        i
        for i in WORLD.invitations
        if url == (i.inviter_url if i.invitee_url == me else i.invitee_url)
    ]
    assert len(invitations) <= 1
    worked_at = {V1_KEYS.get(url, current_key), current_key} - {target_key, ""}
    return {
        "message_count": len(messages),
        "last_message_at": max((utc(m.at) for m in messages), default=None),
        "recommendations_received": sum(
            1 for r in WORLD.recommendations_received if f"{r.first_name} {r.last_name}" == name
        ),
        "recommendations_given": sum(
            1 for r in WORLD.recommendations_given if f"{r.first_name} {r.last_name}" == name
        ),
        "endorsements_received": sorted(
            e.skill for e in WORLD.endorsements_received if e.url == url
        ),
        "endorsements_given": sorted(e.skill for e in WORLD.endorsements_given if e.url == url),
        "note_count": sum(1 for n in WORLD.notes if n.url == url),
        "invitation": invitations[0].direction.lower() if invitations else None,
        "invitation_had_message": bool(invitations and invitations[0].message),
        "shared_employers": len(worked_at & OWNER_COMPANY_KEYS),
    }


def actual_signals(signals: Signals) -> dict:
    return {
        "message_count": signals.message_count,
        "last_message_at": signals.last_message_at,
        "recommendations_received": len(signals.recommendations_received),
        "recommendations_given": len(signals.recommendations_given),
        "endorsements_received": sorted(signals.endorsements_received),
        "endorsements_given": sorted(signals.endorsements_given),
        "note_count": signals.note_count,
        "invitation": signals.invitation,
        "invitation_had_message": signals.invitation_had_message,
        "shared_employers": len(signals.shared_employers),
    }


# -- warm paths -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["gifthealth", "acme", "quillfeather", "juniperwren"])
def test_signals_match_the_synthetic_world(repo: Repository, key: str) -> None:
    target = company_id(repo, WORLD.company_groups[key][0])
    people = repo.connections_at_companies([target])
    assert all(isinstance(person, NetworkPerson) for person in people)
    expected_people = WORLD.connections_at(key, 2)
    assert sorted(p.profile_url or "" for p in people) == sorted(c.url for c in expected_people)

    signals = repo.warmth_signals([p.id for p in people], exclude_company_ids=[target])
    assert set(signals) == {p.id for p in people}
    checked = 0
    for person in people:
        if not person.profile_url:
            continue
        expected = expected_signals(person.profile_url, person.name, key, key)
        assert actual_signals(signals[person.id]) == expected, person.name
        assert signals[person.id].connected_on == person.connected_on
        checked += 1
    assert checked >= 3


def test_every_kind_of_signal_occurs_in_the_fixture(repo: Repository) -> None:
    """Guards the test above against passing on all-empty data."""
    everyone = [
        person
        for key in WORLD.company_groups
        for match in repo.find_companies(WORLD.company_groups[key][0])[:1]
        for person in repo.connections_at_companies([match.id])
    ]
    signals = repo.warmth_signals([p.id for p in everyone]).values()
    assert any(s.message_count for s in signals)
    assert any(s.recommendations_received for s in signals)
    assert any(s.recommendations_given for s in signals)
    assert any(s.endorsements_received for s in signals)
    assert any(s.endorsements_given for s in signals)
    assert any(s.note_count for s in signals)
    assert {s.invitation for s in signals} == {None, "incoming", "outgoing"}
    assert any(s.invitation_had_message for s in signals)
    assert any(s.shared_employers for s in signals)


def test_recommendation_plus_messages_outranks_no_interaction(repo: Repository) -> None:
    target = company_id(repo, "gift health")
    people = repo.connections_at_companies([target])
    signals = repo.warmth_signals([p.id for p in people], exclude_company_ids=[target])
    scores = {p.name: score_warmth(signals[p.id], V2_AT) for p in people}

    casey = scores["Casey Foxglove"]  # a recommendation in 2023 and three messages
    assert casey.reasons[0] == "Wrote you a recommendation in 2023"
    last = max(m.at for m in WORLD.messages if "casey-foxglove" in m.sender_url + m.recipient_urls)
    assert f"You exchanged 3 messages, last on {last.date().isoformat()}" in casey.reasons

    cold = [name for name, warmth in scores.items() if warmth.score == 0]
    assert cold, "the fixture should have gifthealth connections with no interaction"
    for name in cold:
        assert scores[name].reasons[0].startswith("No recorded interaction")
        assert casey.score > 30 > scores[name].score


def test_shared_employer_comes_from_history_and_excludes_the_target(repo: Repository) -> None:
    """Elio Emberly moved from Acme to Gifthealth; the owner worked at both."""
    gift, acme = company_id(repo, "gift health"), company_id(repo, "acme")
    elio = next(p for p in repo.connections_at_companies([gift]) if p.name == "Elio Emberly")

    def shared(*exclude: int) -> tuple[str, ...]:
        return repo.warmth_signals([elio.id], exclude_company_ids=exclude)[elio.id].shared_employers

    assert len(shared()) == 2
    assert len(shared(gift)) == 1 and shared(gift) != shared(acme)
    assert shared(gift, acme) == ()

    former = repo.former_connections_of_companies([acme])
    assert "Elio Emberly" in [p.name for p in former]
    assert all(p.company_id != acme for p in former)
    moved_out = {
        c.url for c in WORLD.connections(2) if V1_KEYS.get(c.url) == "acme" != c.company_key
    }
    assert {p.profile_url for p in former} == moved_out
    assert repo.former_connections_of_companies([]) == []
    assert repo.connections_at_companies([]) == []
    assert repo.warmth_signals([]) == {}


def test_ranking_works_without_messages(
    session: Session, owner: Owner, without_messages: list[ParsedExport]
) -> None:
    for parsed in without_messages:
        import_export(session, owner.id, parsed)
    repo = Repository(session, owner.id)
    assert not repo.has_messages()

    target = company_id(repo, "gift health")
    people = repo.connections_at_companies([target])
    signals = repo.warmth_signals([p.id for p in people], exclude_company_ids=[target])
    assert all(s.message_count == 0 and s.last_message_at is None for s in signals.values())
    scores = {p.name: score_warmth(signals[p.id], V2_AT, use_messages=False) for p in people}

    assert scores["Casey Foxglove"].reasons[0] == "Wrote you a recommendation in 2023"
    assert not any("message" in reason for w in scores.values() for reason in w.reasons)
    assert scores["Casey Foxglove"].score >= 30
    assert min(w.score for w in scores.values()) == 0


def test_has_messages(repo: Repository) -> None:
    assert repo.has_messages()


# -- changes --------------------------------------------------------------------


def urls(items) -> tuple[str, ...]:
    return tuple(sorted(item.person.profile_url for item in items))


@pytest.mark.parametrize("since", [V1_AT, EPOCH, V2_AT - timedelta(seconds=1)])
def test_changes_match_diff_truth(repo: Repository, since: datetime) -> None:
    changes = repo.job_changes(since)
    assert urls(c for c in changes if c.company_changed) == TRUTH.company_changed
    assert urls(c for c in changes if c.title_changed) == TRUTH.title_changed
    assert urls(c for c in changes if c.is_spelling_only) == TRUTH.company_renamed
    assert urls(changes) == TRUTH.job_changed
    assert {c.as_of for c in changes} == {V2_AT}

    assert urls(repo.new_connections(since)) == TRUTH.added
    assert urls(repo.removed_connections(since)) == TRUTH.removed
    assert {e.at for e in repo.removed_connections(since)} == {V2_AT}


def test_job_change_carries_both_states(repo: Repository) -> None:
    before = {c.url: c for c in WORLD.connections(1)}
    after = {c.url: c for c in WORLD.connections(2)}
    changes = repo.job_changes(V1_AT)
    assert changes
    for change in changes:
        url = change.person.profile_url
        assert (change.from_company, change.from_title) == (
            before[url].company,
            before[url].position,
        )
        assert (change.to_company, change.to_title) == (after[url].company, after[url].position)


def test_nothing_changes_after_the_latest_export(repo: Repository) -> None:
    for since in (V2_AT, V2_AT + timedelta(days=1)):
        assert repo.job_changes(since) == []
        assert repo.new_connections(since) == []
        assert repo.removed_connections(since) == []


def test_with_one_import_nothing_is_new(session: Session, owner: Owner, v1: ParsedExport) -> None:
    import_export(session, owner.id, v1)
    repo = Repository(session, owner.id)
    assert [info.exported_at for info in repo.imports_in_order()] == [V1_AT]
    assert repo.new_connections(EPOCH) == []
    assert repo.job_changes(EPOCH) == []
    assert repo.removed_connections(EPOCH) == []


def test_import_order_does_not_matter(session: Session, owner: Owner, v1, v2) -> None:
    import_export(session, owner.id, v2)
    import_export(session, owner.id, v1)  # the older export, loaded last
    repo = Repository(session, owner.id)
    assert [info.exported_at for info in repo.imports_in_order()] == [V1_AT, V2_AT]
    assert repo.latest_import().exported_at == V2_AT
    assert urls(repo.job_changes(V1_AT)) == TRUTH.job_changed
    assert urls(repo.new_connections(V1_AT)) == TRUTH.added
    assert urls(repo.removed_connections(V1_AT)) == TRUTH.removed


def test_applied_companies(repo: Repository) -> None:
    current = Counter(c.company_key for c in WORLD.connections(2))
    expected = [a for a in WORLD.job_applications if a.company_key and current[a.company_key]]
    assert len(expected) >= 2

    found = repo.applied_companies()
    assert [(a.company, a.job_title, a.applied_at) for a in found] == [
        (a.company, a.job_title, utc(a.at).replace(second=0)) for a in expected
    ]  # newest first, as in the world
    for item, application in zip(found, expected, strict=True):
        at_company = sorted(
            (c.last_name, c.first_name) for c in WORLD.connections_at(application.company_key, 2)
        )
        assert item.connection_count == len(at_company)
        assert list(item.connection_names) == [
            f"{first} {last}" for last, first in at_company[:MAX_NAMES_PER_COMPANY]
        ]

    newest, second = found[0].applied_at, found[1].applied_at
    assert [a.applied_at for a in repo.applied_companies(second)] == [newest]
    assert repo.applied_companies(newest) == []


# -- owner isolation ------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    repo_a: Repository
    repo_b: Repository
    b_company_id: int
    b_connection_id: int


@pytest.fixture
def pair(session: Session, make_owner: Callable[..., Owner], v1, v2) -> Pair:
    """Owners A and B with the same exports; B also has `add_b_only_data`."""
    a, b = make_owner("Owner A"), make_owner("Owner B")
    for owner in (a, b):
        import_export(session, owner.id, v1)
        import_export(session, owner.id, v2)
    b_company_id, b_connection_id = add_b_only_data(session, b.id)
    return Pair(Repository(session, a.id), Repository(session, b.id), b_company_id, b_connection_id)


def test_b_sees_its_extra_data(pair: Pair) -> None:
    """The isolation tests below would pass vacuously if B's extra rows were not visible to B."""
    b = pair.repo_b
    assert [p.profile_url for p in b.connections_at_companies([pair.b_company_id])] == [B_ONLY_URL]
    signals = b.warmth_signals([pair.b_connection_id])[pair.b_connection_id]
    assert signals.message_count == 1 and len(signals.recommendations_received) == 1
    assert set(signals.shared_employers) == {"Acme", B_ONLY_COMPANY}
    acme = company_id(b, "acme")
    assert B_ONLY_URL in [p.profile_url for p in b.former_connections_of_companies([acme])]
    assert [c.person.profile_url for c in b.job_changes(V2_AT)] == [B_ONLY_URL]
    assert [a.company for a in b.applied_companies(V2_AT)] == [B_ONLY_COMPANY]
    assert [info.exported_at for info in b.imports_in_order()] == [V1_AT, V2_AT, B_ONLY_AT]
    assert b.latest_import().exported_at == B_ONLY_AT


def test_a_never_sees_b_rows(pair: Pair) -> None:
    a, b = pair.repo_a, pair.repo_b
    b_acme = company_id(b, "acme")
    b_gift = company_id(b, "gift health")
    b_people = [p.id for p in b.connections_at_companies([b_gift])]
    assert b_people

    # Lookups by B's ids return nothing.
    assert a.connections_at_companies([pair.b_company_id, b_acme, b_gift]) == []
    assert a.former_connections_of_companies([b_acme]) == []
    assert a.warmth_signals([pair.b_connection_id, *b_people]) == {}

    # A's own results are exactly the single-owner results.
    a_acme = company_id(a, "acme")
    assert a_acme != b_acme
    assert B_ONLY_URL not in [p.profile_url for p in a.former_connections_of_companies([a_acme])]
    assert len(a.connections_at_companies([a_acme])) == len(WORLD.connections_at("acme", 2))
    a_people = a.connections_at_companies([a_acme])
    for signals in a.warmth_signals(
        [p.id for p in a_people], exclude_company_ids=[b_acme]
    ).values():
        assert B_ONLY_COMPANY not in signals.shared_employers

    assert [info.exported_at for info in a.imports_in_order()] == [V1_AT, V2_AT]
    assert a.latest_import().exported_at == V2_AT
    assert a.job_changes(V2_AT) == []
    assert urls(a.job_changes(EPOCH)) == TRUTH.job_changed
    assert urls(a.new_connections(EPOCH)) == TRUTH.added
    assert urls(a.removed_connections(EPOCH)) == TRUTH.removed
    assert B_ONLY_COMPANY not in [item.company for item in a.applied_companies()]
    assert a.applied_companies(V2_AT) == []
    assert a.find_companies(B_ONLY_COMPANY) == []


def test_b_counts_are_not_doubled_by_a(pair: Pair) -> None:
    b = pair.repo_b
    assert urls(b.new_connections(V1_AT)) == tuple(sorted((*TRUTH.added, B_ONLY_URL)))
    assert urls(b.removed_connections(EPOCH)) == TRUTH.removed
    gift = company_id(b, "gift health")
    assert len(b.connections_at_companies([gift])) == len(WORLD.connections_at("gifthealth", 2))


def test_latest_import_ignores_load_order_and_other_owners(
    pair: Pair, session: Session, make_owner
) -> None:
    other = make_owner("Later")
    session.add(
        Import(owner_id=other.id, source_sha256="x", exported_at=B_ONLY_AT + timedelta(days=30))
    )
    session.flush()
    assert pair.repo_a.latest_import().exported_at == V2_AT
