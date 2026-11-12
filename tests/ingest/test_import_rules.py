"""Importer rules on small hand-built exports: gaps, any order, events, snapshots."""

from __future__ import annotations

import itertools
import time
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kinship.db.models import (
    Connection,
    ConnectionHistory,
    ConnectionNote,
    Endorsement,
    Import,
    Invitation,
    Message,
    MyPosition,
    Profile,
    Recommendation,
    Skill,
)
from kinship.ingest import import_export
from kinship.parser import ParsedExport
from kinship.parser import records as r

from .conftest import dump

URL = "https://www.linkedin.com/in/"
T1, T2, T3, T4 = (datetime(2026, month, 1, tzinfo=UTC) for month in (1, 2, 3, 4))
WHEN = datetime(2025, 6, 1, 9, 30, tzinfo=UTC)


def person(
    slug: str | None, company: str | None = "Acme", position: str | None = "Engineer", **fields: Any
) -> r.Connection:
    name = slug or "nourl"
    values = {
        "first_name": name.capitalize(),
        "last_name": "Test",
        "profile_url": URL + slug if slug else None,
        "email": None,
        "company": company,
        "position": position,
        "connected_on": date(2020, 1, 1),
    }
    return r.Connection(**{**values, **fields})


def export(sha: str, when: datetime, connections: list[r.Connection], **parts: Any) -> ParsedExport:
    counts = {key: len(value) if isinstance(value, list) else 1 for key, value in parts.items()}
    counts["connections"] = len(connections)
    return ParsedExport(
        source_sha256=sha,
        exported_at=when,
        file_counts=counts,
        warnings=[],
        connections=connections,
        **parts,
    )


def note(slug: str, text: str, edited_at: datetime | None = None) -> r.Note:
    return r.Note(slug.capitalize(), "Test", URL + slug, text, WHEN, edited_at)


def endorsement(slug: str, status: str = "ACCEPTED") -> r.EndorsementReceived:
    return r.EndorsementReceived(WHEN, "Python", slug.capitalize(), "Test", URL + slug, status)


def recommendation(first: str, text: str = "Great.", status: str = "VISIBLE") -> r.Recommendation:
    return r.Recommendation(first, "Test", "Acme", "Engineer", text, WHEN, status)


def position(company: str, title: str, description: str | None = None) -> r.Position:
    return r.Position(company, title, description, None, date(2020, 1, 1), None)


def history(session: Session, owner_id: int, slug: str) -> list[tuple[str, str | None, str | None]]:
    """A connection's history as (import sha, company, position), in export order."""
    rows = session.execute(
        select(Import.source_sha256, ConnectionHistory.company_raw, ConnectionHistory.position)
        .join(Import, Import.id == ConnectionHistory.import_id)
        .join(Connection, Connection.id == ConnectionHistory.connection_id)
        .where(
            ConnectionHistory.owner_id == owner_id,
            Import.owner_id == owner_id,
            Connection.owner_id == owner_id,
            Connection.profile_url == URL + slug,
        )
        .order_by(Import.exported_at)
    )
    return [tuple(row) for row in rows]


def connection(session: Session, owner_id: int, slug: str) -> Connection:
    session.expire_all()
    return session.scalars(
        select(Connection).where(
            Connection.owner_id == owner_id, Connection.profile_url == URL + slug
        )
    ).one()


def four_exports() -> list[ParsedExport]:
    """Pat leaves and comes back; Rae leaves for good; Ned connects late; Quin never moves."""
    return [
        export(
            "a",
            T1,
            [person("pat"), person("quin"), person("rae"), person(None)],
            notes=[note("pat", "first draft")],
            endorsements_received=[endorsement("ned", "PENDING")],
            recommendations_received=[recommendation("Ned")],
            positions=[position("Acme", "Engineer", "old text"), position("Old Co", "Intern")],
            skills=[r.Skill("Python"), r.Skill("COBOL")],
            profile=r.Profile("Engineer", None, "Software"),
            profile_summary=[r.ProfileSummary("Summary from the summary file.")],
        ),
        export("x", T2, [person("pat", "Globex", "Manager"), person("quin"), person(None)]),
        export("b", T3, [person("quin", position="Senior Engineer"), person(None)]),
        export(
            "c",
            T4,
            [person("pat"), person("quin", position="Senior Engineer"), person("ned")],
            notes=[note("pat", "second draft", T4)],
            endorsements_received=[endorsement("ned", "ACCEPTED")],
            positions=[position("Acme", "Engineer", "new text"), position("Globex", "Lead")],
            skills=[r.Skill("Python"), r.Skill("SQL")],
            profile=r.Profile("Lead Engineer", "Summary from the profile.", "Software"),
        ),
    ]


def test_any_import_order_gives_the_same_database(session, make_owner) -> None:
    exports = four_exports()
    expected = None
    for order in itertools.permutations(exports):
        owner = make_owner("".join(parsed.source_sha256 for parsed in order))
        for parsed in order:
            import_export(session, owner.id, parsed)
        for parsed in order:  # and every export again changes nothing
            assert import_export(session, owner.id, parsed).already_imported
        result = dump(session, owner.id)
        if expected is None:
            expected = result
            check_four_exports(session, owner.id)
        assert result == expected, [parsed.source_sha256 for parsed in order]


def check_four_exports(session: Session, owner_id: int) -> None:
    imports = {
        row.source_sha256: row
        for row in session.scalars(select(Import).where(Import.owner_id == owner_id))
    }
    # Pat: seen in a, x and c, absent from b. Each row is the state changed *to*.
    assert history(session, owner_id, "pat") == [
        ("a", "Acme", "Engineer"),
        ("x", "Globex", "Manager"),
        ("c", "Acme", "Engineer"),
    ]
    pat = connection(session, owner_id, "pat")
    assert (pat.company_raw, pat.position, pat.removed_at) == ("Acme", "Engineer", None)
    assert pat.first_seen_import_id == imports["a"].id
    assert pat.last_seen_import_id == imports["c"].id

    # Quin: the title change first shows in b; c repeats it and adds no row.
    assert history(session, owner_id, "quin") == [
        ("a", "Acme", "Engineer"),
        ("b", "Acme", "Senior Engineer"),
    ]
    # Rae: only in a, so gone as of the next export, x.
    rae = connection(session, owner_id, "rae")
    assert rae.removed_at == T2
    assert rae.first_seen_import_id == rae.last_seen_import_id == imports["a"].id
    # The URL-less connection is matched by name + connected_on; gone as of c.
    nameless = session.scalars(
        select(Connection).where(Connection.owner_id == owner_id, Connection.profile_url.is_(None))
    ).one()
    assert nameless.removed_at == T4
    assert nameless.last_seen_import_id == imports["b"].id

    # Ned's endorsement and recommendation predate his connection; both get linked.
    ned = connection(session, owner_id, "ned")
    assert ned.first_seen_import_id == imports["c"].id
    endorsed = session.scalars(select(Endorsement).where(Endorsement.owner_id == owner_id)).one()
    assert (endorsed.connection_id, endorsed.status) == (ned.id, "ACCEPTED")
    recommended = session.scalars(
        select(Recommendation).where(Recommendation.owner_id == owner_id)
    ).one()
    assert recommended.connection_id == ned.id

    # The edited note replaced the original in place.
    stored = session.scalars(
        select(ConnectionNote).where(ConnectionNote.owner_id == owner_id)
    ).one()
    assert (stored.note, stored.edited_at, stored.connection_id) == ("second draft", T4, pat.id)

    # Owner snapshot tables mirror c, the newest export that has them.
    positions = session.scalars(select(MyPosition).where(MyPosition.owner_id == owner_id)).all()
    assert {(row.company, row.title, row.description) for row in positions} == {
        ("Acme", "Engineer", "new text"),
        ("Globex", "Lead", None),
    }
    skills = session.scalars(select(Skill.name).where(Skill.owner_id == owner_id)).all()
    assert sorted(skills) == ["Python", "SQL"]
    profile = session.scalars(select(Profile).where(Profile.owner_id == owner_id)).one()
    assert (profile.headline, profile.summary) == ("Lead Engineer", "Summary from the profile.")


def test_removed_then_restored(session, owner) -> None:
    a, _, b, c = four_exports()
    import_export(session, owner.id, a)
    gone = import_export(session, owner.id, b)
    assert (gone.removed, gone.changed, gone.unchanged) == (2, 1, 1)  # Pat and Rae; Quin
    assert connection(session, owner.id, "pat").removed_at == T3

    back = import_export(session, owner.id, c)
    assert (back.restored, back.new, back.removed) == (1, 1, 1)  # Pat; Ned; the URL-less row
    assert connection(session, owner.id, "pat").removed_at is None
    assert connection(session, owner.id, "rae").removed_at == T3


def test_snapshot_rows_keep_their_ids_and_follow_the_newest_export(session, owner) -> None:
    a, _, _, c = four_exports()
    import_export(session, owner.id, a)
    profile = session.scalars(select(Profile).where(Profile.owner_id == owner.id)).one()
    assert profile.summary == "Summary from the summary file."  # Profile.csv had none
    ids = {
        row.company: row.id
        for row in session.scalars(select(MyPosition).where(MyPosition.owner_id == owner.id))
    }
    python_id = session.scalar(
        select(Skill.id).where(Skill.owner_id == owner.id, Skill.name == "Python")
    )

    import_export(session, owner.id, c)
    session.expire_all()
    after = {
        row.company: row
        for row in session.scalars(select(MyPosition).where(MyPosition.owner_id == owner.id))
    }
    assert set(after) == {"Acme", "Globex"}  # "Old Co" is gone from the newest export
    assert after["Acme"].id == ids["Acme"]
    assert after["Acme"].description == "new text"
    assert python_id == session.scalar(
        select(Skill.id).where(Skill.owner_id == owner.id, Skill.name == "Python")
    )
    assert session.get(Profile, profile.id).headline == "Lead Engineer"

    # A newer export without those files leaves the snapshot alone.
    import_export(session, owner.id, export("d", datetime(2026, 5, 1, tzinfo=UTC), [person("pat")]))
    assert (
        len(session.scalars(select(MyPosition).where(MyPosition.owner_id == owner.id)).all()) == 2
    )


def test_events_are_deduplicated_and_never_deleted(session, owner) -> None:
    first = export(
        "a",
        T1,
        [person("pat")],
        endorsements_received=[endorsement("pat"), endorsement("pat")],
        endorsements_given=[r.EndorsementGiven(WHEN, "Python", "Pat", "Test", URL + "pat", "OK")],
        company_follows=[r.CompanyFollow("Acme", WHEN), r.CompanyFollow(None, WHEN)],
        saved_jobs=[r.SavedJob(WHEN, "https://jobs.example/1", "Engineer", "Acme")],
    )
    summary = import_export(session, owner.id, first)
    assert summary.inserted["endorsements"] == 2  # one received (deduplicated), one given
    assert summary.inserted["company_follows"] == 1  # the blank organization is skipped

    # The newer export lost the endorsements; the stored rows stay.
    later = export("b", T2, [person("pat")], endorsements_received=[endorsement("quin")])
    summary = import_export(session, owner.id, later)
    assert summary.inserted["endorsements"] == 1
    rows = session.scalars(select(Endorsement).where(Endorsement.owner_id == owner.id)).all()
    assert sorted((row.direction, row.first_name) for row in rows) == [
        ("given", "Pat"),
        ("received", "Pat"),
        ("received", "Quin"),
    ]


def test_invitation_direction_and_other_person(session, owner) -> None:
    me = URL + "me"
    parsed = export(
        "a",
        T1,
        [person("pat"), person("quin")],
        invitations=[
            r.Invitation("Pat Test", "Me Myself", WHEN, "hi", "INCOMING", URL + "pat", me),
            r.Invitation("Me Myself", "Quin Test", WHEN, None, "Outgoing", me, URL + "quin"),
            r.Invitation("Me Myself", "Stranger", WHEN, None, "outgoing", me, URL + "stranger"),
            r.Invitation("Me Myself", "Nobody", WHEN, None, "SIDEWAYS", me, URL + "nobody"),
        ],
    )
    summary = import_export(session, owner.id, parsed)

    assert summary.inserted["invitations"] == 3
    assert any("unknown Direction" in warning for warning in summary.warnings)
    rows = session.scalars(select(Invitation).where(Invitation.owner_id == owner.id)).all()
    found = {(row.direction, row.profile_url, row.connection_id) for row in rows}
    assert found == {
        ("incoming", URL + "pat", connection(session, owner.id, "pat").id),
        ("outgoing", URL + "quin", connection(session, owner.id, "quin").id),
        ("outgoing", URL + "stranger", None),
    }


def message(
    sender: str,
    to: str,
    body: str,
    sender_url: str | None = None,
    recipient_urls: tuple[str, ...] = (),
) -> r.Message:
    return r.Message("conv-1", None, sender, sender_url, to, recipient_urls, WHEN, None, body, None)


def test_messages_link_to_the_other_participant(session, owner) -> None:
    me = URL + "me"
    parsed = export(
        "a",
        T1,
        [person("pat"), person("quin"), person("sam"), person("sam2", first_name="Sam")],
        messages=[
            message("Pat Test", "Me Myself", "from pat", URL + "pat", (me,)),
            message("Me Myself", "Pat Test", "to pat", me, (URL + "pat",)),
            message("Me Myself", "Pat Test", "to pat", me, (URL + "pat",)),  # duplicate
            message("Me Myself", "Pat Test, Quin Test", "group", me, (URL + "pat", URL + "quin")),
            message("Quin Test", "Me Myself", "old format, unique name"),
            message("Sam Test", "Me Myself", "old format, ambiguous name"),
            message("Recruiter", "Me Myself", "stranger", URL + "recruiter", (me,)),
        ],
    )
    summary = import_export(session, owner.id, parsed)

    assert summary.inserted["messages"] == 6
    pat = connection(session, owner.id, "pat").id
    quin = connection(session, owner.id, "quin").id
    rows = session.scalars(select(Message).where(Message.owner_id == owner.id)).all()
    assert {row.body: (row.connection_id, row.to_names) for row in rows} == {
        "from pat": (pat, ["Me Myself"]),
        "to pat": (pat, ["Pat Test"]),
        "group": (None, ["Pat Test", "Quin Test"]),
        "old format, unique name": (quin, ["Me Myself"]),
        "old format, ambiguous name": (None, ["Me Myself"]),
        "stranger": (None, ["Me Myself"]),
    }

    # The recruiter connects later; the message is re-linked when it is seen again.
    later = export(
        "b",
        T2,
        [person("pat"), person("recruiter", first_name="Recruiter", last_name=None)],
        messages=[message("Recruiter", "Me Myself", "stranger", URL + "recruiter", (me,))],
    )
    summary = import_export(session, owner.id, later)
    assert summary.inserted["messages"] == 0
    session.expire_all()
    stranger = session.scalars(
        select(Message).where(Message.owner_id == owner.id, Message.body == "stranger")
    ).one()
    assert stranger.connection_id == connection(session, owner.id, "recruiter").id


def test_ambiguous_recommendation_names_are_not_linked(session, owner) -> None:
    parsed = export(
        "a",
        T1,
        [person("pat"), person("sam"), person("sam2", first_name="Sam")],
        recommendations_received=[recommendation("Pat"), recommendation("Sam")],
        recommendations_given=[recommendation("Pat"), recommendation("Zed")],
    )
    import_export(session, owner.id, parsed)

    pat = connection(session, owner.id, "pat").id
    rows = session.scalars(select(Recommendation).where(Recommendation.owner_id == owner.id)).all()
    assert sorted((row.direction, row.first_name, row.connection_id) for row in rows) == sorted(
        [
            ("received", "Pat", pat),
            ("received", "Sam", None),
            ("given", "Pat", pat),
            ("given", "Zed", None),
        ]
    )


def test_duplicate_connection_rows_are_collapsed(session, owner) -> None:
    parsed = export("a", T1, [person("pat"), person("pat", "Other"), person(None), person(None)])
    summary = import_export(session, owner.id, parsed)

    assert summary.new == 2
    assert any("duplicate" in warning for warning in summary.warnings)
    assert connection(session, owner.id, "pat").company_raw == "Acme"


def test_a_few_thousand_connections_import_in_seconds(session, owner) -> None:
    people = [person(f"p{number}") for number in range(4000)]
    moved = [person(f"p{number}", "Globex") for number in range(500)]
    older = export("a", T1, people)
    newer = export("b", T2, moved + people[500:3900] + [person(f"q{n}") for n in range(100)])

    started = time.perf_counter()
    import_export(session, owner.id, older)
    summary = import_export(session, owner.id, newer)
    elapsed = time.perf_counter() - started

    assert (summary.new, summary.changed, summary.removed) == (100, 500, 100)
    assert summary.unchanged == 3400
    assert elapsed < 30  # generous for CI; locally this is about a second
