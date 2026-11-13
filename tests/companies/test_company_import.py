"""Companies are assigned during import: synthetic ground truth, rules, isolation."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

from ingest.conftest import dump
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kinship.companies import company_key
from kinship.db.models import (
    Company,
    CompanyFollow,
    Connection,
    ConnectionHistory,
    JobApplication,
    MyPosition,
    SavedJob,
)
from kinship.ingest import CompaniesRebuilt, import_export, rebuild_companies
from kinship.ingest.companies import COMPANY_SOURCES
from kinship.parser import records as r
from kinship.synth import build_world

from .conftest import T1, T2, T3, URL, export, person

WORLD = build_world()
WHEN = datetime(2025, 6, 1, 9, 30, tzinfo=UTC)


def companies(session: Session, owner_id: int) -> dict[str, Company]:
    session.expire_all()
    rows = session.scalars(select(Company).where(Company.owner_id == owner_id))
    return {row.name_normalized: row for row in rows}


def raw_names(session: Session, owner_id: int) -> list[tuple[str, str | None, int | None]]:
    """(table, raw company name, company_id) for every row that can name a company."""
    found = []
    for model, column in COMPANY_SOURCES:
        table = model.__table__
        rows = session.execute(
            select(table.c[column], table.c.company_id).where(table.c.owner_id == owner_id)
        )
        found += [(table.name, raw, company_id) for raw, company_id in rows]
    return found


def check_links(session: Session, owner_id: int) -> None:
    """Every company-naming row points at its own owner's company with the right key."""
    by_id = {row.id: row for row in companies(session, owner_id).values()}
    names = raw_names(session, owner_id)
    assert names
    for table, raw, company_id in names:
        key = company_key(raw)
        if key is None:
            assert company_id is None, (table, raw)
        else:
            assert by_id[company_id].name_normalized == key, (table, raw)
            assert " ".join(raw.split()) in by_id[company_id].aliases
    # No company is left without a row naming it.
    assert {company_id for _, _, company_id in names} - {None} == set(by_id)


def test_synthetic_groups_map_to_one_company_each(session, owner, v1, v2) -> None:
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    stored = companies(session, owner.id)
    names = raw_names(session, owner.id)
    present = {raw for _, raw, _ in names if raw}
    check_links(session, owner.id)

    for group, spellings in WORLD.company_groups.items():
        expected_aliases = sorted(set(spellings) & present)
        assert expected_aliases, group
        matching = [row for row in stored.values() if set(row.aliases) & set(spellings)]
        assert len(matching) == 1, group
        company = matching[0]
        assert company.aliases == expected_aliases, group
        assert company.display_name in spellings
        assert company.name_normalized == company_key(spellings[0])
    # The deliberate variants are all there, so the three groups really merged.
    for group in ("acme", "gifthealth", "quillfeather"):
        assert set(WORLD.company_groups[group]) <= present
    assert set(stored["gifthealth"].aliases) == {"Gifthealth", "Gift Health, Inc.", "gift health"}
    assert set(stored["acme"].aliases) == {"Acme", "Acme, Inc.", "ACME Corp"}
    assert set(stored["quillfeathersystems"].aliases) == {
        "Quillfeather Systems",
        "Quillfeather Systems, LLC",
    }
    # One company per distinct key, and nothing else.
    assert set(stored) == {company_key(raw) for raw in present}


def test_each_connection_links_to_its_ground_truth_company(session, owner, v1, v2) -> None:
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    stored = companies(session, owner.id)
    group_of = {
        row.id: group
        for group, spellings in WORLD.company_groups.items()
        for row in stored.values()
        if set(row.aliases) & set(spellings)
    }
    rows = {
        row.profile_url: row
        for row in session.scalars(select(Connection).where(Connection.owner_id == owner.id))
        if row.profile_url
    }
    # Connections only v1 has keep their v1 company; everyone else is as of v2.
    truth = {person.url: person for person in WORLD.connections(1)}
    truth |= {person.url: person for person in WORLD.connections(2)}
    checked = 0
    for url, expected in truth.items():
        if not url:
            continue
        row = rows[url]
        if expected.company_key:
            assert group_of[row.company_id] == expected.company_key, url
        else:
            assert row.company_id is None, url
        checked += 1
    assert checked == len(rows)

    # `connections_at` is the tools' ground truth for "who works at X now".
    for group in WORLD.company_groups:
        expected_urls = {c.url for c in WORLD.connections_at(group, 2) if c.url}
        actual = {
            url
            for url, row in rows.items()
            if row.removed_at is None and group_of.get(row.company_id) == group
        }
        assert actual == expected_urls, group


def test_history_and_own_positions_are_linked(session, owner, v1, v2) -> None:
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    stored = companies(session, owner.id)
    by_id = {row.id: row for row in stored.values()}

    history = session.execute(
        select(ConnectionHistory.company_raw, ConnectionHistory.company_id).where(
            ConnectionHistory.owner_id == owner.id
        )
    ).all()
    assert any(company_id for _, company_id in history)
    for raw, company_id in history:
        assert (company_id is None) == (company_key(raw) is None)
        if company_id is not None:
            assert by_id[company_id].name_normalized == company_key(raw)

    # A spelling-only rename keeps the company; a real move changes it.
    connection_ids = {
        row.profile_url: row.id
        for row in session.scalars(select(Connection).where(Connection.owner_id == owner.id))
    }

    def timeline(url: str) -> list[int | None]:
        return list(
            session.scalars(
                select(ConnectionHistory.company_id)
                .where(
                    ConnectionHistory.owner_id == owner.id,
                    ConnectionHistory.connection_id == connection_ids[url],
                )
                .order_by(ConnectionHistory.import_id)
            )
        )

    assert WORLD.truth.company_renamed and WORLD.truth.company_changed
    for url in WORLD.truth.company_renamed:
        first, second = timeline(url)
        assert first == second is not None
    for url in WORLD.truth.company_changed:
        first, second = timeline(url)
        assert first != second

    positions = session.scalars(select(MyPosition).where(MyPosition.owner_id == owner.id)).all()
    assert len(positions) == len(WORLD.owner.positions)
    for row in positions:
        assert by_id[row.company_id].name_normalized == company_key(row.company)
    # The generator gives the owner a past employer where connections work.
    shared = {row.company_id for row in positions} & {
        company_id
        for company_id in session.scalars(
            select(Connection.company_id).where(Connection.owner_id == owner.id)
        )
    }
    assert shared

    for model in (CompanyFollow, JobApplication, SavedJob):
        linked = session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.owner_id == owner.id, model.company_id.is_not(None))
        )
        assert linked, model.__name__


def test_rebuild_is_idempotent(session, owner, v1, v2) -> None:
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    before = dump(session, owner.id)
    ids = {key: row.id for key, row in companies(session, owner.id).items()}

    result = rebuild_companies(session, owner.id)
    assert result == CompaniesRebuilt(total=len(ids), created=0, deleted=0, relinked=0)
    assert dump(session, owner.id) == before
    assert {key: row.id for key, row in companies(session, owner.id).items()} == ids

    # Re-importing is a no-op and leaves companies alone too.
    assert import_export(session, owner.id, v1).already_imported
    assert dump(session, owner.id) == before


def test_rebuild_repairs_links_and_companies(session, owner, v1) -> None:
    """As after a change of the normalization rules: stale rows are corrected."""
    import_export(session, owner.id, v1)
    before = dump(session, owner.id)
    stored = companies(session, owner.id)

    stored["acme"].name_normalized = "stale-key"
    stored["gifthealth"].aliases = ["wrong"]
    stored["gifthealth"].display_name = "wrong"
    session.add(Company(owner_id=owner.id, name_normalized="orphan", display_name="Orphan"))
    session.flush()
    for row in session.scalars(select(Connection).where(Connection.owner_id == owner.id).limit(5)):
        row.company_id = None
    session.flush()

    result = rebuild_companies(session, owner.id)
    assert (result.created, result.deleted) == (1, 2)  # acme again; stale-key and orphan gone
    assert result.relinked > 5
    assert dump(session, owner.id) == before
    check_links(session, owner.id)


def test_display_name_and_aliases(session, owner) -> None:
    people = [
        person("a", "Acme, Inc."),
        person("b", "ACME Corp"),
        person("c", "ACME Corp"),
        person("d", "Gift Health, Inc."),
        person("e", "gift health"),
        person("f", None),
        person("g", "   "),
        person("h", "Co"),
    ]
    summary = import_export(
        session,
        owner.id,
        export(
            "one",
            T1,
            people,
            positions=[r.Position("Old Co", "Intern", None, None, None, None)],
            company_follows=[r.CompanyFollow("Acme", WHEN), r.CompanyFollow("Followed Only", WHEN)],
            job_applications=[
                r.JobApplication(WHEN, None, None, "Globex LLC", "Engineer", None, None, None)
            ],
            saved_jobs=[r.SavedJob(WHEN, None, "Engineer", "Globex")],
        ),
    )
    stored = companies(session, owner.id)
    assert summary.inserted["companies"] == len(stored) == 6
    assert set(stored) == {"acme", "gifthealth", "co", "old", "followedonly", "globex"}
    check_links(session, owner.id)

    # Most common spelling among current connections; the follow adds an alias only.
    assert stored["acme"].display_name == "ACME Corp"
    assert stored["acme"].aliases == ["ACME Corp", "Acme", "Acme, Inc."]
    # A tie goes to the longer spelling.
    assert stored["gifthealth"].display_name == "Gift Health, Inc."
    assert stored["gifthealth"].aliases == ["Gift Health, Inc.", "gift health"]
    # No connection there: the most common spelling of any row, longest on a tie.
    assert stored["globex"].display_name == "Globex LLC"
    assert stored["globex"].aliases == ["Globex", "Globex LLC"]
    assert stored["co"].display_name == "Co"
    assert stored["old"].aliases == ["Old Co"]

    blank = session.scalars(
        select(Connection).where(
            Connection.owner_id == owner.id, Connection.profile_url.in_([URL + "f", URL + "g"])
        )
    ).all()
    assert [row.company_id for row in blank] == [None, None]


def test_companies_follow_the_data_across_imports(session, owner) -> None:
    import_export(
        session, owner.id, export("one", T1, [person("a", "Acme"), person("b", "Initech")])
    )
    first = companies(session, owner.id)
    assert set(first) == {"acme", "initech"}

    # b moves to Acme under another spelling; a leaves the network.
    import_export(session, owner.id, export("two", T2, [person("b", "Acme, Inc.")]))
    second = companies(session, owner.id)
    # Initech stays: b's history still names it ("who used to work there").
    assert set(second) == {"acme", "initech"}
    assert {key: row.id for key, row in second.items()} == {
        key: row.id for key, row in first.items()
    }
    assert second["acme"].aliases == ["Acme", "Acme, Inc."]
    # Only current connections vote on the display name; a was removed.
    assert second["acme"].display_name == "Acme, Inc."

    b = session.scalars(
        select(Connection).where(
            Connection.owner_id == owner.id, Connection.profile_url == URL + "b"
        )
    ).one()
    assert b.company_id == second["acme"].id
    history = session.execute(
        select(ConnectionHistory.company_raw, ConnectionHistory.company_id)
        .where(ConnectionHistory.owner_id == owner.id, ConnectionHistory.connection_id == b.id)
        .order_by(ConnectionHistory.import_id)
    ).all()
    assert [tuple(row) for row in history] == [
        ("Initech", second["initech"].id),
        ("Acme, Inc.", second["acme"].id),
    ]


def test_unused_companies_are_deleted(session, owner) -> None:
    positions = [r.Position("Old Co", "Intern", None, None, None, None)]
    import_export(session, owner.id, export("one", T1, [person("a", "Acme")], positions=positions))
    assert set(companies(session, owner.id)) == {"acme", "old"}

    # The newest export's Positions.csv replaces the snapshot: nothing names Old Co now.
    positions = [r.Position("New Co", "Lead", None, None, None, None)]
    import_export(session, owner.id, export("two", T2, [person("a", "Acme")], positions=positions))
    assert set(companies(session, owner.id)) == {"acme", "new"}
    check_links(session, owner.id)


def test_any_import_order_gives_the_same_companies(session, make_owner) -> None:
    exports = [
        export(
            "one",
            T1,
            [person("a", "Acme"), person("b", "Gifthealth"), person("c", "Initech")],
            positions=[r.Position("Gift Health, Inc.", "Intern", None, None, None, None)],
        ),
        export(
            "two",
            T2,
            [person("a", "ACME Corp"), person("b", "Globex"), person("d", "gift health")],
            company_follows=[r.CompanyFollow("Acme, Inc.", WHEN)],
        ),
        export("three", T3, [person("a", "Acme, Inc."), person("d", "gift health")]),
    ]
    expected = None
    for order in itertools.permutations(exports):
        owner = make_owner("-".join(parsed.source_sha256 for parsed in order))
        for parsed in order:
            import_export(session, owner.id, parsed)
        result = dump(session, owner.id)
        check_links(session, owner.id)
        if expected is None:
            expected = result
            stored = companies(session, owner.id)
            assert set(stored) == {"acme", "gifthealth", "initech", "globex"}
            assert stored["acme"].aliases == ["ACME Corp", "Acme", "Acme, Inc."]
            assert stored["acme"].display_name == "Acme, Inc."
            assert stored["gifthealth"].display_name == "gift health"
        assert result["companies"] == expected["companies"]
        assert result == expected, [parsed.source_sha256 for parsed in order]


def test_synthetic_import_order_does_not_change_companies(session, make_owner, v1, v2) -> None:
    forward, backward = make_owner("fwd"), make_owner("back")
    for parsed in (v1, v2):
        import_export(session, forward.id, parsed)
    for parsed in (v2, v1):
        import_export(session, backward.id, parsed)
    expected = dump(session, forward.id)
    assert expected["companies"]
    assert any(dict(row)["company_id"] for row in expected["connections"])
    assert dump(session, backward.id) == expected


def test_owners_get_separate_companies(session, make_owner, v1, v2) -> None:
    ada, bob = make_owner("Ada"), make_owner("Bob")
    import_export(session, ada.id, v1)
    ada_before = dump(session, ada.id)
    ada_ids = {row.id for row in companies(session, ada.id).values()}

    import_export(session, bob.id, v1)
    import_export(session, bob.id, v2)
    bob_companies = companies(session, bob.id)
    bob_ids = {row.id for row in bob_companies.values()}

    # Same names, separate rows; importing for Bob left Ada's alone.
    assert set(companies(session, ada.id)) <= set(bob_companies)
    assert ada_ids and bob_ids and not ada_ids & bob_ids
    assert {row.id for row in companies(session, ada.id).values()} == ada_ids
    assert dump(session, ada.id) == ada_before
    check_links(session, ada.id)
    check_links(session, bob.id)

    # No company link crosses owners, in any table.
    for model, _ in COMPANY_SOURCES:
        crossed = session.scalar(
            select(func.count())
            .select_from(model)
            .join(Company, Company.id == model.company_id)
            .where(Company.owner_id != model.owner_id)
        )
        assert crossed == 0, model.__name__

    # Rebuilding one owner never touches the other, even when their data is stale.
    for row in session.scalars(select(Company).where(Company.owner_id == ada.id)):
        row.display_name = "stale"
    session.flush()
    rebuild_companies(session, bob.id)
    names = session.scalars(select(Company.display_name).where(Company.owner_id == ada.id)).all()
    assert set(names) == {"stale"}
    rebuild_companies(session, ada.id)
    assert dump(session, ada.id) == ada_before
