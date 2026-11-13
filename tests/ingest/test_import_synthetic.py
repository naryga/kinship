"""Importer acceptance tests on the synthetic exports #1 and #2, in both orders."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from kinship.cli import app
from kinship.db.models import (
    Connection,
    ConnectionHistory,
    Endorsement,
    Import,
    Invitation,
    Message,
    Owner,
    Profile,
    Recommendation,
)
from kinship.ingest import (
    EmptyExportError,
    ImportStore,
    OwnerNotFoundError,
    import_export,
    run_import,
)
from kinship.parser import ExportError, parse_export
from kinship.synth import diff_truth, generate_export

from .conftest import V1_AT, V1_DIR, V2_AT, V2_DIR, dump

TRUTH = diff_truth()


def connections_by_url(session: Session, owner_id: int) -> dict[str | None, Connection]:
    rows = session.scalars(select(Connection).where(Connection.owner_id == owner_id))
    return {row.profile_url: row for row in rows if row.profile_url}


def count(session: Session, model, owner_id: int) -> int:
    return session.scalar(select(func.count()).select_from(model).where(model.owner_id == owner_id))


def test_truth_is_the_documented_seed_42_diff() -> None:
    assert (len(TRUTH.added), len(TRUTH.removed), len(TRUTH.job_changed)) == (4, 3, 9)
    assert TRUTH.blank_url_count == 2


def test_first_import_inserts_everything(session, owner, v1) -> None:
    summary = import_export(session, owner.id, v1)

    assert not summary.already_imported
    assert summary.is_newest
    assert (summary.new, summary.changed, summary.removed) == (TRUTH.v1_count, 0, 0)
    assert (summary.unchanged, summary.restored) == (0, 0)
    assert summary.inserted["connections"] == TRUTH.v1_count
    assert summary.inserted["messages"] == len(v1.messages)

    rows = session.scalars(select(Connection).where(Connection.owner_id == owner.id)).all()
    assert len(rows) == TRUTH.v1_count
    assert sum(row.profile_url is None for row in rows) == TRUTH.blank_url_count
    for row in rows:
        assert row.first_seen_import_id == row.last_seen_import_id == summary.import_id
        assert row.removed_at is None
        assert row.company_id is None
    # The first-seen state of every connection opens its history.
    assert count(session, ConnectionHistory, owner.id) == TRUTH.v1_count

    record = session.get(Import, summary.import_id)
    assert record.exported_at == V1_AT
    assert record.file_counts["connections"] == TRUTH.v1_count


def test_same_export_twice_is_a_no_op(session, owner, v1) -> None:
    first = import_export(session, owner.id, v1)
    before = dump(session, owner.id)

    again = import_export(session, owner.id, v1)

    assert again.already_imported
    assert again.import_id == first.import_id
    assert (again.new, again.changed, again.removed) == (0, 0, 0)
    assert "already imported" in again.format()
    assert dump(session, owner.id) == before
    assert count(session, Import, owner.id) == 1


def test_newer_export_diffs_against_the_older_one(session, owner, v1, v2) -> None:
    first = import_export(session, owner.id, v1)
    before = connections_by_url(session, owner.id)
    old_state = {url: (row.company_raw, row.position) for url, row in before.items()}
    old_ids = {url: row.id for url, row in before.items()}

    summary = import_export(session, owner.id, v2)

    assert summary.is_newest
    assert summary.new == len(TRUTH.added) == 4
    assert summary.changed == len(TRUTH.job_changed) == 9
    assert summary.removed == len(TRUTH.removed) == 3
    assert summary.unchanged == TRUTH.unchanged_count
    assert summary.restored == 0
    assert "4 new, 9 changed, 3 removed" in summary.format()

    session.expire_all()
    after = connections_by_url(session, owner.id)
    assert count(session, Connection, owner.id) == TRUTH.v1_count + len(TRUTH.added)
    assert {url: row.id for url, row in after.items() if url in old_ids} == old_ids

    # Removed connections are kept, flagged with the export that no longer had them.
    for url in TRUTH.removed:
        assert after[url].removed_at == V2_AT
        assert after[url].last_seen_import_id == first.import_id
        assert (after[url].company_raw, after[url].position) == old_state[url]
    assert {url for url, row in after.items() if row.removed_at} == set(TRUTH.removed)

    for url in TRUTH.added:
        assert after[url].first_seen_import_id == summary.import_id
    parsed_v2 = {item.profile_url: item for item in v2.connections}
    for url, row in after.items():
        if url in TRUTH.removed:
            continue
        assert row.last_seen_import_id == summary.import_id
        assert (row.company_raw, row.position) == (parsed_v2[url].company, parsed_v2[url].position)

    # One history row per change, pointing at the import that showed the new state.
    history = session.scalars(
        select(ConnectionHistory).where(
            ConnectionHistory.owner_id == owner.id,
            ConnectionHistory.import_id == summary.import_id,
        )
    ).all()
    by_connection = {row.connection_id: row for row in history}
    assert len(history) == len(TRUTH.job_changed) + len(TRUTH.added)
    for url in TRUTH.job_changed:
        row = by_connection[after[url].id]
        assert (row.company_raw, row.position) == (after[url].company_raw, after[url].position)
        assert (row.company_raw, row.position) != old_state[url]
    # A spelling-only company change is still a raw change.
    assert set(TRUTH.company_renamed) <= set(TRUTH.job_changed)
    assert summary.inserted["connection_history"] == len(history)


def test_older_export_after_newer_does_not_regress(session, owner, v1, v2) -> None:
    newest = import_export(session, owner.id, v2)
    before = {
        url: (row.first_name, row.last_name, row.email, row.company_raw, row.position)
        for url, row in connections_by_url(session, owner.id).items()
    }
    profile_before = session.scalar(select(Profile.headline).where(Profile.owner_id == owner.id))

    summary = import_export(session, owner.id, v1)

    assert not summary.is_newest
    assert "older than the newest import" in summary.format()
    # Only the connections that v1 alone has are new, and they arrive already removed.
    assert (summary.new, summary.changed, summary.removed) == (len(TRUTH.removed), 0, 0)
    session.expire_all()
    after = connections_by_url(session, owner.id)
    for url, fields in before.items():
        row = after[url]
        assert (
            row.first_name,
            row.last_name,
            row.email,
            row.company_raw,
            row.position,
        ) == fields
        assert row.removed_at is None
        assert row.last_seen_import_id == newest.import_id
    for url in TRUTH.removed:
        assert after[url].removed_at == V2_AT
        assert after[url].first_seen_import_id == summary.import_id
    for url in TRUTH.job_changed:
        assert after[url].first_seen_import_id == summary.import_id
    for url in TRUTH.added:
        assert after[url].first_seen_import_id == newest.import_id
    assert (
        session.scalar(select(Profile.headline).where(Profile.owner_id == owner.id))
        == profile_before
    )


def test_import_order_does_not_change_the_result(session, make_owner, v1, v2) -> None:
    forward, backward, repeated = make_owner("fwd"), make_owner("back"), make_owner("again")
    for parsed in (v1, v2):
        import_export(session, forward.id, parsed)
    for parsed in (v2, v1):
        import_export(session, backward.id, parsed)
    for parsed in (v2, v2, v1, v2, v1):
        import_export(session, repeated.id, parsed)

    expected = dump(session, forward.id)
    assert expected["connections"]
    assert len(expected["connection_history"]) == (
        TRUTH.v1_count + len(TRUTH.added) + len(TRUTH.job_changed)
    )
    assert dump(session, backward.id) == expected
    assert dump(session, repeated.id) == expected

    # v1 -> v2 -> v1 again, and v2 twice, are no-ops.
    for parsed in (v1, v2):
        assert import_export(session, forward.id, parsed).already_imported
    assert dump(session, forward.id) == expected


def test_links_stay_within_the_owner_and_point_at_the_right_person(session, owner, v1, v2) -> None:
    import_export(session, owner.id, v1)
    import_export(session, owner.id, v2)
    connections = {
        row.id: row
        for row in session.scalars(select(Connection).where(Connection.owner_id == owner.id))
    }

    endorsements = session.scalars(
        select(Endorsement).where(Endorsement.owner_id == owner.id)
    ).all()
    assert {row.direction for row in endorsements} == {"received", "given"}
    assert any(row.connection_id for row in endorsements)
    for row in endorsements:
        if row.connection_id is not None:
            assert connections[row.connection_id].profile_url == row.profile_url

    invitations = session.scalars(select(Invitation).where(Invitation.owner_id == owner.id)).all()
    assert {row.direction for row in invitations} <= {"incoming", "outgoing"}
    for row in invitations:
        other = row.inviter_profile_url if row.direction == "incoming" else row.invitee_profile_url
        assert row.profile_url == other
        if row.connection_id is not None:
            assert connections[row.connection_id].profile_url == row.profile_url
    assert any(row.connection_id for row in invitations)
    assert any(row.connection_id is None for row in invitations)  # never accepted

    recommendations = session.scalars(
        select(Recommendation).where(Recommendation.owner_id == owner.id)
    ).all()
    assert any(row.connection_id for row in recommendations)
    for row in recommendations:
        if row.connection_id is not None:
            person = connections[row.connection_id]
            assert (person.first_name, person.last_name) == (row.first_name, row.last_name)

    messages = session.scalars(select(Message).where(Message.owner_id == owner.id)).all()
    assert messages and all(isinstance(row.to_names, list) for row in messages)
    linked = [row for row in messages if row.connection_id is not None]
    assert linked
    for row in linked:
        person = connections[row.connection_id]
        name = f"{person.first_name} {person.last_name}"
        assert name == row.from_name or name in row.to_names


def test_owners_are_isolated(session, make_owner, v1, v2) -> None:
    ada, bob, cyd = make_owner("Ada"), make_owner("Bob"), make_owner("Cyd")
    ada_v1 = import_export(session, ada.id, v1)
    ada_before = dump(session, ada.id)

    # A different export for another owner leaves Ada's rows alone.
    bob_v2 = import_export(session, bob.id, v2)
    assert (bob_v2.new, bob_v2.changed, bob_v2.removed) == (TRUTH.v2_count, 0, 0)
    assert dump(session, ada.id) == ada_before
    assert count(session, Connection, bob.id) == TRUTH.v2_count

    # The SHA-256 no-op is per owner: the same export is new to Bob and to Cyd.
    bob_v1 = import_export(session, bob.id, v1)
    assert not bob_v1.already_imported
    assert bob_v1.import_id != ada_v1.import_id
    cyd_v1 = import_export(session, cyd.id, v1)
    assert (cyd_v1.already_imported, cyd_v1.new) == (False, TRUTH.v1_count)
    assert dump(session, ada.id) == ada_before == dump(session, cyd.id)

    # Ada's own diff is still exactly the truth, whatever the others imported.
    ada_v2 = import_export(session, ada.id, v2)
    assert (ada_v2.new, ada_v2.changed, ada_v2.removed) == (4, 9, 3)
    assert dump(session, ada.id) == dump(session, bob.id)
    assert dump(session, cyd.id) == ada_before

    # Every link stays inside its owner.
    for model in (Endorsement, Invitation, Recommendation, Message, ConnectionHistory):
        crossed = session.scalar(
            select(func.count())
            .select_from(model)
            .join(Connection, Connection.id == model.connection_id)
            .where(Connection.owner_id != model.owner_id)
        )
        assert crossed == 0, model.__name__
    for owner_row in (ada, bob, cyd):
        imports = set(session.scalars(select(Import.id).where(Import.owner_id == owner_row.id)))
        used = set(
            session.scalars(
                select(Connection.last_seen_import_id).where(Connection.owner_id == owner_row.id)
            )
        )
        assert used <= imports


def test_store_requires_an_owner_and_only_sees_its_rows(session, make_owner, v1) -> None:
    ada, bob = make_owner("Ada"), make_owner("Bob")
    import_export(session, ada.id, v1)

    with pytest.raises(TypeError):
        ImportStore(session, None)  # type: ignore[arg-type]
    store = ImportStore(session, bob.id)
    assert store.rows(Connection) == []
    assert store.find_import(v1.source_sha256) is None
    assert store.sightings() == store.history() == store.messages() == []
    # Ids that belong to another owner are out of reach.
    ada_ids = list(session.scalars(select(Connection.id).where(Connection.owner_id == ada.id)))
    store.delete_ids(Connection, ada_ids)
    linked = select(Message.id).where(
        Message.owner_id == ada.id, Message.connection_id.is_not(None)
    )
    message_id = session.scalar(linked.limit(1))
    store.set_message_connections({message_id: None})
    session.expire_all()
    assert session.get(Message, message_id).connection_id is not None
    assert count(session, Connection, ada.id) == TRUTH.v1_count
    assert not ImportStore(session, 10**9).lock_owner()


def test_unknown_owner_is_an_error(session, v1) -> None:
    with pytest.raises(OwnerNotFoundError):
        import_export(session, 10**9, v1)


def test_export_without_connections_is_refused(session, owner, v1, tmp_path) -> None:
    import_export(session, owner.id, v1)
    before = dump(session, owner.id)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "Connections.csv").write_text("Name,Link\nSomebody,https://example.com\n")
    parsed = parse_export(broken, exported_at=V2_AT)
    assert parsed.connections == []

    with pytest.raises(EmptyExportError, match="no connections"):
        import_export(session, owner.id, parsed)

    assert issubclass(EmptyExportError, ExportError)
    assert dump(session, owner.id) == before


@pytest.fixture
def session_factory(session: Session, monkeypatch: pytest.MonkeyPatch):
    """Sessions that commit into the test's rolled-back transaction; also used by the CLI."""
    factory = sessionmaker(
        bind=session.connection(), join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    monkeypatch.setattr("kinship.db.engine.get_sessionmaker", lambda: factory)
    return factory


def test_run_import_commits_one_transaction(session, owner, session_factory) -> None:
    first = run_import(V1_DIR, owner.id, exported_at=V1_AT, session_factory=session_factory)
    second = run_import(V2_DIR, owner.id, exported_at=V2_AT, session_factory=session_factory)

    assert (first.new, first.owner_id, first.owner_created) == (TRUTH.v1_count, owner.id, False)
    assert (second.new, second.changed, second.removed) == (4, 9, 3)
    assert count(session, Connection, owner.id) == TRUTH.v1_count + 4


def test_failure_mid_import_leaves_nothing_behind(session, session_factory, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch, pytest.raises(RuntimeError, match="boom"):
        patch.setattr("kinship.ingest.importer._import_messages", explode)
        run_import(V1_DIR, "Brand New Owner", exported_at=V1_AT, session_factory=session_factory)

    assert session.scalar(select(Owner.id).where(Owner.display_name == "Brand New Owner")) is None
    for model in (Import, Connection, ConnectionHistory, Endorsement):
        assert session.scalar(select(func.count()).select_from(model)) == 0

    summary = run_import(V1_DIR, "Brand New Owner", exported_at=V1_AT)
    assert summary.owner_created
    assert summary.new == TRUTH.v1_count


def test_zip_exports_carry_their_own_date(session, make_owner, tmp_path, v1, v2) -> None:
    """Zips need no `exported_at`: member timestamps decide which export is newer."""
    older = generate_export(tmp_path, version=1, as_zip=True)
    newer = generate_export(tmp_path, version=2, as_zip=True)
    parsed_old, parsed_new = parse_export(older), parse_export(newer)
    assert parsed_old.exported_at.date() == V1_AT.date()
    assert parsed_new.exported_at.date() == V2_AT.date()

    forward, backward = make_owner("fwd"), make_owner("back")
    import_export(session, forward.id, parsed_old)
    summary = import_export(session, forward.id, parsed_new)
    assert (summary.new, summary.changed, summary.removed) == (4, 9, 3)
    import_export(session, backward.id, parsed_new)
    late = import_export(session, backward.id, parsed_old)
    assert (late.is_newest, late.new, late.removed) == (False, 3, 0)
    assert dump(session, forward.id) == dump(session, backward.id)


def test_cli_import(session, session_factory, tmp_path) -> None:
    runner = CliRunner()
    newer = generate_export(tmp_path, version=2, as_zip=True)

    result = runner.invoke(
        app, ["import", str(V1_DIR), "--owner", "Cli Owner", "--exported-at", V1_AT.isoformat()]
    )
    assert result.exit_code == 0, result.output
    owner_id = session.scalar(select(Owner.id).where(Owner.display_name == "Cli Owner"))
    assert f"owner {owner_id} (created)" in result.output
    assert f"{TRUTH.v1_count} new, 0 changed, 0 removed" in result.output

    # By id this time, from a zip whose date comes from its members.
    result = runner.invoke(app, ["import", str(newer), "--owner", str(owner_id)])
    assert result.exit_code == 0, result.output
    assert "4 new, 9 changed, 3 removed" in result.output
    assert "(created)" not in result.output

    result = runner.invoke(app, ["import", str(newer), "--owner", "Cli Owner"])
    assert result.exit_code == 0, result.output
    assert "already imported" in result.output
    assert (
        session.scalar(
            select(func.count()).select_from(Owner).where(Owner.display_name == "Cli Owner")
        )
        == 1
    )
    assert count(session, Import, owner_id) == 2


def test_cli_import_errors(session, session_factory, tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["import", str(tmp_path / "missing.zip"), "--owner", "Nobody"])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output

    result = runner.invoke(
        app, ["import", str(V1_DIR), "--owner", "999999999", "--exported-at", V1_AT.isoformat()]
    )
    assert result.exit_code == 1
    assert "owner 999999999 does not exist" in result.output

    result = runner.invoke(app, ["import", str(V1_DIR), "--owner", "x", "--exported-at", "soon"])
    assert result.exit_code == 2
    assert session.scalar(select(func.count()).select_from(Owner)) == 0
