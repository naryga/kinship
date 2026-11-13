"""`kinship embed` end to end against the throwaway database.

The command opens its own session and commits, so these tests commit their
owners too and delete them afterwards (ON DELETE CASCADE removes the rest).
"""

from collections.abc import Callable, Iterator

import pytest
import sqlalchemy as sa
from rag_seeding import CHUNKS_WITHOUT_MESSAGES, MESSAGE_CHUNKS, chunk_rows, seed_sources
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

import kinship.config
import kinship.db.engine
from kinship.cli import app
from kinship.config import Settings
from kinship.db.models import Owner


@pytest.fixture
def factory(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """The CLI's session factory and settings, pointed at the test database, with no key."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(kinship.db.engine, "get_sessionmaker", lambda: factory)
    use_settings(monkeypatch)
    return factory


def use_settings(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    settings = Settings(_env_file=None, **{"openai_api_key": "", "kinship_embedder": "", **values})  # type: ignore[call-arg]
    monkeypatch.setattr(kinship.config, "get_settings", lambda: settings)


@pytest.fixture
def committed_owner(factory: sessionmaker[Session]) -> Iterator[Callable[[str, str], int]]:
    """Factory: a committed owner with seeded source rows; returns the owner id."""
    owner_ids: list[int] = []

    def _make(display_name: str, tag: str) -> int:
        with factory() as s:
            owner = Owner(display_name=display_name)
            s.add(owner)
            s.flush()
            owner_ids.append(owner.id)
            seed_sources(s, owner.id, tag)
            s.commit()
            return owner.id

    try:
        yield _make
    finally:
        with factory() as s:
            s.execute(sa.delete(Owner).where(Owner.id.in_(owner_ids)))
            s.commit()


def run(*args: str):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(app, ["embed", *args])


def test_embed_by_id_commits_chunks_and_prints_a_summary(
    factory: sessionmaker[Session], committed_owner: Callable[[str, str], int]
) -> None:
    owner_id = committed_owner("Cli Owner", "alpha")
    other_id = committed_owner("Cli Other", "beta")

    result = run("--owner", str(owner_id), "--embedder", "fake")

    assert result.exit_code == 0, result.output
    assert f"owner {owner_id} with fake-hash-v1 (1536 dimensions)" in result.output
    assert "recommendations" in result.output and "total" in result.output
    assert "Messages were not included" in result.output
    with factory() as s:
        rows = chunk_rows(s, owner_id)
        assert len(rows) == CHUNKS_WITHOUT_MESSAGES
        assert all(
            row.embedding is not None and row.model == "fake-hash-v1" for row in rows.values()
        )
        assert all(row.dim == 1536 for row in rows.values())
        assert chunk_rows(s, other_id) == {}

    again = run("--owner", str(owner_id), "--embedder", "fake")
    assert again.exit_code == 0, again.output
    total = next(line for line in again.output.splitlines() if line.startswith("total"))
    assert total.split() == ["total", "0", "0", "0", "0", str(CHUNKS_WITHOUT_MESSAGES)]


def test_embed_by_display_name_with_messages_then_purge(
    factory: sessionmaker[Session], committed_owner: Callable[[str, str], int]
) -> None:
    owner_id = committed_owner("Cli Named Owner", "alpha")

    result = run("--owner", "Cli Named Owner", "--embedder", "fake", "--include-messages")
    assert result.exit_code == 0, result.output
    assert "Messages were not included" not in result.output
    with factory() as s:
        assert len(chunk_rows(s, owner_id)) == CHUNKS_WITHOUT_MESSAGES + MESSAGE_CHUNKS

    result = run("--owner", "Cli Named Owner", "--embedder", "fake", "--purge-messages")
    assert result.exit_code == 0, result.output
    with factory() as s:
        rows = chunk_rows(s, owner_id)
        assert len(rows) == CHUNKS_WITHOUT_MESSAGES
        assert not any(key[0] == "messages" for key in rows)


def test_embed_unknown_or_ambiguous_owner_fails(
    factory: sessionmaker[Session], committed_owner: Callable[[str, str], int]
) -> None:
    committed_owner("Cli Twin", "alpha")
    committed_owner("Cli Twin", "beta")

    missing = run("--owner", "Nobody By This Name", "--embedder", "fake")
    assert missing.exit_code == 1 and "No owner" in missing.output
    missing_id = run("--owner", "999999999", "--embedder", "fake")
    assert missing_id.exit_code == 1 and "No owner" in missing_id.output
    twins = run("--owner", "Cli Twin", "--embedder", "fake")
    assert twins.exit_code == 1 and "use the id" in twins.output
    with factory() as s:
        assert s.scalar(sa.text("select count(*) from chunks")) == 0


def test_embed_without_a_key_fails_and_never_falls_back_to_fake(
    factory: sessionmaker[Session], committed_owner: Callable[[str, str], int]
) -> None:
    owner_id = committed_owner("Cli Keyless", "alpha")

    result = run("--owner", str(owner_id))

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
    with factory() as s:
        assert chunk_rows(s, owner_id) == {}


def test_embed_needs_the_flag_to_use_fake_vectors(
    factory: sessionmaker[Session],
    committed_owner: Callable[[str, str], int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id = committed_owner("Cli Env Fake", "alpha")
    use_settings(monkeypatch, kinship_embedder="fake")

    result = run("--owner", str(owner_id))
    assert result.exit_code == 1 and "--embedder fake" in result.output
    with factory() as s:
        assert chunk_rows(s, owner_id) == {}

    assert run("--owner", str(owner_id), "--embedder", "fake").exit_code == 0


def test_embed_rejects_bad_options(factory: sessionmaker[Session]) -> None:
    both = run("--owner", "1", "--embedder", "fake", "--include-messages", "--purge-messages")
    assert both.exit_code == 1 and "cannot be used together" in both.output
    unknown = run("--owner", "1", "--embedder", "voyage")
    assert unknown.exit_code == 1 and "Unknown embedder" in unknown.output
    assert run("--embedder", "fake").exit_code == 2  # --owner is required
