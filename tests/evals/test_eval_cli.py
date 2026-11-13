"""`kinship eval` through the CLI: it makes, uses and drops its own database."""

import json
from pathlib import Path

import pytest
from conftest import CI_FLOOR, REPO_ROOT
from sqlalchemy import Engine, text
from typer.testing import CliRunner, Result

import kinship.config
import kinship.db.scratch
from kinship.cli import app
from kinship.config import Settings
from kinship.db.scratch import ScratchDatabaseError, scratch_database

START = "<!-- eval-results:fake:start -->"
END = "<!-- eval-results:fake:end -->"


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the throwaway databases created during the test."""
    names: list[str] = []
    original = kinship.db.scratch.new_database_name

    def recording(prefix: str) -> str:
        names.append(original(prefix))
        return names[-1]

    monkeypatch.setattr(kinship.db.scratch, "new_database_name", recording)
    return names


def databases(engine: Engine, names: list[str]) -> list[str]:
    """Which of `names` exist on the server right now."""
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT datname FROM pg_database WHERE datname = ANY(:names)"), {"names": names}
        )
        return [row[0] for row in rows]


def run(*args: str) -> Result:
    return CliRunner().invoke(app, ["eval", *args])


def test_json_report_and_the_database_is_dropped(engine: Engine, created: list[str]) -> None:
    result = run("--format", "json", "--min-recall", str(CI_FLOOR), "--min-mrr", "0.5")

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["embedder"] == "fake-hash-v1"
    assert report["k"] == 5
    assert report["modes"] == ["fts", "vector", "hybrid"]
    assert len(report["queries"]) >= 20
    assert report["overall"]["hybrid"]["recall_at_k"] >= CI_FLOOR
    assert report["by_kind"]["keyword"]["hybrid"]["recall_at_k"] >= 0.9

    assert len(created) == 1 and created[0].startswith("kinship_eval_")
    assert databases(engine, created) == []


def test_markdown_output_is_the_readme_table(engine: Engine, created: list[str]) -> None:
    """The command generates its own export; the README table must be what it prints."""
    result = run("--format", "markdown")

    assert result.exit_code == 0, result.output
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert result.stdout.strip() == readme.split(START)[1].split(END)[0].strip()
    assert databases(engine, created) == []


def test_below_a_threshold_exits_1_after_printing_the_report(
    engine: Engine, created: list[str]
) -> None:
    result = run("--min-recall", "1.0", "--min-mrr", "1.0", "--k", "3")

    assert result.exit_code == 1
    assert "hybrid r@3" in result.stdout  # the report is still shown
    assert "p01" in result.stdout
    assert "hybrid recall@3" in result.stderr and "is below 1.00" in result.stderr
    assert "hybrid MRR" in result.stderr
    assert len(created) == 1
    assert databases(engine, created) == []


def test_database_is_dropped_when_the_run_fails(
    engine: Engine, created: list[str], tmp_path: Path
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "x", "kind": "keyword", "query": "anything", "expected": '
        '[{"source": "my_positions", "company": "Nowhere", "title": "Nothing"}]}\n'
    )

    result = run("--golden", str(golden))

    assert result.exit_code == 1
    assert "does not match this owner's data" in result.stderr
    assert "Nowhere" in result.stderr
    assert len(created) == 1  # it got as far as creating the database
    assert databases(engine, created) == []


def test_errors_before_any_database_is_created(created: list[str], tmp_path: Path) -> None:
    missing = run("--golden", str(tmp_path / "missing.jsonl"))
    assert missing.exit_code == 1
    assert "cannot read golden set" in missing.stderr

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n")
    malformed = run("--golden", str(bad))
    assert malformed.exit_code == 1
    assert "bad.jsonl, line 1" in malformed.stderr

    assert run("--format", "yaml").exit_code == 2
    assert run("--embedder", "telepathy").exit_code == 1
    assert run("--export", str(tmp_path / "no-such-export")).exit_code == 1
    assert created == []


def test_openai_without_a_key_is_refused(
    created: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None, openai_api_key="", kinship_embedder="")  # type: ignore[call-arg]
    monkeypatch.setattr(kinship.config, "get_settings", lambda: settings)

    result = run("--embedder", "openai")

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.stderr
    assert created == []


def test_unreachable_server_is_a_clear_error(created: list[str]) -> None:
    result = run("--database-url", "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none")

    assert result.exit_code == 1
    assert "could not use the Postgres server" in result.stderr


def test_help_says_what_the_command_needs() -> None:
    result = run("--help")

    assert result.exit_code == 0
    text_ = " ".join(result.stdout.split())
    assert "source checkout" in text_
    assert "throwaway database" in text_
    assert "OPENAI_API_KEY" in text_


# -- kinship.db.scratch -------------------------------------------------------


def test_scratch_database_is_migrated_and_dropped_on_error(
    engine: Engine, created: list[str]
) -> None:
    from sqlalchemy import create_engine, inspect

    with (
        pytest.raises(RuntimeError, match="boom"),
        scratch_database(prefix="kinship_evaltest") as url,
    ):
        scratch = create_engine(url)
        try:
            tables = set(inspect(scratch).get_table_names())
        finally:
            scratch.dispose()
        assert {"owners", "chunks", "alembic_version"} <= tables
        assert databases(engine, created) == created
        raise RuntimeError("boom")

    assert len(created) == 1 and created[0].startswith("kinship_evaltest_")
    assert databases(engine, created) == []


def test_scratch_database_rejects_odd_prefixes() -> None:
    with pytest.raises(ValueError, match="prefix"), scratch_database(prefix='x"; DROP'):
        pass


def test_scratch_database_needs_a_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, created: list[str]
) -> None:
    fake_module = tmp_path / "site-packages" / "kinship" / "db" / "scratch.py"
    monkeypatch.setattr(kinship.db.scratch, "__file__", str(fake_module))

    with pytest.raises(ScratchDatabaseError, match="source checkout"), scratch_database():
        pass
    assert created == []
