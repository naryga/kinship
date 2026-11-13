"""The migrated schema: ownership rules, constraints, FTS and vector columns."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kinship.db.models import (
    EMBEDDING_DIM,
    Base,
    Chunk,
    Connection,
    Endorsement,
    Import,
    Owner,
)

UNOWNED_TABLES = {"owners", "alembic_version"}

EXPECTED_TABLES = {
    "owners",
    "imports",
    "companies",
    "connections",
    "connection_history",
    "connection_sightings",
    "messages",
    "my_positions",
    "profiles",
    "skills",
    "education",
    "endorsements",
    "recommendations",
    "invitations",
    "connection_notes",
    "company_follows",
    "job_applications",
    "saved_jobs",
    "chunks",
}


def owned_tables(engine: Engine) -> list[str]:
    return sorted(set(inspect(engine).get_table_names()) - UNOWNED_TABLES)


def test_migration_applies_to_head(engine: Engine, alembic_cfg: Config) -> None:
    head = ScriptDirectory.from_config(alembic_cfg).get_current_head()
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT version_num FROM alembic_version")) == head
        assert conn.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")) == 1
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES | {"alembic_version"}


def test_models_match_migrations(engine: Engine) -> None:
    """Fails when models.py changes without a migration (or the reverse)."""
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_every_owned_table_has_indexed_owner_fk(engine: Engine) -> None:
    """Introspects the database, so tables added later are covered automatically."""
    inspector = inspect(engine)
    tables = owned_tables(engine)
    assert tables
    for table in tables:
        columns = {c["name"]: c for c in inspector.get_columns(table)}
        assert "owner_id" in columns, f"{table} has no owner_id"
        assert columns["owner_id"]["nullable"] is False, f"{table}.owner_id is nullable"

        assert any(
            fk["constrained_columns"] == ["owner_id"]
            and fk["referred_table"] == "owners"
            and fk["referred_columns"] == ["id"]
            for fk in inspector.get_foreign_keys(table)
        ), f"{table}.owner_id has no foreign key to owners.id"

        # get_indexes() also reports the indexes that back unique constraints.
        assert any(
            index["column_names"][:1] == ["owner_id"] for index in inspector.get_indexes(table)
        ), f"{table} has no index that starts with owner_id"


def test_owner_id_is_required(session: Session) -> None:
    with pytest.raises(IntegrityError):
        session.execute(text("INSERT INTO skills (name) VALUES ('python')"))


def test_imports_unique_per_owner_and_sha(session: Session, make_owner) -> None:
    ada, bob = make_owner("Ada"), make_owner("Bob")
    session.add_all(
        [
            Import(owner_id=ada.id, source_sha256="abc"),
            Import(owner_id=ada.id, source_sha256="def"),
            Import(owner_id=bob.id, source_sha256="abc"),
        ]
    )
    session.flush()

    session.add(Import(owner_id=ada.id, source_sha256="abc"))
    with pytest.raises(IntegrityError, match="uq_imports_owner_id_source_sha256"):
        session.flush()


def test_connections_unique_per_owner_and_profile_url(session: Session, make_owner) -> None:
    ada, bob = make_owner("Ada"), make_owner("Bob")
    url = "https://www.linkedin.com/in/grace-hopper"
    session.add_all(
        [
            Connection(owner_id=ada.id, profile_url=url, first_name="Grace"),
            Connection(owner_id=bob.id, profile_url=url, first_name="Grace"),
            # Rows without a URL never collide with each other.
            Connection(owner_id=ada.id, profile_url=None, first_name="No"),
            Connection(owner_id=ada.id, profile_url=None, first_name="Url"),
        ]
    )
    session.flush()

    session.add(Connection(owner_id=ada.id, profile_url=url, first_name="Dup"))
    with pytest.raises(IntegrityError, match="uq_connections_owner_id_profile_url"):
        session.flush()


def test_direction_is_constrained(session: Session, owner: Owner) -> None:
    session.add_all(
        [
            Endorsement(owner_id=owner.id, direction="received", skill_name="SQL"),
            Endorsement(owner_id=owner.id, direction="given", skill_name="SQL"),
        ]
    )
    session.flush()

    session.add(Endorsement(owner_id=owner.id, direction="sideways", skill_name="SQL"))
    with pytest.raises(IntegrityError, match="ck_endorsements_direction"):
        session.flush()


def test_chunk_tsv_is_generated_and_gin_indexed(
    engine: Engine, session: Session, owner: Owner
) -> None:
    chunk = Chunk(
        owner_id=owner.id,
        source_table="recommendations",
        source_id=1,
        text="She migrated our billing databases with zero downtime.",
    )
    session.add(chunk)
    session.flush()
    session.refresh(chunk)

    assert chunk.chunk_index == 0
    assert chunk.embedding is None
    assert "databas" in chunk.tsv  # stemmed by the 'english' configuration
    hit = session.scalar(
        select(Chunk.id).where(
            Chunk.tsv.bool_op("@@")(text("plainto_tsquery('english', 'database')"))
        )
    )
    assert hit == chunk.id

    column = next(c for c in inspect(engine).get_columns("chunks") if c["name"] == "tsv")
    assert column["computed"]["persisted"] is True
    with engine.connect() as conn:
        indexdef = conn.scalar(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_tsv'")
        )
    assert "USING gin (tsv)" in indexdef


def test_chunk_embedding_round_trips(session: Session, owner: Owner) -> None:
    vector = [0.0] * EMBEDDING_DIM
    vector[0], vector[-1] = 1.0, 0.5
    chunk = Chunk(
        owner_id=owner.id,
        source_table="my_positions",
        source_id=7,
        text="Staff engineer",
        embedding=vector,
        model="fake",
        dim=EMBEDDING_DIM,
    )
    session.add(chunk)
    session.flush()
    session.expire(chunk)

    assert [float(x) for x in chunk.embedding] == vector
    # The operator classes are usable: cosine distance to itself is zero.
    distance = session.scalar(
        select(Chunk.embedding.cosine_distance(vector)).where(Chunk.id == chunk.id)
    )
    assert distance == pytest.approx(0.0)

    with pytest.raises(Exception, match="expected 1536 dimensions"):
        session.execute(
            text("UPDATE chunks SET embedding = '[1,2,3]' WHERE id = :id"), {"id": chunk.id}
        )


def test_chunks_unique_per_source_and_index(session: Session, owner: Owner) -> None:
    def chunk(index: int) -> Chunk:
        return Chunk(
            owner_id=owner.id, source_table="messages", source_id=3, chunk_index=index, text="hi"
        )

    session.add_all([chunk(0), chunk(1)])
    session.flush()
    session.add(chunk(1))
    with pytest.raises(IntegrityError):
        session.flush()


def test_session_fixture_rolls_back_commits(session: Session, make_owner) -> None:
    """Committed work stays inside the test's outer transaction (see conftest)."""
    make_owner("Committed")
    session.commit()
    assert session.scalar(select(Owner.id).where(Owner.display_name == "Committed"))


def test_session_fixture_starts_clean(session: Session) -> None:
    assert session.scalars(select(Owner)).all() == []
