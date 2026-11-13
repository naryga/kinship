"""SQLAlchemy models for the Kinship schema.

Every table except `owners` is an *owned* table: it mixes in `OwnedMixin`, which
adds a NOT NULL, indexed `owner_id` foreign key. `tests/test_schema.py`
introspects the migrated database and fails if a new table forgets it.

Schema changes need an Alembic migration (`uv run alembic revision --autogenerate`);
a test checks that these models and the migrations agree.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

EMBEDDING_DIM = 1536
"""Dimension of `chunks.embedding` (OpenAI text-embedding-3-small)."""

DIRECTIONS = ("received", "given")
"""Allowed values of `endorsements.direction` and `recommendations.direction`."""

INVITATION_DIRECTIONS = ("incoming", "outgoing")
"""Allowed values of `invitations.direction` (LinkedIn's INCOMING / OUTGOING, lower-cased)."""

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _pk() -> Mapped[int]:
    return mapped_column(BigInteger, Identity(), primary_key=True, sort_order=-2)


def _timestamp() -> Mapped[datetime | None]:
    return mapped_column(DateTime(timezone=True))


def _created_now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def _connection_fk() -> Mapped[int | None]:
    """Nullable link to the matching connection, resolved at import via `profile_url`."""
    return mapped_column(ForeignKey("connections.id", ondelete="SET NULL"), index=True)


def _company_fk() -> Mapped[int | None]:
    """Nullable link to the normalized company, resolved at import."""
    return mapped_column(ForeignKey("companies.id", ondelete="SET NULL"), index=True)


def _direction_check(values: tuple[str, ...]) -> CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in values)
    return CheckConstraint(f"direction IN ({allowed})", name="direction")


class OwnedMixin:
    """Adds the mandatory, indexed `owner_id` to an owned table."""

    @declared_attr
    def owner_id(cls) -> Mapped[int]:
        return mapped_column(
            ForeignKey("owners.id", ondelete="CASCADE"), nullable=False, index=True, sort_order=-1
        )


class Owner(Base):
    __tablename__ = "owners"

    id: Mapped[int] = _pk()
    display_name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_now()


class Import(OwnedMixin, Base):
    """One imported LinkedIn export. Re-importing the same archive is a no-op."""

    __tablename__ = "imports"
    __table_args__ = (UniqueConstraint("owner_id", "source_sha256"),)

    id: Mapped[int] = _pk()
    source_sha256: Mapped[str] = mapped_column(Text)
    exported_at: Mapped[datetime | None] = _timestamp()
    imported_at: Mapped[datetime] = _created_now()
    file_counts: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))


class Company(OwnedMixin, Base):
    __tablename__ = "companies"
    __table_args__ = (UniqueConstraint("owner_id", "name_normalized"),)

    id: Mapped[int] = _pk()
    name_normalized: Mapped[str] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text)
    aliases: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))


class Connection(OwnedMixin, Base):
    """A first-degree connection.

    `profile_url` is the canonical profile URL and may be NULL (some export rows
    have none; import then falls back to name + `connected_on`). The unique
    constraint treats NULLs as distinct, so it only applies to rows with a URL.
    """

    __tablename__ = "connections"
    __table_args__ = (UniqueConstraint("owner_id", "profile_url"),)

    id: Mapped[int] = _pk()
    profile_url: Mapped[str | None] = mapped_column(Text)
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)
    company_raw: Mapped[str | None] = mapped_column(Text)
    company_id: Mapped[int | None] = _company_fk()
    position: Mapped[str | None] = mapped_column(Text)
    connected_on: Mapped[date | None] = mapped_column(Date)
    first_seen_import_id: Mapped[int | None] = mapped_column(
        ForeignKey("imports.id", ondelete="SET NULL")
    )
    last_seen_import_id: Mapped[int | None] = mapped_column(
        ForeignKey("imports.id", ondelete="SET NULL")
    )
    removed_at: Mapped[datetime | None] = _timestamp()


class ConnectionHistory(OwnedMixin, Base):
    """A connection's company/position timeline: one row per import that changed it.

    The first import that contains the connection gets a row, and so does every
    later import (by `imports.exported_at`) whose company or position differs
    from the previous import that contained it. See `kinship.ingest`.
    """

    __tablename__ = "connection_history"

    id: Mapped[int] = _pk()
    connection_id: Mapped[int] = mapped_column(
        ForeignKey("connections.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[int] = mapped_column(ForeignKey("imports.id", ondelete="CASCADE"))
    company_raw: Mapped[str | None] = mapped_column(Text)
    position: Mapped[str | None] = mapped_column(Text)


class ConnectionSighting(OwnedMixin, Base):
    """The company/position a connection had in one import: one row per (connection, import).

    This is the importer's observation log. `connection_history` is derived
    from it, which is what lets exports be imported in any order (see
    `kinship.ingest`). Read-side code should use `connection_history`.
    """

    __tablename__ = "connection_sightings"
    __table_args__ = (UniqueConstraint("connection_id", "import_id"),)

    id: Mapped[int] = _pk()
    connection_id: Mapped[int] = mapped_column(ForeignKey("connections.id", ondelete="CASCADE"))
    import_id: Mapped[int] = mapped_column(ForeignKey("imports.id", ondelete="CASCADE"), index=True)
    company_raw: Mapped[str | None] = mapped_column(Text)
    position: Mapped[str | None] = mapped_column(Text)


class Message(OwnedMixin, Base):
    """A direct message. The table may stay empty: `messages.csv` is optional."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_owner_id_conversation_id", "owner_id", "conversation_id"),)

    id: Mapped[int] = _pk()
    conversation_id: Mapped[str | None] = mapped_column(Text)
    from_name: Mapped[str | None] = mapped_column(Text)
    to_names: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    sent_at: Mapped[datetime | None] = _timestamp()
    subject: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    connection_id: Mapped[int | None] = _connection_fk()


class MyPosition(OwnedMixin, Base):
    """A row of the owner's own `Positions.csv`."""

    __tablename__ = "my_positions"

    id: Mapped[int] = _pk()
    company: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    started_on: Mapped[date | None] = mapped_column(Date)
    ended_on: Mapped[date | None] = mapped_column(Date)


class Profile(OwnedMixin, Base):
    """The owner's own profile: one row per owner; only these three fields are kept."""

    __tablename__ = "profiles"
    __table_args__ = (UniqueConstraint("owner_id"),)

    id: Mapped[int] = _pk()
    headline: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    industry: Mapped[str | None] = mapped_column(Text)


class Skill(OwnedMixin, Base):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("owner_id", "name"),)

    id: Mapped[int] = _pk()
    name: Mapped[str] = mapped_column(Text)


class Education(OwnedMixin, Base):
    __tablename__ = "education"

    id: Mapped[int] = _pk()
    school_name: Mapped[str | None] = mapped_column(Text)
    degree_name: Mapped[str | None] = mapped_column(Text)
    started_on: Mapped[date | None] = mapped_column(Date)
    ended_on: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
    activities: Mapped[str | None] = mapped_column(Text)


class Endorsement(OwnedMixin, Base):
    """A skill endorsement the owner received from, or gave to, the named person."""

    __tablename__ = "endorsements"
    __table_args__ = (_direction_check(DIRECTIONS),)

    id: Mapped[int] = _pk()
    direction: Mapped[str] = mapped_column(Text)
    skill_name: Mapped[str | None] = mapped_column(Text)
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    profile_url: Mapped[str | None] = mapped_column(Text)
    endorsed_at: Mapped[datetime | None] = _timestamp()
    status: Mapped[str | None] = mapped_column(Text)
    connection_id: Mapped[int | None] = _connection_fk()


class Recommendation(OwnedMixin, Base):
    """A recommendation received from, or given to, the named person.

    The export has no profile URL here, so `connection_id` is matched by name.
    `text` is kept for embedding.
    """

    __tablename__ = "recommendations"
    __table_args__ = (_direction_check(DIRECTIONS),)

    id: Mapped[int] = _pk()
    direction: Mapped[str] = mapped_column(Text)
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    company: Mapped[str | None] = mapped_column(Text)
    job_title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = _timestamp()
    status: Mapped[str | None] = mapped_column(Text)
    connection_id: Mapped[int | None] = _connection_fk()


class Invitation(OwnedMixin, Base):
    """A connection invitation. `profile_url` is the other person's canonical URL."""

    __tablename__ = "invitations"
    __table_args__ = (_direction_check(INVITATION_DIRECTIONS),)

    id: Mapped[int] = _pk()
    direction: Mapped[str] = mapped_column(Text)
    from_name: Mapped[str | None] = mapped_column(Text)
    to_name: Mapped[str | None] = mapped_column(Text)
    sent_at: Mapped[datetime | None] = _timestamp()
    message: Mapped[str | None] = mapped_column(Text)
    inviter_profile_url: Mapped[str | None] = mapped_column(Text)
    invitee_profile_url: Mapped[str | None] = mapped_column(Text)
    profile_url: Mapped[str | None] = mapped_column(Text)
    connection_id: Mapped[int | None] = _connection_fk()


class ConnectionNote(OwnedMixin, Base):
    """A private note the owner wrote about a connection (`Notes.csv`)."""

    __tablename__ = "connection_notes"

    id: Mapped[int] = _pk()
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    profile_url: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = _timestamp()
    edited_at: Mapped[datetime | None] = _timestamp()
    connection_id: Mapped[int | None] = _connection_fk()


class CompanyFollow(OwnedMixin, Base):
    __tablename__ = "company_follows"

    id: Mapped[int] = _pk()
    organization: Mapped[str] = mapped_column(Text)
    followed_at: Mapped[datetime | None] = _timestamp()
    company_id: Mapped[int | None] = _company_fk()


class JobApplication(OwnedMixin, Base):
    """A job the owner applied to. Contact email/phone columns are not stored."""

    __tablename__ = "job_applications"

    id: Mapped[int] = _pk()
    applied_at: Mapped[datetime | None] = _timestamp()
    company_name: Mapped[str | None] = mapped_column(Text)
    job_title: Mapped[str | None] = mapped_column(Text)
    job_url: Mapped[str | None] = mapped_column(Text)
    resume_name: Mapped[str | None] = mapped_column(Text)
    question_and_answers: Mapped[str | None] = mapped_column(Text)
    company_id: Mapped[int | None] = _company_fk()


class SavedJob(OwnedMixin, Base):
    __tablename__ = "saved_jobs"

    id: Mapped[int] = _pk()
    saved_at: Mapped[datetime | None] = _timestamp()
    job_url: Mapped[str | None] = mapped_column(Text)
    job_title: Mapped[str | None] = mapped_column(Text)
    company_name: Mapped[str | None] = mapped_column(Text)
    company_id: Mapped[int | None] = _company_fk()


class Chunk(OwnedMixin, Base):
    """A retrievable piece of text from another table (`source_table`, `source_id`).

    `tsv` is generated by Postgres from `text`. `embedding` stays NULL until the
    chunk is embedded; `content_hash` lets the embedder detect changed text.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("owner_id", "source_table", "source_id", "chunk_index"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = _pk()
    source_table: Mapped[str] = mapped_column(Text)
    source_id: Mapped[int] = mapped_column(BigInteger)
    chunk_index: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIM))
    model: Mapped[str | None] = mapped_column(Text)
    dim: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _created_now()
