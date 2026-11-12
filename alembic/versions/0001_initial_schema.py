"""initial schema

Revision ID: 0001
Revises: -
Create Date: 2026-11-12 23:39:05.162248
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "owners",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_owners")),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("source_table", sa.Text(), nullable=False),
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1536), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("dim", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["owners.id"], name=op.f("fk_chunks_owner_id_owners"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "owner_id",
            "source_table",
            "source_id",
            "chunk_index",
            name=op.f("uq_chunks_owner_id_source_table_source_id_chunk_index"),
        ),
    )
    op.create_index(op.f("ix_chunks_owner_id"), "chunks", ["owner_id"], unique=False)
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin")
    op.create_table(
        "companies",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("name_normalized", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "aliases",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_companies_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_companies")),
        sa.UniqueConstraint(
            "owner_id", "name_normalized", name=op.f("uq_companies_owner_id_name_normalized")
        ),
    )
    op.create_index(op.f("ix_companies_owner_id"), "companies", ["owner_id"], unique=False)
    op.create_table(
        "education",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("school_name", sa.Text(), nullable=True),
        sa.Column("degree_name", sa.Text(), nullable=True),
        sa.Column("started_on", sa.Date(), nullable=True),
        sa.Column("ended_on", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("activities", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_education_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_education")),
    )
    op.create_index(op.f("ix_education_owner_id"), "education", ["owner_id"], unique=False)
    op.create_table(
        "imports",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("exported_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "imported_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "file_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["owners.id"], name=op.f("fk_imports_owner_id_owners"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_imports")),
        sa.UniqueConstraint(
            "owner_id", "source_sha256", name=op.f("uq_imports_owner_id_source_sha256")
        ),
    )
    op.create_index(op.f("ix_imports_owner_id"), "imports", ["owner_id"], unique=False)
    op.create_table(
        "my_positions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("company", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("started_on", sa.Date(), nullable=True),
        sa.Column("ended_on", sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_my_positions_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_my_positions")),
    )
    op.create_index(op.f("ix_my_positions_owner_id"), "my_positions", ["owner_id"], unique=False)
    op.create_table(
        "profiles",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("headline", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("industry", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_profiles_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_profiles")),
        sa.UniqueConstraint("owner_id", name=op.f("uq_profiles_owner_id")),
    )
    op.create_index(op.f("ix_profiles_owner_id"), "profiles", ["owner_id"], unique=False)
    op.create_table(
        "skills",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["owners.id"], name=op.f("fk_skills_owner_id_owners"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_skills")),
        sa.UniqueConstraint("owner_id", "name", name=op.f("uq_skills_owner_id_name")),
    )
    op.create_index(op.f("ix_skills_owner_id"), "skills", ["owner_id"], unique=False)
    op.create_table(
        "company_follows",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("organization", sa.Text(), nullable=False),
        sa.Column("followed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("company_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_company_follows_company_id_companies"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_company_follows_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_company_follows")),
    )
    op.create_index(
        op.f("ix_company_follows_company_id"), "company_follows", ["company_id"], unique=False
    )
    op.create_index(
        op.f("ix_company_follows_owner_id"), "company_follows", ["owner_id"], unique=False
    )
    op.create_table(
        "connections",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("profile_url", sa.Text(), nullable=True),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("company_raw", sa.Text(), nullable=True),
        sa.Column("company_id", sa.BigInteger(), nullable=True),
        sa.Column("position", sa.Text(), nullable=True),
        sa.Column("connected_on", sa.Date(), nullable=True),
        sa.Column("first_seen_import_id", sa.BigInteger(), nullable=True),
        sa.Column("last_seen_import_id", sa.BigInteger(), nullable=True),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_connections_company_id_companies"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["first_seen_import_id"],
            ["imports.id"],
            name=op.f("fk_connections_first_seen_import_id_imports"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["last_seen_import_id"],
            ["imports.id"],
            name=op.f("fk_connections_last_seen_import_id_imports"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_connections_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connections")),
        sa.UniqueConstraint(
            "owner_id", "profile_url", name=op.f("uq_connections_owner_id_profile_url")
        ),
    )
    op.create_index(op.f("ix_connections_company_id"), "connections", ["company_id"], unique=False)
    op.create_index(op.f("ix_connections_owner_id"), "connections", ["owner_id"], unique=False)
    op.create_table(
        "job_applications",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("company_name", sa.Text(), nullable=True),
        sa.Column("job_title", sa.Text(), nullable=True),
        sa.Column("job_url", sa.Text(), nullable=True),
        sa.Column("resume_name", sa.Text(), nullable=True),
        sa.Column("question_and_answers", sa.Text(), nullable=True),
        sa.Column("company_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_job_applications_company_id_companies"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_job_applications_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_applications")),
    )
    op.create_index(
        op.f("ix_job_applications_company_id"), "job_applications", ["company_id"], unique=False
    )
    op.create_index(
        op.f("ix_job_applications_owner_id"), "job_applications", ["owner_id"], unique=False
    )
    op.create_table(
        "saved_jobs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("saved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("job_url", sa.Text(), nullable=True),
        sa.Column("job_title", sa.Text(), nullable=True),
        sa.Column("company_name", sa.Text(), nullable=True),
        sa.Column("company_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_saved_jobs_company_id_companies"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_saved_jobs_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_saved_jobs")),
    )
    op.create_index(op.f("ix_saved_jobs_company_id"), "saved_jobs", ["company_id"], unique=False)
    op.create_index(op.f("ix_saved_jobs_owner_id"), "saved_jobs", ["owner_id"], unique=False)
    op.create_table(
        "connection_history",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("connection_id", sa.BigInteger(), nullable=False),
        sa.Column("import_id", sa.BigInteger(), nullable=False),
        sa.Column("company_raw", sa.Text(), nullable=True),
        sa.Column("position", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_connection_history_connection_id_connections"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_id"],
            ["imports.id"],
            name=op.f("fk_connection_history_import_id_imports"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_connection_history_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connection_history")),
    )
    op.create_index(
        op.f("ix_connection_history_connection_id"),
        "connection_history",
        ["connection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_connection_history_owner_id"), "connection_history", ["owner_id"], unique=False
    )
    op.create_table(
        "connection_notes",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("profile_url", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("connection_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_connection_notes_connection_id_connections"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_connection_notes_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connection_notes")),
    )
    op.create_index(
        op.f("ix_connection_notes_connection_id"),
        "connection_notes",
        ["connection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_connection_notes_owner_id"), "connection_notes", ["owner_id"], unique=False
    )
    op.create_table(
        "endorsements",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("skill_name", sa.Text(), nullable=True),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("profile_url", sa.Text(), nullable=True),
        sa.Column("endorsed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("connection_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(
            "direction IN ('received', 'given')", name=op.f("ck_endorsements_direction")
        ),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_endorsements_connection_id_connections"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_endorsements_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_endorsements")),
    )
    op.create_index(
        op.f("ix_endorsements_connection_id"), "endorsements", ["connection_id"], unique=False
    )
    op.create_index(op.f("ix_endorsements_owner_id"), "endorsements", ["owner_id"], unique=False)
    op.create_table(
        "invitations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("from_name", sa.Text(), nullable=True),
        sa.Column("to_name", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("inviter_profile_url", sa.Text(), nullable=True),
        sa.Column("invitee_profile_url", sa.Text(), nullable=True),
        sa.Column("profile_url", sa.Text(), nullable=True),
        sa.Column("connection_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(
            "direction IN ('incoming', 'outgoing')", name=op.f("ck_invitations_direction")
        ),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_invitations_connection_id_connections"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_invitations_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_invitations")),
    )
    op.create_index(
        op.f("ix_invitations_connection_id"), "invitations", ["connection_id"], unique=False
    )
    op.create_index(op.f("ix_invitations_owner_id"), "invitations", ["owner_id"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=True),
        sa.Column("from_name", sa.Text(), nullable=True),
        sa.Column(
            "to_names",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("connection_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_messages_connection_id_connections"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_messages_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
    )
    op.create_index(op.f("ix_messages_connection_id"), "messages", ["connection_id"], unique=False)
    op.create_index(op.f("ix_messages_owner_id"), "messages", ["owner_id"], unique=False)
    op.create_index(
        "ix_messages_owner_id_conversation_id",
        "messages",
        ["owner_id", "conversation_id"],
        unique=False,
    )
    op.create_table(
        "recommendations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("company", sa.Text(), nullable=True),
        sa.Column("job_title", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("connection_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(
            "direction IN ('received', 'given')", name=op.f("ck_recommendations_direction")
        ),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_recommendations_connection_id_connections"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_recommendations_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_recommendations")),
    )
    op.create_index(
        op.f("ix_recommendations_connection_id"), "recommendations", ["connection_id"], unique=False
    )
    op.create_index(
        op.f("ix_recommendations_owner_id"), "recommendations", ["owner_id"], unique=False
    )


def downgrade() -> None:
    # Indexes go with their tables. The vector extension is left installed: it may
    # be shared with other schemas and is harmless to keep.
    for table in (
        "recommendations",
        "messages",
        "invitations",
        "endorsements",
        "connection_notes",
        "connection_history",
        "saved_jobs",
        "job_applications",
        "connections",
        "company_follows",
        "skills",
        "profiles",
        "my_positions",
        "imports",
        "education",
        "companies",
        "chunks",
        "owners",
    ):
        op.drop_table(table)
