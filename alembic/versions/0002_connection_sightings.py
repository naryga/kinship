"""connection sightings

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12 23:55:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "connection_sightings",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("owner_id", sa.BigInteger(), nullable=False),
        sa.Column("connection_id", sa.BigInteger(), nullable=False),
        sa.Column("import_id", sa.BigInteger(), nullable=False),
        sa.Column("company_raw", sa.Text(), nullable=True),
        sa.Column("position", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["connection_id"],
            ["connections.id"],
            name=op.f("fk_connection_sightings_connection_id_connections"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_id"],
            ["imports.id"],
            name=op.f("fk_connection_sightings_import_id_imports"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["owners.id"],
            name=op.f("fk_connection_sightings_owner_id_owners"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connection_sightings")),
        sa.UniqueConstraint(
            "connection_id",
            "import_id",
            name=op.f("uq_connection_sightings_connection_id_import_id"),
        ),
    )
    op.create_index(
        op.f("ix_connection_sightings_import_id"),
        "connection_sightings",
        ["import_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_connection_sightings_owner_id"),
        "connection_sightings",
        ["owner_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_connection_sightings_owner_id"), table_name="connection_sightings")
    op.drop_index(op.f("ix_connection_sightings_import_id"), table_name="connection_sightings")
    op.drop_table("connection_sightings")
