"""company links on connection_history and my_positions

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-13 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("connection_history", "my_positions")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("company_id", sa.BigInteger(), nullable=True))
        op.create_index(op.f(f"ix_{table}_company_id"), table, ["company_id"], unique=False)
        op.create_foreign_key(
            op.f(f"fk_{table}_company_id_companies"),
            table,
            "companies",
            ["company_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_constraint(op.f(f"fk_{table}_company_id_companies"), table, type_="foreignkey")
        op.drop_index(op.f(f"ix_{table}_company_id"), table_name=table)
        op.drop_column(table, "company_id")
