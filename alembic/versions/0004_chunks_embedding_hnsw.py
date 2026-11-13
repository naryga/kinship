"""HNSW index on chunks.embedding for cosine distance

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-13 14:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "ix_chunks_embedding_hnsw"


def upgrade() -> None:
    # Default build parameters (m = 16, ef_construction = 64). Rows whose
    # embedding is NULL are not in the index.
    op.create_index(
        INDEX,
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(INDEX, table_name="chunks", postgresql_using="hnsw")
