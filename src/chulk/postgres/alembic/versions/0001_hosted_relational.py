"""Create the hosted relational reference schema.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from alembic import op

from chulk.postgres.schema import create_schema


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    create_schema(op.get_bind())


def downgrade() -> None:
    raise NotImplementedError(
        "PostgreSQL hosted migrations are forward-only; restore a backup"
    )
