"""Add durable agent evaluation reports.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op

from chulk.postgres.schema import EVAL_SCHEMA_STATEMENTS


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for statement in EVAL_SCHEMA_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for table in ("eval_baselines", "eval_grades", "eval_turns", "eval_trials", "eval_runs"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
