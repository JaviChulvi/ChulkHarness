"""Add resumable evaluation lifecycle and provenance.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from alembic import op

from chulk.postgres.schema import EVAL_RESUME_SCHEMA_STATEMENTS


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for statement in EVAL_RESUME_SCHEMA_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_eval_runs_status")
    op.drop_column("eval_grades", "grader_identity")
    op.drop_column("eval_grades", "grader_version")
    op.drop_column("eval_trials", "target_fingerprint")
    for column in (
        "sampling_json",
        "grader_versions_json",
        "target_fingerprints_json",
        "suite_fingerprint",
        "git_revision",
        "chulk_version",
        "updated_at",
        "status",
    ):
        op.drop_column("eval_runs", column)
