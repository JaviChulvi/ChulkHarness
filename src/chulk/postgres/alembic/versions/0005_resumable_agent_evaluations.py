"""Add resumable evaluation lifecycle and provenance.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from alembic import op


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


_EVAL_RESUME_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'completed'",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS updated_at TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS chulk_version TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS git_revision TEXT",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS suite_fingerprint TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS target_fingerprints_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS grader_versions_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS sampling_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE eval_trials ADD COLUMN IF NOT EXISTS target_fingerprint TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE eval_grades ADD COLUMN IF NOT EXISTS grader_version TEXT NOT NULL DEFAULT '1'",
    "ALTER TABLE eval_grades ADD COLUMN IF NOT EXISTS grader_identity TEXT NOT NULL DEFAULT ''",
    "UPDATE eval_runs SET updated_at = ended_at WHERE updated_at = ''",
    """CREATE INDEX IF NOT EXISTS idx_eval_runs_status
    ON eval_runs(tenant_id, workspace_id, status, updated_at, id)""",
)


def upgrade() -> None:
    for statement in _EVAL_RESUME_SCHEMA_STATEMENTS:
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
