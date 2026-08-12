"""Add durable agent evaluation reports.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


_EVAL_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        suite_name TEXT NOT NULL,
        dataset_digest TEXT NOT NULL,
        mode TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT NOT NULL,
        passed INTEGER NOT NULL,
        case_count INTEGER NOT NULL,
        pass_rate DOUBLE PRECISION NOT NULL,
        total_cost DOUBLE PRECISION NOT NULL,
        report_json TEXT NOT NULL,
        CHECK (passed IN (0, 1))
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_eval_runs_scope
    ON eval_runs(tenant_id, workspace_id, suite_name, started_at, id)""",
    """
    CREATE TABLE IF NOT EXISTS eval_trials (
        run_id TEXT NOT NULL,
        target_name TEXT NOT NULL,
        case_id TEXT NOT NULL,
        trial_number INTEGER NOT NULL,
        passed INTEGER NOT NULL,
        duration_seconds DOUBLE PRECISION NOT NULL,
        exception TEXT,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (run_id, target_name, case_id, trial_number),
        FOREIGN KEY (run_id) REFERENCES eval_runs(id) ON DELETE CASCADE,
        CHECK (passed IN (0, 1))
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_eval_trials_case
    ON eval_trials(run_id, target_name, case_id, trial_number)""",
    """
    CREATE TABLE IF NOT EXISTS eval_turns (
        run_id TEXT NOT NULL,
        target_name TEXT NOT NULL,
        case_id TEXT NOT NULL,
        trial_number INTEGER NOT NULL,
        turn_index INTEGER NOT NULL,
        status TEXT NOT NULL,
        duration_seconds DOUBLE PRECISION NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (run_id, target_name, case_id, trial_number, turn_index),
        FOREIGN KEY (run_id, target_name, case_id, trial_number)
            REFERENCES eval_trials(run_id, target_name, case_id, trial_number)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_grades (
        run_id TEXT NOT NULL,
        target_name TEXT NOT NULL,
        case_id TEXT NOT NULL,
        trial_number INTEGER NOT NULL,
        grader TEXT NOT NULL,
        score DOUBLE PRECISION NOT NULL,
        passed INTEGER NOT NULL,
        error TEXT,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (run_id, target_name, case_id, trial_number, grader),
        FOREIGN KEY (run_id, target_name, case_id, trial_number)
            REFERENCES eval_trials(run_id, target_name, case_id, trial_number)
            ON DELETE CASCADE,
        CHECK (passed IN (0, 1))
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_eval_grades_grader
    ON eval_grades(run_id, grader, score)""",
    """
    CREATE TABLE IF NOT EXISTS eval_baselines (
        tenant_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        suite_name TEXT NOT NULL,
        report_id TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (tenant_id, workspace_id, suite_name),
        FOREIGN KEY (report_id) REFERENCES eval_runs(id) ON DELETE CASCADE
    )
    """,
)


def upgrade() -> None:
    for statement in _EVAL_SCHEMA_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for table in ("eval_baselines", "eval_grades", "eval_turns", "eval_trials", "eval_runs"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
