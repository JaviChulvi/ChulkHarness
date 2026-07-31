"""Add durable parent-child run orchestration.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from alembic import op


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS durable_run_parents (
            parent_run_id TEXT PRIMARY KEY,
            policy_json TEXT NOT NULL,
            aggregation_status TEXT NOT NULL DEFAULT 'open',
            aggregation_revision INTEGER NOT NULL DEFAULT 0,
            aggregation_key TEXT,
            aggregate_result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (parent_run_id)
                REFERENCES durable_runs(id) ON DELETE CASCADE,
            CHECK (aggregation_status IN ('open', 'completed')),
            CHECK (aggregation_revision >= 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS durable_child_runs (
            child_run_id TEXT PRIMARY KEY,
            parent_run_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            idempotency_key TEXT NOT NULL,
            definition_revision TEXT NOT NULL,
            definition_digest TEXT NOT NULL,
            input_digest TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            terminal_evidence_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (parent_run_id, ordinal),
            FOREIGN KEY (parent_run_id)
                REFERENCES durable_run_parents(parent_run_id) ON DELETE CASCADE,
            FOREIGN KEY (child_run_id)
                REFERENCES durable_runs(id) ON DELETE CASCADE,
            CHECK (ordinal > 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_durable_child_runs_parent
        ON durable_child_runs(parent_run_id, ordinal)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS durable_child_progress (
            child_run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            actor TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (child_run_id, sequence),
            FOREIGN KEY (child_run_id)
                REFERENCES durable_child_runs(child_run_id) ON DELETE CASCADE,
            CHECK (sequence > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS durable_parent_completion_outbox (
            id TEXT PRIMARY KEY,
            parent_run_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            worker_id TEXT,
            lease_token TEXT,
            lease_until TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            delivered_at TEXT,
            FOREIGN KEY (parent_run_id)
                REFERENCES durable_run_parents(parent_run_id) ON DELETE CASCADE,
            CHECK (status IN ('pending', 'claimed', 'delivered', 'unknown')),
            CHECK (attempt_count >= 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_durable_parent_completion_claim
        ON durable_parent_completion_outbox(status, created_at, id)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            uq_durable_child_runs_idempotency_digest
        ON durable_child_runs (
            chulk_idempotency_digest(parent_run_id, idempotency_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_durable_child_runs_idempotency_hash
        ON durable_child_runs USING HASH (idempotency_key)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            uq_durable_child_progress_idempotency_digest
        ON durable_child_progress (
            chulk_idempotency_digest(child_run_id, idempotency_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_durable_child_progress_idempotency_hash
        ON durable_child_progress USING HASH (idempotency_key)
        """
    )


def downgrade() -> None:
    raise NotImplementedError(
        "PostgreSQL hosted migrations are forward-only; restore a backup"
    )
