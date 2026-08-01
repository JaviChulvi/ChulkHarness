"""Use fixed-size digests for hosted idempotency uniqueness.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    (
        "durable_runs",
        "durable_runs_tenant_id_workspace_id_idempotency_key_key",
    ),
    (
        "durable_run_events",
        "durable_run_events_run_id_idempotency_key_key",
    ),
    (
        "durable_approval_requests",
        "durable_approval_requests_tenant_id_workspace_id_decision_k_key",
    ),
    (
        "durable_audit_events",
        "durable_audit_events_tenant_id_workspace_id_idempotency_key_key",
    ),
    (
        "gateway_inbox",
        "gateway_inbox_adapter_account_id_idempotency_key_key",
    ),
    (
        "automation_run_requests",
        "automation_run_requests_profile_id_idempotency_key_key",
    ),
    (
        "automation_control_actions",
        "automation_control_actions_pkey",
    ),
)

_DIGEST_INDEXES: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    (
        "uq_durable_runs_idempotency_digest",
        "durable_runs",
        ("tenant_id", "workspace_id", "idempotency_key"),
        False,
    ),
    (
        "uq_durable_run_events_idempotency_digest",
        "durable_run_events",
        ("run_id", "idempotency_key"),
        True,
    ),
    (
        "uq_durable_approvals_decision_digest",
        "durable_approval_requests",
        ("tenant_id", "workspace_id", "decision_key"),
        True,
    ),
    (
        "uq_durable_audit_idempotency_digest",
        "durable_audit_events",
        ("tenant_id", "workspace_id", "idempotency_key"),
        True,
    ),
    (
        "uq_gateway_inbox_idempotency_digest",
        "gateway_inbox",
        ("adapter", "account_id", "idempotency_key"),
        False,
    ),
    (
        "uq_automation_requests_idempotency_digest",
        "automation_run_requests",
        ("profile_id", "idempotency_key"),
        False,
    ),
    (
        "uq_automation_controls_idempotency_digest",
        "automation_control_actions",
        ("profile_id", "idempotency_key"),
        False,
    ),
)

_KEY_HASH_INDEXES: tuple[tuple[str, str, str], ...] = (
    (
        "idx_durable_runs_idempotency_hash",
        "durable_runs",
        "idempotency_key",
    ),
    (
        "idx_durable_run_events_idempotency_hash",
        "durable_run_events",
        "idempotency_key",
    ),
    (
        "idx_durable_approvals_decision_hash",
        "durable_approval_requests",
        "decision_key",
    ),
    (
        "idx_durable_audit_idempotency_hash",
        "durable_audit_events",
        "idempotency_key",
    ),
    (
        "idx_gateway_inbox_idempotency_hash",
        "gateway_inbox",
        "idempotency_key",
    ),
    (
        "idx_automation_requests_idempotency_hash",
        "automation_run_requests",
        "idempotency_key",
    ),
    (
        "idx_automation_controls_idempotency_hash",
        "automation_control_actions",
        "idempotency_key",
    ),
)


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION chulk_idempotency_digest(VARIADIC parts TEXT[])
        RETURNS BYTEA
        LANGUAGE SQL
        IMMUTABLE STRICT PARALLEL SAFE
        AS $$
            SELECT sha256(
                string_agg(
                    sha256(convert_to(part, 'UTF8')),
                    ''::bytea
                    ORDER BY ordinal
                )
            )
            FROM unnest(parts) WITH ORDINALITY AS item(part, ordinal)
        $$
        """
    )
    for table, constraint in _CONSTRAINTS:
        op.execute(
            f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}"
        )
    for name, table, columns, nullable in _DIGEST_INDEXES:
        arguments = ", ".join(columns)
        expression = f"chulk_idempotency_digest({arguments})"
        if nullable:
            key = columns[-1]
            expression = (
                f"(CASE WHEN {key} IS NULL THEN NULL ELSE {expression} END)"
            )
        op.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {name} "
            f"ON {table} ({expression})"
        )
    for name, table, column in _KEY_HASH_INDEXES:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {name} "
            f"ON {table} USING HASH ({column})"
        )


def downgrade() -> None:
    raise NotImplementedError(
        "PostgreSQL hosted migrations are forward-only; restore a backup"
    )
