"""PostgreSQL schema owned by the optional hosted persistence reference."""

from __future__ import annotations

from typing import Any


SCHEMA_REVISION = "0001"


POSTGRES_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS durable_runs (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        agent_version TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        scope_json TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        definition_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0,
        cancellation_requested INTEGER NOT NULL DEFAULT 0,
        waiting_reason TEXT,
        next_retry_at TEXT,
        budget_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        error TEXT,
        claim_token TEXT,
        worker_id TEXT,
        lease_until TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE (tenant_id, workspace_id, idempotency_key),
        CHECK (cancellation_requested IN (0, 1))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_runs_claim
    ON durable_runs(status, next_retry_at, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_runs_scope
    ON durable_runs(
        tenant_id, workspace_id, agent_id, agent_version, updated_at, id
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_run_steps (
        run_id TEXT NOT NULL,
        id TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        retry_json TEXT NOT NULL,
        next_retry_at TEXT,
        last_checkpoint_id TEXT,
        error TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        PRIMARY KEY (run_id, id),
        FOREIGN KEY (run_id) REFERENCES durable_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_steps_status
    ON durable_run_steps(run_id, status, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_run_attempts (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        number INTEGER NOT NULL,
        status TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        lease_token TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        error TEXT,
        UNIQUE (run_id, step_id, number),
        FOREIGN KEY (run_id, step_id)
            REFERENCES durable_run_steps(run_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_run_checkpoints (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, sequence),
        FOREIGN KEY (attempt_id)
            REFERENCES durable_run_attempts(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_effects (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        logical_key TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        tool_version TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        arguments_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        result_digest TEXT,
        reconciliation TEXT,
        reconciled_by TEXT,
        reconciliation_reason TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (run_id, logical_key),
        FOREIGN KEY (attempt_id)
            REFERENCES durable_run_attempts(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_effects_status
    ON durable_effects(run_id, status, step_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_run_events (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        name TEXT NOT NULL,
        actor TEXT NOT NULL,
        step_id TEXT,
        payload_json TEXT NOT NULL,
        causation_id TEXT,
        correlation_id TEXT,
        idempotency_key TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, sequence),
        UNIQUE (run_id, idempotency_key),
        FOREIGN KEY (run_id) REFERENCES durable_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_approval_requests (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        effect_id TEXT,
        scope_json TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        tool_version TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        arguments_digest TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        preview_json TEXT NOT NULL,
        status TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0,
        decision TEXT,
        decided_by TEXT,
        decision_reason TEXT,
        decision_key TEXT,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        decided_at TEXT,
        consumed_at TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE (tenant_id, workspace_id, decision_key),
        FOREIGN KEY (run_id) REFERENCES durable_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_approvals_run
    ON durable_approval_requests(run_id, status, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS durable_audit_events (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        run_id TEXT,
        step_id TEXT,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        correlation_id TEXT,
        causation_id TEXT,
        idempotency_key TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (tenant_id, workspace_id, idempotency_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_durable_audit_scope
    ON durable_audit_events(tenant_id, workspace_id, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_adapters (
        adapter TEXT NOT NULL,
        account_id TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'stopped',
        instance_token TEXT,
        lease_until TEXT,
        cursor TEXT,
        legacy_adopted_at TEXT,
        stop_requested INTEGER NOT NULL DEFAULT 0,
        started_at TEXT,
        stopped_at TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (adapter, account_id),
        CHECK (state IN ('stopped', 'running')),
        CHECK (stop_requested IN (0, 1))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_inbox (
        id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        adapter TEXT NOT NULL,
        account_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        conversation_key TEXT NOT NULL,
        principal_id TEXT NOT NULL,
        destination_id TEXT NOT NULL,
        thread_id TEXT,
        envelope_json TEXT NOT NULL,
        state TEXT NOT NULL,
        execution_token TEXT,
        execution_lease_until TEXT,
        cancellation_requested INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        executed_at TEXT,
        execution_scope_json TEXT,
        agent_definition_id TEXT,
        agent_definition_version TEXT,
        agent_definition_digest TEXT,
        run_id TEXT,
        dead_lettered_at TEXT,
        UNIQUE (adapter, account_id, idempotency_key),
        CHECK (
            state IN (
                'queued', 'processing', 'executed', 'ignored',
                'cancelled', 'uncertain'
            )
        ),
        CHECK (cancellation_requested IN (0, 1))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_inbox_queue
    ON gateway_inbox(state, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_inbox_profile_queue
    ON gateway_inbox(profile_id, state, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_inbox_conversation
    ON gateway_inbox(conversation_key, state, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_inbox_run ON gateway_inbox(run_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_outbox (
        id TEXT PRIMARY KEY,
        inbox_id TEXT NOT NULL,
        profile_id TEXT NOT NULL,
        adapter TEXT NOT NULL,
        account_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        envelope_json TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        checkpoint TEXT,
        delivery_token TEXT,
        delivery_lease_until TEXT,
        next_attempt_at TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        delivered_at TEXT,
        reconciliation_required INTEGER NOT NULL DEFAULT 0,
        dead_lettered_at TEXT,
        UNIQUE (inbox_id, sequence),
        FOREIGN KEY (inbox_id) REFERENCES gateway_inbox(id) ON DELETE CASCADE,
        CHECK (sequence >= 0),
        CHECK (attempt_count >= 0),
        CHECK (reconciliation_required IN (0, 1)),
        CHECK (state IN ('pending', 'delivering', 'delivered', 'failed'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_outbox_delivery
    ON gateway_outbox(
        adapter, account_id, state, next_attempt_at, created_at, id
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_outbox_profile
    ON gateway_outbox(profile_id, state, created_at, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_outbox_reconciliation
    ON gateway_outbox(reconciliation_required, state, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_delivery_events (
        id TEXT PRIMARY KEY,
        outbox_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        state TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (outbox_id) REFERENCES gateway_outbox(id) ON DELETE CASCADE,
        CHECK (attempt >= 1)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_gateway_delivery_events_outbox
    ON gateway_delivery_events(outbox_id, recorded_at, id)
    """,
)

AUTOMATION_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS automation_jobs (
        id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        adapter TEXT NOT NULL,
        account_id TEXT NOT NULL,
        destination_id TEXT NOT NULL,
        thread_id TEXT,
        prompt TEXT NOT NULL,
        recurrence_json TEXT NOT NULL,
        next_run_at TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        status TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0,
        run_count INTEGER NOT NULL DEFAULT 0,
        max_runs INTEGER,
        budget_json TEXT NOT NULL,
        retry_json TEXT NOT NULL,
        requires_approval INTEGER NOT NULL DEFAULT 0,
        approved_at TEXT,
        claim_token TEXT,
        lease_until TEXT,
        active_run_id TEXT,
        last_run_at TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (revision >= 0),
        CHECK (run_count >= 0),
        CHECK (max_runs IS NULL OR max_runs > 0),
        CHECK (requires_approval IN (0, 1)),
        CHECK (
            (claim_token IS NULL AND lease_until IS NULL AND active_run_id IS NULL)
            OR
            (claim_token IS NOT NULL AND lease_until IS NOT NULL
                AND active_run_id IS NOT NULL)
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_jobs_due
    ON automation_jobs(profile_id, status, next_run_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_jobs_destination
    ON automation_jobs(
        profile_id, adapter, account_id, destination_id, created_at
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_runs (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        profile_id TEXT NOT NULL,
        occurrence_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        claim_token TEXT,
        worker_id TEXT,
        lease_until TEXT,
        started_at TEXT,
        finished_at TEXT,
        result_json TEXT NOT NULL DEFAULT '{}',
        trace_id TEXT,
        usage_json TEXT NOT NULL DEFAULT '{}',
        cost_json TEXT NOT NULL DEFAULT '{}',
        error TEXT,
        artifact_refs_json TEXT NOT NULL DEFAULT '[]',
        delivery_state TEXT NOT NULL DEFAULT 'none',
        delivery_error TEXT,
        trigger_event_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES automation_jobs(id) ON DELETE CASCADE,
        CHECK (attempt > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_runs_job
    ON automation_runs(profile_id, job_id, created_at DESC, id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_runs_status
    ON automation_runs(profile_id, status, lease_until)
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_delivery_attempts (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        profile_id TEXT NOT NULL,
        state TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES automation_runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_delivery_attempts
    ON automation_delivery_attempts(profile_id, run_id, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_run_requests (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        profile_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        occurrence_at TEXT NOT NULL,
        available_at TEXT NOT NULL,
        trigger_event_id TEXT,
        idempotency_key TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES automation_jobs(id) ON DELETE CASCADE,
        UNIQUE (profile_id, idempotency_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_run_requests_pending
    ON automation_run_requests(
        profile_id, status, available_at, created_at
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_job_events (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        profile_id TEXT NOT NULL,
        action TEXT NOT NULL,
        revision INTEGER NOT NULL,
        actor TEXT NOT NULL,
        run_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES automation_jobs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_job_events
    ON automation_job_events(profile_id, job_id, created_at, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_control_actions (
        profile_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        job_id TEXT NOT NULL,
        action TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        result_revision INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (profile_id, idempotency_key),
        FOREIGN KEY (job_id) REFERENCES automation_jobs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_triggers (
        id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        source_resource_id TEXT,
        secret_digest TEXT,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES automation_jobs(id) ON DELETE CASCADE,
        CHECK (enabled IN (0, 1))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_automation_triggers_source
    ON automation_triggers(profile_id, kind, source_resource_id, enabled)
    """,
    """
    CREATE TABLE IF NOT EXISTS automation_trigger_events (
        id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        trigger_id TEXT NOT NULL,
        source_event_id TEXT,
        trust TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (trigger_id)
            REFERENCES automation_triggers(id) ON DELETE CASCADE,
        UNIQUE (profile_id, trigger_id, source_event_id)
    )
    """,
)


def create_schema(connection: Any) -> None:
    """Create the initial hosted schema on a SQLAlchemy connection."""

    from sqlalchemy import text

    for statement in POSTGRES_SCHEMA_STATEMENTS + AUTOMATION_SCHEMA_STATEMENTS:
        connection.execute(text(statement))


__all__ = [
    "AUTOMATION_SCHEMA_STATEMENTS",
    "POSTGRES_SCHEMA_STATEMENTS",
    "SCHEMA_REVISION",
    "create_schema",
]
