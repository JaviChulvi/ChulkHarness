"""Outside-in contracts for the shared durable hosted-run owner."""

from __future__ import annotations

from datetime import timedelta

import pytest

from chulk import ExecutionScope
from chulk.runs import (
    EffectStatus,
    ReconciliationDecision,
    RetryPolicy,
    InMemoryRunStore,
    RunConflictError,
    RunLeaseError,
    RunStatus,
    RunSubmission,
    SQLiteRunStore,
    StepDefinition,
    StepStatus,
)


def _scope(
    *,
    run_id: str = "run-1",
    tenant_id: str = "tenant-a",
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id="workspace",
        actor_id="operator",
        agent_id="support-agent",
        agent_version="1.0.0",
        run_id=run_id,
        grants=frozenset({"catalog:read", "tickets:write"}),
    )


def _submission(
    *,
    idempotency_key: str = "trigger-1",
    input_digest: str = "sha256:input",
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=idempotency_key,
        input_digest=input_digest,
        definition_digest="sha256:definition",
        steps=(
            StepDefinition(
                id="lookup",
                name="Look up the ticket",
                retry_policy=RetryPolicy(
                    max_attempts=2,
                    initial_delay_seconds=0,
                    max_delay_seconds=0,
                ),
            ),
        ),
        budget={"max_tool_calls": 2},
        metadata={"source": "test"},
        source_event_id="event-1",
        correlation_id="correlation-1",
    )


def test_duplicate_submission_returns_existing_run_and_rejects_drift(
    tmp_path,
) -> None:
    store = SQLiteRunStore(tmp_path / "control.sqlite")
    first = store.submit(_scope(), _submission())
    duplicate = store.submit(
        _scope(run_id="run-2"),
        _submission(),
    )

    assert duplicate == first
    assert duplicate.id == "run-1"
    assert [event.name for event in store.events(_scope(), first.id)] == [
        "run.queued"
    ]

    with pytest.raises(RunConflictError, match="different input"):
        store.submit(
            _scope(run_id="run-3"),
            _submission(input_digest="sha256:changed"),
        )
    with pytest.raises(LookupError):
        store.get(_scope(tenant_id="tenant-b"), first.id)


def test_in_memory_run_store_is_filesystem_free(tmp_path) -> None:
    store = InMemoryRunStore()
    created = store.submit(_scope(), _submission())

    assert created.status is RunStatus.QUEUED
    assert list(tmp_path.iterdir()) == []
    store.close()


def test_stale_worker_cannot_checkpoint_or_complete_after_requeue(tmp_path) -> None:
    db_path = tmp_path / "control.sqlite"
    store = SQLiteRunStore(db_path)
    scope = _scope()
    store.submit(scope, _submission())
    claim = store.claim(scope, worker_id="worker-a", lease_seconds=1)
    assert claim is not None
    store.start_step(scope, claim, "lookup")

    restarted = SQLiteRunStore(db_path)
    reconciled = restarted.reconcile_expired(
        now=claim.lease_until + timedelta(seconds=1)
    )

    assert reconciled[0].status is RunStatus.QUEUED
    assert reconciled[0].step("lookup").status is StepStatus.QUEUED
    with pytest.raises(RunLeaseError):
        restarted.checkpoint(
            scope,
            claim,
            "lookup",
            kind="progress",
            payload={"ordinal": 1},
        )
    with pytest.raises(RunLeaseError):
        restarted.complete(scope, claim, result={"answer": "stale"})


def test_effect_intent_is_idempotent_and_unknown_requires_reconciliation(
    tmp_path,
) -> None:
    store = SQLiteRunStore(tmp_path / "control.sqlite")
    scope = _scope()
    store.submit(scope, _submission())
    claim = store.claim(scope, worker_id="worker-a", lease_seconds=30)
    assert claim is not None
    store.start_step(scope, claim, "lookup")
    effect = store.begin_effect(
        scope,
        claim,
        "lookup",
        logical_key="ticket:42:update",
        tool_name="update_ticket",
        tool_version="1.0.0",
        schema_version="1.0.0",
        arguments_digest="sha256:arguments",
    )
    duplicate = store.begin_effect(
        scope,
        claim,
        "lookup",
        logical_key="ticket:42:update",
        tool_name="update_ticket",
        tool_version="1.0.0",
        schema_version="1.0.0",
        arguments_digest="sha256:arguments",
    )
    assert duplicate.id == effect.id

    store.mark_effect_started(scope, claim, effect.id)
    unknown = store.mark_effect_unknown(
        scope,
        claim,
        effect.id,
        reason="transport disconnected after dispatch",
    )
    assert unknown.status is EffectStatus.UNKNOWN
    assert store.get(scope, scope.run_id).status is RunStatus.UNKNOWN

    with pytest.raises(RunLeaseError):
        store.begin_effect(
            scope,
            claim,
            "lookup",
            logical_key="ticket:42:update",
            tool_name="update_ticket",
            tool_version="1.0.0",
            schema_version="1.0.0",
            arguments_digest="sha256:arguments",
        )

    reconciled = store.reconcile_effect(
        scope,
        effect.id,
        decision=ReconciliationDecision.RETRY,
        actor="operator",
        reason="external system confirms no write occurred",
    )
    assert reconciled.effect.status is EffectStatus.INTENDED
    assert reconciled.run.status is RunStatus.QUEUED

    second_claim = store.claim(scope, worker_id="worker-b")
    assert second_claim is not None
    second_attempt = store.start_step(scope, second_claim, "lookup")
    reused = store.begin_effect(
        scope,
        second_claim,
        "lookup",
        logical_key="ticket:42:update",
        tool_name="update_ticket",
        tool_version="1.0.0",
        schema_version="1.0.0",
        arguments_digest="sha256:arguments",
    )
    assert reused.id == effect.id
    assert reused.attempt_id == second_attempt.id
    store.mark_effect_started(scope, second_claim, reused.id)
    store.complete_effect(
        scope,
        second_claim,
        reused.id,
        result_digest="sha256:result",
    )
    store.complete_step(
        scope,
        second_claim,
        "lookup",
        result={"ticket": "updated"},
    )
    completed = store.complete(
        scope,
        second_claim,
        result={"answer": "done"},
    )
    assert completed.status is RunStatus.COMPLETED

    event_names = [event.name for event in store.events(scope, scope.run_id)]
    assert event_names == [
        "run.queued",
        "run.started",
        "step.started",
        "effect.intended",
        "effect.started",
        "effect.unknown",
        "run.unknown",
        "effect.reconciled",
        "run.started",
        "step.started",
        "effect.started",
        "effect.completed",
        "step.completed",
        "run.completed",
    ]
    assert [event.sequence for event in store.events(scope, scope.run_id)] == list(
        range(1, len(event_names) + 1)
    )


def test_pause_resume_retry_and_cancellation_have_durable_boundaries(
    tmp_path,
) -> None:
    store = SQLiteRunStore(tmp_path / "control.sqlite")
    scope = _scope()
    store.submit(scope, _submission())
    first_claim = store.claim(scope, worker_id="worker-a")
    assert first_claim is not None
    store.start_step(scope, first_claim, "lookup")
    paused = store.pause_for_approval(
        scope,
        first_claim,
        "lookup",
        approval_id="approval-1",
        payload={"arguments_digest": "sha256:arguments"},
    )
    assert paused.status is RunStatus.WAITING_FOR_APPROVAL
    assert paused.step("lookup").status is StepStatus.WAITING_FOR_APPROVAL
    assert store.claim(scope, worker_id="worker-b") is None

    resumed = store.resume(
        scope,
        scope.run_id,
        actor="approver",
        reason="approval granted",
    )
    assert resumed.status is RunStatus.QUEUED
    second_claim = store.claim(scope, worker_id="worker-b")
    assert second_claim is not None
    store.start_step(scope, second_claim, "lookup")
    waiting = store.fail_step(
        scope,
        second_claim,
        "lookup",
        reason="provider overloaded",
        retryable=True,
    )
    assert waiting.status is RunStatus.WAITING_FOR_RETRY
    resumed_retry = store.resume(
        scope,
        scope.run_id,
        actor="scheduler",
        reason="retry due",
    )
    assert resumed_retry.status is RunStatus.QUEUED

    cancelled = store.request_cancellation(
        scope,
        scope.run_id,
        actor="operator",
        reason="request withdrawn",
    )
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.cancellation_requested


def test_cancellation_before_during_and_after_uncertain_effects(tmp_path) -> None:
    queued_store = SQLiteRunStore(tmp_path / "queued.sqlite")
    scope = _scope()
    queued_store.submit(scope, _submission())
    before = queued_store.request_cancellation(
        scope,
        scope.run_id,
        actor="operator",
        reason="cancel before claim",
    )
    assert before.status is RunStatus.CANCELLED

    uncertain_store = SQLiteRunStore(tmp_path / "uncertain.sqlite")
    uncertain_store.submit(scope, _submission())
    claim = uncertain_store.claim(scope, worker_id="worker-a")
    assert claim is not None
    uncertain_store.start_step(scope, claim, "lookup")
    effect = uncertain_store.begin_effect(
        scope,
        claim,
        "lookup",
        logical_key="ticket:42:update",
        tool_name="update_ticket",
        tool_version="1.0.0",
        schema_version="1.0.0",
        arguments_digest="sha256:arguments",
    )
    uncertain_store.mark_effect_started(scope, claim, effect.id)
    during = uncertain_store.cancel(
        scope,
        scope.run_id,
        actor="operator",
        reason="cancel during dispatch",
        claim=claim,
    )
    assert during.status is RunStatus.UNKNOWN
    assert during.cancellation_requested
    reconciled = uncertain_store.reconcile_effect(
        scope,
        effect.id,
        decision=ReconciliationDecision.CANCELLED,
        actor="operator",
        reason="external system confirmed cancellation",
    )
    assert reconciled.run.status is RunStatus.CANCELLED

    completed_effect_store = SQLiteRunStore(
        tmp_path / "completed-effect.sqlite"
    )
    completed_effect_store.submit(scope, _submission())
    final_claim = completed_effect_store.claim(scope, worker_id="worker-a")
    assert final_claim is not None
    completed_effect_store.start_step(scope, final_claim, "lookup")
    final_effect = completed_effect_store.begin_effect(
        scope,
        final_claim,
        "lookup",
        logical_key="ticket:42:update",
        tool_name="update_ticket",
        tool_version="1.0.0",
        schema_version="1.0.0",
        arguments_digest="sha256:arguments",
    )
    completed_effect_store.mark_effect_started(
        scope,
        final_claim,
        final_effect.id,
    )
    completed_effect_store.complete_effect(
        scope,
        final_claim,
        final_effect.id,
        result_digest="sha256:result",
    )
    after_effect = completed_effect_store.cancel(
        scope,
        scope.run_id,
        actor="operator",
        reason="stop remaining work",
        claim=final_claim,
    )
    assert after_effect.status is RunStatus.CANCELLED
