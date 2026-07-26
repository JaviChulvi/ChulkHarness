"""Outside-in contracts for restart-safe hosted approvals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from chulk import ExecutionScope
from chulk.approvals import (
    ApprovalConflictError,
    ApprovalDecision,
    ApprovalOutcomeKind,
    ApprovalStatus,
    ApprovalSubmission,
    ApprovalValidation,
    AsyncSQLiteApprovalStore,
    DurableApprovalService,
    ImmediateApprovalAdapter,
    SQLiteApprovalStore,
)
from chulk.runs import (
    AsyncSQLiteRunStore,
    RunStatus,
    RunSubmission,
    SQLiteRunStore,
    StepDefinition,
)
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
)


def _scope() -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant-a",
        workspace_id="workspace",
        actor_id="operator",
        agent_id="support-agent",
        agent_version="1.0.0",
        run_id="run-approval",
        grants=frozenset({"tickets:write"}),
    )


def _submission() -> RunSubmission:
    return RunSubmission(
        idempotency_key="approval-trigger",
        input_digest="sha256:input",
        definition_digest="sha256:definition",
        steps=(StepDefinition(id="update", name="Update ticket"),),
    )


def _approval(
    *,
    expires_at: datetime | None = None,
) -> ApprovalSubmission:
    return ApprovalSubmission(
        step_id="update",
        tool_name="update_ticket",
        tool_version="2.0.0",
        schema_version="3",
        arguments_digest="sha256:arguments",
        policy_version="policy-7",
        preview={"ticket_id": "42", "token": "must-not-leak"},
        expires_at=expires_at
        or datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _validation(
    scope: ExecutionScope,
    **changes: object,
) -> ApprovalValidation:
    values = {
        "scope": scope,
        "tool_name": "update_ticket",
        "tool_version": "2.0.0",
        "schema_version": "3",
        "arguments_digest": "sha256:arguments",
        "policy_version": "policy-7",
        "authority_valid": True,
        "credentials_available": True,
    }
    values.update(changes)
    return ApprovalValidation(**values)  # type: ignore[arg-type]


def _running(
    runs: SQLiteRunStore,
    scope: ExecutionScope,
):
    runs.submit(scope, _submission())
    claim = runs.claim(scope, worker_id="worker-a")
    assert claim is not None
    runs.start_step(scope, claim, "update")
    return claim


def test_approval_can_be_decided_in_another_process_and_resume_after_restart(
    tmp_path,
) -> None:
    path = tmp_path / "control.sqlite"
    scope = _scope()
    runs = SQLiteRunStore(path)
    claim = _running(runs, scope)
    released: list[str] = []
    service = DurableApprovalService(
        SQLiteApprovalStore(path),
        runs,
        release_budget=lambda _, run: released.append(run.id),
    )

    paused = service.request(scope, claim, _approval())
    assert paused.kind is ApprovalOutcomeKind.PAUSED
    assert paused.run.status is RunStatus.WAITING_FOR_APPROVAL
    assert released == [scope.run_id]

    operator = DurableApprovalService(
        SQLiteApprovalStore(path),
        SQLiteRunStore(path),
    )
    operator.decide(
        scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator-b",
        reason="verified ticket change",
        idempotency_key="decision-1",
    )

    restarted = DurableApprovalService(
        SQLiteApprovalStore(path),
        SQLiteRunStore(path),
    )
    resumed = restarted.resume(
        scope,
        paused.approval.id,
        _validation(scope),
        actor="worker-b",
    )

    assert resumed.kind is ApprovalOutcomeKind.RESUMED
    assert resumed.approval.status is ApprovalStatus.CONSUMED
    assert resumed.run.status is RunStatus.QUEUED
    names = [
        event.name
        for event in SQLiteRunStore(path).events(scope, scope.run_id)
    ]
    assert names == [
        "run.queued",
        "run.started",
        "step.started",
        "approval.requested",
        "run.paused",
        "approval.decided",
        "approval.consumed",
        "run.resumed",
    ]


def test_consumed_approval_recovers_resume_but_cannot_be_consumed_twice(
    tmp_path,
) -> None:
    path = tmp_path / "control.sqlite"
    scope = _scope()
    runs = SQLiteRunStore(path)
    claim = _running(runs, scope)
    approvals = SQLiteApprovalStore(path)
    service = DurableApprovalService(approvals, runs)
    paused = service.request(scope, claim, _approval())
    approved = service.decide(
        scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator",
        reason="approved",
        idempotency_key="decision-1",
    )
    consumed = approvals.consume(
        scope,
        approved.id,
        expected_revision=approved.revision,
    )

    with pytest.raises(ApprovalConflictError, match="already consumed"):
        approvals.consume(
            scope,
            approved.id,
            expected_revision=consumed.revision,
        )

    recovered = DurableApprovalService(
        SQLiteApprovalStore(path),
        SQLiteRunStore(path),
    ).resume(
        scope,
        approved.id,
        _validation(scope),
        actor="recovery-worker",
    )
    assert recovered.kind is ApprovalOutcomeKind.RESUMED
    assert recovered.run.status is RunStatus.QUEUED


@pytest.mark.parametrize(
    ("changes", "reason", "expected_kind"),
    [
        (
            {"tool_version": "2.1.0"},
            "tool version changed",
            ApprovalOutcomeKind.INVALIDATED,
        ),
        (
            {"schema_version": "4"},
            "schema version changed",
            ApprovalOutcomeKind.INVALIDATED,
        ),
        (
            {"arguments_digest": "sha256:changed"},
            "arguments digest changed",
            ApprovalOutcomeKind.INVALIDATED,
        ),
        (
            {"policy_version": "policy-8"},
            "policy version changed",
            ApprovalOutcomeKind.INVALIDATED,
        ),
        (
            {"authority_valid": False},
            "authority was revoked",
            ApprovalOutcomeKind.REVOKED_AUTHORITY,
        ),
        (
            {"credentials_available": False},
            "credentials",
            ApprovalOutcomeKind.UNAVAILABLE_INTEGRATION,
        ),
    ],
)
def test_resume_revalidates_every_execution_fact(
    tmp_path,
    changes,
    reason,
    expected_kind,
) -> None:
    path = tmp_path / f"{reason.replace(' ', '-')}.sqlite"
    scope = _scope()
    runs = SQLiteRunStore(path)
    claim = _running(runs, scope)
    service = DurableApprovalService(SQLiteApprovalStore(path), runs)
    paused = service.request(scope, claim, _approval())
    service.decide(
        scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator",
        reason="approved",
        idempotency_key="decision-1",
    )

    outcome = service.resume(
        scope,
        paused.approval.id,
        _validation(scope, **changes),
        actor="worker-b",
    )

    assert outcome.kind is expected_kind
    assert outcome.approval.status is ApprovalStatus.INVALIDATED
    assert reason in outcome.approval.decision_reason
    assert outcome.run.status is RunStatus.CANCELLED


def test_denial_and_expiry_produce_typed_terminal_outcomes(tmp_path) -> None:
    denial_path = tmp_path / "denial.sqlite"
    scope = _scope()
    denial_runs = SQLiteRunStore(denial_path)
    denial_claim = _running(denial_runs, scope)
    denial_service = DurableApprovalService(
        SQLiteApprovalStore(denial_path),
        denial_runs,
    )
    denied = denial_service.request(scope, denial_claim, _approval())
    denial_service.decide(
        scope,
        denied.approval.id,
        ApprovalDecision.DENY,
        decided_by="operator",
        reason="unsafe change",
        idempotency_key="deny-1",
    )
    denied_outcome = denial_service.resume(
        scope,
        denied.approval.id,
        _validation(scope),
        actor="worker",
    )
    assert denied_outcome.kind is ApprovalOutcomeKind.DENIED
    assert denied_outcome.run.status is RunStatus.CANCELLED

    expiry_path = tmp_path / "expiry.sqlite"
    expiry_runs = SQLiteRunStore(expiry_path)
    expiry_claim = _running(expiry_runs, scope)
    expiry_store = SQLiteApprovalStore(expiry_path)
    expiry_service = DurableApprovalService(expiry_store, expiry_runs)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    expiring = expiry_service.request(
        scope,
        expiry_claim,
        _approval(expires_at=expires_at),
    )
    expiry_store.expire(now=expires_at + timedelta(seconds=1))
    expired_outcome = expiry_service.resume(
        scope,
        expiring.approval.id,
        _validation(scope),
        actor="worker",
    )
    assert expired_outcome.kind is ApprovalOutcomeKind.EXPIRED
    assert expired_outcome.run.status is RunStatus.CANCELLED


def test_immediate_adapter_records_decision_without_releasing_worker(
    tmp_path,
) -> None:
    path = tmp_path / "control.sqlite"
    scope = _scope()
    runs = SQLiteRunStore(path)
    _running(runs, scope)
    approvals = SQLiteApprovalStore(path)
    adapter = ImmediateApprovalAdapter(
        approvals,
        runs,
        scope=scope,
        step_id="update",
        decide=lambda request: request.tool_name == "update_ticket",
    )
    request = PermissionRequest(
        tool_name="update_ticket",
        permission_level=ToolPermissionLevel.WRITE,
        arguments={"ticket_id": "42"},
        reason="tool uses write permission",
        tool_identity={"version": "2.0.0", "schema_version": "3"},
        tool_policy={"version": "policy-7"},
        arguments_digest="sha256:arguments",
    )
    record = PermissionDecisionRecord(
        tool_name="update_ticket",
        permission_level=ToolPermissionLevel.WRITE,
        decision=PermissionDecision.ASK,
        reason="requires approval",
    )

    assert adapter(request, record)
    stored = approvals.list(scope)
    assert len(stored) == 1
    assert stored[0].status is ApprovalStatus.CONSUMED
    assert runs.get(scope, scope.run_id).status is RunStatus.RUNNING


def test_audit_log_contains_safe_metadata_not_raw_arguments_or_secrets(
    tmp_path,
) -> None:
    path = tmp_path / "control.sqlite"
    scope = _scope()
    runs = SQLiteRunStore(path)
    claim = _running(runs, scope)
    service = DurableApprovalService(SQLiteApprovalStore(path), runs)
    paused = service.request(scope, claim, _approval())
    service.decide(
        scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator",
        reason="reviewed",
        idempotency_key="decision-1",
    )

    conn = sqlite3.connect(path)
    payloads = [
        json.loads(row[0])
        for row in conn.execute(
            "SELECT payload_json FROM durable_audit_events ORDER BY created_at"
        )
    ]
    conn.close()
    serialized = json.dumps(payloads)
    assert "must-not-leak" not in serialized
    assert "raw_arguments" not in serialized
    assert "sha256:arguments" in serialized


@pytest.mark.asyncio
async def test_async_sqlite_adapters_cover_run_and_approval_transitions(
    tmp_path,
) -> None:
    path = tmp_path / "async-control.sqlite"
    scope = _scope()
    runs = AsyncSQLiteRunStore(path)
    await runs.submit(scope, _submission())
    claim = await runs.claim(scope, worker_id="async-worker")
    assert claim is not None
    await runs.start_step(scope, claim, "update")
    approvals = AsyncSQLiteApprovalStore(path)
    created = await approvals.create(scope, _approval())
    decided = await approvals.decide(
        scope,
        created.id,
        ApprovalDecision.APPROVE,
        decided_by="async-operator",
        reason="approved",
        idempotency_key="async-decision",
    )
    consumed = await approvals.consume(
        scope,
        decided.id,
        expected_revision=decided.revision,
    )

    assert consumed.status is ApprovalStatus.CONSUMED
    assert len(await approvals.list(scope)) == 1
