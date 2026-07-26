"""Durable approval coordination across run and approval stores."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from chulk.approvals.models import (
    ApprovalDecision,
    ApprovalOutcomeKind,
    ApprovalRequest,
    ApprovalResumeOutcome,
    ApprovalStatus,
    ApprovalSubmission,
    ApprovalValidation,
    DurableApprovalPaused,
    PausedRunOutcome,
)
from chulk.approvals.protocols import ApprovalStore, AsyncApprovalStore
from chulk.hosting.scope import ExecutionScope, ExecutionScopeError
from chulk.runs.models import RunClaim, RunRecord, RunStatus
from chulk.runs.protocols import AsyncRunStore, RunStore
from chulk.tools.permissions import (
    PermissionDecision as ToolPermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
)
from chulk.tools.policy import schema_digest
from chulk.tools.registry import ToolExecutionContext


BudgetRelease = Callable[[ExecutionScope, RunRecord], None]
AsyncBudgetRelease = Callable[
    [ExecutionScope, RunRecord],
    Awaitable[None],
]
ImmediateDecision = Callable[[ApprovalRequest], bool]


class DurableApprovalService:
    """Pause durable runs and safely consume decisions after fresh validation."""

    def __init__(
        self,
        store: ApprovalStore,
        runs: RunStore,
        *,
        release_budget: BudgetRelease | None = None,
    ) -> None:
        self.store = store
        self.runs = runs
        self.release_budget = release_budget

    def request(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        submission: ApprovalSubmission,
    ) -> PausedRunOutcome:
        approval = self.store.create(scope, submission)
        self._record(
            scope,
            approval,
            "approval.requested",
            "runtime",
            {
                "approval_id": approval.id,
                "effect_id": approval.effect_id,
                "tool_name": approval.tool_name,
                "tool_version": approval.tool_version,
                "schema_version": approval.schema_version,
                "arguments_digest": approval.arguments_digest,
                "policy_version": approval.policy_version,
                "expires_at": approval.expires_at.isoformat(),
            },
            idempotency_key=f"approval-requested:{approval.id}",
        )
        try:
            run = self.runs.pause_for_approval(
                scope,
                claim,
                submission.step_id,
                approval_id=approval.id,
                payload={
                    "tool_name": approval.tool_name,
                    "tool_version": approval.tool_version,
                    "arguments_digest": approval.arguments_digest,
                    "policy_version": approval.policy_version,
                },
            )
        except BaseException:
            self.store.invalidate(
                scope,
                approval.id,
                reason="run could not be paused for approval",
            )
            raise
        if self.release_budget is not None:
            self.release_budget(scope, run)
        return PausedRunOutcome(
            kind=ApprovalOutcomeKind.PAUSED,
            run=run,
            approval=approval,
            reason="run paused until a durable approval is decided",
        )

    def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest:
        approval = self.store.decide(
            scope,
            approval_id,
            decision,
            decided_by=decided_by,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        self._record(
            scope,
            approval,
            "approval.decided",
            decided_by,
            {
                "approval_id": approval.id,
                "decision": decision.value,
                "reason": reason,
                "arguments_digest": approval.arguments_digest,
            },
            idempotency_key=f"approval-decided:{idempotency_key}",
        )
        return approval

    def resume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        validation: ApprovalValidation,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        approval = self.store.get(scope, approval_id)
        if approval.expires_at <= _utc_now() and approval.status in {
            ApprovalStatus.PENDING,
            ApprovalStatus.APPROVED,
        }:
            self.store.expire(now=_utc_now())
            approval = self.store.get(scope, approval_id)
        terminal_kind = _terminal_kind(approval.status)
        if terminal_kind is not None and approval.status is not ApprovalStatus.CONSUMED:
            return self._terminal_outcome(
                scope,
                approval,
                terminal_kind,
                actor=actor,
            )
        mismatch = _validation_mismatch(approval, validation)
        if mismatch is not None:
            mismatch_kind, mismatch_reason = mismatch
            if approval.status is ApprovalStatus.CONSUMED:
                run = self.runs.get(scope, approval.run_id)
                return ApprovalResumeOutcome(
                    kind=mismatch_kind,
                    approval=approval,
                    run=run,
                    reason=(
                        "consumed approval no longer validates: "
                        f"{mismatch_reason}"
                    ),
                )
            approval = self.store.invalidate(
                scope,
                approval.id,
                reason=mismatch_reason,
            )
            self._record(
                scope,
                approval,
                "approval.invalidated",
                actor,
                {"approval_id": approval.id, "reason": mismatch_reason},
                idempotency_key=(
                    f"approval-invalidated:{approval.id}:{approval.revision}"
                ),
            )
            return self._terminal_outcome(
                scope,
                approval,
                mismatch_kind,
                actor=actor,
            )
        if approval.status is ApprovalStatus.PENDING:
            run = self.runs.get(scope, approval.run_id)
            return ApprovalResumeOutcome(
                kind=ApprovalOutcomeKind.PAUSED,
                approval=approval,
                run=run,
                reason="approval is still pending",
            )
        if approval.status is ApprovalStatus.CONSUMED:
            return self._recover_consumed(scope, approval, actor=actor)
        if approval.status is not ApprovalStatus.APPROVED:
            return self._terminal_outcome(
                scope,
                approval,
                _terminal_kind(approval.status) or ApprovalOutcomeKind.INVALIDATED,
                actor=actor,
            )
        consumed = self.store.consume(
            scope,
            approval.id,
            expected_revision=approval.revision,
        )
        self._record(
            scope,
            consumed,
            "approval.consumed",
            actor,
            {
                "approval_id": consumed.id,
                "arguments_digest": consumed.arguments_digest,
            },
            idempotency_key=f"approval-consumed:{consumed.id}",
        )
        return self._recover_consumed(scope, consumed, actor=actor)

    def _recover_consumed(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        run = self.runs.get(scope, approval.run_id)
        if run.status is RunStatus.WAITING_FOR_APPROVAL:
            run = self.runs.resume(
                scope,
                run.id,
                actor=actor,
                reason=f"approval {approval.id} consumed",
            )
        if run.status in {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.COMPLETED,
        }:
            return ApprovalResumeOutcome(
                kind=ApprovalOutcomeKind.RESUMED,
                approval=approval,
                run=run,
                reason="approval consumed and durable run released",
            )
        return ApprovalResumeOutcome(
            kind=ApprovalOutcomeKind.INVALIDATED,
            approval=approval,
            run=run,
            reason=f"run cannot resume from {run.status.value}",
        )

    def _terminal_outcome(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        kind: ApprovalOutcomeKind,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        run = self.runs.get(scope, approval.run_id)
        if run.status is RunStatus.WAITING_FOR_APPROVAL:
            run = self.runs.cancel(
                scope,
                run.id,
                actor=actor,
                reason=f"approval {approval.id} is {kind.value}",
            )
        return ApprovalResumeOutcome(
            kind=kind,
            approval=approval,
            run=run,
            reason=f"approval is {kind.value}",
        )

    def _record(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        name: str,
        actor: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> None:
        self.runs.record_event(
            scope,
            approval.run_id,
            name=name,
            actor=actor,
            step_id=approval.step_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )


class AsyncDurableApprovalService:
    """Async durable approval coordination without blocking a worker."""

    def __init__(
        self,
        store: AsyncApprovalStore,
        runs: AsyncRunStore,
        *,
        release_budget: AsyncBudgetRelease | None = None,
    ) -> None:
        self.store = store
        self.runs = runs
        self.release_budget = release_budget

    async def request(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        submission: ApprovalSubmission,
    ) -> PausedRunOutcome:
        approval = await self.store.create(scope, submission)
        await self._record(
            scope,
            approval,
            "approval.requested",
            "runtime",
            {
                "approval_id": approval.id,
                "effect_id": approval.effect_id,
                "tool_name": approval.tool_name,
                "tool_version": approval.tool_version,
                "schema_version": approval.schema_version,
                "arguments_digest": approval.arguments_digest,
                "policy_version": approval.policy_version,
                "expires_at": approval.expires_at.isoformat(),
            },
            idempotency_key=f"approval-requested:{approval.id}",
        )
        try:
            run = await self.runs.pause_for_approval(
                scope,
                claim,
                submission.step_id,
                approval_id=approval.id,
                payload={
                    "tool_name": approval.tool_name,
                    "tool_version": approval.tool_version,
                    "arguments_digest": approval.arguments_digest,
                    "policy_version": approval.policy_version,
                },
            )
        except BaseException:
            await self.store.invalidate(
                scope,
                approval.id,
                reason="run could not be paused for approval",
            )
            raise
        if self.release_budget is not None:
            await self.release_budget(scope, run)
        return PausedRunOutcome(
            kind=ApprovalOutcomeKind.PAUSED,
            run=run,
            approval=approval,
            reason="run paused until a durable approval is decided",
        )

    async def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest:
        approval = await self.store.decide(
            scope,
            approval_id,
            decision,
            decided_by=decided_by,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        await self._record(
            scope,
            approval,
            "approval.decided",
            decided_by,
            {
                "approval_id": approval.id,
                "decision": decision.value,
                "reason": reason,
                "arguments_digest": approval.arguments_digest,
            },
            idempotency_key=f"approval-decided:{idempotency_key}",
        )
        return approval

    async def resume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        validation: ApprovalValidation,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        approval = await self.store.get(scope, approval_id)
        if approval.expires_at <= _utc_now() and approval.status in {
            ApprovalStatus.PENDING,
            ApprovalStatus.APPROVED,
        }:
            await self.store.expire(now=_utc_now())
            approval = await self.store.get(scope, approval_id)
        terminal_kind = _terminal_kind(approval.status)
        if (
            terminal_kind is not None
            and approval.status is not ApprovalStatus.CONSUMED
        ):
            return await self._terminal_outcome(
                scope,
                approval,
                terminal_kind,
                actor=actor,
            )
        mismatch = _validation_mismatch(approval, validation)
        if mismatch is not None:
            mismatch_kind, mismatch_reason = mismatch
            if approval.status is ApprovalStatus.CONSUMED:
                run = await self.runs.get(scope, approval.run_id)
                return ApprovalResumeOutcome(
                    kind=mismatch_kind,
                    approval=approval,
                    run=run,
                    reason=(
                        "consumed approval no longer validates: "
                        f"{mismatch_reason}"
                    ),
                )
            approval = await self.store.invalidate(
                scope,
                approval.id,
                reason=mismatch_reason,
            )
            await self._record(
                scope,
                approval,
                "approval.invalidated",
                actor,
                {"approval_id": approval.id, "reason": mismatch_reason},
                idempotency_key=(
                    f"approval-invalidated:{approval.id}:{approval.revision}"
                ),
            )
            return await self._terminal_outcome(
                scope,
                approval,
                mismatch_kind,
                actor=actor,
            )
        if approval.status is ApprovalStatus.PENDING:
            run = await self.runs.get(scope, approval.run_id)
            return ApprovalResumeOutcome(
                kind=ApprovalOutcomeKind.PAUSED,
                approval=approval,
                run=run,
                reason="approval is still pending",
            )
        if approval.status is ApprovalStatus.CONSUMED:
            return await self._recover_consumed(
                scope,
                approval,
                actor=actor,
            )
        if approval.status is not ApprovalStatus.APPROVED:
            return await self._terminal_outcome(
                scope,
                approval,
                _terminal_kind(approval.status)
                or ApprovalOutcomeKind.INVALIDATED,
                actor=actor,
            )
        consumed = await self.store.consume(
            scope,
            approval.id,
            expected_revision=approval.revision,
        )
        await self._record(
            scope,
            consumed,
            "approval.consumed",
            actor,
            {
                "approval_id": consumed.id,
                "arguments_digest": consumed.arguments_digest,
            },
            idempotency_key=f"approval-consumed:{consumed.id}",
        )
        return await self._recover_consumed(
            scope,
            consumed,
            actor=actor,
        )

    async def _recover_consumed(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        run = await self.runs.get(scope, approval.run_id)
        if run.status is RunStatus.WAITING_FOR_APPROVAL:
            run = await self.runs.resume(
                scope,
                run.id,
                actor=actor,
                reason=f"approval {approval.id} consumed",
            )
        if run.status in {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.COMPLETED,
        }:
            return ApprovalResumeOutcome(
                kind=ApprovalOutcomeKind.RESUMED,
                approval=approval,
                run=run,
                reason="approval consumed and durable run released",
            )
        return ApprovalResumeOutcome(
            kind=ApprovalOutcomeKind.INVALIDATED,
            approval=approval,
            run=run,
            reason=f"run cannot resume from {run.status.value}",
        )

    async def _terminal_outcome(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        kind: ApprovalOutcomeKind,
        *,
        actor: str,
    ) -> ApprovalResumeOutcome:
        run = await self.runs.get(scope, approval.run_id)
        if run.status is RunStatus.WAITING_FOR_APPROVAL:
            run = await self.runs.cancel(
                scope,
                run.id,
                actor=actor,
                reason=f"approval {approval.id} is {kind.value}",
            )
        return ApprovalResumeOutcome(
            kind=kind,
            approval=approval,
            run=run,
            reason=f"approval is {kind.value}",
        )

    async def _record(
        self,
        scope: ExecutionScope,
        approval: ApprovalRequest,
        name: str,
        actor: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> None:
        await self.runs.record_event(
            scope,
            approval.run_id,
            name=name,
            actor=actor,
            step_id=approval.step_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )


class DurableApprovalCoordinator:
    """Connect one synchronous tool permission decision to durable state."""

    def __init__(
        self,
        service: DurableApprovalService,
        effects: Any,
        *,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        lifetime: timedelta = timedelta(minutes=5),
    ) -> None:
        self.service = service
        self.effects = effects
        self.scope = scope
        self.claim = claim
        self.step_id = step_id
        self.lifetime = lifetime

    def resolve(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        turn: Any,
        record: PermissionDecisionRecord,
    ) -> tuple[PermissionDecisionRecord, object]:
        token = self.effects.prepare(
            tool=tool,
            arguments=arguments,
            context=context,
            turn=turn,
        )
        approval = _approval_for_effect(
            self.service.store.list(
                self.scope,
                run_id=self.scope.run_id,
            ),
            token.effect.id,
        )
        validation = _validation(
            self.scope,
            tool,
            arguments,
            credentials_available=True,
        )
        if approval is not None:
            outcome = self.service.resume(
                self.scope,
                approval.id,
                validation,
                actor=self.claim.worker_id,
            )
            if outcome.resumed:
                return _allowed(record), token
            raise DurableApprovalPaused(outcome)
        pause_outcome = self.service.request(
            self.scope,
            self.claim,
            _submission(
                self.step_id,
                token.effect.id,
                tool,
                arguments,
                record,
                lifetime=self.lifetime,
            ),
        )
        raise DurableApprovalPaused(pause_outcome)


class AsyncDurableApprovalCoordinator:
    """Async equivalent used by the native async hosted tool path."""

    def __init__(
        self,
        service: AsyncDurableApprovalService,
        effects: Any,
        *,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        lifetime: timedelta = timedelta(minutes=5),
    ) -> None:
        self.service = service
        self.effects = effects
        self.scope = scope
        self.claim = claim
        self.step_id = step_id
        self.lifetime = lifetime

    async def resolve_async(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        turn: Any,
        record: PermissionDecisionRecord,
    ) -> tuple[PermissionDecisionRecord, object]:
        token = await self.effects.prepare_async(
            tool=tool,
            arguments=arguments,
            context=context,
            turn=turn,
        )
        approval = _approval_for_effect(
            await self.service.store.list(
                self.scope,
                run_id=self.scope.run_id,
            ),
            token.effect.id,
        )
        validation = _validation(
            self.scope,
            tool,
            arguments,
            credentials_available=True,
        )
        if approval is not None:
            outcome = await self.service.resume(
                self.scope,
                approval.id,
                validation,
                actor=self.claim.worker_id,
            )
            if outcome.resumed:
                return _allowed(record), token
            raise DurableApprovalPaused(outcome)
        pause_outcome = await self.service.request(
            self.scope,
            self.claim,
            _submission(
                self.step_id,
                token.effect.id,
                tool,
                arguments,
                record,
                lifetime=self.lifetime,
            ),
        )
        raise DurableApprovalPaused(pause_outcome)


class ImmediateApprovalAdapter:
    """Local permission callback backed by the same durable approval ledger."""

    def __init__(
        self,
        store: ApprovalStore,
        runs: RunStore,
        *,
        scope: ExecutionScope,
        step_id: str,
        decide: ImmediateDecision,
        actor: str = "local-operator",
        lifetime: timedelta = timedelta(minutes=5),
    ) -> None:
        self.store = store
        self.runs = runs
        self.scope = scope
        self.step_id = step_id
        self.decide_callback = decide
        self.actor = actor
        self.lifetime = lifetime

    def __call__(
        self,
        request: PermissionRequest,
        record: PermissionDecisionRecord,
    ) -> bool:
        identity = request.tool_identity or {}
        policy = request.tool_policy or {}
        arguments_digest = request.arguments_digest
        if not arguments_digest:
            raise ValueError("immediate durable approval requires arguments_digest")
        submission = ApprovalSubmission(
            step_id=self.step_id,
            tool_name=request.tool_name,
            tool_version=str(identity.get("version") or "unversioned"),
            schema_version=str(identity.get("schema_version") or "1"),
            arguments_digest=arguments_digest,
            policy_version=str(policy.get("version") or record.policy_name),
            preview={
                "tool_name": request.tool_name,
                "permission_level": request.permission_level.value,
                "reason": request.reason,
            },
            expires_at=_utc_now() + self.lifetime,
        )
        approval = self.store.create(self.scope, submission)
        self.runs.record_event(
            self.scope,
            self.scope.run_id,
            name="approval.requested",
            actor="runtime",
            step_id=self.step_id,
            payload={
                "approval_id": approval.id,
                "tool_name": approval.tool_name,
                "arguments_digest": approval.arguments_digest,
            },
            idempotency_key=f"immediate-approval-requested:{approval.id}",
        )
        allowed = bool(self.decide_callback(approval))
        decision = (
            ApprovalDecision.APPROVE if allowed else ApprovalDecision.DENY
        )
        decided = self.store.decide(
            self.scope,
            approval.id,
            decision,
            decided_by=self.actor,
            reason="immediate local approval decision",
            idempotency_key=f"immediate:{approval.id}",
        )
        self.runs.record_event(
            self.scope,
            self.scope.run_id,
            name="approval.decided",
            actor=self.actor,
            step_id=self.step_id,
            payload={
                "approval_id": decided.id,
                "decision": decision.value,
            },
            idempotency_key=f"immediate-approval-decided:{approval.id}",
        )
        if not allowed:
            return False
        consumed = self.store.consume(
            self.scope,
            decided.id,
            expected_revision=decided.revision,
        )
        self.runs.record_event(
            self.scope,
            self.scope.run_id,
            name="approval.consumed",
            actor=self.actor,
            step_id=self.step_id,
            payload={"approval_id": consumed.id},
            idempotency_key=f"immediate-approval-consumed:{approval.id}",
        )
        return True


def _validation_mismatch(
    approval: ApprovalRequest,
    validation: ApprovalValidation,
) -> tuple[ApprovalOutcomeKind, str] | None:
    try:
        validation.scope.assert_resumable(approval.scope)
    except ExecutionScopeError:
        return (
            ApprovalOutcomeKind.REVOKED_AUTHORITY,
            "execution scope or authority changed",
        )
    comparisons = (
        ("tool name", validation.tool_name, approval.tool_name),
        ("tool version", validation.tool_version, approval.tool_version),
        ("schema version", validation.schema_version, approval.schema_version),
        (
            "arguments digest",
            validation.arguments_digest,
            approval.arguments_digest,
        ),
        ("policy version", validation.policy_version, approval.policy_version),
    )
    for label, observed, expected in comparisons:
        if observed != expected:
            return (
                ApprovalOutcomeKind.INVALIDATED,
                f"{label} changed after approval was requested",
            )
    if not validation.authority_valid:
        return (
            ApprovalOutcomeKind.REVOKED_AUTHORITY,
            "execution authority was revoked",
        )
    if not validation.credentials_available:
        return (
            ApprovalOutcomeKind.UNAVAILABLE_INTEGRATION,
            "required credentials are no longer available",
        )
    return None


def _approval_for_effect(
    approvals: tuple[ApprovalRequest, ...],
    effect_id: str,
) -> ApprovalRequest | None:
    matches = [item for item in approvals if item.effect_id == effect_id]
    return matches[-1] if matches else None


def _submission(
    step_id: str,
    effect_id: str,
    tool: Any,
    arguments: Mapping[str, object],
    record: PermissionDecisionRecord,
    *,
    lifetime: timedelta,
) -> ApprovalSubmission:
    identity = tool.resolved_identity()
    policy = tool.resolved_policy()
    return ApprovalSubmission(
        step_id=step_id,
        effect_id=effect_id,
        tool_name=identity.name,
        tool_version=identity.version,
        schema_version=identity.input_schema_version,
        arguments_digest=schema_digest(arguments),
        policy_version=policy.version,
        preview={
            "tool_name": identity.name,
            "permission_level": record.permission_level.value,
            "reason": record.reason,
        },
        expires_at=_utc_now() + lifetime,
    )


def _validation(
    scope: ExecutionScope,
    tool: Any,
    arguments: Mapping[str, object],
    *,
    credentials_available: bool,
) -> ApprovalValidation:
    identity = tool.resolved_identity()
    policy = tool.resolved_policy()
    return ApprovalValidation(
        scope=scope,
        tool_name=identity.name,
        tool_version=identity.version,
        schema_version=identity.input_schema_version,
        arguments_digest=schema_digest(arguments),
        policy_version=policy.version,
        authority_valid=True,
        credentials_available=credentials_available,
    )


def _allowed(record: PermissionDecisionRecord) -> PermissionDecisionRecord:
    return PermissionDecisionRecord(
        tool_name=record.tool_name,
        permission_level=record.permission_level,
        decision=ToolPermissionDecision.ALLOW,
        reason="tool call approved by consumed durable approval",
        policy_name=record.policy_name,
        requires_confirmation=record.requires_confirmation,
        capability_category=record.capability_category,
        capability_enabled=record.capability_enabled,
        tool_identity=record.tool_identity,
        tool_policy=record.tool_policy,
        arguments_digest=record.arguments_digest,
    )


def _terminal_kind(status: ApprovalStatus) -> ApprovalOutcomeKind | None:
    return {
        ApprovalStatus.DENIED: ApprovalOutcomeKind.DENIED,
        ApprovalStatus.EXPIRED: ApprovalOutcomeKind.EXPIRED,
        ApprovalStatus.INVALIDATED: ApprovalOutcomeKind.INVALIDATED,
        ApprovalStatus.CANCELLED: ApprovalOutcomeKind.CANCELLED,
        ApprovalStatus.CONSUMED: ApprovalOutcomeKind.RESUMED,
    }.get(status)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "AsyncBudgetRelease",
    "AsyncDurableApprovalCoordinator",
    "AsyncDurableApprovalService",
    "BudgetRelease",
    "DurableApprovalCoordinator",
    "DurableApprovalService",
    "ImmediateApprovalAdapter",
    "ImmediateDecision",
]
