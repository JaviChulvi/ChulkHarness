"""External-effect durability and reconciliation operations."""

from __future__ import annotations

from uuid import uuid4

from chulk.hosting.scope import ExecutionScope
from chulk.runs.models import (
    AttemptStatus,
    EffectRecord,
    EffectStatus,
    ReconciliationDecision,
    ReconciliationRecord,
    RunClaim,
    RunStatus,
    StepStatus,
)

import chulk.runs._store_clock as _clock
from chulk.runs.errors import EffectConflictError, RunNotFoundError
from chulk.runs._store_support import (
    _RunStoreBackend,
    _active_attempt,
    _assert_scope,
    _effect_from_row,
    _effect_row,
    _insert_event,
    _iso,
    _mark_active_attempt,
    _owned_run,
    _required,
    _run_from_conn,
    _run_row,
    _set_run_state,
    _touch_run,
)


class _RunStoreEffectMixin(_RunStoreBackend):
    def begin_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        logical_key: str,
        tool_name: str,
        tool_version: str,
        schema_version: str,
        arguments_digest: str,
    ) -> EffectRecord:
        logical_key = _required(logical_key, "logical effect key")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            attempt = _active_attempt(conn, claim.run_id, step_id)
            existing = conn.execute(
                """
                SELECT * FROM durable_effects
                WHERE run_id = ? AND logical_key = ?
                """,
                (claim.run_id, logical_key),
            ).fetchone()
            if existing is not None:
                effect = _effect_from_row(existing)
                expected = (
                    tool_name,
                    tool_version,
                    schema_version,
                    arguments_digest,
                )
                actual = (
                    effect.tool_name,
                    effect.tool_version,
                    effect.schema_version,
                    effect.arguments_digest,
                )
                if actual != expected:
                    raise EffectConflictError(
                        "logical effect key was reused for different tool input"
                    )
                if effect.status in {
                    EffectStatus.UNKNOWN,
                    EffectStatus.CANCELLED,
                }:
                    raise EffectConflictError(
                        f"effect is {effect.status.value}; reconciliation is required"
                    )
                if effect.status is EffectStatus.FAILED:
                    conn.execute(
                        """
                        UPDATE durable_effects
                        SET attempt_id = ?, status = 'intended',
                            reconciliation = NULL, reconciled_by = NULL,
                            reconciliation_reason = NULL, updated_at = ?
                        WHERE id = ? AND status = 'failed'
                        """,
                        (str(attempt["id"]), _iso(now), effect.id),
                    )
                    _insert_event(
                        conn,
                        claim.run_id,
                        name="effect.retried",
                        actor=claim.worker_id,
                        step_id=step_id,
                        payload={
                            "effect_id": effect.id,
                            "logical_key": effect.logical_key,
                        },
                        now=now,
                    )
                    return _effect_from_row(_effect_row(conn, effect.id))
                if effect.status is EffectStatus.INTENDED and effect.attempt_id != str(
                    attempt["id"]
                ):
                    conn.execute(
                        """
                        UPDATE durable_effects
                        SET attempt_id = ?, updated_at = ?
                        WHERE id = ? AND status = 'intended'
                        """,
                        (str(attempt["id"]), _iso(now), effect.id),
                    )
                    existing = _effect_row(conn, effect.id)
                    effect = _effect_from_row(existing)
                return effect
            effect_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO durable_effects (
                    id, run_id, step_id, attempt_id, logical_key, tool_name,
                    tool_version, schema_version, arguments_digest, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'intended', ?, ?)
                """,
                (
                    effect_id,
                    claim.run_id,
                    step_id,
                    str(attempt["id"]),
                    logical_key,
                    _required(tool_name, "tool name"),
                    _required(tool_version, "tool version"),
                    _required(schema_version, "schema version"),
                    _required(arguments_digest, "arguments digest"),
                    _iso(now),
                    _iso(now),
                ),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name="effect.intended",
                actor=claim.worker_id,
                step_id=step_id,
                payload={
                    "effect_id": effect_id,
                    "logical_key": logical_key,
                    "tool_name": tool_name,
                    "tool_version": tool_version,
                    "schema_version": schema_version,
                    "arguments_digest": arguments_digest,
                },
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))

    def mark_effect_started(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.INTENDED},
            status=EffectStatus.EXECUTING,
            event_name="effect.started",
        )

    def complete_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        result_digest: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.EXECUTING},
            status=EffectStatus.COMPLETED,
            event_name="effect.completed",
            result_digest=_required(result_digest, "result digest"),
        )

    def fail_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.INTENDED, EffectStatus.EXECUTING},
            status=EffectStatus.FAILED,
            event_name="effect.failed",
            reason=_required(reason, "effect failure reason"),
        )

    def mark_effect_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        reason = _required(reason, "unknown effect reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            effect = _effect_from_row(_effect_row(conn, effect_id))
            if effect.run_id != claim.run_id:
                raise EffectConflictError("effect does not belong to the claimed run")
            if effect.status not in {
                EffectStatus.INTENDED,
                EffectStatus.EXECUTING,
            }:
                raise EffectConflictError(
                    f"effect cannot become unknown from {effect.status.value}"
                )
            conn.execute(
                """
                UPDATE durable_effects
                SET status = 'unknown', reconciliation_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (reason, _iso(now), effect_id),
            )
            _mark_active_attempt(
                conn,
                claim.run_id,
                effect.step_id,
                AttemptStatus.UNKNOWN,
                now,
                reason,
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = 'unknown', revision = revision + 1,
                    error = ?, updated_at = ?
                WHERE run_id = ? AND id = ?
                """,
                (reason, _iso(now), claim.run_id, effect.step_id),
            )
            _set_run_state(
                conn,
                claim.run_id,
                RunStatus.UNKNOWN,
                now=now,
                error=reason,
                clear_lease=True,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="effect.unknown",
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={"effect_id": effect_id, "reason": reason},
                now=now,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="run.unknown",
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={"effect_id": effect_id, "reason": reason},
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))

    def reconcile_effect(
        self,
        scope: ExecutionScope,
        effect_id: str,
        *,
        decision: ReconciliationDecision,
        actor: str,
        reason: str,
        result_digest: str | None = None,
    ) -> ReconciliationRecord:
        decision = ReconciliationDecision(decision)
        actor = _required(actor, "reconciliation actor")
        reason = _required(reason, "reconciliation reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT run_id FROM durable_effects WHERE id = ?",
                (effect_id,),
            ).fetchone()
            if owner is None:
                raise RunNotFoundError(f"durable effect {effect_id!r} was not found")
            run_id = str(owner["run_id"])
            run = _run_from_conn(conn, _run_row(conn, run_id))
            effect = _effect_from_row(_effect_row(conn, effect_id))
            _assert_scope(scope, run.scope)
            if effect.status is not EffectStatus.UNKNOWN:
                raise EffectConflictError("only unknown effects can be reconciled")
            if decision is ReconciliationDecision.CONFIRMED:
                new_effect_status = EffectStatus.COMPLETED
                new_step_status = (
                    StepStatus.CANCELLED
                    if run.cancellation_requested
                    else StepStatus.QUEUED
                )
                new_run_status = (
                    RunStatus.CANCELLED
                    if run.cancellation_requested
                    else RunStatus.QUEUED
                )
                if result_digest is None:
                    raise ValueError(
                        "confirmed reconciliation requires a result digest"
                    )
            elif decision is ReconciliationDecision.RETRY:
                if run.cancellation_requested:
                    new_effect_status = EffectStatus.CANCELLED
                    new_step_status = StepStatus.CANCELLED
                    new_run_status = RunStatus.CANCELLED
                else:
                    new_effect_status = EffectStatus.INTENDED
                    new_step_status = StepStatus.QUEUED
                    new_run_status = RunStatus.QUEUED
            elif decision is ReconciliationDecision.FAILED:
                new_effect_status = EffectStatus.FAILED
                new_step_status = StepStatus.FAILED
                new_run_status = RunStatus.FAILED
            else:
                new_effect_status = EffectStatus.CANCELLED
                new_step_status = StepStatus.CANCELLED
                new_run_status = RunStatus.CANCELLED
            updated = conn.execute(
                """
                UPDATE durable_effects
                SET status = ?, result_digest = ?, reconciliation = ?,
                    reconciled_by = ?, reconciliation_reason = ?, updated_at = ?
                WHERE id = ? AND status = 'unknown'
                """,
                (
                    new_effect_status.value,
                    result_digest,
                    decision.value,
                    actor,
                    reason,
                    _iso(now),
                    effect_id,
                ),
            )
            if updated.rowcount != 1:
                raise EffectConflictError("only unknown effects can be reconciled")
            terminal = new_step_status in {
                StepStatus.FAILED,
                StepStatus.CANCELLED,
            }
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = ?, revision = revision + 1, error = ?,
                    next_retry_at = NULL, updated_at = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE NULL END
                WHERE run_id = ? AND id = ?
                """,
                (
                    new_step_status.value,
                    reason if terminal else None,
                    _iso(now),
                    int(terminal),
                    _iso(now),
                    effect.run_id,
                    effect.step_id,
                ),
            )
            _set_run_state(
                conn,
                effect.run_id,
                new_run_status,
                now=now,
                error=reason if terminal else None,
                clear_lease=True,
                completed=terminal,
            )
            _insert_event(
                conn,
                effect.run_id,
                name="effect.reconciled",
                actor=actor,
                step_id=effect.step_id,
                payload={
                    "effect_id": effect_id,
                    "decision": decision.value,
                    "reason": reason,
                    "result_digest": result_digest,
                },
                now=now,
            )
            updated_effect = _effect_from_row(_effect_row(conn, effect_id))
            updated_run = _run_from_conn(conn, _run_row(conn, effect.run_id))
        return ReconciliationRecord(
            effect=updated_effect,
            run=updated_run,
            decision=decision,
        )

    def _transition_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        expected: set[EffectStatus],
        status: EffectStatus,
        event_name: str,
        result_digest: str | None = None,
        reason: str | None = None,
    ) -> EffectRecord:
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            effect = _effect_from_row(_effect_row(conn, effect_id))
            if effect.run_id != claim.run_id:
                raise EffectConflictError("effect does not belong to the claimed run")
            if effect.status is status:
                return effect
            if effect.status not in expected:
                raise EffectConflictError(
                    f"effect cannot transition from {effect.status.value} "
                    f"to {status.value}"
                )
            conn.execute(
                """
                UPDATE durable_effects
                SET status = ?, result_digest = COALESCE(?, result_digest),
                    reconciliation_reason = COALESCE(?, reconciliation_reason),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    result_digest,
                    reason,
                    _iso(now),
                    effect_id,
                ),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name=event_name,
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={
                    "effect_id": effect_id,
                    "logical_key": effect.logical_key,
                    "result_digest": result_digest,
                    "reason": reason,
                },
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))


__all__ = ["_RunStoreEffectMixin"]
