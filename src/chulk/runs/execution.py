"""Shared durable boundary for hosted agent execution and tool effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

from chulk.core.signals import DurableApprovalPaused
from chulk.hosting.async_utils import call_async_service
from chulk.runs.errors import EffectConflictError
from chulk.hosting.scope import ExecutionScope
from chulk.redaction import redact_data
from chulk.runs.models import (
    EffectRecord,
    EffectStatus,
    RunClaim,
    RunRecord,
    RunStatus,
    RunSubmission,
)
from chulk.runs.events import AsyncRunEventPublisher, RunEventPublisher
from chulk.runs.protocols import AsyncRunStore, RunStore
from chulk.tools.policy import ToolEffect, schema_digest
from chulk.tools.registry import (
    ToolExecutionContext,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class DurableEffectToken:
    effect: EffectRecord
    effect_class: ToolEffect
    recovered_result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class DurableExecutionOutcome:
    """Result of one attempt to execute a submitted hosted run."""

    run: RunRecord
    result: Any | None
    claimed: bool
    duplicate: bool
    approval: Any | None = None


class DurableEffectCoordinator:
    """Persist effect intent immediately around runtime tool transport."""

    def __init__(
        self,
        runs: RunStore,
        *,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        publisher: RunEventPublisher | None = None,
    ) -> None:
        self.runs = runs
        self.scope = scope
        self.claim = claim
        self.step_id = step_id
        self.publisher = publisher

    def prepare(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        turn: Any,
    ) -> DurableEffectToken:
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        arguments_digest = schema_digest(arguments)
        logical_key = context.effect_key
        operation_id = turn.extension_metadata.get("goal_pending", {}).get("operation_id")
        if logical_key is None and turn.extension_metadata.get("goal_id") and operation_id:
            logical_key = f"goal:{turn.extension_metadata['goal_id']}:{operation_id}:attempt:{len(turn.extension_metadata.get('goal_tool_attempts', [])) + 1}"
        if logical_key is None:
            if policy.effect is not ToolEffect.READ:
                raise ValueError(
                    "durable mutating tool execution requires a host-derived "
                    "logical effect key"
                )
            logical_key = (
                f"read:{turn.turn_id}:{turn.tool_call_count}:"
                f"{identity.digest}:{arguments_digest}"
            )
        effect = self.runs.begin_effect(
            self.scope,
            self.claim,
            self.step_id,
            logical_key=logical_key,
            tool_name=identity.name,
            tool_version=identity.version,
            schema_version=identity.input_schema_version,
            arguments_digest=arguments_digest,
        )
        self._publish()
        stored = self.runs.effect_result(self.scope, effect.id)
        recovered = ToolResult.from_dict(stored) if stored is not None else None
        if recovered is not None and effect.status is EffectStatus.EXECUTING:
            if recovered.success:
                effect = self.runs.complete_effect(self.scope, self.claim, effect.id, result_digest=effect.result_digest or "")
            elif policy.effect is ToolEffect.READ:
                effect = self.runs.fail_effect(self.scope, self.claim, effect.id, reason=recovered.error or "recovered read failure")
            else:
                self.runs.mark_effect_unknown(self.scope, self.claim, effect.id, reason="recovered mutation returned failure")
                raise EffectConflictError("Mutation outcome requires host reconciliation")
        if effect.status in {EffectStatus.COMPLETED, EffectStatus.EXECUTING} and recovered is None:
            raise EffectConflictError("logical effect already dispatched or completed without a recoverable result; reconciliation required")
        return DurableEffectToken(effect=effect, effect_class=policy.effect, recovered_result=recovered)

    def effect_id(self, token: object) -> str:
        return _token(token).effect.id

    def recover(self, token: object) -> ToolResult | None:
        return _token(token).recovered_result

    def started(self, token: object) -> None:
        durable = _token(token)
        self.runs.mark_effect_started(
            self.scope,
            self.claim,
            durable.effect.id,
        )
        self._publish()

    def completed(self, token: object, result: ToolResult) -> None:
        durable = _token(token)
        digest = self.runs.record_effect_result(self.scope, self.claim, durable.effect.id, result=result.to_dict())
        if result.success:
            self.runs.complete_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                result_digest=digest,
            )
            self._publish()
        elif durable.effect_class is ToolEffect.READ:
            self.runs.fail_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=result.failure_kind or result.error or "tool returned failure",
            )
            self._publish()
        else:
            self.runs.mark_effect_unknown(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=(
                    result.failure_kind
                    or result.error
                    or "mutating tool returned an uncertain failure"
                ),
            )
            self._publish()

    def failed(self, token: object, error: BaseException) -> None:
        durable = _token(token)
        reason = f"tool transport raised {type(error).__name__}"
        if durable.effect_class is ToolEffect.READ:
            self.runs.fail_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=reason,
            )
            self._publish()
            return
        self.runs.mark_effect_unknown(
            self.scope,
            self.claim,
            durable.effect.id,
            reason=reason,
        )
        self._publish()

    def _publish(self) -> None:
        if self.publisher is not None:
            self.publisher.publish()

    async def prepare_async(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        turn: Any,
    ) -> DurableEffectToken:
        return await call_async_service(
            self, "prepare",
            tool=tool,
            arguments=arguments,
            context=context,
            turn=turn,
        )

    async def started_async(self, token: object) -> None:
        await call_async_service(self, "started", token)

    async def completed_async(
        self,
        token: object,
        result: ToolResult,
    ) -> None:
        await call_async_service(self, "completed", token, result)

    async def failed_async(
        self,
        token: object,
        error: BaseException,
    ) -> None:
        await call_async_service(self, "failed", token, error)


class DurableHostedExecutor:
    """Execute an SDK agent turn through the shared durable-run owner."""

    def __init__(self, agent: Any, runs: RunStore) -> None:
        self.agent = agent
        self.runs = runs

    def execute(
        self,
        message: str,
        submission: RunSubmission,
        *,
        worker_id: str,
        step_id: str,
        lease_seconds: int = 120,
    ) -> DurableExecutionOutcome:
        from chulk.approvals.service import (
            DurableApprovalCoordinator,
            DurableApprovalService,
        )

        scope = _scope(self.agent)
        created = self.runs.submit(scope, submission, actor="host")
        sink = self.agent.runtime.events.public_event_sink
        publisher = (
            RunEventPublisher(self.runs, sink, scope=scope)
            if sink is not None
            else None
        )
        if publisher is not None:
            publisher.publish()
        duplicate = created.revision > 0 or created.status is not RunStatus.QUEUED
        if created.terminal:
            return DurableExecutionOutcome(
                run=created,
                result=None,
                claimed=False,
                duplicate=True,
            )
        claim = self.runs.claim(
            scope,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            run_id=created.id,
        )
        if claim is None:
            return DurableExecutionOutcome(
                run=self.runs.get(scope, created.id),
                result=None,
                claimed=False,
                duplicate=duplicate,
            )
        self.runs.start_step(scope, claim, step_id)
        self.runs.checkpoint(
            scope,
            claim,
            step_id,
            kind="agent_input",
            payload={"message_digest": _digest(message)},
        )
        if publisher is not None:
            publisher.publish()
        coordinator = DurableEffectCoordinator(
            self.runs,
            scope=scope,
            claim=claim,
            step_id=step_id,
            publisher=publisher,
        )
        runtime = self.agent.runtime
        approvals = DurableApprovalCoordinator(
            DurableApprovalService(runtime.resolved_services.approvals, self.runs),
            coordinator,
            scope=scope,
            claim=claim,
            step_id=step_id,
        )
        tool_executor = runtime._tool_executor
        previous = tool_executor.durable_effects
        previous_approvals = tool_executor.durable_approvals
        tool_executor.durable_effects = coordinator
        tool_executor.durable_approvals = approvals
        try:
            if runtime.goal_execution is not None and runtime.state.turns and runtime.state.turns[-1].status == "yielded":
                result = self.agent.continue_goal_slice()
            else:
                result = self.agent.run_result(message)
        except DurableApprovalPaused as exc:
            current = self.runs.get(scope, created.id)
            if publisher is not None:
                publisher.publish()
            return DurableExecutionOutcome(
                run=current,
                result=None,
                claimed=True,
                duplicate=duplicate,
                approval=exc.outcome,
            )
        except BaseException as exc:
            current = self.runs.get(scope, created.id)
            if current.status is RunStatus.RUNNING:
                current = self.runs.fail_step(
                    scope,
                    claim,
                    step_id,
                    reason=f"agent execution raised {type(exc).__name__}",
                    retryable=False,
                )
            raise
        finally:
            tool_executor.durable_effects = previous
            tool_executor.durable_approvals = previous_approvals
        current = self.runs.get(scope, created.id)
        if current.status is RunStatus.UNKNOWN:
            if publisher is not None:
                publisher.publish()
            return DurableExecutionOutcome(
                run=current,
                result=result,
                claimed=True,
                duplicate=duplicate,
            )
        result_status = str(getattr(result.status, "value", result.status))
        if result_status == "completed":
            current = self.runs.complete_step(
                scope,
                claim,
                step_id,
                result={"status": result_status},
            )
            if all(step.status.value == "completed" for step in current.steps):
                current = self.runs.complete(
                    scope,
                    claim,
                    result=_safe_result(result),
                )
        elif result_status == "yielded":
            current = self.runs.yield_step(scope, claim, step_id, continuation={"turn_id": result.turn_id})
        elif result_status == "cancelled":
            current = self.runs.cancel(
                scope,
                created.id,
                actor=worker_id,
                reason="agent execution was cancelled",
                claim=claim,
            )
        else:
            current = self.runs.fail_step(
                scope,
                claim,
                step_id,
                reason=_result_error(result),
                retryable=False,
            )
        if publisher is not None:
            publisher.publish()
        return DurableExecutionOutcome(
            run=current,
            result=result,
            claimed=True,
            duplicate=duplicate,
        )


class AsyncDurableEffectCoordinator:
    """Native async effect boundary for ``AsyncHostedRuntime``."""

    def __init__(
        self,
        runs: AsyncRunStore,
        *,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        publisher: AsyncRunEventPublisher | None = None,
    ) -> None:
        self.runs = runs
        self.scope = scope
        self.claim = claim
        self.step_id = step_id
        self.publisher = publisher

    def prepare(self, **kwargs: Any) -> object:
        raise RuntimeError("async durable effects require the async tool path")

    def effect_id(self, token: object) -> str:
        return _token(token).effect.id

    def recover(self, token: object) -> ToolResult | None:
        return _token(token).recovered_result

    def started(self, token: object) -> None:
        raise RuntimeError("async durable effects require the async tool path")

    def completed(self, token: object, result: ToolResult) -> None:
        raise RuntimeError("async durable effects require the async tool path")

    def failed(self, token: object, error: BaseException) -> None:
        raise RuntimeError("async durable effects require the async tool path")

    async def prepare_async(
        self,
        *,
        tool: Any,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
        turn: Any,
    ) -> DurableEffectToken:
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        arguments_digest = schema_digest(arguments)
        logical_key = context.effect_key
        operation_id = turn.extension_metadata.get("goal_pending", {}).get("operation_id")
        if logical_key is None and turn.extension_metadata.get("goal_id") and operation_id:
            logical_key = f"goal:{turn.extension_metadata['goal_id']}:{operation_id}:attempt:{len(turn.extension_metadata.get('goal_tool_attempts', [])) + 1}"
        if logical_key is None:
            if policy.effect is not ToolEffect.READ:
                raise ValueError(
                    "durable mutating tool execution requires a host-derived "
                    "logical effect key"
                )
            logical_key = (
                f"read:{turn.turn_id}:{turn.tool_call_count}:"
                f"{identity.digest}:{arguments_digest}"
            )
        effect = await self.runs.begin_effect(
            self.scope,
            self.claim,
            self.step_id,
            logical_key=logical_key,
            tool_name=identity.name,
            tool_version=identity.version,
            schema_version=identity.input_schema_version,
            arguments_digest=arguments_digest,
        )
        await self._publish()
        stored = await self.runs.effect_result(self.scope, effect.id)
        recovered = ToolResult.from_dict(stored) if stored is not None else None
        if recovered is not None and effect.status is EffectStatus.EXECUTING:
            if recovered.success:
                effect = await self.runs.complete_effect(self.scope, self.claim, effect.id, result_digest=effect.result_digest or "")
            elif policy.effect is ToolEffect.READ:
                effect = await self.runs.fail_effect(self.scope, self.claim, effect.id, reason=recovered.error or "recovered read failure")
            else:
                await self.runs.mark_effect_unknown(self.scope, self.claim, effect.id, reason="recovered mutation returned failure")
                raise EffectConflictError("Mutation outcome requires host reconciliation")
        if effect.status in {EffectStatus.COMPLETED, EffectStatus.EXECUTING} and recovered is None:
            raise EffectConflictError("logical effect already dispatched or completed without a recoverable result; reconciliation required")
        return DurableEffectToken(effect=effect, effect_class=policy.effect, recovered_result=recovered)

    async def started_async(self, token: object) -> None:
        durable = _token(token)
        await self.runs.mark_effect_started(
            self.scope,
            self.claim,
            durable.effect.id,
        )
        await self._publish()

    async def completed_async(
        self,
        token: object,
        result: ToolResult,
    ) -> None:
        durable = _token(token)
        digest = await self.runs.record_effect_result(self.scope, self.claim, durable.effect.id, result=result.to_dict())
        if result.success:
            await self.runs.complete_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                result_digest=digest,
            )
        elif durable.effect_class is ToolEffect.READ:
            await self.runs.fail_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=result.failure_kind or result.error or "tool returned failure",
            )
        else:
            await self.runs.mark_effect_unknown(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=(
                    result.failure_kind
                    or result.error
                    or "mutating tool returned an uncertain failure"
                ),
            )
        await self._publish()

    async def failed_async(
        self,
        token: object,
        error: BaseException,
    ) -> None:
        durable = _token(token)
        reason = f"tool transport raised {type(error).__name__}"
        if durable.effect_class is ToolEffect.READ:
            await self.runs.fail_effect(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=reason,
            )
        else:
            await self.runs.mark_effect_unknown(
                self.scope,
                self.claim,
                durable.effect.id,
                reason=reason,
            )
        await self._publish()

    async def _publish(self) -> None:
        if self.publisher is not None:
            await self.publisher.publish()


class AsyncDurableHostedExecutor:
    """Run an async SDK agent through native async durable services."""

    def __init__(self, agent: Any, runs: AsyncRunStore) -> None:
        self.agent = agent
        self.runs = runs

    async def execute(
        self,
        message: str,
        submission: RunSubmission,
        *,
        worker_id: str,
        step_id: str,
        lease_seconds: int = 120,
    ) -> DurableExecutionOutcome:
        from chulk.approvals.service import (
            AsyncDurableApprovalCoordinator,
            AsyncDurableApprovalService,
        )

        scope = _scope(self.agent)
        created = await self.runs.submit(scope, submission, actor="host")
        sink = self.agent.runtime.events.public_event_sink
        publisher = (
            AsyncRunEventPublisher(self.runs, sink, scope=scope)
            if sink is not None
            else None
        )
        if publisher is not None:
            await publisher.publish()
        duplicate = created.revision > 0 or created.status is not RunStatus.QUEUED
        if created.terminal:
            return DurableExecutionOutcome(
                run=created,
                result=None,
                claimed=False,
                duplicate=True,
            )
        claim = await self.runs.claim(
            scope,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            run_id=created.id,
        )
        if claim is None:
            return DurableExecutionOutcome(
                run=await self.runs.get(scope, created.id),
                result=None,
                claimed=False,
                duplicate=duplicate,
            )
        await self.runs.start_step(scope, claim, step_id)
        await self.runs.checkpoint(
            scope,
            claim,
            step_id,
            kind="agent_input",
            payload={"message_digest": _digest(message)},
        )
        if publisher is not None:
            await publisher.publish()
        coordinator = AsyncDurableEffectCoordinator(
            self.runs,
            scope=scope,
            claim=claim,
            step_id=step_id,
            publisher=publisher,
        )
        approvals = AsyncDurableApprovalCoordinator(
            AsyncDurableApprovalService(
                self.agent.runtime.resolved_services.approvals,
                self.runs,
            ),
            coordinator,
            scope=scope,
            claim=claim,
            step_id=step_id,
        )
        runtime = self.agent.runtime
        tool_executor = runtime._tool_executor
        previous = tool_executor.durable_effects
        previous_approvals = tool_executor.durable_approvals
        tool_executor.durable_effects = coordinator
        tool_executor.durable_approvals = approvals
        try:
            if runtime.goal_execution is not None and runtime.state.turns and runtime.state.turns[-1].status == "yielded":
                result = await self.agent.continue_goal_slice()
            else:
                result = await self.agent.run_result(message)
        except DurableApprovalPaused as exc:
            current = await self.runs.get(scope, created.id)
            if publisher is not None:
                await publisher.publish()
            return DurableExecutionOutcome(
                run=current,
                result=None,
                claimed=True,
                duplicate=duplicate,
                approval=exc.outcome,
            )
        except BaseException as exc:
            current = await self.runs.get(scope, created.id)
            if current.status is RunStatus.RUNNING:
                await self.runs.fail_step(
                    scope,
                    claim,
                    step_id,
                    reason=f"agent execution raised {type(exc).__name__}",
                    retryable=False,
                )
            raise
        finally:
            tool_executor.durable_effects = previous
            tool_executor.durable_approvals = previous_approvals
        current = await self.runs.get(scope, created.id)
        if current.status is RunStatus.UNKNOWN:
            if publisher is not None:
                await publisher.publish()
            return DurableExecutionOutcome(
                run=current,
                result=result,
                claimed=True,
                duplicate=duplicate,
            )
        result_status = str(getattr(result.status, "value", result.status))
        if result_status == "completed":
            current = await self.runs.complete_step(
                scope,
                claim,
                step_id,
                result={"status": result_status},
            )
            if all(step.status.value == "completed" for step in current.steps):
                current = await self.runs.complete(
                    scope,
                    claim,
                    result=_safe_result(result),
                )
        elif result_status == "yielded":
            current = await self.runs.yield_step(scope, claim, step_id, continuation={"turn_id": result.turn_id})
        elif result_status == "cancelled":
            current = await self.runs.cancel(
                scope,
                created.id,
                actor=worker_id,
                reason="agent execution was cancelled",
                claim=claim,
            )
        else:
            current = await self.runs.fail_step(
                scope,
                claim,
                step_id,
                reason=_result_error(result),
                retryable=False,
            )
        if publisher is not None:
            await publisher.publish()
        return DurableExecutionOutcome(
            run=current,
            result=result,
            claimed=True,
            duplicate=duplicate,
        )


def _token(value: object) -> DurableEffectToken:
    if not isinstance(value, DurableEffectToken):
        raise TypeError("invalid durable effect token")
    return value


def _scope(agent: Any) -> ExecutionScope:
    scope = getattr(agent, "execution_scope", None)
    if not isinstance(scope, ExecutionScope):
        raise TypeError("durable hosted execution requires an ExecutionScope")
    return scope


def _safe_result(result: Any) -> dict[str, Any]:
    value = result.to_dict()
    safe = redact_data(value)
    if not isinstance(safe, dict):
        raise ValueError("durable result must serialize to an object")
    return safe


def _result_error(result: Any) -> str:
    errors = getattr(result, "errors", ())
    if errors:
        return str(errors[-1])
    return f"agent execution ended with status {result.status}"


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


__all__ = [
    "AsyncDurableEffectCoordinator",
    "AsyncDurableHostedExecutor",
    "DurableEffectCoordinator",
    "DurableEffectToken",
    "DurableExecutionOutcome",
    "DurableHostedExecutor",
]
