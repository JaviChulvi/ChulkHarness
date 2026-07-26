"""Permission-aware sync and async tool transports."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import inspect
import time
from typing import Protocol

from chulk.core.events import TraceEvent
from chulk.core.state import TurnState, utc_now
from chulk.goals.models import GoalActionCheckpoint
from chulk.hosting import ExecutionScope
from chulk.tools import ToolRegistry
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
    ToolPermissionPolicy,
)
from chulk.tools.registry import ToolExecutionContext, ToolFailureKind, ToolResult
from chulk.tools.policy import (
    DataClassification,
    ToolAuthorization,
    ToolIdentity,
    ToolPolicy,
    ToolPolicyHooks,
    schema_digest,
)
from chulk.usage import BudgetExceededError, ModelUsageAccounting


class GoalExecutionPort(Protocol):
    def begin_tool(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
        tool_name: str,
    ) -> GoalActionCheckpoint: ...

    def finish_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        result: ToolResult,
    ) -> GoalActionCheckpoint: ...

    def abort_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        error: BaseException,
    ) -> GoalActionCheckpoint: ...


@dataclass
class ToolExecutor:
    """Execute tools with explicit permission, retry, and context policies."""

    registry: ToolRegistry
    permission_policy: ToolPermissionPolicy
    permission_callback: Callable[
        [PermissionRequest, PermissionDecisionRecord],
        PermissionDecision | bool,
    ] | None
    trace: Callable[[str, dict | None], None]
    get_context: Callable[[TurnState], ToolExecutionContext | None]
    usage_accounting: ModelUsageAccounting | None = None
    goal_execution: GoalExecutionPort | None = None
    execution_scope: ExecutionScope | None = None
    policy_hooks: ToolPolicyHooks | None = None

    def execute(self, tool_name: str, arguments: dict, turn: TurnState) -> ToolResult:
        """Execute a tool through the blocking transport and retry policy."""
        tool = self._registered_tool(tool_name)
        retry_policy = getattr(tool, "retry_policy", None)
        max_attempts, non_idempotent_guard = _attempt_policy(tool, retry_policy)
        attempts: list[dict] = []
        result: ToolResult | None = None
        for attempt_number in range(1, max_attempts + 1):
            started_at = utc_now()
            self._reserve_tool_attempt(
                turn,
                tool_name=tool_name,
                attempt=attempt_number,
            )
            try:
                goal_checkpoint = self._begin_goal_tool(
                    turn,
                    tool_name=tool_name,
                    attempt=attempt_number,
                )
            except BaseException:
                self._release_tool_attempt(turn, attempt=attempt_number)
                raise
            try:
                result = self._authorization_result(tool, arguments)
                context: ToolExecutionContext | None = None
                if result is None:
                    result = self._permission_result(
                        tool_name,
                        arguments,
                        turn,
                    )
                if result is None:
                    context = self._authorized_context(tool, arguments, turn)
                    result = self.registry.run(
                        tool_name,
                        arguments,
                        context=context,
                    )
                    if context.effect_key is not None:
                        result = replace(
                            result,
                            metadata={
                                **result.metadata,
                                "effect_key": context.effect_key,
                            },
                        )
                result = self._redacted_result(tool, arguments, result)
            except BaseException as exc:
                self._release_tool_attempt(turn, attempt=attempt_number)
                self._abort_goal_tool(goal_checkpoint, exc)
                raise
            try:
                self._finish_goal_tool(goal_checkpoint, result)
                self._commit_tool_attempt(
                    turn,
                    tool_name=tool_name,
                    attempt=attempt_number,
                    result=result,
                )
            except BaseException:
                self._release_tool_attempt(turn, attempt=attempt_number)
                raise
            retry = _should_retry(result, retry_policy, attempt_number, max_attempts)
            record = _attempt_payload(
                attempt_number,
                started_at,
                result,
                retry=retry,
                non_idempotent_guard=non_idempotent_guard,
            )
            attempts.append(record)
            self.trace(
                TraceEvent.TOOL_CALL_ATTEMPT,
                {"turn_id": turn.turn_id, "tool_name": tool_name, **record},
            )
            if not retry:
                break
            if retry_policy is not None and retry_policy.backoff_seconds:
                time.sleep(retry_policy.backoff_seconds)
        assert result is not None
        return self._versioned_result(tool, result, attempts)

    async def execute_async(
        self,
        tool_name: str,
        arguments: dict,
        turn: TurnState,
    ) -> ToolResult:
        """Execute a tool through the async transport and retry policy."""
        tool = self._registered_tool(tool_name)
        retry_policy = getattr(tool, "retry_policy", None)
        max_attempts, non_idempotent_guard = _attempt_policy(tool, retry_policy)
        attempts: list[dict] = []
        result: ToolResult | None = None
        for attempt_number in range(1, max_attempts + 1):
            started_at = utc_now()
            self._reserve_tool_attempt(
                turn,
                tool_name=tool_name,
                attempt=attempt_number,
            )
            try:
                goal_checkpoint = self._begin_goal_tool(
                    turn,
                    tool_name=tool_name,
                    attempt=attempt_number,
                )
            except BaseException:
                self._release_tool_attempt(turn, attempt=attempt_number)
                raise
            try:
                result = await self._authorization_result_async(tool, arguments)
                context = None
                if result is None:
                    result = await self._permission_result_async(
                        tool_name,
                        arguments,
                        turn,
                    )
                if result is None:
                    context = await self._authorized_context_async(
                        tool,
                        arguments,
                        turn,
                    )
                    result = await self.registry.run_async(
                        tool_name,
                        arguments,
                        context=context,
                    )
                    if context.effect_key is not None:
                        result = replace(
                            result,
                            metadata={
                                **result.metadata,
                                "effect_key": context.effect_key,
                            },
                        )
                result = await self._redacted_result_async(
                    tool,
                    arguments,
                    result,
                )
            except BaseException as exc:
                self._release_tool_attempt(turn, attempt=attempt_number)
                self._abort_goal_tool(goal_checkpoint, exc)
                raise
            try:
                self._finish_goal_tool(goal_checkpoint, result)
                self._commit_tool_attempt(
                    turn,
                    tool_name=tool_name,
                    attempt=attempt_number,
                    result=result,
                )
            except BaseException:
                self._release_tool_attempt(turn, attempt=attempt_number)
                raise
            retry = _should_retry(result, retry_policy, attempt_number, max_attempts)
            record = _attempt_payload(
                attempt_number,
                started_at,
                result,
                retry=retry,
                non_idempotent_guard=non_idempotent_guard,
            )
            attempts.append(record)
            self.trace(
                TraceEvent.TOOL_CALL_ATTEMPT,
                {"turn_id": turn.turn_id, "tool_name": tool_name, **record},
            )
            if not retry:
                break
            if retry_policy is not None and retry_policy.backoff_seconds:
                await asyncio.sleep(retry_policy.backoff_seconds)
        assert result is not None
        return self._versioned_result(tool, result, attempts)

    def _authorization_result(
        self,
        tool,
        arguments: Mapping[str, object],
    ) -> ToolResult | None:
        if tool is None:
            return None
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        traced = self.policy_hooks is not None or bool(policy.required_grants)
        if traced:
            self.trace(
                TraceEvent.TOOL_AUTHORIZATION_REQUESTED,
                _host_authorization_payload(identity, policy, arguments),
            )
        denied_reason = self._missing_grants_reason(policy)
        if denied_reason is not None:
            if traced:
                self.trace(
                    TraceEvent.TOOL_AUTHORIZATION_DECIDED,
                    {
                        **_host_authorization_payload(
                            identity,
                            policy,
                            arguments,
                        ),
                        "decision": "deny",
                        "reason": denied_reason,
                    },
                )
            return _authorization_denied_result(identity, policy, denied_reason)
        if (
            self.execution_scope is None
            or self.policy_hooks is None
            or self.policy_hooks.authorize is None
        ):
            if traced:
                self.trace(
                    TraceEvent.TOOL_AUTHORIZATION_DECIDED,
                    {
                        **_host_authorization_payload(
                            identity,
                            policy,
                            arguments,
                        ),
                        "decision": "allow",
                        "reason": "execution scope grants satisfied",
                    },
                )
            return None
        decision = self.policy_hooks.authorize(
            self.execution_scope,
            identity,
            policy,
            arguments,
        )
        if inspect.isawaitable(decision):
            close = getattr(decision, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "async tool authorizer cannot be used by the synchronous runtime"
            )
        authorization = _authorization(decision)
        self.trace(
            TraceEvent.TOOL_AUTHORIZATION_DECIDED,
            {
                **_host_authorization_payload(
                    identity,
                    policy,
                    arguments,
                ),
                "decision": "allow" if authorization.allowed else "deny",
                "reason": authorization.reason,
            },
        )
        if authorization.allowed:
            return None
        return _authorization_denied_result(
            identity,
            policy,
            authorization.reason or "tool call denied by host authorizer",
        )

    async def _authorization_result_async(
        self,
        tool,
        arguments: Mapping[str, object],
    ) -> ToolResult | None:
        if tool is None:
            return None
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        traced = self.policy_hooks is not None or bool(policy.required_grants)
        if traced:
            self.trace(
                TraceEvent.TOOL_AUTHORIZATION_REQUESTED,
                _host_authorization_payload(identity, policy, arguments),
            )
        denied_reason = self._missing_grants_reason(policy)
        if denied_reason is not None:
            if traced:
                self.trace(
                    TraceEvent.TOOL_AUTHORIZATION_DECIDED,
                    {
                        **_host_authorization_payload(
                            identity,
                            policy,
                            arguments,
                        ),
                        "decision": "deny",
                        "reason": denied_reason,
                    },
                )
            return _authorization_denied_result(identity, policy, denied_reason)
        if (
            self.execution_scope is None
            or self.policy_hooks is None
            or self.policy_hooks.authorize is None
        ):
            if traced:
                self.trace(
                    TraceEvent.TOOL_AUTHORIZATION_DECIDED,
                    {
                        **_host_authorization_payload(
                            identity,
                            policy,
                            arguments,
                        ),
                        "decision": "allow",
                        "reason": "execution scope grants satisfied",
                    },
                )
            return None
        decision = self.policy_hooks.authorize(
            self.execution_scope,
            identity,
            policy,
            arguments,
        )
        if inspect.isawaitable(decision):
            decision = await decision
        authorization = _authorization(decision)
        self.trace(
            TraceEvent.TOOL_AUTHORIZATION_DECIDED,
            {
                **_host_authorization_payload(
                    identity,
                    policy,
                    arguments,
                ),
                "decision": "allow" if authorization.allowed else "deny",
                "reason": authorization.reason,
            },
        )
        if authorization.allowed:
            return None
        return _authorization_denied_result(
            identity,
            policy,
            authorization.reason or "tool call denied by host authorizer",
        )

    def _authorized_context(
        self,
        tool,
        arguments: Mapping[str, object],
        turn: TurnState,
    ) -> ToolExecutionContext:
        context = self.get_context(turn) or ToolExecutionContext()
        if tool is None or self.execution_scope is None:
            return context
        credentials: Mapping[str, object] = {}
        effect_key: str | None = None
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        if self.policy_hooks is not None:
            if self.policy_hooks.derive_effect_key is not None:
                resolved_key = self.policy_hooks.derive_effect_key(
                    self.execution_scope,
                    identity,
                    policy,
                    arguments,
                )
                if inspect.isawaitable(resolved_key):
                    close = getattr(resolved_key, "close", None)
                    if callable(close):
                        close()
                    raise RuntimeError(
                        "async effect-key hook cannot be used by the synchronous runtime"
                    )
                effect_key = _effect_key(resolved_key)
            if self.policy_hooks.resolve_credentials is not None:
                resolved = self.policy_hooks.resolve_credentials(
                    self.execution_scope,
                    identity,
                    policy,
                    arguments,
                )
                if inspect.isawaitable(resolved):
                    close = getattr(resolved, "close", None)
                    if callable(close):
                        close()
                    raise RuntimeError(
                        "async credential resolver cannot be used by the synchronous runtime"
                    )
                credentials = _credentials(resolved)
        return replace(
            context,
            scope=self.execution_scope,
            credentials=credentials,
            effect_key=effect_key,
        )

    async def _authorized_context_async(
        self,
        tool,
        arguments: Mapping[str, object],
        turn: TurnState,
    ) -> ToolExecutionContext:
        context = self.get_context(turn) or ToolExecutionContext()
        if tool is None or self.execution_scope is None:
            return context
        credentials: Mapping[str, object] = {}
        effect_key: str | None = None
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        if self.policy_hooks is not None:
            if self.policy_hooks.derive_effect_key is not None:
                resolved_key = self.policy_hooks.derive_effect_key(
                    self.execution_scope,
                    identity,
                    policy,
                    arguments,
                )
                if inspect.isawaitable(resolved_key):
                    resolved_key = await resolved_key
                effect_key = _effect_key(resolved_key)
            if self.policy_hooks.resolve_credentials is not None:
                resolved = self.policy_hooks.resolve_credentials(
                    self.execution_scope,
                    identity,
                    policy,
                    arguments,
                )
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                credentials = _credentials(resolved)
        return replace(
            context,
            scope=self.execution_scope,
            credentials=credentials,
            effect_key=effect_key,
        )

    def _missing_grants_reason(self, policy: ToolPolicy) -> str | None:
        if not policy.required_grants:
            return None
        if self.execution_scope is None:
            return "tool requires a hosted execution scope"
        missing = sorted(policy.required_grants - self.execution_scope.grants)
        if not missing:
            return None
        return "execution scope is missing required grants: " + ", ".join(missing)

    def _redacted_result(
        self,
        tool,
        arguments: Mapping[str, object],
        result: ToolResult,
    ) -> ToolResult:
        if tool is None:
            return result
        policy = tool.resolved_policy()
        if policy.output_classification is DataClassification.SECRET:
            return _withheld_secret_result(result)
        if (
            self.execution_scope is None
            or self.policy_hooks is None
            or self.policy_hooks.redact is None
        ):
            return result
        redacted = self.policy_hooks.redact(
            self.execution_scope,
            tool.resolved_identity(),
            policy,
            _redaction_payload(arguments, result),
        )
        if inspect.isawaitable(redacted):
            close = getattr(redacted, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "async redaction hook cannot be used by the synchronous runtime"
            )
        return _coerce_redacted_result(result, redacted)

    async def _redacted_result_async(
        self,
        tool,
        arguments: Mapping[str, object],
        result: ToolResult,
    ) -> ToolResult:
        if tool is None:
            return result
        policy = tool.resolved_policy()
        if policy.output_classification is DataClassification.SECRET:
            return _withheld_secret_result(result)
        if (
            self.execution_scope is None
            or self.policy_hooks is None
            or self.policy_hooks.redact is None
        ):
            return result
        redacted = self.policy_hooks.redact(
            self.execution_scope,
            tool.resolved_identity(),
            policy,
            _redaction_payload(arguments, result),
        )
        if inspect.isawaitable(redacted):
            redacted = await redacted
        return _coerce_redacted_result(result, redacted)

    @staticmethod
    def _versioned_result(
        tool,
        result: ToolResult,
        attempts: list[dict],
    ) -> ToolResult:
        if tool is None:
            return replace(
                result,
                metadata={**result.metadata, "attempt_history": attempts},
            )
        identity = tool.resolved_identity()
        policy = tool.resolved_policy()
        return replace(
            result,
            metadata={
                **result.metadata,
                "attempt_history": attempts,
                "tool_identity": identity.to_dict(),
                "tool_identity_digest": identity.digest,
                "tool_policy": policy.to_dict(),
                "tool_policy_digest": policy.digest,
            },
        )

    def _begin_goal_tool(
        self,
        turn: TurnState,
        *,
        tool_name: str,
        attempt: int,
    ) -> GoalActionCheckpoint | None:
        if self.goal_execution is None:
            return None
        return self.goal_execution.begin_tool(
            turn_id=turn.turn_id,
            tool_call_index=turn.tool_call_count,
            attempt=attempt,
            tool_name=tool_name,
        )

    def _finish_goal_tool(
        self,
        checkpoint: GoalActionCheckpoint | None,
        result: ToolResult,
    ) -> None:
        if self.goal_execution is not None and checkpoint is not None:
            self.goal_execution.finish_tool(checkpoint, result)

    def _abort_goal_tool(
        self,
        checkpoint: GoalActionCheckpoint | None,
        error: BaseException,
    ) -> None:
        if self.goal_execution is not None and checkpoint is not None:
            self.goal_execution.abort_tool(checkpoint, error)

    def _reserve_tool_attempt(
        self,
        turn: TurnState,
        *,
        tool_name: str,
        attempt: int,
    ) -> None:
        if self.usage_accounting is None:
            return
        try:
            reservation = self.usage_accounting.reserve_tool_call(
                turn_id=turn.turn_id,
                tool_call_index=turn.tool_call_count,
                attempt=attempt,
                tool_name=tool_name,
            )
        except BudgetExceededError as exc:
            payload = {
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "tool_call_index": turn.tool_call_count,
                "attempt": attempt,
                "resource_kind": "tool",
                "scope": exc.scope.value,
                "dimension": exc.dimension,
                "limit": exc.limit,
                "committed": exc.committed,
                "reserved": exc.reserved,
                "requested": exc.requested,
                "message": str(exc),
            }
            turn.extension_metadata["budget_exhausted"] = payload
            self.trace(TraceEvent.BUDGET_EXHAUSTED, payload)
            raise
        self.trace(
            TraceEvent.BUDGET_RESERVED,
            {
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "tool_call_index": turn.tool_call_count,
                "attempt": attempt,
                "resource_kind": "tool",
                "scope": reservation.budget.scope.value,
                "reservation_id": reservation.id,
                "reserved_tool_calls": reservation.reserved_tool_calls,
            },
        )

    def _commit_tool_attempt(
        self,
        turn: TurnState,
        *,
        tool_name: str,
        attempt: int,
        result: ToolResult,
    ) -> None:
        if self.usage_accounting is None:
            return
        entries = self.usage_accounting.commit_tool_call(
            turn_id=turn.turn_id,
            tool_call_index=turn.tool_call_count,
            attempt=attempt,
            tool_name=tool_name,
            success=result.success,
            failure_kind=result.failure_kind,
        )
        self.trace(
            TraceEvent.BUDGET_COMMITTED,
            {
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "tool_call_index": turn.tool_call_count,
                "attempt": attempt,
                "resource_kind": "tool",
                "entry_ids": [item.id for item in entries],
            },
        )

    def _release_tool_attempt(self, turn: TurnState, *, attempt: int) -> None:
        if self.usage_accounting is None:
            return
        reservation = self.usage_accounting.release_tool_call(
            turn_id=turn.turn_id,
            tool_call_index=turn.tool_call_count,
            attempt=attempt,
        )
        if reservation is not None:
            self.trace(
                TraceEvent.BUDGET_RELEASED,
                {
                    "turn_id": turn.turn_id,
                    "tool_call_index": turn.tool_call_count,
                    "attempt": attempt,
                    "resource_kind": "tool",
                    "reservation_id": reservation.id,
                    "reason": "tool_result_unavailable",
                },
            )

    async def _permission_result_async(
        self,
        tool_name: str,
        arguments: dict,
        turn: TurnState,
    ) -> ToolResult | None:
        """Resolve blocking host approvals without stalling the agent event loop."""
        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            return None
        request = self.permission_policy.request_for_tool(tool, arguments)
        self.trace(
            TraceEvent.TOOL_PERMISSION_REQUESTED,
            {"turn_id": turn.turn_id, "request": request.to_dict()},
        )
        record = self.permission_policy.decide(request)
        if record.decision == PermissionDecision.ASK:
            record = await asyncio.to_thread(self._resolve_approval, request, record)
        self.trace(
            TraceEvent.TOOL_PERMISSION_DECIDED,
            {"turn_id": turn.turn_id, "decision": record.to_dict()},
        )
        if record.decision == PermissionDecision.ALLOW:
            return None
        return _permission_denied_result(request, record)

    def resolve_hosted_mcp_approval(self, approval: dict, turn: TurnState) -> bool:
        """Resolve one provider-hosted MCP approval through the same policy."""
        server_label = str(approval.get("server_label") or "unknown")
        tool_name = str(approval.get("name") or approval.get("tool_name") or "unknown")
        arguments = {
            "server_label": server_label,
            "tool_name": tool_name,
            "arguments": approval.get("arguments"),
            "approval_request_id": (
                approval.get("approval_request_id") or approval.get("id")
            ),
        }
        request = PermissionRequest(
            tool_name=f"mcp:{server_label}:{tool_name}",
            permission_level=ToolPermissionLevel.EXTERNAL_SERVICE,
            arguments=arguments,
            requires_confirmation=True,
            policy_name=self.permission_policy.name,
            reason="hosted MCP tool call requires approval",
        )
        self.trace(
            TraceEvent.MCP_APPROVAL_REQUESTED,
            {
                "turn_id": turn.turn_id,
                "request": request.to_dict(),
                "provider_request": approval,
            },
        )
        self.trace(
            TraceEvent.TOOL_PERMISSION_REQUESTED,
            {"turn_id": turn.turn_id, "request": request.to_dict()},
        )
        record = self.permission_policy.decide(request)
        if record.decision == PermissionDecision.ASK:
            record = self._resolve_approval(request, record)
        self.trace(
            TraceEvent.TOOL_PERMISSION_DECIDED,
            {"turn_id": turn.turn_id, "decision": record.to_dict()},
        )
        self.trace(
            TraceEvent.MCP_APPROVAL_DECIDED,
            {
                "turn_id": turn.turn_id,
                "decision": record.to_dict(),
                "approval_request_id": arguments["approval_request_id"],
            },
        )
        return record.decision == PermissionDecision.ALLOW

    def _registered_tool(self, tool_name: str):
        names = {item.name for item in self.registry.list_tools()}
        return self.registry.get(tool_name) if tool_name in names else None

    def _permission_result(
        self,
        tool_name: str,
        arguments: dict,
        turn: TurnState,
    ) -> ToolResult | None:
        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            return None
        request = self.permission_policy.request_for_tool(tool, arguments)
        self.trace(
            TraceEvent.TOOL_PERMISSION_REQUESTED,
            {"turn_id": turn.turn_id, "request": request.to_dict()},
        )
        record = self.permission_policy.decide(request)
        if record.decision == PermissionDecision.ASK:
            record = self._resolve_approval(request, record)
        self.trace(
            TraceEvent.TOOL_PERMISSION_DECIDED,
            {"turn_id": turn.turn_id, "decision": record.to_dict()},
        )
        if record.decision == PermissionDecision.ALLOW:
            return None
        return _permission_denied_result(request, record)

    def _resolve_approval(
        self,
        request: PermissionRequest,
        record: PermissionDecisionRecord,
    ) -> PermissionDecisionRecord:
        if self.permission_callback is None:
            return PermissionDecisionRecord(
                tool_name=request.tool_name,
                permission_level=request.permission_level,
                decision=PermissionDecision.DENY,
                reason="tool call requires approval but no permission callback is configured",
                policy_name=record.policy_name,
                requires_confirmation=request.requires_confirmation,
                capability_category=request.capability_category,
                capability_enabled=request.capability_enabled,
                tool_identity=request.tool_identity,
                tool_policy=request.tool_policy,
                arguments_digest=request.arguments_digest,
            )
        callback_decision = self.permission_callback(request, record)
        if isinstance(callback_decision, bool):
            decision = (
                PermissionDecision.ALLOW if callback_decision else PermissionDecision.DENY
            )
        elif isinstance(callback_decision, PermissionDecision):
            decision = callback_decision
        else:
            decision = PermissionDecision(str(callback_decision))
        reason = (
            "tool call approved by permission callback"
            if decision == PermissionDecision.ALLOW
            else "tool call denied by permission callback"
        )
        return PermissionDecisionRecord(
            tool_name=request.tool_name,
            permission_level=request.permission_level,
            decision=decision,
            reason=reason,
            policy_name=record.policy_name,
            requires_confirmation=request.requires_confirmation,
            capability_category=request.capability_category,
            capability_enabled=request.capability_enabled,
            tool_identity=request.tool_identity,
            tool_policy=request.tool_policy,
            arguments_digest=request.arguments_digest,
        )


def _attempt_policy(tool, retry_policy) -> tuple[int, bool]:
    max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
    non_idempotent_guard = bool(
        retry_policy is not None
        and retry_policy.require_idempotent
        and tool is not None
        and not (
            tool.idempotent
            or tool.resolved_policy().idempotency.value != "none"
        )
    )
    return (1 if non_idempotent_guard else max_attempts), non_idempotent_guard


def _authorization(value: ToolAuthorization | bool) -> ToolAuthorization:
    if isinstance(value, ToolAuthorization):
        return value
    if isinstance(value, bool):
        return ToolAuthorization(value)
    raise TypeError("tool authorizer must return ToolAuthorization or bool")


def _credentials(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("credential resolver must return a mapping")
    return dict(value)


def _effect_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("effect-key hook must return a non-empty string or None")
    return value.strip()


def _authorization_denied_result(
    identity: ToolIdentity,
    policy: ToolPolicy,
    reason: str,
) -> ToolResult:
    return ToolResult(
        tool_name=identity.name,
        success=False,
        observation=(
            f"Tool call blocked by host authorization: {reason}. "
            "Do not claim the tool ran."
        ),
        error="authorization_denied",
        failure_kind=ToolFailureKind.USER_BLOCKED,
        metadata={
            "reason": reason,
            "tool_identity": identity.to_dict(),
            "tool_identity_digest": identity.digest,
            "tool_policy": policy.to_dict(),
            "tool_policy_digest": policy.digest,
        },
    )


def _host_authorization_payload(
    identity: ToolIdentity,
    policy: ToolPolicy,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    return {
        "tool_name": identity.name,
        "tool_identity": identity.to_dict(),
        "tool_identity_digest": identity.digest,
        "tool_policy": policy.to_dict(),
        "tool_policy_digest": policy.digest,
        "arguments_digest": schema_digest(arguments),
    }


def _redaction_payload(
    arguments: Mapping[str, object],
    result: ToolResult,
) -> dict[str, object]:
    return {
        "arguments": dict(arguments),
        "success": result.success,
        "observation": result.observation,
        "value": result.value,
        "metadata": dict(result.metadata),
    }


def _coerce_redacted_result(
    original: ToolResult,
    redacted: object,
) -> ToolResult:
    if isinstance(redacted, ToolResult):
        return redacted
    if isinstance(redacted, str):
        return replace(original, observation=redacted, value=None)
    if isinstance(redacted, Mapping):
        observation = redacted.get("observation", original.observation)
        if not isinstance(observation, str):
            raise TypeError("tool redaction observation must be a string")
        return replace(
            original,
            observation=observation,
            value=redacted.get("value"),
            metadata=dict(redacted.get("metadata", {}))
            if isinstance(redacted.get("metadata"), Mapping)
            else {},
        )
    raise TypeError(
        "tool redaction hook must return ToolResult, string, or mapping"
    )


def _withheld_secret_result(result: ToolResult) -> ToolResult:
    metadata = {
        key: value
        for key, value in result.metadata.items()
        if key not in {"structured_output"}
    }
    metadata["secret_output_withheld"] = True
    return replace(
        result,
        observation="[secret tool output withheld]",
        stdout=None,
        stderr=None,
        error=None if result.success else result.error,
        metadata=metadata,
        value=None,
    )


def _permission_denied_result(
    request: PermissionRequest,
    record: PermissionDecisionRecord,
) -> ToolResult:
    return ToolResult(
        tool_name=request.tool_name,
        success=False,
        observation=(
            f"Tool call blocked by permission policy: {record.reason}. "
            "Do not claim the tool ran. Either request approval, choose an allowed tool, "
            "or explain the limitation."
        ),
        error="permission_denied",
        failure_kind=ToolFailureKind.USER_BLOCKED,
        metadata={
            "permission_request": request.to_dict(),
            "permission_decision": record.to_dict(),
        },
    )


def _should_retry(result, retry_policy, attempt_number: int, max_attempts: int) -> bool:
    if result.success or retry_policy is None or attempt_number >= max_attempts:
        return False
    if result.failure_kind in {
        ToolFailureKind.CANCELLED,
        ToolFailureKind.USER_BLOCKED,
        ToolFailureKind.FATAL_SAFETY,
        ToolFailureKind.UNKNOWN_TOOL,
        ToolFailureKind.INVALID_ARGUMENTS,
        ToolFailureKind.ASYNC_REQUIRED,
    }:
        return False
    return result.failure_kind in retry_policy.retryable_failure_kinds


def _attempt_payload(
    attempt_number: int,
    started_at: str,
    result: ToolResult,
    *,
    retry: bool,
    non_idempotent_guard: bool,
) -> dict:
    return {
        "attempt": attempt_number,
        "started_at": started_at,
        "ended_at": utc_now(),
        "success": result.success,
        "failure_kind": result.failure_kind,
        "error": result.error,
        "permission_decision": (
            "denied"
            if result.failure_kind == ToolFailureKind.USER_BLOCKED
            else "allowed"
        ),
        "retry_scheduled": retry,
        "retry_disposition": (
            "non_idempotent_guard"
            if non_idempotent_guard and not result.success
            else "scheduled"
            if retry
            else "finished"
        ),
    }
