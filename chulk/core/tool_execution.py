"""Permission-aware sync and async tool transports."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
import time

from chulk.core.events import TraceEvent
from chulk.core.state import TurnState, utc_now
from chulk.tools import ToolRegistry
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
    ToolPermissionPolicy,
)
from chulk.tools.registry import ToolExecutionContext, ToolFailureKind, ToolResult


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

    def execute(self, tool_name: str, arguments: dict, turn: TurnState) -> ToolResult:
        """Execute a tool through the blocking transport and retry policy."""
        tool = self._registered_tool(tool_name)
        retry_policy = getattr(tool, "retry_policy", None)
        max_attempts, non_idempotent_guard = _attempt_policy(tool, retry_policy)
        attempts: list[dict] = []
        result: ToolResult | None = None
        for attempt_number in range(1, max_attempts + 1):
            started_at = utc_now()
            result = self._permission_result(tool_name, arguments, turn) or self.registry.run(
                tool_name,
                arguments,
                context=self.get_context(turn),
            )
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
        return replace(result, metadata={**result.metadata, "attempt_history": attempts})

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
            result = await self._permission_result_async(
                tool_name,
                arguments,
                turn,
            ) or await self.registry.run_async(
                tool_name,
                arguments,
                context=self.get_context(turn),
            )
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
        return replace(result, metadata={**result.metadata, "attempt_history": attempts})

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
        )


def _attempt_policy(tool, retry_policy) -> tuple[int, bool]:
    max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
    non_idempotent_guard = bool(
        retry_policy is not None
        and retry_policy.require_idempotent
        and tool is not None
        and not tool.idempotent
    )
    return (1 if non_idempotent_guard else max_attempts), non_idempotent_guard


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
