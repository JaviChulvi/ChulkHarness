"""Tool permission primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolPermissionLevel(str, Enum):
    """Coarse permission level for a tool."""

    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    MEMORY = "memory"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    DESTRUCTIVE = "destructive"


class PermissionDecision(str, Enum):
    """Decision returned by a permission policy or user approval callback."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionRequest:
    """A pending permission decision for one requested tool call."""

    tool_name: str
    permission_level: ToolPermissionLevel
    arguments: dict[str, Any]
    requires_confirmation: bool = False
    policy_name: str = "default"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "permission_level": self.permission_level.value,
            "arguments": self.arguments,
            "requires_confirmation": self.requires_confirmation,
            "policy_name": self.policy_name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PermissionDecisionRecord:
    """Inspectable record of a tool permission decision."""

    tool_name: str
    permission_level: ToolPermissionLevel
    decision: PermissionDecision
    reason: str
    policy_name: str = "default"
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "permission_level": self.permission_level.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "policy_name": self.policy_name,
            "requires_confirmation": self.requires_confirmation,
        }


class ToolPermissionPolicy:
    """Base policy for deciding whether a tool call may execute."""

    def __init__(
        self,
        *,
        name: str = "default",
        default_decision: PermissionDecision = PermissionDecision.ALLOW,
        confirmation_decision: PermissionDecision = PermissionDecision.ASK,
    ) -> None:
        self.name = name
        self.default_decision = default_decision
        self.confirmation_decision = confirmation_decision

    def request_for_tool(self, tool, arguments: dict[str, Any]) -> PermissionRequest:
        level = normalize_permission_level(getattr(tool, "permission_level", ToolPermissionLevel.READ))
        reason = "tool requires confirmation" if getattr(tool, "requires_confirmation", False) else "tool allowed by default"
        return PermissionRequest(
            tool_name=tool.name,
            permission_level=level,
            arguments=arguments,
            requires_confirmation=bool(getattr(tool, "requires_confirmation", False)),
            policy_name=self.name,
            reason=reason,
        )

    def decide(self, request: PermissionRequest) -> PermissionDecisionRecord:
        decision = self.confirmation_decision if request.requires_confirmation else self.default_decision
        if decision == PermissionDecision.ALLOW:
            reason = "tool call allowed by permission policy"
        elif decision == PermissionDecision.ASK:
            reason = "tool call requires user approval"
        else:
            reason = "tool call denied by permission policy"
        return PermissionDecisionRecord(
            tool_name=request.tool_name,
            permission_level=request.permission_level,
            decision=decision,
            reason=reason,
            policy_name=self.name,
            requires_confirmation=request.requires_confirmation,
        )


def normalize_permission_level(value: ToolPermissionLevel | str) -> ToolPermissionLevel:
    """Return a known permission level."""
    if isinstance(value, ToolPermissionLevel):
        return value
    try:
        return ToolPermissionLevel(str(value))
    except ValueError as exc:
        supported = ", ".join(level.value for level in ToolPermissionLevel)
        raise ValueError(f"Unknown tool permission level {value!r}. Supported levels: {supported}") from exc
