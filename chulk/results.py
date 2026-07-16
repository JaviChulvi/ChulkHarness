"""Immutable, typed public result snapshots for the Chulk SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar


class RunStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PLAN_REJECTED = "plan_rejected"
    CANCELLED = "cancelled"
    NO_PENDING_PLAN = "no_pending_plan"
    UNKNOWN = "unknown"


class PlanStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class MemoryProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False
    cache_split_estimated: bool = False
    source: str = "provider"
    raw: Mapping[str, Any] = field(default_factory=dict)
    cache_write_input_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", freeze_mapping(self.raw))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class Cost:
    amount: Decimal | None = None
    currency: str = "USD"
    pricing_known: bool = False
    estimated: bool = False
    input_cost: Decimal | None = None
    cached_input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    provider: str | None = None
    model: str | None = None
    pricing_source: str | None = None
    pricing_last_checked: str | None = None
    cache_write_input_cost: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class ToolAttempt:
    attempt: int
    started_at: str
    ended_at: str
    success: bool
    failure_kind: str | None = None
    error: str | None = None
    permission_decision: str | None = None
    retry_scheduled: bool = False
    retry_disposition: str = "finished"

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: Mapping[str, Any]
    iteration: int
    phase: str = "execution"
    plan_step_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    resolved_tool_name: str | None = None
    success: bool | None = None
    error: str | None = None
    failure_kind: str | None = None
    attempts: tuple[ToolAttempt, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", freeze_mapping(self.arguments))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class Observation:
    tool_name: str
    content: str
    output_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_metadata", freeze_mapping(self.output_metadata))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class ContextBudget:
    enabled: bool = False
    context_window_tokens: int = 0
    max_prompt_tokens: int = 0
    response_reserve_tokens: int = 0
    input_token_budget: int | None = None
    max_input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class ContextSection:
    name: str
    label: str
    char_count: int
    estimated_tokens: int
    item_count: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class ContextReport:
    total_char_count: int
    estimated_tokens: int
    section_estimated_tokens: int
    budget: ContextBudget
    over_budget_tokens: int
    trimmed: bool
    included_message_count: int
    omitted_message_count: int
    omitted_observation_count: int
    sections: tuple[ContextSection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class PlanStepEvidence:
    content: str
    tool_name: str | None = None
    tool_call_iteration: int | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class PlanStep:
    id: str
    title: str
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    depends_on: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    retry_limit: int = 0
    evidence: tuple[PlanStepEvidence, ...] = ()
    started_at: str | None = None
    completed_at: str | None = None
    blocked_at: str | None = None
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", enum_value(PlanStepStatus, self.status, PlanStepStatus.UNKNOWN))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "acceptance_criteria", tuple(self.acceptance_criteria))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class Plan:
    summary: str
    status: PlanStatus
    steps: tuple[PlanStep, ...] = ()
    created_at: str | None = None
    approved_at: str | None = None
    rejected_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", enum_value(PlanStatus, self.status, PlanStatus.UNKNOWN))
        object.__setattr__(self, "steps", tuple(self.steps))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


PlanSnapshot = Plan


@dataclass(frozen=True)
class RunResult:
    content: str
    status: RunStatus
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    usage: Usage | None = None
    cost: Cost | None = None
    context_report: ContextReport | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    observations: tuple[Observation, ...] = ()
    loaded_skill_names: tuple[str, ...] = ()
    loaded_memory_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    plan: Plan | None = None
    extension_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", enum_value(RunStatus, self.status, RunStatus.UNKNOWN))
        object.__setattr__(self, "trace_path", Path(self.trace_path) if self.trace_path is not None else None)
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "loaded_skill_names", tuple(self.loaded_skill_names))
        object.__setattr__(self, "loaded_memory_ids", tuple(self.loaded_memory_ids))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "extension_metadata", freeze_mapping(self.extension_metadata))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class PlanResult:
    content: str
    status: RunStatus
    plan: Plan | None
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    context_report: ContextReport | None = None
    loaded_skill_names: tuple[str, ...] = ()
    loaded_memory_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", enum_value(RunStatus, self.status, RunStatus.UNKNOWN))
        object.__setattr__(self, "trace_path", Path(self.trace_path) if self.trace_path is not None else None)
        object.__setattr__(self, "loaded_skill_names", tuple(self.loaded_skill_names))
        object.__setattr__(self, "loaded_memory_ids", tuple(self.loaded_memory_ids))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


@dataclass(frozen=True)
class MemoryProposal:
    id: str
    content: str
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]
    importance: int
    source: str
    confidence: float
    evidence: str | None
    conversation_id: str | None
    turn_id: str | None
    status: MemoryProposalStatus
    created_at: str
    reviewed_at: str | None = None
    accepted_memory_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(
            self,
            "status",
            enum_value(MemoryProposalStatus, self.status, MemoryProposalStatus.UNKNOWN),
        )

    def to_dict(self) -> dict[str, Any]:
        return plain_data(self)


EnumT = TypeVar("EnumT", bound=StrEnum)


def enum_value(enum_type: type[EnumT], value: str | EnumT, unknown: EnumT) -> EnumT:
    try:
        return enum_type(value)
    except ValueError:
        return unknown


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): freeze_value(item) for key, item in (value or {}).items()})


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(freeze_value(item) for item in value)
    return value


def plain_data(value: Any) -> Any:
    """Return fresh JSON-oriented plain data for a public snapshot."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain_data(item) for item in value]
    if is_dataclass(value):
        return {item.name: plain_data(getattr(value, item.name)) for item in fields(value)}
    return value


__all__ = [
    "ContextBudget",
    "ContextReport",
    "ContextSection",
    "Cost",
    "MemoryProposal",
    "MemoryProposalStatus",
    "Observation",
    "Plan",
    "PlanResult",
    "PlanSnapshot",
    "PlanStatus",
    "PlanStep",
    "PlanStepEvidence",
    "PlanStepStatus",
    "RunResult",
    "RunStatus",
    "ToolCall",
    "ToolAttempt",
    "Usage",
]
