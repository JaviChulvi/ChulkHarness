"""Convert mutable runtime records into immutable public SDK snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, SupportsIndex, SupportsInt, TypeAlias, cast

from chulk.llm.usage import cost_snapshot_data, usage_snapshot_data
from chulk.results import (
    ContextBudget,
    ContextReport,
    ContextSection,
    Cost,
    MemoryProposal,
    MemoryProposalStatus,
    Observation,
    Plan,
    PlanResult,
    PlanSnapshot,
    PlanStatus,
    PlanStep,
    PlanStepEvidence,
    PlanStepStatus,
    RunResult,
    RunStatus,
    ToolAttempt,
    ToolCall,
    Usage,
)


def usage_snapshot(value: object) -> Usage | None:
    payload = usage_snapshot_data(value)
    if payload is None:
        return None
    return Usage(**payload)


def cost_snapshot(value: object) -> Cost | None:
    payload = cost_snapshot_data(value)
    if payload is None:
        return None
    return Cost(
        amount=_decimal(payload.get("amount")),
        currency=str(payload.get("currency") or "USD"),
        pricing_known=bool(payload.get("pricing_known")),
        estimated=bool(payload.get("estimated")),
        input_cost=_decimal(payload.get("input_cost")),
        cached_input_cost=_decimal(payload.get("cached_input_cost")),
        output_cost=_decimal(payload.get("output_cost")),
        provider=_optional_str(payload.get("provider")),
        model=_optional_str(payload.get("model")),
        pricing_source=_optional_str(payload.get("pricing_source")),
        pricing_last_checked=_optional_str(payload.get("pricing_last_checked")),
    )


def tool_call_snapshot(value: object) -> ToolCall:
    payload = _mapping(value)
    metadata = _dict(payload.get("metadata"))
    attempt_values = metadata.pop("attempt_history", ())
    attempts = tuple(
        ToolAttempt(
            attempt=_int(attempt_payload.get("attempt")),
            started_at=str(attempt_payload.get("started_at") or ""),
            ended_at=str(attempt_payload.get("ended_at") or ""),
            success=bool(attempt_payload.get("success")),
            failure_kind=_optional_str(attempt_payload.get("failure_kind")),
            error=_optional_str(attempt_payload.get("error")),
            permission_decision=_optional_str(attempt_payload.get("permission_decision")),
            retry_scheduled=bool(attempt_payload.get("retry_scheduled")),
            retry_disposition=str(attempt_payload.get("retry_disposition") or "finished"),
        )
        for item in attempt_values or ()
        if (attempt_payload := _mapping(item))
    )
    return ToolCall(
        tool_name=str(payload.get("tool_name") or "unknown"),
        arguments=_dict(payload.get("arguments")),
        iteration=_int(payload.get("iteration")),
        phase=str(payload.get("phase") or "execution"),
        plan_step_id=_optional_str(payload.get("plan_step_id")),
        started_at=_optional_str(payload.get("started_at")),
        ended_at=_optional_str(payload.get("ended_at")),
        resolved_tool_name=_optional_str(payload.get("resolved_tool_name")),
        success=payload.get("success") if isinstance(payload.get("success"), bool) else None,
        error=_optional_str(payload.get("error")),
        failure_kind=_optional_str(payload.get("failure_kind")),
        attempts=attempts,
        metadata=metadata,
    )


def observation_snapshot(value: object) -> Observation:
    payload = _mapping(value)
    return Observation(
        tool_name=str(payload.get("tool_name") or "unknown"),
        content=str(payload.get("content") or ""),
        output_metadata=_dict(payload.get("output_metadata")),
        created_at=_optional_str(payload.get("created_at")),
    )


def context_report_snapshot(value: object) -> ContextReport | None:
    if value is None:
        return None
    payload = _mapping(value)
    budget_payload = _dict(payload.get("budget"))
    budget = ContextBudget(
        enabled=bool(budget_payload.get("enabled")),
        context_window_tokens=_int(budget_payload.get("context_window_tokens")),
        max_prompt_tokens=_int(budget_payload.get("max_prompt_tokens")),
        response_reserve_tokens=_int(budget_payload.get("response_reserve_tokens")),
        input_token_budget=_optional_int(budget_payload.get("input_token_budget")),
    )
    sections = tuple(
        ContextSection(
            name=str(section.get("name") or "unknown"),
            label=str(section.get("label") or ""),
            char_count=_int(section.get("char_count")),
            estimated_tokens=_int(section.get("estimated_tokens")),
            item_count=_int(section.get("item_count"), default=1),
            metadata=_dict(section.get("metadata")),
        )
        for item in payload.get("sections") or ()
        if (section := _mapping(item))
    )
    return ContextReport(
        total_char_count=_int(payload.get("total_char_count")),
        estimated_tokens=_int(payload.get("estimated_tokens")),
        section_estimated_tokens=_int(payload.get("section_estimated_tokens")),
        budget=budget,
        over_budget_tokens=_int(payload.get("over_budget_tokens")),
        trimmed=bool(payload.get("trimmed")),
        included_message_count=_int(payload.get("included_message_count")),
        omitted_message_count=_int(payload.get("omitted_message_count")),
        omitted_observation_count=_int(payload.get("omitted_observation_count")),
        sections=sections,
    )


def plan_snapshot(value: object | None) -> Plan | None:
    if value is None:
        return None
    payload = _mapping(value)
    steps = tuple(_plan_step_snapshot(item) for item in payload.get("steps") or ())
    return Plan(
        summary=str(payload.get("summary") or ""),
        status=_enum(PlanStatus, payload.get("status"), PlanStatus.UNKNOWN),
        steps=steps,
        created_at=_optional_str(payload.get("created_at")),
        approved_at=_optional_str(payload.get("approved_at")),
        rejected_at=_optional_str(payload.get("rejected_at")),
    )


def memory_proposal_snapshot(value: object) -> MemoryProposal:
    payload = _mapping(value)
    return MemoryProposal(
        id=str(payload.get("id") or ""),
        content=str(payload.get("content") or ""),
        tags=tuple(str(item) for item in payload.get("tags") or ()),
        metadata=_dict(payload.get("metadata")),
        importance=_int(payload.get("importance"), default=1),
        source=str(payload.get("source") or "manual_review"),
        confidence=float(payload.get("confidence") or 0),
        evidence=_optional_str(payload.get("evidence")),
        conversation_id=_optional_str(payload.get("conversation_id")),
        turn_id=_optional_str(payload.get("turn_id")),
        status=_enum(MemoryProposalStatus, payload.get("status"), MemoryProposalStatus.UNKNOWN),
        created_at=str(payload.get("created_at") or ""),
        reviewed_at=_optional_str(payload.get("reviewed_at")),
        accepted_memory_id=_optional_str(payload.get("accepted_memory_id")),
    )


def run_result_from_runtime(runtime: Any, content: str | None = None) -> RunResult:
    """Build one detached public snapshot from a completed runtime turn."""
    state = runtime.state
    turn = state.turns[-1] if state.turns else None
    usage_totals = turn.model_usage_totals if turn is not None else state.last_usage_report or {}
    terminal_content = content
    if terminal_content is None and turn is not None:
        terminal_content = turn.final_answer or (turn.errors[-1] if turn.errors else None)
    if terminal_content is None:
        terminal_content = state.final_answer or (state.errors[-1] if state.errors else "")
    trace_logger = getattr(runtime, "trace_logger", None)
    return RunResult(
        content=terminal_content,
        status=_enum(RunStatus, turn.status if turn is not None else "unknown", RunStatus.UNKNOWN),
        turn_id=turn.turn_id if turn is not None else state.current_turn_id,
        conversation_id=state.conversation_id,
        trace_path=getattr(trace_logger, "path", None),
        usage=usage_snapshot(usage_totals.get("usage") if isinstance(usage_totals, dict) else None),
        cost=cost_snapshot(usage_totals.get("cost") if isinstance(usage_totals, dict) else None),
        context_report=context_report_snapshot(
            turn.context_reports[-1] if turn is not None and turn.context_reports else state.last_context_report
        ),
        tool_calls=tuple(tool_call_snapshot(record) for record in turn.tool_calls) if turn is not None else (),
        observations=(tuple(observation_snapshot(record) for record in turn.observations) if turn is not None else ()),
        loaded_skill_names=tuple(turn.loaded_skill_names) if turn is not None else tuple(state.loaded_skill_names),
        loaded_memory_ids=tuple(turn.loaded_memory_ids) if turn is not None else tuple(state.loaded_memory_ids),
        errors=tuple(turn.errors) if turn is not None else tuple(state.errors),
        plan=plan_snapshot(turn.active_plan if turn is not None else state.active_plan),
        extension_metadata=turn.extension_metadata if turn is not None else {},
    )


def plan_result_from_runtime(runtime: Any, content: str) -> PlanResult:
    state = runtime.state
    turn = state.turns[-1] if state.turns else None
    trace_logger = getattr(runtime, "trace_logger", None)
    return PlanResult(
        content=content,
        status=_enum(RunStatus, turn.status if turn is not None else "unknown", RunStatus.UNKNOWN),
        plan=plan_snapshot(turn.active_plan if turn is not None else state.active_plan),
        turn_id=turn.turn_id if turn is not None else state.current_turn_id,
        conversation_id=state.conversation_id,
        trace_path=getattr(trace_logger, "path", None),
        context_report=context_report_snapshot(
            turn.context_reports[-1] if turn is not None and turn.context_reports else state.last_context_report
        ),
        loaded_skill_names=tuple(turn.loaded_skill_names) if turn is not None else tuple(state.loaded_skill_names),
        loaded_memory_ids=tuple(turn.loaded_memory_ids) if turn is not None else tuple(state.loaded_memory_ids),
        errors=tuple(turn.errors) if turn is not None else tuple(state.errors),
    )


def _plan_step_snapshot(value: object) -> PlanStep:
    payload = _mapping(value)
    evidence = tuple(
        PlanStepEvidence(
            content=str(item_payload.get("content") or ""),
            tool_name=_optional_str(item_payload.get("tool_name")),
            tool_call_iteration=_optional_int(item_payload.get("tool_call_iteration")),
            created_at=_optional_str(item_payload.get("created_at")),
            metadata=_dict(item_payload.get("metadata")),
        )
        for item in payload.get("evidence") or ()
        if (item_payload := _mapping(item))
    )
    return PlanStep(
        id=str(payload.get("id") or ""),
        title=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        status=_enum(PlanStepStatus, payload.get("status"), PlanStepStatus.UNKNOWN),
        depends_on=tuple(str(item) for item in payload.get("depends_on") or ()),
        acceptance_criteria=tuple(str(item) for item in payload.get("acceptance_criteria") or ()),
        retry_limit=_int(payload.get("retry_limit")),
        evidence=evidence,
        started_at=_optional_str(payload.get("started_at")),
        completed_at=_optional_str(payload.get("completed_at")),
        blocked_at=_optional_str(payload.get("blocked_at")),
        blocked_reason=_optional_str(payload.get("blocked_reason")),
    )


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return {}


def _dict(value: object) -> dict[str, Any]:
    return _mapping(value)


_IntInput: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex


def _int(value: object, *, default: int = 0) -> int:
    try:
        return int(cast(_IntInput, value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    return None if value is None else _int(value)


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _enum(enum_type, value: object, unknown):
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return unknown


__all__ = [
    "PlanResult",
    "PlanSnapshot",
    "RunResult",
    "context_report_snapshot",
    "cost_snapshot",
    "memory_proposal_snapshot",
    "observation_snapshot",
    "plan_result_from_runtime",
    "plan_snapshot",
    "run_result_from_runtime",
    "tool_call_snapshot",
    "usage_snapshot",
]
