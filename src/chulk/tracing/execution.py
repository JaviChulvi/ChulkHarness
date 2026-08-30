"""Offline execution of deterministic replay fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from chulk.core.action_loop import run_action_loop, run_action_loop_async
from chulk.core.action_runtime import ActionLoopRuntime
from chulk.core.events import TraceEvent
from chulk.core.model_transport import ModelTransport
from chulk.core.plan_execution import PlanExecution
from chulk.core.reflection import ReflectionResult
from chulk.core.state import AgentState, TurnState
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects
from chulk.llm import LLMClient
from chulk.memory import ConversationMemory
from chulk.tools.registry import ToolFailureKind, ToolResult
from chulk.tracing.fixtures import (
    RecordedToolResult,
    ReplayFixture,
    ReplayFixtureError,
    normalize_replay_value,
)


_CORE_EVENT_TYPES = frozenset(
    {
        TraceEvent.TURN_STARTED,
        TraceEvent.USER_MESSAGE,
        TraceEvent.LLM_FALLBACK_ATTEMPTS,
        TraceEvent.TOOL_PERMISSION_REQUESTED,
        TraceEvent.TOOL_PERMISSION_DECIDED,
        TraceEvent.TOOL_CALL_STARTED,
        TraceEvent.TOOL_CALL_ATTEMPT,
        TraceEvent.TOOL_CALL,
        TraceEvent.TOOL_CALL_COMPLETED,
        TraceEvent.TOOL_CALL_FAILED,
        TraceEvent.TOOL_OBSERVATION,
        TraceEvent.PLAN_CREATED,
        TraceEvent.PLAN_APPROVED,
        TraceEvent.PLAN_REJECTED,
        TraceEvent.PLAN_REVISION_REQUESTED,
        TraceEvent.PLAN_STEP_STARTED,
        TraceEvent.PLAN_STEP_COMPLETED,
        TraceEvent.PLAN_STEP_BLOCKED,
        TraceEvent.REFLECTION_STARTED,
        TraceEvent.REFLECTION_COMPLETED,
        TraceEvent.REFLECTION_FAILED,
        TraceEvent.REFLECTION_REVISION_REQUESTED,
        TraceEvent.FINAL_ANSWER,
        TraceEvent.TURN_FAILED,
        TraceEvent.TURN_FINISHED,
    }
)
_STATE_KEYS = (
    "conversation_id",
    "current_turn_id",
    "message_count",
    "turn_count",
    "loaded_memory_ids",
    "loaded_skill_names",
    "available_tool_names",
    "error_count",
    "final_answer",
    "active_plan",
    "pending_plan_turn_id",
    "conversation_summary",
)


class ReplayExecutionError(ReplayFixtureError):
    """Raised when a valid fixture cannot be executed deterministically."""


@dataclass(frozen=True)
class ReplayExecutionReport:
    """Comparison between one executable replay and its recorded outcome."""

    ok: bool
    asynchronous: bool
    comparisons: dict[str, bool]
    mismatches: tuple[str, ...]
    actual: dict[str, Any]
    expected: dict[str, Any]
    model_actions_consumed: int
    tool_results_consumed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": "matched" if self.ok else "mismatch",
            "mode": "executable_fixture",
            "executed": True,
            "offline": True,
            "asynchronous": self.asynchronous,
            "comparisons": self.comparisons,
            "mismatches": list(self.mismatches),
            "model_actions_consumed": self.model_actions_consumed,
            "tool_results_consumed": self.tool_results_consumed,
            "actual": self.actual,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class _ReplayObservation:
    content: str
    output_metadata: dict[str, Any]


class _OfflineLLMClient(LLMClient):
    provider = "replay"
    model = "offline"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        del messages, max_output_tokens
        raise ReplayExecutionError("Executable replay attempted a provider call")


class _ScriptedModelTransport:
    def __init__(self, fixture: ReplayFixture, *, trace) -> None:
        self.llm_client: LLMClient = _OfflineLLMClient()
        self._trace = trace
        self._actions = list(fixture.model_actions)
        self._action_index = 0
        self._reflection_records = _reflection_records(fixture)
        self._reflection_index = 0
        self._request_indices: list[int] = []
        self._fallback_events = [
            event
            for event in fixture.expected.events
            if event.get("type") == TraceEvent.LLM_FALLBACK_ATTEMPTS
        ]
        self._fallback_index = 0

    @property
    def consumed(self) -> int:
        return self._action_index

    @property
    def request_indices(self) -> tuple[int, ...]:
        return tuple(self._request_indices)

    def assert_exhausted(self) -> None:
        if self._action_index != len(self._actions):
            remaining = len(self._actions) - self._action_index
            raise ReplayExecutionError(
                f"Replay stopped with {remaining} unconsumed model action(s)"
            )
        if self._reflection_index != len(self._reflection_records):
            remaining = len(self._reflection_records) - self._reflection_index
            raise ReplayExecutionError(
                f"Replay stopped with {remaining} unconsumed reflection result(s)"
            )
        if self._fallback_index != len(self._fallback_events):
            remaining = len(self._fallback_events) - self._fallback_index
            raise ReplayExecutionError(
                f"Replay stopped with {remaining} unconsumed fallback event(s)"
            )

    def build_prompt(self, turn: TurnState, *, require_plan: bool) -> None:
        del turn, require_plan
        return None

    def compact_prompt(
        self,
        prompt: None,
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> None:
        del turn, require_plan
        return prompt

    async def compact_prompt_async(
        self,
        prompt: None,
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> None:
        return self.compact_prompt(prompt, turn, require_plan=require_plan)

    def request_action(
        self,
        turn: TurnState,
        prompt: None,
        *,
        require_plan: bool,
    ):
        del prompt, require_plan
        if self._action_index >= len(self._actions):
            raise ReplayExecutionError("Replay model action script is exhausted")
        record = self._actions[self._action_index]
        self._action_index += 1
        request_index = record.request_index or (turn.model_request_count + 1)
        if request_index <= turn.model_request_count:
            raise ReplayExecutionError(
                "Recorded model request indices are not strictly increasing"
            )
        turn.model_request_count = request_index
        self._request_indices.append(request_index)
        while self._fallback_index < len(self._fallback_events):
            event = self._fallback_events[self._fallback_index]
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            event_request_index = payload.get("request_index")
            if event_request_index != request_index:
                break
            self._fallback_index += 1
            self._trace(TraceEvent.LLM_FALLBACK_ATTEMPTS, dict(payload))
        return record.to_action()

    async def request_action_async(
        self,
        turn: TurnState,
        prompt: None,
        *,
        require_plan: bool,
    ):
        return self.request_action(turn, prompt, require_plan=require_plan)

    def reflect(self, proposed_answer: str, turn: TurnState) -> ReflectionResult:
        if self._reflection_index >= len(self._reflection_records):
            raise ReplayExecutionError("Replay reflection script is exhausted")
        payload = self._reflection_records[self._reflection_index]
        self._reflection_index += 1
        turn.reflection_count += 1
        request_index = payload.get("request_index")
        if not isinstance(request_index, int):
            request_index = turn.model_request_count + 1
        turn.model_request_count = max(turn.model_request_count, request_index)
        self._request_indices.append(request_index)
        approved = payload.get("approved")
        reason = payload.get("reason")
        feedback = payload.get("feedback")
        if not isinstance(approved, bool) or not isinstance(reason, str):
            raise ReplayExecutionError(
                "Recorded reflection result is missing approved/reason fields"
            )
        reflection = ReflectionResult(
            approved=approved,
            reason=reason,
            feedback=feedback if isinstance(feedback, str) else None,
        )
        record = {
            **reflection.to_dict(),
            "attempt": turn.reflection_count,
            "proposed_answer": proposed_answer,
        }
        self._trace(
            TraceEvent.REFLECTION_STARTED,
            {
                "turn_id": turn.turn_id,
                "reflection_attempt": turn.reflection_count,
                "proposed_answer": proposed_answer,
            },
        )
        if payload.get("type") == TraceEvent.REFLECTION_FAILED:
            error = payload.get("error")
            raw_response = payload.get("raw_response")
            record.update({"error": error, "raw_response": raw_response})
            turn.errors.append(reason)
            self._trace(
                TraceEvent.REFLECTION_FAILED,
                {
                    "turn_id": turn.turn_id,
                    "request_index": request_index,
                    **record,
                },
            )
        else:
            self._trace(
                TraceEvent.REFLECTION_COMPLETED,
                {"turn_id": turn.turn_id, **record},
            )
        turn.reflections.append(record)
        return reflection

    async def reflect_async(
        self,
        proposed_answer: str,
        turn: TurnState,
    ) -> ReflectionResult:
        return self.reflect(proposed_answer, turn)


class _ReplayCancelled(asyncio.CancelledError):
    pass


class _ScriptedToolExecutor:
    def __init__(
        self,
        fixture: ReplayFixture,
        *,
        trace,
        cancellation_expected: bool,
    ) -> None:
        self._results = list(fixture.tool_results)
        self._index = 0
        self._trace = trace
        self._transport_groups = _tool_transport_groups(fixture)
        self._transport_index = 0
        self._permissions: list[dict[str, Any]] = []
        self._cancellation_expected = cancellation_expected

    @property
    def consumed(self) -> int:
        return self._index

    @property
    def permissions(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._permissions)

    def assert_exhausted(self) -> None:
        if self._index != len(self._results):
            remaining = len(self._results) - self._index
            raise ReplayExecutionError(
                f"Replay stopped with {remaining} unconsumed tool result(s)"
            )
        if self._transport_index != len(self._transport_groups):
            remaining = len(self._transport_groups) - self._transport_index
            raise ReplayExecutionError(
                f"Replay stopped with {remaining} unconsumed tool transport group(s)"
            )

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        turn: TurnState,
    ) -> ToolResult:
        del arguments
        return self._next(tool_name, turn)

    async def execute_async(
        self,
        tool_name: str,
        arguments: dict,
        turn: TurnState,
    ) -> ToolResult:
        del arguments
        return self._next(tool_name, turn)

    def _next(self, tool_name: str, turn: TurnState) -> ToolResult:
        self._emit_transport_group(tool_name)
        if self._index >= len(self._results):
            if self._cancellation_expected:
                raise _ReplayCancelled()
            raise ReplayExecutionError("Replay tool result script is exhausted")
        record = self._results[self._index]
        self._index += 1
        _validate_tool_result(record, tool_name=tool_name, turn=turn)
        return ToolResult(
            tool_name=record.tool_name,
            success=record.success,
            observation="",
            exit_code=record.exit_code,
            error=record.error,
            failure_kind=record.failure_kind,
            metadata=dict(record.metadata or {}),
            value=_ReplayObservation(
                content=record.observation,
                output_metadata=dict(record.output_metadata or {}),
            ),
        )

    def _emit_transport_group(self, tool_name: str) -> None:
        if self._transport_index >= len(self._transport_groups):
            return
        group = self._transport_groups[self._transport_index]
        self._transport_index += 1
        for event in group:
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            recorded_name = payload.get("tool_name")
            if isinstance(recorded_name, str) and recorded_name != tool_name:
                raise ReplayExecutionError(
                    f"Recorded tool transport expected {recorded_name!r}, "
                    f"got {tool_name!r}"
                )
            event_type = str(event.get("type"))
            self._trace(event_type, dict(payload))
            if event_type in {
                TraceEvent.TOOL_PERMISSION_REQUESTED,
                TraceEvent.TOOL_PERMISSION_DECIDED,
            }:
                self._permissions.append(
                    {
                        "type": event_type,
                        "turn_id": event.get("turn_id"),
                        "payload": dict(payload),
                    }
                )


class _ReplayTurnEffects(TurnEffects):
    def _format_observation(
        self,
        requested_tool_name: str,
        result: ToolResult,
    ) -> tuple[str, dict]:
        del requested_tool_name
        replay = result.value
        if not isinstance(replay, _ReplayObservation):
            raise ReplayExecutionError("Replay tool result lost its recorded observation")
        return replay.content, dict(replay.output_metadata)


@dataclass
class _ReplayRuntime:
    fixture: ReplayFixture
    state: AgentState
    memory: ConversationMemory
    turn: TurnState
    model: _ScriptedModelTransport
    tools: _ScriptedToolExecutor
    effects: _ReplayTurnEffects
    events: list[dict[str, Any]]


def execute_replay_fixture(fixture: ReplayFixture) -> ReplayExecutionReport:
    """Execute one fixture without providers, registered tools, or adapters."""
    runtime = _build_runtime(fixture)
    _run(runtime, asynchronous=False)
    return _report(runtime, asynchronous=False)


async def execute_replay_fixture_async(
    fixture: ReplayFixture,
) -> ReplayExecutionReport:
    """Execute one fixture through the async action-loop driver."""
    runtime = _build_runtime(fixture)
    await _run_async(runtime)
    return _report(runtime, asynchronous=True)


def _build_runtime(fixture: ReplayFixture) -> _ReplayRuntime:
    expected_state = fixture.expected.state
    state = AgentState()
    state.loaded_memory_ids = list(expected_state.get("loaded_memory_ids") or [])
    state.loaded_skill_names = list(expected_state.get("loaded_skill_names") or [])
    state.available_tool_names = list(
        expected_state.get("available_tool_names") or _tool_names(fixture)
    )
    state.conversation_summary = expected_state.get("conversation_summary")
    memory = ConversationMemory()
    memory.conversation_summary = state.conversation_summary
    turn = TurnState(
        user_message=_user_message(fixture),
        available_tool_names=list(state.available_tool_names),
    )
    state.current_turn_id = turn.turn_id
    state.turns.append(turn)
    events: list[dict[str, Any]] = []

    def trace(event_type: str, payload: dict | None = None) -> None:
        events.append(
            {
                "type": event_type,
                "turn_id": turn.turn_id,
                "payload": payload or {},
            }
        )

    plan = PlanExecution(state=state, memory=memory, trace=trace)
    effects = _ReplayTurnEffects(
        state=state,
        memory=memory,
        get_llm_client=_OfflineLLMClient,
        plan=plan,
        trace=trace,
        redact_text=lambda _event, text, _metadata: (text, {"redacted": False}),
        artifact_writer=lambda _content, _reason: (_raise_artifact_error()),
        planning_tool_names=lambda: frozenset(
            record.tool_name
            for record in fixture.tool_results
            if record.phase == "planning"
        ),
        max_tool_calls_per_turn=_max_tool_calls(fixture),
        max_reflection_attempts=len(_reflection_records(fixture)),
        max_observation_chars=12_000,
        max_tool_stdout_chars=8_000,
        max_tool_stderr_chars=4_000,
    )
    model = _ScriptedModelTransport(fixture, trace=trace)
    cancellation_expected = fixture.expected.result.get("status") == "cancelled"
    tools = _ScriptedToolExecutor(
        fixture,
        trace=trace,
        cancellation_expected=cancellation_expected,
    )
    trace(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})
    memory.add_user_message(turn.user_message)
    trace(
        TraceEvent.USER_MESSAGE,
        {"turn_id": turn.turn_id, "content": turn.user_message},
    )
    return _ReplayRuntime(
        fixture=fixture,
        state=state,
        memory=memory,
        turn=turn,
        model=model,
        tools=tools,
        effects=effects,
        events=events,
    )


def _run(runtime: _ReplayRuntime, *, asynchronous: bool) -> None:
    del asynchronous
    action_runtime = ActionLoopRuntime(
        model=cast(ModelTransport, runtime.model),
        tools=cast(ToolExecutor, runtime.tools),
        effects=runtime.effects,
    )
    try:
        run_action_loop(
            action_runtime,
            runtime.turn,
            require_plan=_requires_plan(runtime.fixture),
        )
        _continue_plan_sync(runtime, action_runtime)
    except _ReplayCancelled as exc:
        _terminalize_cancellation(runtime, exc)
    runtime.model.assert_exhausted()
    runtime.tools.assert_exhausted()


async def _run_async(runtime: _ReplayRuntime) -> None:
    action_runtime = ActionLoopRuntime(
        model=cast(ModelTransport, runtime.model),
        tools=cast(ToolExecutor, runtime.tools),
        effects=runtime.effects,
    )
    try:
        await run_action_loop_async(
            action_runtime,
            runtime.turn,
            require_plan=_requires_plan(runtime.fixture),
        )
        await _continue_plan_async(runtime, action_runtime)
    except _ReplayCancelled as exc:
        _terminalize_cancellation(runtime, exc)
    runtime.model.assert_exhausted()
    runtime.tools.assert_exhausted()


def _continue_plan_sync(
    runtime: _ReplayRuntime,
    action_runtime: ActionLoopRuntime,
) -> None:
    if runtime.turn.status != "waiting_for_approval":
        return
    if _has_event(runtime.fixture, TraceEvent.PLAN_REJECTED):
        _reject_plan(runtime)
        return
    if not _has_event(runtime.fixture, TraceEvent.PLAN_APPROVED):
        return
    _approve_plan(runtime)
    run_action_loop(action_runtime, runtime.turn, require_plan=False)


async def _continue_plan_async(
    runtime: _ReplayRuntime,
    action_runtime: ActionLoopRuntime,
) -> None:
    if runtime.turn.status != "waiting_for_approval":
        return
    if _has_event(runtime.fixture, TraceEvent.PLAN_REJECTED):
        _reject_plan(runtime)
        return
    if not _has_event(runtime.fixture, TraceEvent.PLAN_APPROVED):
        return
    _approve_plan(runtime)
    await run_action_loop_async(action_runtime, runtime.turn, require_plan=False)


def _approve_plan(runtime: _ReplayRuntime) -> None:
    turn = runtime.turn
    if turn.active_plan is None:
        raise ReplayExecutionError("Recorded plan approval has no proposed plan")
    turn.approve_plan()
    runtime.state.pending_plan_turn_id = None
    runtime.state.active_plan = turn.active_plan
    runtime.effects.trace(
        TraceEvent.PLAN_APPROVED,
        {
            "turn_id": turn.turn_id,
            "plan": turn.active_plan.to_dict(),
            "turn": turn.to_dict(),
        },
    )


def _reject_plan(runtime: _ReplayRuntime) -> None:
    turn = runtime.turn
    if turn.active_plan is None:
        raise ReplayExecutionError("Recorded plan rejection has no proposed plan")
    message = "Plan rejected. No tools were run."
    turn.reject_plan(message)
    runtime.state.final_answer = message
    runtime.memory.add_assistant_message(message)
    runtime.state.messages = runtime.memory.recent()
    runtime.effects.trace(
        TraceEvent.PLAN_REJECTED,
        {
            "turn_id": turn.turn_id,
            "plan": turn.active_plan.to_dict(),
            "message": message,
            "status": turn.status,
            "turn": turn.to_dict(),
        },
    )
    runtime.effects.trace(
        TraceEvent.TURN_FINISHED,
        runtime.effects.state_snapshot(turn),
    )


def _terminalize_cancellation(
    runtime: _ReplayRuntime,
    exc: BaseException,
) -> None:
    message = "Turn cancelled."
    turn = runtime.turn
    turn.cancel(message)
    runtime.state.errors.append(message)
    runtime.state.final_answer = message
    runtime.memory.add_assistant_message(message)
    runtime.state.messages = runtime.memory.recent()
    runtime.effects.plan.clear(turn)
    runtime.effects.trace(
        TraceEvent.TURN_FAILED,
        {
            "turn_id": turn.turn_id,
            "message": message,
            "status": turn.status,
            "exception_type": type(exc).__name__,
            "turn": turn.to_dict(),
        },
    )
    runtime.effects.trace(
        TraceEvent.TURN_FINISHED,
        runtime.effects.state_snapshot(turn),
    )


def _report(
    runtime: _ReplayRuntime,
    *,
    asynchronous: bool,
) -> ReplayExecutionReport:
    fixture = runtime.fixture
    actual_snapshot = runtime.effects.state_snapshot(runtime.turn)
    actual = {
        "state": _state_projection(actual_snapshot.get("agent_state", {})),
        "events": _event_sequence(runtime.events),
        "permissions": list(runtime.tools.permissions),
        "plans": _plan_sequence(runtime.events),
        "usage": _accounting_for_requests(
            fixture.expected.usage,
            runtime.model.request_indices,
        ),
        "costs": _accounting_for_requests(
            fixture.expected.costs,
            runtime.model.request_indices,
        ),
        "result": {
            "status": runtime.turn.status,
            "content": runtime.turn.final_answer,
            "errors": list(runtime.turn.errors),
            "plan": (
                runtime.turn.active_plan.to_dict()
                if runtime.turn.active_plan is not None
                else None
            ),
        },
    }
    expected_turn = _expected_finished_turn(fixture)
    expected = {
        "state": _state_projection(fixture.expected.state),
        "events": _event_sequence(fixture.expected.events),
        "permissions": list(fixture.expected.permissions),
        "plans": _plan_sequence(fixture.expected.plans),
        "usage": list(fixture.expected.usage),
        "costs": list(fixture.expected.costs),
        "result": (
            dict(fixture.expected.result)
            if fixture.expected.result
            else {
                "status": expected_turn.get("status"),
                "content": expected_turn.get("final_answer"),
                "errors": expected_turn.get("errors", []),
                "plan": expected_turn.get("active_plan"),
            }
        ),
    }
    actual_normalized = normalize_replay_value(actual)
    expected_normalized = normalize_replay_value(expected)
    comparisons = {
        key: actual_normalized[key] == expected_normalized[key]
        for key in expected_normalized
    }
    mismatches = tuple(key for key, matched in comparisons.items() if not matched)
    return ReplayExecutionReport(
        ok=not mismatches,
        asynchronous=asynchronous,
        comparisons=comparisons,
        mismatches=mismatches,
        actual=actual_normalized,
        expected=expected_normalized,
        model_actions_consumed=runtime.model.consumed,
        tool_results_consumed=runtime.tools.consumed,
    )


def _state_projection(value: object) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {key: payload.get(key) for key in _STATE_KEYS}


def _event_sequence(events: Sequence[object]) -> list[str]:
    return [
        str(event.get("type"))
        for event in events
        if isinstance(event, dict) and event.get("type") in _CORE_EVENT_TYPES
    ]


def _plan_sequence(events: Sequence[object]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in {
            TraceEvent.PLAN_CREATED,
            TraceEvent.PLAN_APPROVED,
            TraceEvent.PLAN_REJECTED,
            TraceEvent.PLAN_REVISION_REQUESTED,
            TraceEvent.PLAN_STEP_STARTED,
            TraceEvent.PLAN_STEP_COMPLETED,
            TraceEvent.PLAN_STEP_BLOCKED,
        }:
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        values.append(
            {
                "type": event["type"],
                "plan": payload.get("plan"),
                "step": payload.get("step"),
            }
        )
    return values


def _accounting_for_requests(
    records: tuple[dict[str, Any], ...],
    request_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    requested = set(request_indices)
    return [
        dict(record)
        for record in records
        if record.get("request_index") in requested
    ]


def _expected_finished_turn(fixture: ReplayFixture) -> dict[str, Any]:
    for event in reversed(fixture.expected.events):
        if event.get("type") != TraceEvent.TURN_FINISHED:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("turn"), dict):
            return dict(payload["turn"])
    return {}


def _user_message(fixture: ReplayFixture) -> str:
    for event in fixture.expected.events:
        if event.get("type") != TraceEvent.USER_MESSAGE:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("content"), str):
            content = payload["content"].strip()
            if content:
                return content
    raise ReplayExecutionError(
        "Replay fixture has no non-empty user_message event"
    )


def _reflection_records(fixture: ReplayFixture) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in fixture.expected.events:
        event_type = event.get("type")
        if event_type not in {
            TraceEvent.REFLECTION_COMPLETED,
            TraceEvent.REFLECTION_FAILED,
        }:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            records.append({"type": event_type, **payload})
    return records


def _tool_transport_groups(
    fixture: ReplayFixture,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    transport_types = {
        TraceEvent.TOOL_PERMISSION_REQUESTED,
        TraceEvent.TOOL_PERMISSION_DECIDED,
        TraceEvent.TOOL_CALL_ATTEMPT,
    }
    terminal_types = {
        TraceEvent.TOOL_CALL_COMPLETED,
        TraceEvent.TOOL_CALL_FAILED,
        TraceEvent.TURN_FAILED,
    }
    for event in fixture.expected.events:
        event_type = event.get("type")
        if event_type == TraceEvent.TOOL_CALL_STARTED:
            if current is not None:
                groups.append(current)
            current = []
            continue
        if current is None:
            continue
        if event_type in transport_types:
            current.append(dict(event))
            continue
        if event_type in terminal_types:
            groups.append(current)
            current = None
    if current is not None:
        groups.append(current)
    return groups


def _tool_names(fixture: ReplayFixture) -> list[str]:
    names = {
        str(record.action.get("tool_name"))
        for record in fixture.model_actions
        if record.action.get("type") == "tool_call"
    }
    return sorted(name for name in names if name and name != "None")


def _max_tool_calls(fixture: ReplayFixture) -> int:
    for event in fixture.expected.events:
        if event.get("type") != TraceEvent.TOOL_CALL_STARTED:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get("max_tool_calls_per_turn")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return max(5, len(fixture.tool_results))


def _requires_plan(fixture: ReplayFixture) -> bool:
    return _has_event(fixture, TraceEvent.PLAN_CREATED)


def _has_event(fixture: ReplayFixture, event_type: str) -> bool:
    return any(event.get("type") == event_type for event in fixture.expected.events)


def _validate_tool_result(
    record: RecordedToolResult,
    *,
    tool_name: str,
    turn: TurnState,
) -> None:
    if record.tool_name != tool_name:
        raise ReplayExecutionError(
            f"Recorded tool result expected {record.tool_name!r}, got {tool_name!r}"
        )
    if record.iteration != turn.tool_call_count:
        raise ReplayExecutionError(
            f"Recorded tool result iteration {record.iteration} does not match "
            f"action-loop iteration {turn.tool_call_count}"
        )
    expected_phase = turn.tool_calls[-1].phase if turn.tool_calls else None
    if record.phase != expected_phase:
        raise ReplayExecutionError(
            f"Recorded tool phase {record.phase!r} does not match {expected_phase!r}"
        )
    if record.failure_kind == ToolFailureKind.CANCELLED:
        raise _ReplayCancelled()


def _raise_artifact_error() -> None:
    raise ReplayExecutionError("Executable replay attempted to write an artifact")


__all__ = [
    "ReplayExecutionError",
    "ReplayExecutionReport",
    "execute_replay_fixture",
    "execute_replay_fixture_async",
]
