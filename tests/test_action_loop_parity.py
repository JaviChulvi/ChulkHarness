"""Characterization tests shared by the sync and async orchestration drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from copy import deepcopy
import re
from typing import Any

import pytest

from chulk.capabilities import ToolRetryPolicy
from chulk.core import Agent
from chulk.core.plan_execution import PlanStepVerification, PlanStepVerificationRequest
from chulk.llm import LLMClient, LLMError
from chulk.testing import ScriptedLLMClient, ScriptedResponse
from chulk.tools import Tool, ToolFailureKind, ToolRegistry
from chulk.tools.registry import ToolResult


_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)
_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"
)


def _final_answer(content: str) -> dict[str, Any]:
    return {"type": "final_answer", "content": content}


def _tool_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_call", "tool_name": name, "arguments": arguments or {}}


def _plan() -> dict[str, Any]:
    return {
        "type": "plan",
        "plan": {
            "summary": "Implement the requested change.",
            "steps": [
                {
                    "id": "implementation",
                    "title": "Implement behavior",
                    "description": "Implement and verify the requested behavior.",
                    "acceptance_criteria": ["The requested behavior is verified."],
                }
            ],
        },
    }


def _complete_plan_step() -> dict[str, Any]:
    return {
        "type": "plan_step_update",
        "step_update": {
            "step_id": "implementation",
            "status": "completed",
            "evidence": "The requested behavior is verified.",
        },
    }


def _success_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Return a deterministic lookup result.",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            callable=lambda arguments: ToolResult(
                tool_name="lookup",
                success=True,
                observation=f"Found {arguments['query']}.",
                value={"query": arguments["query"]},
            ),
        )
    )
    return registry


def _failure_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="unstable_lookup",
            description="Return a deterministic tool failure.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=lambda _arguments: ToolResult(
                tool_name="unstable_lookup",
                success=False,
                observation="The lookup backend is unavailable.",
                error="backend_unavailable",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            ),
        )
    )
    return registry


def _fatal_safety_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="dangerous_operation",
            description="Return a deterministic fatal safety failure.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=lambda _arguments: ToolResult(
                tool_name="dangerous_operation",
                success=False,
                observation="The operation violated a hard safety boundary.",
                error="blocked_command",
                failure_kind=ToolFailureKind.FATAL_SAFETY,
            ),
        )
    )
    return registry


def _ordinary_denial_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="approval_required",
            description="Return a recoverable permission denial.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=lambda _arguments: ToolResult(
                tool_name="approval_required",
                success=False,
                observation="The user did not approve this operation.",
                error="permission_denied",
                failure_kind=ToolFailureKind.USER_BLOCKED,
            ),
        )
    )
    return registry


def _retry_success_registry() -> ToolRegistry:
    attempts = 0

    def flaky_lookup(_arguments: dict[str, Any]) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                tool_name="flaky_lookup",
                success=False,
                observation="The first lookup attempt failed.",
                error="temporary_failure",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            )
        return ToolResult(
            tool_name="flaky_lookup",
            success=True,
            observation="The retry succeeded.",
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="flaky_lookup",
            description="Fail once and then return a deterministic result.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=flaky_lookup,
            retry_policy=ToolRetryPolicy(max_attempts=3),
            idempotent=True,
        )
    )
    return registry


def _retry_exhaustion_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="unavailable_lookup",
            description="Fail every retry deterministically.",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=lambda _arguments: ToolResult(
                tool_name="unavailable_lookup",
                success=False,
                observation="The lookup remains unavailable.",
                error="temporary_failure",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            ),
            retry_policy=ToolRetryPolicy(max_attempts=3),
            idempotent=True,
        )
    )
    return registry


def _empty_registry() -> ToolRegistry:
    return ToolRegistry()


class _FailingLLMClient(LLMClient):
    provider = "parity-provider"
    model = "parity-model"

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        raise LLMError(
            "provider unavailable",
            provider=self.provider,
            model=self.model,
            code="server_error",
            retryable=True,
            fallback_eligible=True,
        )


class _HangingAsyncLLMClient(LLMClient):
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        raise AssertionError("the sync provider transport must not run")

    async def acomplete_action(self, messages: list[dict[str, str]], **_kwargs: Any) -> Any:
        self.started.set()
        await asyncio.Future()
        raise AssertionError(f"unreachable: {messages!r}")


def _normalize_runtime_values(value: Any) -> Any:
    """Replace nondeterministic identifiers and timestamps in state and traces."""
    if isinstance(value, dict):
        return {key: _normalize_runtime_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_runtime_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_runtime_values(item) for item in value)
    if isinstance(value, str):
        value = _UUID_PATTERN.sub("<uuid>", value)
        return _TIMESTAMP_PATTERN.sub("<timestamp>", value)
    return value


def _run_driver(
    *,
    async_driver: bool,
    responses: Sequence[ScriptedResponse],
    registry_factory: Callable[[], ToolRegistry],
    planned: bool = False,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = Agent(
        ScriptedLLMClient(deepcopy(responses)),
        tool_registry=registry_factory(),
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
        **(agent_kwargs or {}),
    )

    if async_driver:
        if planned:
            pending_response = asyncio.run(agent.run_planned_turn_async("Do the work"))
            response = asyncio.run(agent.approve_plan_async())
        else:
            pending_response = None
            response = asyncio.run(agent.run_turn_async("Do the work"))
    elif planned:
        pending_response = agent.run_planned_turn("Do the work")
        response = agent.approve_plan()
    else:
        pending_response = None
        response = agent.run_turn("Do the work")

    turn = agent.state.turns[-1]
    return _normalize_runtime_values(
        {
            "pending_response": pending_response,
            "response": response,
            "turn": turn.to_dict(),
            "events": events,
        }
    )


def _assert_driver_parity(
    responses: Sequence[ScriptedResponse],
    registry_factory: Callable[[], ToolRegistry] = _empty_registry,
    *,
    planned: bool = False,
    agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sync_result = _run_driver(
        async_driver=False,
        responses=responses,
        registry_factory=registry_factory,
        planned=planned,
        agent_kwargs=agent_kwargs,
    )
    async_result = _run_driver(
        async_driver=True,
        responses=responses,
        registry_factory=registry_factory,
        planned=planned,
        agent_kwargs=agent_kwargs,
    )

    assert async_result["turn"] == sync_result["turn"]
    assert [event_type for event_type, _payload in async_result["events"]] == [
        event_type for event_type, _payload in sync_result["events"]
    ]
    assert async_result["events"] == sync_result["events"]
    assert async_result["pending_response"] == sync_result["pending_response"]
    assert async_result["response"] == sync_result["response"]
    return sync_result


def test_sync_and_async_direct_final_answer_have_parity() -> None:
    result = _assert_driver_parity([_final_answer("Work complete.")])

    assert result["response"] == "Work complete."
    assert result["turn"]["status"] == "completed"
    assert [event_type for event_type, _payload in result["events"]][-2:] == [
        "final_answer",
        "turn_finished",
    ]


@pytest.mark.parametrize(
    ("responses", "registry_factory", "completion_event", "success"),
    [
        (
            [_tool_call("lookup", {"query": "answer"}), _final_answer("Lookup complete.")],
            _success_registry,
            "tool_call_completed",
            True,
        ),
        (
            [_tool_call("unstable_lookup"), _final_answer("Lookup failed safely.")],
            _failure_registry,
            "tool_call_failed",
            False,
        ),
    ],
    ids=["tool-success", "tool-failure"],
)
def test_sync_and_async_tool_outcomes_have_parity(
    responses: Sequence[ScriptedResponse],
    registry_factory: Callable[[], ToolRegistry],
    completion_event: str,
    success: bool,
) -> None:
    result = _assert_driver_parity(responses, registry_factory)

    event_types = [event_type for event_type, _payload in result["events"]]
    assert completion_event in event_types
    assert result["turn"]["tool_calls"][0]["success"] is success
    assert result["turn"]["tool_call_count"] == 1
    assert result["turn"]["model_request_count"] == 2


@pytest.mark.parametrize("planned", [False, True], ids=["unplanned", "planned"])
def test_sync_and_async_fatal_safety_stops_without_second_model_request(
    planned: bool,
) -> None:
    responses = (
        [_plan(), _tool_call("dangerous_operation")]
        if planned
        else [_tool_call("dangerous_operation")]
    )

    result = _assert_driver_parity(
        responses,
        _fatal_safety_registry,
        planned=planned,
    )

    assert result["turn"]["status"] == "failed"
    assert result["turn"]["model_request_count"] == (2 if planned else 1)
    assert result["turn"]["tool_call_count"] == 1
    assert result["turn"]["tool_calls"][0]["failure_kind"] == "fatal_safety"
    assert result["response"].startswith("Fatal safety policy stopped the turn.")
    assert [event_type for event_type, _payload in result["events"]][-2:] == [
        "turn_failed",
        "turn_finished",
    ]


def test_ordinary_permission_denial_remains_recoverable() -> None:
    result = _assert_driver_parity(
        [
            _tool_call("approval_required"),
            _final_answer("Explained the approval requirement."),
        ],
        _ordinary_denial_registry,
    )

    assert result["turn"]["status"] == "completed"
    assert result["turn"]["model_request_count"] == 2
    assert result["turn"]["tool_calls"][0]["failure_kind"] == "user_blocked"


def test_unchanged_non_retryable_failure_stops_sync_and_async_without_reexecution() -> None:
    result = _assert_driver_parity(
        [_tool_call("missing_tool"), _tool_call("missing_tool")],
    )

    assert result["turn"]["status"] == "failed"
    assert result["turn"]["model_request_count"] == 2
    assert result["turn"]["tool_call_count"] == 1
    assert result["turn"]["tool_calls"][0]["failure_kind"] == "unknown_tool"
    assert result["response"].startswith("No-progress guard stopped")


def test_environment_failure_may_be_retried_unchanged_by_the_model() -> None:
    result = _assert_driver_parity(
        [
            _tool_call("unstable_lookup"),
            _tool_call("unstable_lookup"),
            _final_answer("Explained the persistent outage."),
        ],
        _failure_registry,
    )

    assert result["turn"]["status"] == "completed"
    assert result["turn"]["model_request_count"] == 3
    assert result["turn"]["tool_call_count"] == 2


@pytest.mark.parametrize(
    ("responses", "registry_factory", "expected_attempts", "success"),
    [
        (
            [_tool_call("flaky_lookup"), _final_answer("Retry complete.")],
            _retry_success_registry,
            2,
            True,
        ),
        (
            [_tool_call("unavailable_lookup"), _final_answer("Retries exhausted safely.")],
            _retry_exhaustion_registry,
            3,
            False,
        ),
    ],
    ids=["retry-success", "retry-exhaustion"],
)
def test_sync_and_async_tool_retries_have_parity(
    responses: Sequence[ScriptedResponse],
    registry_factory: Callable[[], ToolRegistry],
    expected_attempts: int,
    success: bool,
) -> None:
    result = _assert_driver_parity(responses, registry_factory)

    tool_call = result["turn"]["tool_calls"][0]
    attempts = tool_call["metadata"]["attempt_history"]
    assert len(attempts) == expected_attempts
    assert [attempt["attempt"] for attempt in attempts] == list(range(1, expected_attempts + 1))
    assert all(attempt["retry_scheduled"] for attempt in attempts[:-1])
    assert attempts[-1]["retry_scheduled"] is False
    assert tool_call["success"] is success


def test_sync_and_async_plan_creation_and_approval_have_parity() -> None:
    result = _assert_driver_parity(
        [_plan(), _complete_plan_step(), _final_answer("Approved work complete.")],
        planned=True,
    )

    event_types = [event_type for event_type, _payload in result["events"]]
    assert "plan_created" in event_types
    assert "plan_approved" in event_types
    assert event_types.index("plan_created") < event_types.index("plan_approved")
    assert result["turn"]["plan_approved"] is True
    assert result["turn"]["active_plan"]["status"] == "completed"
    assert result["turn"]["status"] == "completed"


def test_sync_verifier_has_sync_and_async_plan_execution_parity() -> None:
    def verifier(request: PlanStepVerificationRequest) -> PlanStepVerification:
        assert request.step_id == "implementation"
        return PlanStepVerification(
            passed=True,
            evidence="Host acceptance check passed.",
        )

    result = _assert_driver_parity(
        [_plan(), _complete_plan_step(), _final_answer("Verified work complete.")],
        planned=True,
        agent_kwargs={"plan_step_verifier": verifier},
    )

    evidence = result["turn"]["active_plan"]["steps"][0]["evidence"]
    assert [record["tool_name"] for record in evidence] == [
        "plan_step_update",
        "plan_step_verifier",
    ]


def test_sync_and_async_reflection_revision_have_parity() -> None:
    result = _assert_driver_parity(
        [
            _final_answer("Work complete."),
            {
                "approved": False,
                "reason": "The answer does not identify its evidence.",
                "feedback": "State that no tool evidence was collected.",
            },
            _final_answer("No tool evidence was collected."),
        ],
        agent_kwargs={"max_reflection_attempts": 1},
    )

    event_types = [event_type for event_type, _payload in result["events"]]
    assert result["response"] == "No tool evidence was collected."
    assert result["turn"]["reflection_count"] == 1
    assert result["turn"]["model_request_count"] == 3
    assert result["turn"]["observations"][-1]["tool_name"] == "reflection_feedback"
    assert event_types.index("reflection_started") < event_types.index("reflection_completed")
    assert event_types.index("reflection_completed") < event_types.index("reflection_revision_requested")


def test_sync_and_async_action_protocol_repair_have_parity() -> None:
    result = _assert_driver_parity(["not valid action JSON", _final_answer("Repaired response.")])

    assert result["response"] == "Repaired response."
    assert result["turn"]["model_request_count"] == 1
    assert result["turn"]["errors"] == ["JSON repair attempt: model response was not valid JSON"]
    model_response = next(payload for event_type, payload in result["events"] if event_type == "model_response")
    assert model_response["repair_attempts"] == 1
    assert model_response["repair_errors"] == ["model response was not valid JSON"]


def test_sync_and_async_terminal_action_protocol_failure_have_parity() -> None:
    result = _assert_driver_parity(["invalid one", "invalid two", "invalid three"])

    assert result["turn"]["status"] == "failed"
    assert result["turn"]["model_request_count"] == 1
    assert result["turn"]["errors"][-1].startswith("Model response was not valid action JSON after repair.")
    assert [event_type for event_type, _payload in result["events"]][-2:] == [
        "turn_failed",
        "turn_finished",
    ]


def _run_provider_failure(*, async_driver: bool) -> dict[str, Any]:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = Agent(
        _FailingLLMClient(),
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
    )

    with pytest.raises(LLMError) as error:
        if async_driver:
            asyncio.run(agent.run_turn_async("Trigger the provider failure"))
        else:
            agent.run_turn("Trigger the provider failure")

    return _normalize_runtime_values(
        {
            "exception": (type(error.value).__name__, str(error.value)),
            "turn": agent.state.turns[-1].to_dict(),
            "events": events,
        }
    )


def test_sync_and_async_provider_exception_terminalization_have_parity() -> None:
    sync_result = _run_provider_failure(async_driver=False)
    async_result = _run_provider_failure(async_driver=True)

    assert async_result == sync_result
    assert sync_result["exception"] == ("LLMError", "provider unavailable")
    assert sync_result["turn"]["status"] == "failed"
    assert sync_result["turn"]["errors"] == ["Turn failed with LLMError: provider unavailable"]
    assert [event_type for event_type, _payload in sync_result["events"]][-2:] == [
        "turn_failed",
        "turn_finished",
    ]


@pytest.mark.asyncio
async def test_async_cancellation_propagates_and_terminalizes_the_turn() -> None:
    started = asyncio.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    agent = Agent(
        _HangingAsyncLLMClient(started),
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
    )
    task = asyncio.create_task(agent.run_turn_async("Cancel this turn"))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    turn = _normalize_runtime_values(agent.state.turns[-1].to_dict())
    assert task.cancelled() is True
    assert turn["status"] == "cancelled"
    assert turn["ended_at"] == "<timestamp>"
    assert turn["errors"] == ["Turn cancelled."]
    assert [event_type for event_type, _payload in events][-2:] == [
        "turn_failed",
        "turn_finished",
    ]
