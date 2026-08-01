"""Tests for structured output, timeout, and bounded retry policies."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import time

import pytest

from chulk import Agent, AgentConfig, Capabilities, Tool, ToolOutputPolicy, ToolRetryPolicy
from chulk.core.state import TurnState
from chulk.core.tool_execution import ToolExecutor
from chulk.llm import LLMClient
from chulk.tools import ToolExecutionContext, ToolRegistry
from chulk.tools.permissions import ToolPermissionPolicy
from chulk.tools.policy import schema_digest as policy_schema_digest
from chulk.tools.schema import schema_digest


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer", "minimum": 0}},
    "required": ["count"],
    "additionalProperties": False,
}


def test_schema_digest_compatibility_export_uses_schema_owner() -> None:
    assert policy_schema_digest is schema_digest


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _tool_call(name: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": "{}",
        }
    )


def _final() -> str:
    return json.dumps({"type": "final_answer", "content": "done"})


def test_structured_output_contract_accepts_valid_values():
    @Tool(output_policy=ToolOutputPolicy(OUTPUT_SCHEMA))
    def count_items() -> dict:
        """Count items."""
        return {"count": 2}

    registry = ToolRegistry()
    registry.register(count_items)

    result = registry.run("count_items", {})

    assert result.success is True
    assert result.value == {"count": 2}
    assert result.metadata["output_validated"] is True
    assert result.metadata["structured_output"] == {"count": 2}


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "string", "minLength": 1}, "ready"),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2, 3]),
    ],
)
def test_structured_output_contract_accepts_non_object_root_schemas(schema, value):
    @Tool(output_schema=schema)
    def structured_value():
        """Return a primitive or array structured value."""
        return value

    registry = ToolRegistry()
    registry.register(structured_value)

    result = registry.run("structured_value", {})

    assert result.success is True
    assert result.value == value
    assert result.metadata["structured_output"] == value


def test_structured_output_contract_rejects_invalid_values_with_fields():
    @Tool(output_schema=OUTPUT_SCHEMA)
    def count_items() -> dict:
        """Return an invalid count."""
        return {"count": "two"}

    registry = ToolRegistry()
    registry.register(count_items)

    result = registry.run("count_items", {})

    assert result.success is False
    assert result.failure_kind == "invalid_output"
    assert result.value is None
    assert result.metadata["validation_errors"][0]["path"] == "count"


def test_sync_and_async_timeouts_are_reported_per_attempt():
    @Tool(timeout_seconds=0.01)
    def slow_sync() -> str:
        """Run too slowly."""
        time.sleep(0.05)
        return "late"

    @Tool(timeout_seconds=0.01)
    async def slow_async() -> str:
        """Run too slowly asynchronously."""
        await asyncio.sleep(0.05)
        return "late"

    registry = ToolRegistry()
    registry.register(slow_sync)
    registry.register(slow_async)

    sync_result = registry.run("slow_sync", {})

    assert sync_result.failure_kind == "timeout"

    async def run_async():
        return await registry.run_async("slow_async", {})

    async_result = asyncio.run(run_async())
    assert async_result.failure_kind == "timeout"


def test_retry_success_rechecks_permission_and_records_attempts(tmp_path):
    calls = 0
    approvals = []

    @Tool(
        requires_confirmation=True,
        retry_policy=ToolRetryPolicy(max_attempts=3),
        idempotent=True,
    )
    def flaky() -> str:
        """Fail once, then succeed."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return "ok"

    facade = Agent(
        config=AgentConfig(project_root=tmp_path, permission_profile="workspace-write"),
        capabilities=Capabilities.none(),
        llm=FakeLLM([_tool_call("flaky"), _final()]),
        tools=[flaky],
        skills=[],
        permission_callback=lambda request, record: approvals.append(request.tool_name) or True,
    )

    result = facade.run_result("run flaky")

    assert calls == 2
    assert approvals == ["flaky", "flaky"]
    assert result.tool_calls[0].success is True
    assert len(result.tool_calls[0].attempts) == 2
    assert result.tool_calls[0].attempts[0].retry_scheduled is True
    assert result.tool_calls[0].attempts[1].success is True
    assert all(attempt.permission_decision == "allowed" for attempt in result.tool_calls[0].attempts)


def test_retry_exhaustion_is_bounded(tmp_path):
    calls = 0

    @Tool(retry_policy=ToolRetryPolicy(max_attempts=3), idempotent=True)
    def always_fails() -> str:
        """Always fail."""
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary")

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("always_fails"), _final()]),
        tools=[always_fails],
        skills=[],
    )

    result = facade.run_result("run")

    assert calls == 3
    assert len(result.tool_calls[0].attempts) == 3
    assert result.tool_calls[0].attempts[-1].retry_scheduled is False


def test_invalid_output_can_be_explicitly_retried(tmp_path):
    calls = 0

    @Tool(
        output_schema=OUTPUT_SCHEMA,
        retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_kinds=("invalid_output",)),
        idempotent=True,
    )
    def eventually_valid() -> dict:
        """Return valid output after one invalid attempt."""
        nonlocal calls
        calls += 1
        return {"count": "bad"} if calls == 1 else {"count": 1}

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("eventually_valid"), _final()]),
        tools=[eventually_valid],
        skills=[],
    )

    result = facade.run_result("run")

    assert calls == 2
    assert result.tool_calls[0].success is True
    assert result.tool_calls[0].attempts[0].failure_kind == "invalid_output"


def test_permission_denial_and_non_idempotent_tools_are_never_retried(tmp_path):
    denied_calls = 0

    @Tool(
        requires_confirmation=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, retryable_failure_kinds=("user_blocked",)),
        idempotent=True,
    )
    def denied() -> str:
        """Must not run."""
        nonlocal denied_calls
        denied_calls += 1
        return "unsafe"

    denied_agent = Agent(
        config=AgentConfig(project_root=tmp_path / "denied", permission_profile="workspace-write"),
        llm=FakeLLM([_tool_call("denied"), _final()]),
        tools=[denied],
        skills=[],
        permission_callback=lambda request, record: False,
    )
    denied_result = denied_agent.run_result("run")

    assert denied_calls == 0
    assert len(denied_result.tool_calls[0].attempts) == 1
    assert denied_result.tool_calls[0].attempts[0].permission_decision == "denied"

    write_calls = 0

    @Tool(retry_policy=ToolRetryPolicy(max_attempts=3))
    def non_idempotent_write() -> str:
        """Fail without retrying."""
        nonlocal write_calls
        write_calls += 1
        raise RuntimeError("failed")

    write_agent = Agent(
        config=AgentConfig(project_root=tmp_path / "write"),
        llm=FakeLLM([_tool_call("non_idempotent_write"), _final()]),
        tools=[non_idempotent_write],
        skills=[],
    )
    write_result = write_agent.run_result("run")

    assert write_calls == 1
    assert write_result.tool_calls[0].attempts[0].retry_disposition == "non_idempotent_guard"


@pytest.mark.asyncio
async def test_async_cancellation_is_not_converted_or_retried():
    started = asyncio.Event()

    @Tool(retry_policy=ToolRetryPolicy(max_attempts=3), idempotent=True)
    async def cancellable() -> str:
        """Wait until cancelled."""
        started.set()
        await asyncio.sleep(10)
        return "late"

    registry = ToolRegistry()
    registry.register(cancellable)
    task = asyncio.create_task(
        registry.run_async("cancellable", {}, context=ToolExecutionContext())
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_async_tool_cleanup_preserves_cancellation_and_aborts_goal():
    started = asyncio.Event()
    released = False
    aborted: list[BaseException] = []

    @Tool
    async def cancellable() -> str:
        """Wait until cancelled."""
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    class Usage:
        async def reserve_tool_call(self, **_kwargs):
            return SimpleNamespace(
                id="reservation-1",
                budget=SimpleNamespace(
                    scope=SimpleNamespace(value="turn"),
                ),
                reserved_tool_calls=1,
            )

        async def release_tool_call(self, **_kwargs):
            nonlocal released
            released = True
            raise RuntimeError("tool release failed")

    class Goal:
        def begin_tool(self, **_kwargs):
            return object()

        def finish_tool(self, _checkpoint, _result):
            raise AssertionError("cancelled tools cannot finish")

        def abort_tool(self, checkpoint, error):
            aborted.append(error)
            return checkpoint

    registry = ToolRegistry()
    registry.register(cancellable)
    executor = ToolExecutor(
        registry=registry,
        permission_policy=ToolPermissionPolicy(),
        permission_callback=None,
        trace=lambda _name, _payload=None: None,
        get_context=lambda _turn: None,
        goal_execution=Goal(),
        async_usage_accounting=Usage(),
    )
    task = asyncio.create_task(
        executor.execute_async("cancellable", {}, TurnState("cancel"))
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert released
    assert aborted == [error.value]
    assert any(
        "tool release failed" in note
        for note in getattr(error.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_async_durable_effect_cleanup_preserves_cancellation():
    started = asyncio.Event()
    failed_with: list[BaseException] = []

    @Tool
    async def cancellable() -> str:
        """Wait until cancelled."""
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    class DurableEffects:
        async def prepare_async(self, **_kwargs):
            return "effect-1"

        async def started_async(self, _token):
            return None

        async def failed_async(self, _token, error):
            failed_with.append(error)
            raise RuntimeError("effect quarantine failed")

    registry = ToolRegistry()
    registry.register(cancellable)
    executor = ToolExecutor(
        registry=registry,
        permission_policy=ToolPermissionPolicy(),
        permission_callback=None,
        trace=lambda _name, _payload=None: None,
        get_context=lambda _turn: ToolExecutionContext(),
        durable_effects=DurableEffects(),
    )
    task = asyncio.create_task(
        executor.execute_async("cancellable", {}, TurnState("cancel"))
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert failed_with == [error.value]
    assert any(
        "effect quarantine failed" in note
        for note in getattr(error.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_async_tool_flushes_authorization_before_dispatch():
    tool_calls = 0
    trace_events: list[str] = []

    @Tool
    async def side_effect() -> str:
        """Perform one external side effect."""
        nonlocal tool_calls
        tool_calls += 1
        return "done"

    async def reject_authorization_journal() -> None:
        assert "tool_permission_decided" in trace_events
        raise RuntimeError("authorization journal unavailable")

    registry = ToolRegistry()
    registry.register(side_effect)
    executor = ToolExecutor(
        registry=registry,
        permission_policy=ToolPermissionPolicy(),
        permission_callback=None,
        trace=lambda name, _payload=None: trace_events.append(name),
        get_context=lambda _turn: None,
        flush_async=reject_authorization_journal,
    )

    with pytest.raises(RuntimeError, match="authorization journal unavailable"):
        await executor.execute_async(
            "side_effect",
            {},
            TurnState("run side effect"),
        )

    assert tool_calls == 0
