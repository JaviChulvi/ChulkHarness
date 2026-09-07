"""Lifecycle tests for public Agent facades."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager

import pytest

from chulk import (
    Agent,
    AgentConfig,
    AgentHandle,
    AsyncAgent,
    AsyncAgentHandle,
    ConfigurationError,
    ProviderError,
)
from chulk.core import TraceEvent
from chulk.media import UserInput
from chulk.llm import LLMClient, LLMError
from chulk.tracing import JSONLTraceLogger
import chulk.runtime as runtime_module
from tests.core_agent import build_core_agent as CoreAgent, create_runtime_agent


class FakeLLMClient(LLMClient):
    def __init__(self, content: str = "done") -> None:
        self.content = content
        self.close_count = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps({"type": "final_answer", "content": self.content})

    def close(self) -> None:
        self.close_count += 1


class FailingLLMClient(LLMClient):
    def complete(self, messages, *, max_output_tokens=None) -> str:
        raise LLMError(
            "provider unavailable",
            provider="failing",
            model="broken",
            code="server_error",
            retryable=True,
            fallback_eligible=True,
        )


class HangingAsyncLLMClient(LLMClient):
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started
        self.close_count = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        raise AssertionError("the sync provider path must not run")

    async def acomplete_action(self, messages, **kwargs):
        self.started.set()
        await asyncio.Future()
        raise AssertionError(f"unreachable: {messages!r} {kwargs!r}")

    async def aclose(self) -> None:
        self.close_count += 1


class AsyncCloseLLMClient(LLMClient):
    def __init__(self) -> None:
        self.close_count = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return "done"

    async def aclose(self) -> None:
        self.close_count += 1


def _agent(tmp_path, *, llm: FakeLLMClient | None = None) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm or FakeLLMClient(),
        tools=[],
        skills=[],
    )


def test_public_agent_close_is_idempotent_and_rejects_new_work(tmp_path):
    facade = _agent(tmp_path)

    facade.close()
    facade.close()

    assert facade.closed
    assert facade.runtime.closed
    for operation in (
        lambda: facade.run("hello"),
        lambda: facade.run_result("hello"),
        lambda: facade.plan("hello"),
        lambda: facade.plan_result("hello"),
        facade.approve,
        facade.approve_result,
        facade.reject,
        facade.reject_result,
    ):
        with pytest.raises(ConfigurationError, match="Agent is closed"):
            operation()


def test_sync_context_manager_closes_on_normal_and_exceptional_exit(tmp_path):
    normal = _agent(tmp_path / "normal")
    with normal as active:
        assert active is normal
        assert active.run("hello") == "done"
    assert normal.closed
    trace_events = [
        json.loads(line)
        for line in normal.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert trace_events[0]["type"] == "session_started"
    assert trace_events[-1]["type"] == "session_finished"

    exceptional = _agent(tmp_path / "exceptional")
    with pytest.raises(LookupError, match="boom"):
        with exceptional:
            raise LookupError("boom")
    assert exceptional.closed


@pytest.mark.asyncio
async def test_async_context_manager_closes_once_and_rejects_work(tmp_path):
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient("async done"),
        tools=[],
        skills=[],
    )

    async with facade as active:
        assert active is facade
        assert await active.run("hello") == "async done"

    await facade.close()
    assert facade.closed
    with pytest.raises(ConfigurationError, match="Agent is closed"):
        await facade.run("again")


def test_compatibility_handle_closes_owned_resource_once():
    resource = FakeLLMClient()
    handle = AgentHandle(CoreAgent(resource, owned_resources=[resource]))

    handle.close()
    handle.close()

    assert resource.close_count == 1


@pytest.mark.asyncio
async def test_async_compatibility_handle_awaits_owned_resource_once():
    resource = AsyncCloseLLMClient()
    handle = AsyncAgentHandle(
        AgentHandle(CoreAgent(resource, owned_resources=[resource]))
    )

    await handle.close()
    await handle.close()

    assert resource.close_count == 1


@pytest.mark.asyncio
async def test_async_cancellation_then_close_awaits_owned_provider(tmp_path):
    started = asyncio.Event()
    resource = HangingAsyncLLMClient(started)
    handle = AsyncAgentHandle(
        AgentHandle(CoreAgent(resource, owned_resources=[resource]))
    )
    task = asyncio.create_task(handle.run("wait"))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await handle.close()

    assert resource.close_count == 1


def test_runtime_finishes_trace_after_owned_resource_cleanup(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "cleanup-order")

    class TraceAwareResource(FakeLLMClient):
        def close(self) -> None:
            trace_logger.log("resource_closed")
            super().close()

    resource = TraceAwareResource()
    handle = AgentHandle(
        CoreAgent(
            resource,
            trace_logger=trace_logger,
            owned_resources=[resource],
        )
    )

    handle.close()

    events = [
        json.loads(line)
        for line in trace_logger.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events[-2:]] == ["resource_closed", "session_finished"]


def test_caller_injected_llm_is_not_owned_by_public_facade(tmp_path):
    client = FakeLLMClient()
    facade = _agent(tmp_path, llm=client)

    facade.close()

    assert client.close_count == 0


def test_runtime_construction_failure_closes_factory_owned_client(tmp_path, monkeypatch):
    client = FakeLLMClient()

    def fail_agent(*args, **kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(runtime_module, "Agent", fail_agent)
    config = AgentConfig(project_root=tmp_path).to_config()

    with pytest.raises(RuntimeError, match="construction failed"):
        create_runtime_agent(config, llm_client_factory=lambda _: client)

    assert client.close_count == 1


def test_runtime_construction_failure_closes_async_factory_transport(tmp_path, monkeypatch):
    client = AsyncCloseLLMClient()

    def fail_agent(*args, **kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(runtime_module, "Agent", fail_agent)
    config = AgentConfig(project_root=tmp_path).to_config()

    with pytest.raises(RuntimeError, match="construction failed"):
        create_runtime_agent(config, llm_client_factory=lambda _: client)

    assert client.close_count == 1


def test_provider_failure_terminalizes_and_persists_turn_before_reraising(tmp_path):
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FailingLLMClient(),
        tools=[],
        skills=[],
    )

    with pytest.raises(ProviderError, match="provider unavailable"):
        facade.run("fail this turn")

    turn = facade.state.turns[-1]
    assert turn.status == "failed"
    assert turn.ended_at is not None
    assert turn.errors == ["Turn failed with LLMError: provider unavailable"]
    restored = facade.runtime._components.session_store.load_turns(facade.conversation_id)[-1]
    assert restored.status == "failed"
    assert restored.ended_at == turn.ended_at
    trace_types = [
        json.loads(line)["type"]
        for line in facade.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert trace_types[-2:] == [TraceEvent.TURN_FAILED, TraceEvent.TURN_FINISHED]


def test_callback_failure_terminalizes_turn_and_preserves_original_exception():
    events: list[str] = []

    def callback(event_type: str, payload: dict) -> None:
        events.append(event_type)
        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            raise RuntimeError("observer failed")

    agent = CoreAgent(FakeLLMClient(), event_callback=callback)

    with pytest.raises(RuntimeError, match="observer failed"):
        agent.run_turn("trigger callback")

    turn = agent.state.turns[-1]
    assert turn.status == "failed"
    assert turn.ended_at is not None
    assert events[-2:] == [TraceEvent.TURN_FAILED, TraceEvent.TURN_FINISHED]


def test_exception_terminalization_applies_configured_redaction_before_retention():
    secret = "customer-447-private"
    redaction_calls: list[tuple[str, dict]] = []

    def redact(event_type: str, text: str, metadata: dict) -> str:
        redaction_calls.append((event_type, metadata))
        return text.replace(secret, "[tenant-redacted]")

    def fail_callback(event_type: str, payload: dict) -> None:
        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            raise RuntimeError(f"observer exposed {secret}")

    agent = CoreAgent(
        FakeLLMClient(),
        event_callback=fail_callback,
        redaction_callback=redact,
    )

    with pytest.raises(RuntimeError, match=secret):
        agent.run_turn("trigger redacted failure")

    retained = json.dumps(
        {
            "turn": agent.state.turns[-1].to_dict(),
            "state_errors": agent.state.errors,
            "state_final_answer": agent.state.final_answer,
            "messages": agent.memory.recent(),
        }
    )
    assert secret not in retained
    assert "[tenant-redacted]" in retained
    assert any(
        event_type == TraceEvent.TURN_FAILED and metadata.get("path") == "exception.message"
        for event_type, metadata in redaction_calls
    )


def test_event_callbacks_receive_mandatory_baseline_redaction():
    events: list[dict] = []
    agent = CoreAgent(
        FakeLLMClient(),
        event_callback=lambda _event_type, payload: events.append(payload),
    )

    agent._trace(
        TraceEvent.TOOL_CALL_STARTED,
        {
            "arguments": {
                "api_key": "sk-callback-secret-123456",
                "accesskey": "compact-access-secret",
                "privatekey": "compact-private-secret",
                "SSHKey": "compact-ssh-secret",
            },
            "observation": "access_key=AKIAIOSFODNN7EXAMPLE",
        },
    )

    serialized = json.dumps(events)
    assert "sk-callback-secret-123456" not in serialized
    assert "compact-access-secret" not in serialized
    assert "compact-private-secret" not in serialized
    assert "compact-ssh-secret" not in serialized
    assert "AKIAIOSFODNN7EXAMPLE" not in serialized
    assert serialized.count("[redacted]") == 5


def test_custom_event_redactor_cannot_reintroduce_a_secret():
    events: list[dict] = []

    def redact(_event_type: str, _text: str, _metadata: dict) -> str:
        return "private_key=callback-secret"

    agent = CoreAgent(
        FakeLLMClient(),
        event_callback=lambda _event_type, payload: events.append(payload),
        redaction_callback=redact,
    )

    agent._trace(TraceEvent.TOOL_OBSERVATION, {"observation": "ordinary text"})

    serialized = json.dumps(events)
    assert "callback-secret" not in serialized
    assert "[redacted]" in serialized


def test_exception_terminalization_honors_fail_closed_redaction():
    secret = "customer-992-private"

    def redact(_event_type: str, text: str, metadata: dict) -> str:
        if metadata.get("path") == "exception.message":
            raise RuntimeError("redactor unavailable")
        return text

    def fail_callback(event_type: str, payload: dict) -> None:
        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            raise RuntimeError(f"observer exposed {secret}")

    agent = CoreAgent(
        FakeLLMClient(),
        event_callback=fail_callback,
        redaction_callback=redact,
        redaction_fail_closed=True,
    )

    with pytest.raises(RuntimeError, match=secret):
        agent.run_turn("trigger fail-closed redaction")

    retained = json.dumps(
        {
            "turn": agent.state.turns[-1].to_dict(),
            "state_errors": agent.state.errors,
            "state_final_answer": agent.state.final_answer,
            "messages": agent.memory.recent(),
        }
    )
    assert secret not in retained
    assert "[redaction failed]" in retained


@pytest.mark.asyncio
async def test_async_cancellation_terminalizes_and_persists_turn(tmp_path):
    started = asyncio.Event()
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=HangingAsyncLLMClient(started),
        tools=[],
        skills=[],
    )
    task = asyncio.create_task(facade.run("cancel this turn"))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    turn = facade.state.turns[-1]
    assert turn.status == "cancelled"
    assert turn.ended_at is not None
    assert turn.errors == ["Turn cancelled."]
    restored = facade.runtime._components.session_store.load_turns(facade.conversation_id)[-1]
    assert restored.status == "cancelled"
    trace_types = [
        json.loads(line)["type"]
        for line in facade.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert trace_types[-2:] == [TraceEvent.TURN_FAILED, TraceEvent.TURN_FINISHED]


class CountingGate:
    def __init__(self):
        self.entries = 0

    @contextmanager
    def hold(self):
        self.entries += 1
        assert self.entries == 1, "facade acquired the execution gate twice"
        yield

    async def __aenter__(self):
        self.entries += 1
        assert self.entries == 1, "facade acquired the execution gate twice"

    async def __aexit__(self, *args):
        pass


_GATED_OPERATIONS = [
    ("run", "run_turn", ("hello",)),
    ("run_result", "run_turn", ("hello",)),
    ("run_input", "run_input", (UserInput.text("hello"),)),
    ("run_input_result", "run_input", (UserInput.text("hello"),)),
    ("plan", "run_planned_turn", ("hello",)),
    ("plan_result", "run_planned_turn", ("hello",)),
    ("approve", "approve_plan", ()),
    ("approve_result", "approve_plan", ()),
    ("reject", "reject_plan", ()),
    ("reject_result", "reject_plan", ()),
    ("close", "close", ()),
]


@pytest.mark.parametrize("method,runtime_method,args", _GATED_OPERATIONS)
@pytest.mark.parametrize("fails", [False, True])
def test_sync_facade_enters_gate_once_and_preserves_error_operation(
    tmp_path, monkeypatch, method, runtime_method, args, fails,
):
    facade = _agent(tmp_path)
    gate = CountingGate()
    facade._run_gate = gate

    def operation(*args, **kwargs):
        if fails:
            raise LLMError("injected failure")
        return "done"

    try:
        with monkeypatch.context() as patch:
            patch.setattr(facade.runtime, runtime_method, operation)
            if fails:
                with pytest.raises(ProviderError) as caught:
                    getattr(facade, method)(*args)
                assert caught.value.details.extensions["operation"] == method
            else:
                getattr(facade, method)(*args)
        assert gate.entries == 1
    finally:
        facade.runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,runtime_method,args", _GATED_OPERATIONS)
@pytest.mark.parametrize("fails", [False, True])
async def test_async_facade_enters_gate_once_and_preserves_error_operation(
    tmp_path, monkeypatch, method, runtime_method, args, fails,
):
    facade = AsyncAgent(config=AgentConfig(project_root=tmp_path), llm=FakeLLMClient(), tools=[], skills=[])
    gate = CountingGate()
    facade._async_run_gate = gate

    async def operation(*args, **kwargs):
        if fails:
            raise LLMError("injected failure")
        return "done"

    try:
        with monkeypatch.context() as patch:
            patch.setattr(facade.runtime, "aclose" if runtime_method == "close" else runtime_method + "_async", operation)
            if fails:
                with pytest.raises(ProviderError) as caught:
                    await getattr(facade, method)(*args)
                assert caught.value.details.extensions["operation"] == method
            else:
                await getattr(facade, method)(*args)
        assert gate.entries == 1
    finally:
        await facade.runtime.aclose()


def test_public_runtime_property_remains_read_only(tmp_path):
    facade = _agent(tmp_path)
    try:
        with pytest.raises(AttributeError):
            facade.runtime = facade.runtime
    finally:
        facade.close()


def test_compatibility_handle_keeps_raw_errors_and_sync_failed_close_state(monkeypatch):
    runtime = CoreAgent(FakeLLMClient())
    handle = AgentHandle(runtime)
    failure = RuntimeError("close failed")

    def fail():
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "close", fail)
        with pytest.raises(RuntimeError) as caught:
            handle.close()
        assert caught.value is failure
        assert handle.closed
        handle.close()
    runtime.close()


@pytest.mark.asyncio
async def test_compatibility_handle_retries_failed_async_close(monkeypatch):
    runtime = CoreAgent(FakeLLMClient())
    handle = AsyncAgentHandle(AgentHandle(runtime))
    failure = RuntimeError("close failed")

    async def fail():
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "aclose", fail)
        with pytest.raises(RuntimeError) as caught:
            await handle.close()
        assert caught.value is failure
        assert not handle.closed
    await handle.close()
    assert handle.closed
