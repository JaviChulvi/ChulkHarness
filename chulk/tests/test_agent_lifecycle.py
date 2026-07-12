"""Lifecycle tests for public Agent facades."""

from __future__ import annotations

import json

import pytest

from chulk import Agent, AgentConfig, AgentHandle, AsyncAgent, ConfigurationError
from chulk.core import Agent as CoreAgent
from chulk.llm import LLMClient
from chulk.tracing import JSONLTraceLogger
import chulk.runtime as runtime_module


class FakeLLMClient(LLMClient):
    def __init__(self, content: str = "done") -> None:
        self.content = content
        self.close_count = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps({"type": "final_answer", "content": self.content})

    def close(self) -> None:
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
        runtime_module.create_agent(config, llm_client_factory=lambda _: client)

    assert client.close_count == 1
