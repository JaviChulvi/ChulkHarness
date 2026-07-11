"""Tests for host-only typed tool dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from chulk import Agent, AgentConfig, AsyncAgent, Capabilities, ProviderError, Tool, ToolContext
from chulk.llm import LLMClient, LLMError
from chulk.tools import ToolRegistry


@dataclass(frozen=True)
class Dependencies:
    tenant: str


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _tool_call(name: str, **arguments) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": json.dumps(arguments),
        }
    )


def _final(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def test_typed_dependencies_are_hidden_from_schema_and_injected(tmp_path):
    received = []

    @Tool
    def tenant_lookup(query: str, context: ToolContext[Dependencies]) -> str:
        """Look up data for the current tenant."""
        received.append((query, context.deps.tenant))
        return f"{context.deps.tenant}:{query}"

    assert set(tenant_lookup.args_schema["properties"]) == {"query"}
    assert tenant_lookup.accepts_context is True
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=Capabilities.none(),
        deps=Dependencies("tenant-a"),
        llm=FakeLLM([_tool_call("tenant_lookup", query="invoices"), _final()]),
        tools=[tenant_lookup],
        skills=[],
    )

    result = facade.run_result("look up invoices")

    assert result.tool_calls[0].success is True
    assert received == [("invoices", "tenant-a")]


def test_run_dependencies_override_agent_dependencies(tmp_path):
    received = []

    @Tool
    def tenant_lookup(context: ToolContext[Dependencies]) -> str:
        """Return the current tenant."""
        received.append(context.deps.tenant)
        return context.deps.tenant

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        deps=Dependencies("default"),
        llm=FakeLLM([_tool_call("tenant_lookup"), _final()]),
        tools=[tenant_lookup],
        skills=[],
    )

    facade.run("identify tenant", deps=Dependencies("override"))

    assert received == ["override"]
    assert facade.runtime._tool_contexts == {}


def test_request_dependencies_are_released_when_a_turn_raises(tmp_path):
    class FailingLLM(LLMClient):
        def complete(self, messages, *, max_output_tokens=None) -> str:
            raise LLMError("provider failed")

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FailingLLM(),
        tools=[],
        skills=[],
    )

    with pytest.raises(ProviderError):
        facade.run("fail", deps=Dependencies("sensitive"))

    assert facade.runtime._tool_contexts == {}


def test_missing_dependencies_fail_before_side_effects(tmp_path):
    calls = []

    @Tool
    def guarded_write(value: str, context: ToolContext[Dependencies]) -> str:
        """Perform a guarded write."""
        calls.append((value, context.deps.tenant))
        return "written"

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("guarded_write", value="x"), _final()]),
        tools=[guarded_write],
        skills=[],
    )

    result = facade.run_result("write")

    assert calls == []
    assert result.tool_calls[0].success is False
    assert "dependencies" in (result.tool_calls[0].error or "").lower()


def test_model_cannot_supply_injected_context_argument():
    @Tool
    def guarded_lookup(value: str, context: ToolContext[Dependencies]) -> str:
        """Look up safely."""
        return f"{context.deps.tenant}:{value}"

    registry = ToolRegistry()
    registry.register(guarded_lookup)

    result = registry.run(
        "guarded_lookup",
        {"value": "x", "context": {"deps": {"tenant": "attacker"}}},
        context=ToolContext(deps=Dependencies("trusted")),
    )

    assert result.success is False
    assert result.failure_kind == "invalid_arguments"


@pytest.mark.asyncio
async def test_async_tools_receive_typed_dependencies(tmp_path):
    received = []

    @Tool
    async def tenant_lookup(query: str, context: ToolContext[Dependencies]) -> str:
        """Look up asynchronously."""
        received.append(context.deps.tenant)
        return query

    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        deps=Dependencies("async-tenant"),
        llm=FakeLLM([_tool_call("tenant_lookup", query="docs"), _final()]),
        tools=[tenant_lookup],
        skills=[],
    )

    result = await facade.run_result("look up docs")

    assert result.tool_calls[0].success is True
    assert received == ["async-tenant"]
    assert facade.runtime._tool_contexts == {}
