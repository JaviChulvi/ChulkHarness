"""Tests for explicit SDK capability selection and permission layering."""

from __future__ import annotations

import json

import pytest

from chulk import Agent, AgentConfig, Capabilities, MCP, Tool
from chulk.llm import LLMCapabilities, LLMClient


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [json.dumps({"type": "final_answer", "content": "done"})]

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _names(tmp_path, capabilities: Capabilities) -> set[str]:
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=capabilities,
        llm=FakeLLM(),
        skills=[],
    )
    return {tool.name for tool in facade.tool_registry.list_tools()}


def test_capability_matrix_advertises_only_enabled_default_tools(tmp_path):
    assert _names(tmp_path / "none", Capabilities.none()) == set()
    assert _names(
        tmp_path / "files",
        Capabilities(files="read", memory="off", utilities=False),
    ) == {"read_file", "list_files", "search_files"}
    assert _names(
        tmp_path / "write",
        Capabilities(files="write", memory="off", utilities=False),
    ) == {"read_file", "list_files", "search_files", "write_file", "apply_patch"}
    assert _names(
        tmp_path / "shell",
        Capabilities(files="off", shell=True, memory="off", utilities=False),
    ) == {"run_cmd"}
    assert _names(
        tmp_path / "memory-read",
        Capabilities(files="off", memory="read-only", utilities=False),
    ) == {"search_memory", "list_memories", "summarize_memories"}
    assert _names(
        tmp_path / "memory-manual",
        Capabilities(files="off", memory="manual", utilities=False),
    ) == {"save_memory", "search_memory", "list_memories", "summarize_memories"}


def test_safe_sdk_default_is_read_oriented(tmp_path):
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM(),
        skills=[],
    )
    names = {tool.name for tool in facade.tool_registry.list_tools()}

    assert facade.capabilities == Capabilities.read_only()
    assert {"read_file", "list_files", "search_files"} <= names
    assert {"run_cmd", "write_file", "apply_patch", "save_memory"}.isdisjoint(names)


def test_capabilities_do_not_bypass_permission_policy(tmp_path):
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "run_cmd",
                    "arguments_json": json.dumps({"command": "printf unsafe"}),
                }
            ),
            json.dumps({"type": "final_answer", "content": "blocked"}),
        ]
    )
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=Capabilities(files="off", shell=True, memory="off"),
        llm=llm,
        skills=[],
    )

    result = facade.run_result("run a command")

    assert "run_cmd" in {tool.name for tool in facade.tool_registry.list_tools()}
    assert result.tool_calls[0].success is False
    assert result.tool_calls[0].failure_kind == "user_blocked"
    decision = facade.runtime.permission_policy.decide(
        facade.runtime.permission_policy.request_for_tool(
            facade.tool_registry.get("run_cmd"),
            {"command": "printf unsafe"},
        )
    )
    assert decision.to_dict()["capability"] == {"category": "shell", "enabled": True}


def test_explicit_custom_tools_remain_authoritative(tmp_path):
    @Tool
    def custom_lookup(value: str) -> str:
        """Look up a value."""
        return value

    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=Capabilities.none(),
        llm=FakeLLM(),
        tools=[custom_lookup],
        skills=[],
    )

    assert [tool.name for tool in facade.tool_registry.list_tools()] == ["custom_lookup"]


def test_external_services_capability_controls_mcp_visibility(tmp_path):
    class HostedMCPFakeLLM(FakeLLM):
        capabilities = LLMCapabilities(
            supports_native_tool_calling=True,
            supports_hosted_mcp_tools=True,
        )

    server = MCP.streamable_http(label="docs", server_url="https://mcp.example.com")
    disabled = Agent(
        config=AgentConfig(project_root=tmp_path / "disabled"),
        capabilities=Capabilities.none(),
        llm=FakeLLM(),
        tools=[],
        skills=[],
        mcp=[server],
    )
    enabled = Agent(
        config=AgentConfig(project_root=tmp_path / "enabled"),
        capabilities=Capabilities(files="off", memory="off", external_services=True),
        llm=HostedMCPFakeLLM(),
        tools=[],
        skills=[],
        mcp=[server],
    )

    assert disabled.runtime.mcp_servers == ()
    assert enabled.runtime.mcp_servers == (server,)


@pytest.mark.parametrize("field,value", [("files", "maybe"), ("memory", "forever")])
def test_invalid_capability_values_fail_early(field, value):
    with pytest.raises(ValueError):
        Capabilities(**{field: value})
