"""MCP routing tests across runtime assembly and model dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from chulk import runtime as runtime_module
from chulk.cli import terminal as terminal_module
from chulk.config import load_config
from chulk.core.actions import FinalAnswerAction, PlanAction
from chulk.core.state import Plan, PlanStep
from chulk.llm import LLMActionResult, LLMCapabilities, LLMClient
from chulk.mcp import MCPServerConfig
from chulk.tools import Tool, ToolPermissionLevel, ToolResult


class RecordingNativeBridgeClient(LLMClient):
    capabilities = LLMCapabilities(
        supports_native_tool_calling=True,
        supports_hosted_mcp_tools=False,
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete_action(
        self,
        messages,
        *,
        tools=None,
        hosted_mcp_servers=None,
        mcp_approval_callback=None,
        **_kwargs,
    ) -> LLMActionResult:
        self._record(
            messages,
            tools=tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        return _final_result()

    async def acomplete_action(
        self,
        messages,
        *,
        tools=None,
        hosted_mcp_servers=None,
        mcp_approval_callback=None,
        **_kwargs,
    ) -> LLMActionResult:
        self._record(
            messages,
            tools=tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        return _final_result()

    def _record(
        self,
        messages,
        *,
        tools,
        hosted_mcp_servers,
        mcp_approval_callback,
    ) -> None:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "hosted_mcp_servers": hosted_mcp_servers,
                "mcp_approval_callback": mcp_approval_callback,
            }
        )


class RecordingJSONBridgeClient(RecordingNativeBridgeClient):
    capabilities = LLMCapabilities(
        supports_native_tool_calling=False,
        supports_hosted_mcp_tools=False,
    )


class RecordingHostedPlanningClient(RecordingNativeBridgeClient):
    capabilities = LLMCapabilities(
        supports_native_tool_calling=True,
        supports_hosted_mcp_tools=True,
    )

    def complete_action(
        self,
        messages,
        *,
        tools=None,
        hosted_mcp_servers=None,
        mcp_approval_callback=None,
        **_kwargs,
    ) -> LLMActionResult:
        self._record(
            messages,
            tools=tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        return LLMActionResult(
            action=PlanAction(
                type="plan",
                plan=Plan(
                    summary="Make the requested change.",
                    steps=[
                        PlanStep(
                            id="1",
                            title="Make the change",
                            description="Complete the requested change.",
                        )
                    ],
                ),
            ),
            raw_response='{"type":"plan"}',
        )


class TupleHostedClient(RecordingNativeBridgeClient):
    def __init__(self) -> None:
        super().__init__()
        self.providers = (RecordingHostedPlanningClient(),)


def test_non_hosted_native_client_uses_bridge_without_hosted_options_sync(
    monkeypatch,
    tmp_path,
):
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", _bridge_tools)
    client = RecordingNativeBridgeClient()
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=client,
        tool_specs=[],
        mcp_servers=(server,),
    )

    assert agent.run_turn("search the docs") == "done"

    assert agent.mcp_bridge_tool_names == ["mcp_docs_search_docs"]
    assert [tool.name for tool in client.calls[0]["tools"]] == ["mcp_docs_search_docs"]
    assert client.calls[0]["hosted_mcp_servers"] is None
    assert client.calls[0]["mcp_approval_callback"] is None


def test_non_hosted_native_client_uses_bridge_without_hosted_options_async(
    monkeypatch,
    tmp_path,
):
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", _bridge_tools)
    client = RecordingNativeBridgeClient()
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=client,
        tool_specs=[],
        mcp_servers=(server,),
    )

    assert asyncio.run(agent.run_turn_async("search the docs")) == "done"

    assert agent.mcp_bridge_tool_names == ["mcp_docs_search_docs"]
    assert [tool.name for tool in client.calls[0]["tools"]] == ["mcp_docs_search_docs"]
    assert client.calls[0]["hosted_mcp_servers"] is None
    assert client.calls[0]["mcp_approval_callback"] is None


def test_single_injected_json_client_uses_its_capabilities_for_mcp_route(
    monkeypatch,
    tmp_path,
):
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", _bridge_tools)
    client = RecordingJSONBridgeClient()
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=client,
        tool_specs=[],
        mcp_servers=(server,),
    )

    assert agent.run_turn("search the docs") == "done"

    assert runtime_module._mcp_provider_path(
        config,
        (server,),
        llm_client=client,
    ) == "bridge"
    assert agent.mcp_bridge_tool_names == ["mcp_docs_search_docs"]
    assert client.calls[0]["tools"] is None
    assert client.calls[0]["hosted_mcp_servers"] is None
    assert client.calls[0]["mcp_approval_callback"] is None
    assert "<name>mcp_docs_search_docs</name>" in client.calls[0]["messages"][0]["content"]


def test_cli_reports_the_same_effective_route_as_runtime(monkeypatch, tmp_path):
    config = replace(_config(monkeypatch, tmp_path), mcp_servers=(_server(),))
    client = RecordingNativeBridgeClient()

    runtime_path = runtime_module._mcp_provider_path(
        config,
        config.mcp_servers,
        llm_client=client,
    )

    assert runtime_path == "bridge"
    assert terminal_module._mcp_provider_path(config, llm_client=client) == runtime_path


def test_hosted_mcp_is_not_exposed_before_plan_approval(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    client = RecordingHostedPlanningClient()
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=client,
        tool_specs=[],
        mcp_servers=(server,),
    )

    response = agent.run_planned_turn("plan a remote change")

    assert "Use /approve" in response
    assert client.calls[0]["tools"] == []
    assert client.calls[0]["hosted_mcp_servers"] is None
    assert client.calls[0]["mcp_approval_callback"] is None
    native_tools_section = next(
        section
        for section in agent.state.last_context_report["sections"]
        if section["name"] == "native_tools"
    )
    assert native_tools_section["metadata"]["tool_names"] == [
        "chulk_propose_plan"
    ]


def test_tuple_aggregate_uses_the_same_hosted_route_at_runtime_and_dispatch(
    monkeypatch,
    tmp_path,
):
    config = _config(monkeypatch, tmp_path)
    client = TupleHostedClient()
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=client,
        tool_specs=[],
        mcp_servers=(server,),
    )

    assert runtime_module.resolve_mcp_route(
        config,
        (server,),
        llm_client=client,
    ).provider_path == "hosted"
    assert agent.run_turn("search remotely") == "done"
    assert client.calls[0]["tools"] == []
    assert client.calls[0]["hosted_mcp_servers"] == (server,)


def test_replacing_hosted_client_with_bridge_client_fails_closed(
    monkeypatch,
    tmp_path,
):
    config = _config(monkeypatch, tmp_path)
    server = _server()
    agent = runtime_module.create_agent(
        config,
        llm_client=RecordingHostedPlanningClient(),
        tool_specs=[],
        mcp_servers=(server,),
    )
    replacement = RecordingJSONBridgeClient()
    agent.llm_client = replacement

    try:
        agent.run_turn("search remotely")
    except RuntimeError as exc:
        assert "Rebuild the agent for the replacement client" in str(exc)
    else:
        raise AssertionError("Expected an incompatible MCP client replacement to fail")

    assert replacement.calls == []


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    return load_config()


def _server() -> MCPServerConfig:
    return MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )


def _bridge_tools(_servers) -> list[Tool]:
    return [
        Tool(
            name="mcp_docs_search_docs",
            description="Search the documentation.",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            callable=lambda _arguments: ToolResult(
                tool_name="mcp_docs_search_docs",
                success=True,
                observation="found",
            ),
            requires_confirmation=True,
            permission_level=ToolPermissionLevel.EXTERNAL_SERVICE,
            metadata={"mcp_bridge": True},
        )
    ]


def _final_result() -> LLMActionResult:
    return LLMActionResult(
        action=FinalAnswerAction(type="final_answer", content="done"),
        raw_response='{"type":"final_answer","content":"done"}',
    )
