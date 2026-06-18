"""Tests for MCP configuration and bridge tools."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chulk.mcp import MCPConfigError, MCPServerConfig, create_mcp_bridge_tools, load_mcp_servers
from chulk.tools import ToolPermissionLevel, ToolRegistry


def test_mcp_config_parses_streamable_http_server(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "label": "docs",
                        "transport": "streamable_http",
                        "server_url": "https://mcp.example.com",
                        "description": "Docs search",
                        "allowed_tools": ["search_docs"],
                        "authorization_env": "DOCS_MCP_TOKEN",
                        "approval": "always",
                        "defer_loading": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    servers = load_mcp_servers(config_path, {"DOCS_MCP_TOKEN": "secret-token"})

    assert len(servers) == 1
    server = servers[0]
    assert server.label == "docs"
    assert server.transport == "streamable_http"
    assert server.allowed_tools == ("search_docs",)
    assert server.authorization == "secret-token"
    assert server.to_dict()["authorization"] == "set"
    assert server.to_openai_tool() == {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://mcp.example.com",
        "require_approval": "always",
        "server_description": "Docs search",
        "allowed_tools": ["search_docs"],
        "authorization": "secret-token",
        "defer_loading": True,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"servers": [{"label": "docs", "server_url": "https://one.example"}, {"label": "docs", "server_url": "https://two.example"}]}, "Duplicate"),
        ({"servers": [{"label": "bad-label", "server_url": "https://mcp.example"}]}, "label"),
        ({"servers": [{"label": "docs", "transport": "stdio", "server_url": "https://mcp.example"}]}, "unsupported transport"),
        ({"servers": [{"label": "docs", "server_url": "file:///tmp/server"}]}, "http"),
        ({"servers": [{"label": "docs", "server_url": "https://mcp.example", "authorization_env": "MISSING"}]}, "MISSING"),
    ],
)
def test_mcp_config_validation_errors(tmp_path, payload, message):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MCPConfigError, match=message):
        load_mcp_servers(config_path, {})


def test_mcp_bridge_tools_discover_filter_and_execute():
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
        allowed_tools=("search_docs",),
    )
    client = FakeMCPClient(
        [
            {
                "name": "search_docs",
                "description": "Search docs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {"name": "blocked_tool", "inputSchema": {"type": "object", "properties": {}}},
        ]
    )

    tools = create_mcp_bridge_tools([server], client_factory=lambda _server: client)

    assert [tool.name for tool in tools] == ["mcp_docs_search_docs"]
    tool = tools[0]
    assert tool.requires_confirmation is True
    assert tool.permission_level == ToolPermissionLevel.EXTERNAL_SERVICE
    assert tool.metadata["mcp_bridge"] is True
    assert tool.metadata["server_url"] == "https://mcp.example.com"
    result = tool.callable({"query": "MCP"})
    assert result.success is True
    assert result.observation == "found MCP docs"
    assert client.calls == [("search_docs", {"query": "MCP"})]


def test_mcp_bridge_tool_registration_and_error_formatting():
    server = MCPServerConfig(label="docs", transport="streamable_http", server_url="https://mcp.example.com")
    client = FakeMCPClient(
        [{"name": "fail", "description": "Fails", "inputSchema": {"type": "object", "properties": {}}}],
        result=SimpleNamespace(isError=True, content=[SimpleNamespace(type="text", text="remote error")]),
    )
    registry = ToolRegistry()
    tool = create_mcp_bridge_tools([server], client_factory=lambda _server: client)[0]

    registry.register(tool)
    result = registry.run("mcp_docs_fail", {})

    assert result.success is False
    assert result.error == "mcp_tool_error"
    assert "remote error" in result.to_observation()
    assert result.metadata["mcp_bridge"] is True


class FakeMCPClient:
    def __init__(self, tools, result=None) -> None:
        self.tools = tools
        self.result = result or SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text="found MCP docs")],
        )
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return self.tools

    def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return self.result
