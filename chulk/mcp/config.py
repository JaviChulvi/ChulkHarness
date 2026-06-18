"""MCP server configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse


LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SUPPORTED_TRANSPORTS = {"streamable_http"}
SUPPORTED_APPROVALS = {"always", "never"}


class MCPConfigError(ValueError):
    """Raised when MCP configuration is invalid."""


@dataclass(frozen=True)
class MCPServerConfig:
    """One configured remote MCP server."""

    label: str
    transport: str
    server_url: str
    server_description: str = ""
    allowed_tools: tuple[str, ...] = ()
    authorization_env: str | None = None
    authorization: str | None = field(default=None, repr=False)
    approval: str = "always"
    defer_loading: bool = False

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        return {
            "label": self.label,
            "transport": self.transport,
            "server_url": self.server_url,
            "server_description": self.server_description,
            "allowed_tools": list(self.allowed_tools),
            "authorization_env": self.authorization_env,
            "authorization": "set" if redact and self.authorization else self.authorization,
            "approval": self.approval,
            "defer_loading": self.defer_loading,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.label,
            "server_url": self.server_url,
            "require_approval": self.approval,
        }
        if self.server_description:
            tool["server_description"] = self.server_description
        if self.allowed_tools:
            tool["allowed_tools"] = list(self.allowed_tools)
        if self.authorization:
            tool["authorization"] = self.authorization
        if self.defer_loading:
            tool["defer_loading"] = True
        return tool


def load_mcp_servers(path: Path, env: Mapping[str, str] | None = None) -> tuple[MCPServerConfig, ...]:
    """Load MCP server definitions from a JSON file."""
    if not path.exists():
        return ()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"MCP config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MCPConfigError("MCP config must be a JSON object")

    raw_servers = payload.get("servers", [])
    if not isinstance(raw_servers, list):
        raise MCPConfigError("MCP config field 'servers' must be a list")

    servers: list[MCPServerConfig] = []
    labels: set[str] = set()
    env_values = env or {}
    for index, raw_server in enumerate(raw_servers, start=1):
        if not isinstance(raw_server, dict):
            raise MCPConfigError(f"MCP server #{index} must be an object")
        server = _parse_server(raw_server, env_values, index=index)
        if server.label in labels:
            raise MCPConfigError(f"Duplicate MCP server label: {server.label}")
        labels.add(server.label)
        servers.append(server)
    return tuple(servers)


def _parse_server(raw: dict[str, Any], env: Mapping[str, str], *, index: int) -> MCPServerConfig:
    label = _string_field(raw, "label", index=index)
    if not LABEL_RE.fullmatch(label):
        raise MCPConfigError(
            f"MCP server #{index} label must start with a letter and contain only letters, numbers, and underscores"
        )

    transport = str(raw.get("transport") or "streamable_http").strip()
    if transport not in SUPPORTED_TRANSPORTS:
        supported = ", ".join(sorted(SUPPORTED_TRANSPORTS))
        raise MCPConfigError(f"MCP server {label} uses unsupported transport {transport!r}; supported: {supported}")

    server_url = _string_field(raw, "server_url", index=index)
    parsed_url = urlparse(server_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise MCPConfigError(f"MCP server {label} server_url must be an http(s) URL")

    allowed_tools = raw.get("allowed_tools", [])
    if allowed_tools is None:
        allowed_tools = []
    if not isinstance(allowed_tools, list) or not all(isinstance(item, str) and item.strip() for item in allowed_tools):
        raise MCPConfigError(f"MCP server {label} allowed_tools must be a list of strings")

    approval = str(raw.get("approval") or "always").strip().lower()
    if approval not in SUPPORTED_APPROVALS:
        supported = ", ".join(sorted(SUPPORTED_APPROVALS))
        raise MCPConfigError(f"MCP server {label} approval must be one of: {supported}")

    authorization_env = raw.get("authorization_env")
    authorization = None
    if authorization_env is not None:
        if not isinstance(authorization_env, str) or not authorization_env.strip():
            raise MCPConfigError(f"MCP server {label} authorization_env must be a non-empty string")
        authorization_env = authorization_env.strip()
        authorization = env.get(authorization_env)
        if not authorization:
            raise MCPConfigError(f"MCP server {label} requires missing environment variable {authorization_env}")

    server_description = raw.get("server_description") or raw.get("description") or ""
    if not isinstance(server_description, str):
        raise MCPConfigError(f"MCP server {label} server_description must be a string")
    defer_loading = raw.get("defer_loading", False)
    if not isinstance(defer_loading, bool):
        raise MCPConfigError(f"MCP server {label} defer_loading must be true or false")

    return MCPServerConfig(
        label=label,
        transport=transport,
        server_url=server_url,
        server_description=server_description.strip(),
        allowed_tools=tuple(item.strip() for item in allowed_tools),
        authorization_env=authorization_env,
        authorization=authorization,
        approval=approval,
        defer_loading=defer_loading,
    )


def _string_field(raw: dict[str, Any], field_name: str, *, index: int) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MCPConfigError(f"MCP server #{index} field {field_name!r} must be a non-empty string")
    return value.strip()
