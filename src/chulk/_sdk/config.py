"""Public SDK configuration values and builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from chulk.config import AgentConfig as AgentConfig, Config
from chulk.mcp import MCPServerConfig, build_mcp_server_config


@dataclass(frozen=True)
class AgentPreset:
    """Reusable collection of prompt, tools, skills, and default behavior."""

    system_prompt: str | None = None
    tools: tuple[object, ...] = field(default_factory=tuple)
    skills: tuple[object, ...] | None = None

    @classmethod
    def chat(cls, *, system_prompt: str | None = None) -> "AgentPreset":
        """Return a no-tool, no-skill preset for plain chat embedding."""
        return cls(system_prompt=system_prompt, tools=(), skills=())


class MCP:
    """Public MCP server builders."""

    @staticmethod
    def streamable_http(
        *,
        label: str,
        server_url: str,
        server_description: str = "",
        allowed_tools: Iterable[str] = (),
        authorization: str | None = None,
        authorization_env: str | None = None,
        approval: str = "always",
        defer_loading: bool = False,
    ) -> MCPServerConfig:
        return build_mcp_server_config(
            label=label,
            server_url=server_url,
            server_description=server_description,
            allowed_tools=allowed_tools,
            authorization=authorization,
            authorization_env=authorization_env,
            approval=approval,
            defer_loading=defer_loading,
        )


def coerce_config(config: Config | AgentConfig | None) -> Config:
    if config is None:
        return AgentConfig.from_env().to_config()
    if isinstance(config, AgentConfig):
        return config.to_config()
    return config


def ensure_chat_kwargs(kwargs: dict[str, Any]) -> None:
    disallowed = [name for name in ("preset", "tools", "skills") if name in kwargs and kwargs[name] is not None]
    if disallowed:
        joined = ", ".join(disallowed)
        raise ValueError(f"ChatAgent does not accept {joined}; use Agent for configured tools or skills")


__all__ = ["AgentConfig", "AgentPreset", "MCP"]
