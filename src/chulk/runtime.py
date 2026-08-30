"""Compatibility entrypoint for configured Chulk agent runtimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from chulk._runtime.assembly import (
    assemble_agent,
    assemble_async_hosted_agent,
)
from chulk._runtime.request import AgentAssemblyRequest
from chulk._runtime.skills import SkillSpecResolution as SkillSpecResolution
from chulk._runtime.tools import RuntimeToolContext as RuntimeToolContext
from chulk._version import __version__ as __version__
from chulk.config import Config
from chulk.core import Agent
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.llm import LLMClient, provider_capabilities
from chulk.llm.capabilities import (
    client_requires_mcp_bridge,
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
)
from chulk.mcp import create_mcp_bridge_tools


class LLMClientFactory(Protocol):
    """Factory used by tests and the CLI to inject an LLM client."""

    def __call__(self, config: Config) -> LLMClient:
        """Return an LLM client for the given runtime config."""


@dataclass(frozen=True)
class MCPRoute:
    """Effective MCP transport across the clients that may handle a request."""

    provider_path: str
    bridge_required: bool


def create_agent(request: AgentAssemblyRequest) -> Agent:
    """Create one configured runtime from normalized internal input."""
    return assemble_agent(
        request,
        agent_factory=Agent,
        bridge_tool_factory=create_mcp_bridge_tools,
        mcp_bridge_required=_mcp_bridge_required,
        mcp_provider_path=_mcp_provider_path,
    )


async def create_async_hosted_agent(
    request: AgentAssemblyRequest,
) -> tuple[Agent, ResolvedRuntimeServices]:
    """Create a native-async hosted runtime from normalized internal input."""
    return await assemble_async_hosted_agent(
        request,
        agent_factory=Agent,
        bridge_tool_factory=create_mcp_bridge_tools,
        mcp_bridge_required=_mcp_bridge_required,
    )


def _mcp_bridge_required(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> bool:
    return resolve_mcp_route(
        config,
        mcp_servers,
        llm_client=llm_client,
    ).bridge_required


def _mcp_provider_path(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> str:
    return resolve_mcp_route(
        config,
        mcp_servers,
        llm_client=llm_client,
    ).provider_path


def resolve_mcp_route(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> MCPRoute:
    """Resolve one MCP route from the effective bound client path when available."""
    if not tuple(mcp_servers):
        return MCPRoute(provider_path="none", bridge_required=False)

    if llm_client is not None:
        native_protocol = client_supports_native_tool_calling(llm_client)
        has_hosted = client_supports_hosted_mcp_tools(llm_client)
        has_bridge = client_requires_mcp_bridge(llm_client)
    else:
        provider_names = [
            config.llm_provider,
            *(provider.provider for provider in config.llm_fallback_providers),
        ]
        native_support = [
            _supports_native_tool_calling(provider) for provider in provider_names
        ]
        hosted_support = [_supports_hosted_mcp(provider) for provider in provider_names]
        native_protocol = all(native_support)
        has_hosted = any(hosted_support)
        has_bridge = any(not item for item in hosted_support)

    if not native_protocol:
        return MCPRoute(provider_path="bridge", bridge_required=True)

    if has_hosted and has_bridge:
        route_path = "hosted+bridge"
    else:
        route_path = "hosted" if has_hosted else "bridge"
    return MCPRoute(provider_path=route_path, bridge_required=has_bridge)


def _supports_hosted_mcp(provider: str) -> bool:
    return provider_capabilities(provider).supports_hosted_mcp_tools


def _supports_native_tool_calling(provider: str) -> bool:
    return provider_capabilities(provider).supports_native_tool_calling
