"""Tool registry construction for runtime assembly."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chulk.capabilities import Capabilities
from chulk.config import Config
from chulk.llm import LLMClient
from chulk.mcp import MCPServerConfig
from chulk.memory import MemoryPolicy, SQLiteMemoryStore
from chulk.sessions import SessionSearchService
from chulk.tools import (
    ShellExecutionPolicy,
    Tool,
    ToolRegistry,
    create_default_tool_registry,
)
from chulk.tracing.artifacts import TraceArtifactStore


class MCPBridgeRequired(Protocol):
    """Route decision seam supplied by the public runtime facade."""

    def __call__(
        self,
        config: Config,
        mcp_servers: Iterable[object],
        *,
        llm_client: LLMClient | None = None,
    ) -> bool: ...


@dataclass(frozen=True)
class RuntimeToolContext:
    """Context required to bind project-scoped tool references."""

    project_root: Path
    shell_timeout_seconds: int
    max_tool_stdout_bytes: int
    max_tool_stderr_bytes: int
    shell_execution_policy: ShellExecutionPolicy | None = None
    require_shell_containment: bool = False
    memory_store: SQLiteMemoryStore | None = None
    session_search_service: SessionSearchService | None = None
    artifact_store: TraceArtifactStore | None = None
    deps: object | None = None


def create_tool_registry(
    config: Config,
    memory_store: SQLiteMemoryStore,
    tool_specs: Iterable[object] | None,
    mcp_servers: Iterable[MCPServerConfig],
    *,
    llm_client: LLMClient | None = None,
    capabilities: Capabilities,
    memory_policy: MemoryPolicy,
    session_search_service: SessionSearchService,
    deps: object | None,
    shell_execution_policy: ShellExecutionPolicy | None,
    require_shell_containment: bool,
    artifact_store: TraceArtifactStore,
    bridge_tool_factory: Callable[[Iterable[MCPServerConfig]], Iterable[Tool]],
    mcp_bridge_required: MCPBridgeRequired,
) -> tuple[ToolRegistry, list[str]]:
    """Build explicit or default tools and add any required MCP bridge tools."""
    if tool_specs is None:
        registry = create_default_tool_registry(
            config.project_root,
            config.shell_timeout_seconds,
            memory_store=memory_store,
            capabilities=capabilities,
            memory_policy=memory_policy,
            session_search_service=session_search_service,
            max_tool_stdout_bytes=config.max_tool_stdout_chars,
            max_tool_stderr_bytes=config.max_tool_stderr_chars,
            shell_execution_policy=shell_execution_policy,
            require_shell_containment=require_shell_containment,
        )
        return _register_mcp_bridge_tools(
            config,
            registry,
            mcp_servers,
            llm_client=llm_client,
            bridge_tool_factory=bridge_tool_factory,
            mcp_bridge_required=mcp_bridge_required,
        )

    context = RuntimeToolContext(
        project_root=config.project_root,
        shell_timeout_seconds=config.shell_timeout_seconds,
        max_tool_stdout_bytes=config.max_tool_stdout_chars,
        max_tool_stderr_bytes=config.max_tool_stderr_chars,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        memory_store=memory_store,
        session_search_service=session_search_service,
        artifact_store=artifact_store,
        deps=deps,
    )
    registry = ToolRegistry()
    for spec in tool_specs:
        registry.register(_resolve_tool_spec(spec, context))
    return _register_mcp_bridge_tools(
        config,
        registry,
        mcp_servers,
        llm_client=llm_client,
        bridge_tool_factory=bridge_tool_factory,
        mcp_bridge_required=mcp_bridge_required,
    )


def _register_mcp_bridge_tools(
    config: Config,
    registry: ToolRegistry,
    mcp_servers: Iterable[MCPServerConfig],
    *,
    llm_client: LLMClient | None,
    bridge_tool_factory: Callable[[Iterable[MCPServerConfig]], Iterable[Tool]],
    mcp_bridge_required: MCPBridgeRequired,
) -> tuple[ToolRegistry, list[str]]:
    servers = tuple(mcp_servers)
    if not servers or not mcp_bridge_required(
        config,
        servers,
        llm_client=llm_client,
    ):
        return registry, []
    bridge_tool_names: list[str] = []
    for tool in bridge_tool_factory(servers):
        registry.register(tool)
        bridge_tool_names.append(tool.name)
    return registry, bridge_tool_names


def _resolve_tool_spec(spec: object, context: RuntimeToolContext) -> Tool:
    if isinstance(spec, Tool):
        return spec
    if hasattr(spec, "to_tool"):
        return spec.to_tool(context)  # type: ignore[no-any-return, attr-defined]
    raise TypeError(f"Unsupported tool spec: {spec!r}")
