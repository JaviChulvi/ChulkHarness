"""ChulkHarness package and public API."""

__version__ = "0.1.0"

from chulk import skills as Skills
from chulk import tools as Tools
from chulk.api import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentHandle,
    AgentPreset,
    AsyncAgent,
    AsyncChatAgent,
    AsyncAgentHandle,
    ChatAgent,
    MCP,
    PlanResult,
    PlanSnapshot,
    RunResult,
    agent,
    async_agent,
    async_chat_agent,
    chat_agent,
)
from chulk.tools import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
    tool,
)

Tool = tool
skills = Skills
tools = Tools

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "AgentHandle",
    "AgentPreset",
    "AsyncAgent",
    "AsyncAgentHandle",
    "AsyncChatAgent",
    "ChatAgent",
    "MCP",
    "PermissionDecision",
    "PermissionDecisionRecord",
    "PermissionRequest",
    "PlanResult",
    "PlanSnapshot",
    "RunResult",
    "Skills",
    "Tool",
    "ToolPermissionLevel",
    "Tools",
    "__version__",
    "agent",
    "async_agent",
    "async_chat_agent",
    "chat_agent",
    "skills",
    "tool",
    "tools",
]
