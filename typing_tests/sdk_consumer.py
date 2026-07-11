"""External-style static typing fixture for the installed SDK contract."""

from pathlib import Path

from chulk import (
    Agent,
    AgentConfig,
    ChulkError,
    ConfigurationError,
    MemoryError,
    PermissionDeniedError,
    ProviderError,
    SafetyError,
    Skills,
    Tool,
    ToolExecutionError,
    Tools,
    TraceError,
)


assert Tool is not None

config: AgentConfig = AgentConfig.local(
    project_root=Path.cwd(),
    runtime_dir=".chulk",
    permission_profile="read-only",
)

agent = Agent(
    config=config,
    tools=[Tools.calculator],
    skills=[Skills.files],
)

result: str = agent.run("Calculate 2 + 2")


def error_category(error: ChulkError) -> str:
    if isinstance(error, ConfigurationError):
        return "configuration"
    if isinstance(error, ProviderError):
        return "provider"
    if isinstance(error, ToolExecutionError):
        return "tool"
    if isinstance(error, PermissionDeniedError):
        return "permission"
    if isinstance(error, SafetyError):
        return "safety"
    if isinstance(error, TraceError):
        return "trace"
    if isinstance(error, MemoryError):
        return "memory"
    return error.category
