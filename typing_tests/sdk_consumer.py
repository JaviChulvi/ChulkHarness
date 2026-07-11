"""External-style static typing fixture for the installed SDK contract."""

from pathlib import Path
from typing import assert_never

from chulk import (
    Agent,
    AgentConfig,
    ChulkError,
    ConfigurationError,
    ContextReport,
    Cost,
    MemoryError,
    Observation,
    PermissionDeniedError,
    ProviderError,
    Plan,
    PlanResult,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    SafetyError,
    Skills,
    Tool,
    ToolCall,
    ToolExecutionError,
    Tools,
    TraceError,
    RunCompletedPayload,
    RunResult,
    RunStatus,
    Usage,
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


def consume_result(result: RunResult) -> int:
    usage: Usage | None = result.usage
    cost: Cost | None = result.cost
    context: ContextReport | None = result.context_report
    calls: tuple[ToolCall, ...] = result.tool_calls
    observations: tuple[Observation, ...] = result.observations
    plan: Plan | None = result.plan
    _ = (cost, context, calls, observations, plan)
    return usage.total_tokens if usage is not None else 0


def consume_plan(result: PlanResult) -> tuple[PlanStatus, PlanStepStatus] | None:
    if result.plan is None or not result.plan.steps:
        return None
    step: PlanStep = result.plan.steps[0]
    return result.plan.status, step.status


def consume_terminal(payload: RunCompletedPayload) -> str:
    return payload.result.content


def exhaustive_run_status(status: RunStatus) -> str:
    match status:
        case RunStatus.IN_PROGRESS:
            return "in_progress"
        case RunStatus.COMPLETED:
            return "completed"
        case RunStatus.FAILED:
            return "failed"
        case RunStatus.BLOCKED:
            return "blocked"
        case RunStatus.WAITING_FOR_APPROVAL:
            return "waiting"
        case RunStatus.PLAN_REJECTED:
            return "rejected"
        case RunStatus.NO_PENDING_PLAN:
            return "no_plan"
        case RunStatus.UNKNOWN:
            return "unknown"
    assert_never(status)
