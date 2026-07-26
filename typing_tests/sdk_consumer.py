"""External-style static typing fixture for the installed SDK contract."""

from pathlib import Path
from dataclasses import dataclass
from typing import assert_never

from chulk import (
    Agent,
    AgentCompiler,
    AgentConfig,
    AgentDefinition,
    AgentDefinitionRuntime,
    Capabilities,
    ChulkError,
    ConfigurationError,
    CompiledAgentPackage,
    CompilerRequest,
    ContextReport,
    Cost,
    ExecutionScope,
    HostedRuntime,
    MemoryError,
    MemoryMode,
    MemoryProposal,
    Observation,
    PermissionDeniedError,
    ProviderError,
    Plan,
    PlanResult,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    PluginLifecycleReceipt,
    PluginUpdatePlan,
    SafetyError,
    Skills,
    Tool,
    ToolCall,
    ToolAttempt,
    ToolContext,
    ToolExecutionError,
    ToolEffect,
    ToolIdentity,
    ToolPolicy,
    ToolRetryPolicy,
    Tools,
    TraceError,
    RunCompletedPayload,
    RunResult,
    RunStatus,
    RuntimeProfile,
    SkillActivationRecord,
    Usage,
    VersionedReference,
)


assert Tool is not None
assert AgentCompiler is not None
assert AgentDefinitionRuntime is not None
assert RuntimeProfile is not None

config: AgentConfig = AgentConfig.local(
    project_root=Path.cwd(),
    runtime_dir=".chulk",
    permission_profile="read-only",
)

agent = Agent(
    config=config,
    capabilities=Capabilities(files="read", memory=MemoryMode.READ_ONLY),
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
        case RunStatus.CANCELLED:
            return "cancelled"
        case RunStatus.NO_PENDING_PLAN:
            return "no_plan"
        case RunStatus.UNKNOWN:
            return "unknown"
    assert_never(status)


@dataclass(frozen=True)
class Dependencies:
    tenant: str


def dependency_tool(query: str, context: ToolContext[Dependencies]) -> str:
    return f"{context.require_deps().tenant}:{query}"


retry_policy = ToolRetryPolicy(max_attempts=2)
hosted_scope: ExecutionScope = ExecutionScope(
    tenant_id="tenant",
    workspace_id="workspace",
    actor_id="actor",
    agent_id="assistant",
    agent_version="1.0.0",
    run_id="run",
)
hosted_policy: ToolPolicy = ToolPolicy(
    required_grants=frozenset({"catalog:read"}),
    effect=ToolEffect.READ,
)
hosted_identity: ToolIdentity = ToolIdentity.from_schemas(
    "catalog_lookup",
    input_schema={"type": "object", "properties": {}},
)
assert HostedRuntime is not None


def consume_proposal(proposal: MemoryProposal) -> str:
    return proposal.status.value


def consume_attempts(call: ToolCall) -> tuple[ToolAttempt, ...]:
    return call.attempts


def consume_compiled_package(
    package: CompiledAgentPackage,
) -> tuple[AgentDefinition, str, bool]:
    definition: AgentDefinition = package.definition
    request_type: type[CompilerRequest] = CompilerRequest
    _ = request_type
    return definition, package.skill.digest, package.publishable


def consume_skill_activation(
    activation: SkillActivationRecord,
) -> tuple[VersionedReference, str]:
    return activation.reference, activation.activated_by


def consume_plugin_lifecycle(
    receipt: PluginLifecycleReceipt,
    plan: PluginUpdatePlan,
) -> tuple[str, bool]:
    return receipt.action.value, plan.requires_reapproval
