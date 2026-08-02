"""External-style static typing fixture for the installed SDK contract."""

from pathlib import Path
from dataclasses import dataclass
from typing import assert_never

from chulk import (
    Agent,
    AgentCompiler,
    AgentConfig,
    ApprovalStore,
    ApprovalSubmission,
    ApprovalValidation,
    AgentDefinition,
    AgentDefinitionRuntime,
    AsyncHostedRuntime,
    AsyncLearningProposalService,
    AsyncLearningReviewer,
    AsyncPluginService,
    AsyncRuntimeServices,
    AsyncServiceBinding,
    AsyncSkillLifecycleService,
    AsyncSkillLifecycleStore,
    AsyncSkillService,
    ApplicationEventIntent,
    ApplicationEventSchema,
    Capabilities,
    ChildRunRecord,
    ChulkError,
    ConfigurationError,
    CompiledAgentPackage,
    CompilerRequest,
    ContextReport,
    Cost,
    ExecutionScope,
    GatewayRunSubmitter,
    GatewayRunTarget,
    GatewayScopeResolver,
    GatewayStore,
    HostedScheduledOccurrence,
    HostedScheduleRunSubmitter,
    DurableHostedExecutor,
    DurableRunStatus,
    HostedRuntime,
    HostResource,
    MemoryError,
    MemoryMode,
    MemoryProposal,
    FinalAnswerChunk,
    FinalAnswerDelivery,
    FinalAnswerPolicyDecision,
    FinalAnswerStreamingMode,
    Observation,
    PermissionDeniedError,
    ProviderError,
    Plan,
    PlanResult,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    ParentRunPolicy,
    ParentRunRecord,
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
    TurnContextSection,
    Tools,
    TraceError,
    RunCompletedPayload,
    RunBudget,
    RunRecord,
    RunStore,
    RunSubmission,
    RunResult,
    RunStatus,
    BudgetScope,
    RuntimeProfile,
    SkillActivationRecord,
    StepDefinition,
    Usage,
    VersionedReference,
)
from chulk.postgres import (
    AsyncPostgreSQLApprovalStore,
    AsyncPostgreSQLGatewayStore,
    AsyncPostgreSQLRunStore,
    AsyncPostgreSQLScheduleStore,
    PostgreSQLApprovalStore,
    PostgreSQLGatewayStore,
    PostgreSQLRunStore,
    PostgreSQLScheduleStore,
    async_complete_run_and_enqueue,
    async_ingest_and_submit_run,
    complete_run_and_enqueue,
    create_async_postgres_engine,
    create_postgres_engine,
    ingest_and_submit_run,
    upgrade_postgres,
)


assert Tool is not None
assert AgentCompiler is not None
assert AgentDefinitionRuntime is not None
assert RuntimeProfile is not None
assert PostgreSQLRunStore is not None
assert PostgreSQLApprovalStore is not None
assert PostgreSQLGatewayStore is not None
assert PostgreSQLScheduleStore is not None
assert AsyncPostgreSQLRunStore is not None
assert AsyncPostgreSQLApprovalStore is not None
assert AsyncPostgreSQLGatewayStore is not None
assert AsyncPostgreSQLScheduleStore is not None
assert create_postgres_engine is not None
assert create_async_postgres_engine is not None
assert upgrade_postgres is not None
assert ingest_and_submit_run is not None
assert async_ingest_and_submit_run is not None
assert complete_run_and_enqueue is not None
assert async_complete_run_and_enqueue is not None

config: AgentConfig = AgentConfig.local(
    project_root=Path.cwd(),
    runtime_dir=".chulk",
    permission_profile="read-only",
)


class OutputPolicy:
    def process(self, chunk: FinalAnswerChunk) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision(text=chunk.text)

    def complete(
        self, *, turn_id: str, next_sequence: int
    ) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision()

    def reset(self, *, turn_id: str) -> None:
        return None

agent = Agent(
    config=config,
    capabilities=Capabilities(files="read", memory=MemoryMode.READ_ONLY),
    tools=[Tools.calculator],
    skills=[Skills.files],
    final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
    output_policy=OutputPolicy(),
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
    resources: tuple[HostResource, ...] = result.resources
    plan: Plan | None = result.plan
    delivery: FinalAnswerDelivery | None = result.final_answer_delivery
    _ = (cost, context, calls, observations, resources, plan, delivery)
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
host_resource = HostResource(
    id="resource-1",
    kind="document",
    title="Evidence",
    source="host",
)
host_context = TurnContextSection(
    id="context-1",
    content="private evidence",
    persist_content=False,
    resource=host_resource,
)
application_schema = ApplicationEventSchema(
    namespace="acme.events",
    name="resource.ready",
    version=1,
    payload_schema={"type": "object"},
)
application_intent = ApplicationEventIntent(
    namespace="acme.events",
    name="resource.ready",
    schema_version=1,
    payload={"resource_id": host_resource.id},
    idempotency_key="resource-1:ready",
)
assert host_context.resource is host_resource
assert application_schema.key == application_intent.schema_key
assert HostedRuntime is not None
assert AsyncHostedRuntime is not None
assert AsyncRuntimeServices is not None
assert AsyncServiceBinding is not None
assert AsyncLearningProposalService is not None
assert AsyncLearningReviewer is not None
assert AsyncPluginService is not None
assert AsyncSkillLifecycleService is not None
assert AsyncSkillLifecycleStore is not None
assert AsyncSkillService is not None
assert DurableHostedExecutor is not None
assert GatewayStore is not None
assert GatewayScopeResolver is not None
assert GatewayRunSubmitter is not None
assert GatewayRunTarget is not None
assert HostedScheduledOccurrence is not None
assert HostedScheduleRunSubmitter is not None

durable_submission: RunSubmission = RunSubmission(
    idempotency_key="trigger",
    input_digest="sha256:input",
    definition_digest="sha256:definition",
    steps=(StepDefinition(id="agent", name="Agent turn"),),
)


def consume_durable_run(
    store: RunStore,
    scope: ExecutionScope,
) -> DurableRunStatus:
    record: RunRecord = store.get(scope, scope.run_id)
    return record.status


def consume_parent_child_runs(
    store: RunStore,
    parent_scope: ExecutionScope,
    child_scope: ExecutionScope,
) -> tuple[ParentRunRecord, ChildRunRecord]:
    policy = ParentRunPolicy(
        required_children=1,
        max_children=2,
        budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=4,
        ),
    )
    parent = store.submit_parent(
        parent_scope,
        durable_submission,
        policy=policy,
    )
    child = store.submit_child(
        parent_scope,
        child_scope,
        RunSubmission(
            idempotency_key="child-trigger",
            input_digest="sha256:child-input",
            definition_digest="sha256:child-definition",
            steps=(StepDefinition(id="agent", name="Agent turn"),),
            budget=RunBudget(
                scope=BudgetScope.CHILD_TASK,
                max_model_calls=1,
            ).to_dict(),
        ),
        definition_revision=child_scope.agent_version,
    )
    return parent, child


def consume_approval_contracts(
    store: ApprovalStore,
    scope: ExecutionScope,
    submission: ApprovalSubmission,
    validation: ApprovalValidation,
) -> str:
    _ = (scope, submission, validation)
    return type(store).__name__


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
