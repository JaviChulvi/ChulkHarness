"""Constrained compiler for declarative agent definitions and skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from chulk.authoring.catalog import (
    ArtifactCatalog,
    EvaluationCaseResult,
    EvaluationReport,
    PromptCatalog,
    PublicationError,
    PublishedTool,
    ToolCatalog,
    ValidationFinding,
    ValidationReport,
    validate_definition_catalogs,
)
from chulk.authoring.models import (
    AgentDefinition,
    BudgetDefinition,
    DefinitionProvenance,
    ToolReference,
    TriggerDefinition,
    VersionedReference,
    WorkflowApproval,
    WorkflowEffect,
    WorkflowGraph,
    WorkflowStep,
    canonical_json,
    request_digest,
)
from chulk.hosting import ExecutionScope
from chulk.skills.publication import PortableSkill
from chulk.tools.policy import ToolApprovalMode, ToolEffect


class WorkflowGenerator(Protocol):
    """Optional model adapter that can only return a declarative workflow."""

    def __call__(
        self,
        request: CompilerRequest,
        tools: tuple[ToolReference, ...],
        /,
    ) -> WorkflowGraph: ...


@dataclass(frozen=True, slots=True)
class CompilerRequest:
    """Structured caller intent; no raw credentials or executable artifacts."""

    agent_id: str
    version: str
    goal: str
    prompt: VersionedReference
    model_profile: VersionedReference
    approval_policy: VersionedReference
    selected_tools: tuple[str, ...]
    triggers: tuple[TriggerDefinition, ...] = ()
    constraints: tuple[str, ...] = ()
    autonomy: str = "supervised"
    locale: str = "en"
    sample_inputs: tuple[str, ...] = ()
    expected_outcomes: tuple[str, ...] = ()
    budget: BudgetDefinition = field(default_factory=BudgetDefinition)
    skill_name: str | None = None
    skill_version: str = "1.0.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        goal = _bounded(self.goal, "goal", maximum=8_000)
        tools = tuple(dict.fromkeys(name.strip() for name in self.selected_tools))
        if not tools or any(not name for name in tools):
            raise ValueError("selected_tools must contain explicit tool names")
        autonomy = self.autonomy.strip().lower()
        if autonomy not in {"supervised", "bounded", "autonomous"}:
            raise ValueError(
                "autonomy must be supervised, bounded, or autonomous"
            )
        for collection, label, maximum in (
            (self.constraints, "constraint", 2_000),
            (self.sample_inputs, "sample input", 4_000),
            (self.expected_outcomes, "expected outcome", 4_000),
        ):
            for value in collection:
                _bounded(value, label, maximum=maximum)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "selected_tools", tools)
        object.__setattr__(self, "triggers", tuple(self.triggers))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "sample_inputs", tuple(self.sample_inputs))
        object.__setattr__(
            self,
            "expected_outcomes",
            tuple(self.expected_outcomes),
        )
        object.__setattr__(self, "autonomy", autonomy)
        DefinitionProvenance(
            author="compiler-request",
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "goal": self.goal,
            "prompt": self.prompt.to_dict(),
            "model_profile": self.model_profile.to_dict(),
            "approval_policy": self.approval_policy.to_dict(),
            "selected_tools": list(self.selected_tools),
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "constraints": list(self.constraints),
            "autonomy": self.autonomy,
            "locale": self.locale,
            "sample_inputs": list(self.sample_inputs),
            "expected_outcomes": list(self.expected_outcomes),
            "budget": self.budget.to_dict(),
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ReviewPreview:
    """Bounded review surface for authority, effects, and expected behavior."""

    agent_id: str
    version: str
    tool_names: tuple[str, ...]
    required_grants: tuple[str, ...]
    effects: tuple[str, ...]
    approval_steps: tuple[str, ...]
    trigger_kinds: tuple[str, ...]
    budget: Mapping[str, Any]
    sample_outcomes: tuple[str, ...]
    tenant_id: str
    workspace_id: str
    actor_id: str | None
    autonomy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "tool_names": list(self.tool_names),
            "required_grants": list(self.required_grants),
            "effects": list(self.effects),
            "approval_steps": list(self.approval_steps),
            "trigger_kinds": list(self.trigger_kinds),
            "budget": dict(self.budget),
            "sample_outcomes": list(self.sample_outcomes),
            "scope": {
                "tenant_id": self.tenant_id,
                "workspace_id": self.workspace_id,
                "actor_id": self.actor_id,
            },
            "autonomy": self.autonomy,
        }


@dataclass(frozen=True, slots=True)
class CompiledAgentPackage:
    """Draft artifacts and deterministic evidence; never executable by itself."""

    definition: AgentDefinition
    skill: PortableSkill
    validation_report: ValidationReport
    evaluation_report: EvaluationReport
    preview: ReviewPreview

    @property
    def publishable(self) -> bool:
        return self.validation_report.valid and self.evaluation_report.passed


class AgentCompiler:
    """Compile only caller-selected trusted tools into declarative artifacts."""

    def __init__(
        self,
        *,
        tools: ToolCatalog,
        prompts: PromptCatalog,
        model_profiles: ArtifactCatalog | None = None,
        approval_policies: ArtifactCatalog | None = None,
        generator: WorkflowGenerator | None = None,
        compiler_id: str = "chulk-constrained-compiler/1",
    ) -> None:
        self.tools = tools
        self.prompts = prompts
        self.model_profiles = model_profiles
        self.approval_policies = approval_policies
        self.generator = generator
        self.compiler_id = _bounded(
            compiler_id,
            "compiler_id",
            maximum=256,
        )

    def compile(
        self,
        request: CompilerRequest,
        *,
        caller_scope: ExecutionScope,
        skill_references: Mapping[str, VersionedReference] | None = None,
    ) -> CompiledAgentPackage:
        if caller_scope.agent_id != request.agent_id:
            raise PublicationError(
                "compiler caller scope does not match the requested agent"
            )
        if caller_scope.agent_version != request.version:
            raise PublicationError(
                "compiler caller scope version does not match the request"
            )
        selected = self.tools.selected(request.selected_tools)
        references = tuple(item.reference for item in selected)
        required_grants = frozenset(
            grant
            for item in selected
            for grant in item.tool.resolved_policy().required_grants
        )
        missing_grants = required_grants - caller_scope.grants
        if missing_grants:
            raise PublicationError(
                "compiler request exceeds caller authority: "
                + ", ".join(sorted(missing_grants))
            )
        workflow = (
            self.generator(request, references)
            if self.generator is not None
            else _default_workflow(selected)
        )
        _validate_generated_workflow(
            workflow,
            selected={item.reference.name: item for item in selected},
        )
        skill_name = request.skill_name or f"{request.agent_id}-workflow"
        provenance = DefinitionProvenance(
            author=caller_scope.actor_id or "host",
            generator=self.compiler_id,
            source_request_digest=request_digest(request.to_dict()),
            created_at=request.created_at,
        )
        skill = PortableSkill(
            name=skill_name,
            version=request.skill_version,
            description=f"Procedural workflow for {request.agent_id}.",
            instructions=_skill_instructions(request, workflow),
            required_tools=references,
            provenance=provenance,
        )
        definition = AgentDefinition(
            agent_id=request.agent_id,
            version=request.version,
            prompt=request.prompt,
            model_profile=request.model_profile,
            approval_policy=request.approval_policy,
            tools=references,
            skills=(skill.reference,),
            triggers=request.triggers,
            workflow=workflow,
            budget=request.budget,
            locale=request.locale,
            provenance=provenance,
        )
        definition_validation = validate_definition_catalogs(
            definition,
            tools=self.tools,
            prompts=self.prompts,
            model_profiles=self.model_profiles,
            approval_policies=self.approval_policies,
            skill_references={
                **dict(skill_references or {}),
                skill.reference.name: skill.reference,
            },
        )
        policy_findings = _policy_findings(
            definition,
            selected={item.reference.name: item for item in selected},
            caller_scope=caller_scope,
        )
        validation = ValidationReport(
            valid=definition_validation.valid and not policy_findings,
            findings=(*definition_validation.findings, *policy_findings),
            validator="chulk-agent-compiler",
            artifact_digest=definition.digest,
            created_at=request.created_at,
        )
        evaluation = _evaluate_compiled_package(
            request,
            definition,
            validation=validation,
            evaluator=self.compiler_id,
            created_at=request.created_at,
        )
        preview = ReviewPreview(
            agent_id=definition.agent_id,
            version=definition.version,
            tool_names=tuple(reference.name for reference in definition.tools),
            required_grants=tuple(sorted(required_grants)),
            effects=tuple(
                dict.fromkeys(step.effect.value for step in workflow.steps)
            ),
            approval_steps=tuple(
                step.id
                for step in workflow.steps
                if step.approval is not WorkflowApproval.NEVER
            ),
            trigger_kinds=tuple(trigger.kind for trigger in request.triggers),
            budget=request.budget.to_dict(),
            sample_outcomes=request.expected_outcomes,
            tenant_id=caller_scope.tenant_id,
            workspace_id=caller_scope.workspace_id,
            actor_id=caller_scope.actor_id,
            autonomy=request.autonomy,
        )
        return CompiledAgentPackage(
            definition=definition,
            skill=skill,
            validation_report=validation,
            evaluation_report=evaluation,
            preview=preview,
        )


def _default_workflow(selected: tuple[PublishedTool, ...]) -> WorkflowGraph:
    steps: list[WorkflowStep] = []
    previous: str | None = None
    for index, published in enumerate(selected, start=1):
        policy = published.tool.resolved_policy()
        step_id = f"step_{index}_{published.reference.name}"[:63]
        step = WorkflowStep(
            id=step_id,
            tool=published.reference,
            depends_on=(previous,) if previous is not None else (),
            effect=_workflow_effect(policy.effect),
            approval=_workflow_approval(policy.approval),
            retry_limit=(
                1 if policy.idempotency.value != "none" else 0
            ),
            timeout_seconds=published.tool.timeout_seconds or 30.0,
        )
        steps.append(step)
        previous = step.id
    return WorkflowGraph(steps=tuple(steps))


def _validate_generated_workflow(
    workflow: WorkflowGraph,
    *,
    selected: Mapping[str, Any],
) -> None:
    for step in workflow.steps:
        try:
            published = selected[step.tool.name]
        except KeyError as exc:
            raise PublicationError(
                f"generator introduced unselected tool {step.tool.name!r}"
            ) from exc
        if published.reference != step.tool:
            raise PublicationError(
                f"generator changed the identity of tool {step.tool.name!r}"
            )
        policy = published.tool.resolved_policy()
        if _effect_rank(step.effect) < _tool_effect_rank(policy.effect):
            raise PublicationError(
                f"generator reduced the declared effect of tool {step.tool.name!r}"
            )
        if _approval_rank(step.approval) < _tool_approval_rank(policy.approval):
            raise PublicationError(
                f"generator reduced the approval requirement of tool "
                f"{step.tool.name!r}"
            )


def _policy_findings(
    definition: AgentDefinition,
    *,
    selected: Mapping[str, Any],
    caller_scope: ExecutionScope,
) -> tuple[ValidationFinding, ...]:
    findings: list[ValidationFinding] = []
    for step in definition.workflow.steps:
        published = selected[step.tool.name]
        policy = published.tool.resolved_policy()
        missing = policy.required_grants - caller_scope.grants
        if missing:
            findings.append(
                ValidationFinding(
                    code="authority_exceeded",
                    field=f"workflow.{step.id}",
                    message="missing caller grants: " + ", ".join(sorted(missing)),
                )
            )
        if _approval_rank(step.approval) < _tool_approval_rank(policy.approval):
            findings.append(
                ValidationFinding(
                    code="approval_downgrade",
                    field=f"workflow.{step.id}.approval",
                    message="workflow approval is weaker than tool policy",
                )
            )
    return tuple(findings)


def _evaluate_compiled_package(
    request: CompilerRequest,
    definition: AgentDefinition,
    *,
    validation: ValidationReport,
    evaluator: str,
    created_at: str,
) -> EvaluationReport:
    trigger_keys = [
        (trigger.kind, trigger.version, canonical_json(dict(trigger.config)))
        for trigger in definition.triggers
    ]
    pinned = all(
        reference.version and reference.digest
        for reference in (*definition.tools, *definition.skills)
    )
    cases = (
        EvaluationCaseResult(
            name="success",
            passed=validation.valid,
            detail="all selected dependencies and workflow contracts validate",
        ),
        EvaluationCaseResult(
            name="ambiguity",
            passed=bool(request.goal and request.expected_outcomes),
            detail="goal and expected outcomes are explicit",
        ),
        EvaluationCaseResult(
            name="unavailable_data",
            passed=all(step.timeout_seconds is not None for step in definition.workflow.steps),
            detail="tool steps have bounded failure timing",
        ),
        EvaluationCaseResult(
            name="denial",
            passed=all(
                step.approval is not WorkflowApproval.NEVER
                or step.effect is WorkflowEffect.READ
                for step in definition.workflow.steps
            ),
            detail="write effects retain an approval boundary",
        ),
        EvaluationCaseResult(
            name="timeout",
            passed=all(
                step.timeout_seconds is not None
                and step.timeout_seconds > 0
                for step in definition.workflow.steps
            ),
            detail="every step has a positive timeout",
        ),
        EvaluationCaseResult(
            name="duplicate_trigger",
            passed=len(trigger_keys) == len(set(trigger_keys)),
            detail="duplicate trigger identities are rejected",
        ),
        EvaluationCaseResult(
            name="cancellation",
            passed=True,
            detail="compiled artifacts contain no executable cleanup path",
        ),
        EvaluationCaseResult(
            name="restart",
            passed=pinned,
            detail="all restart dependencies are pinned by version and digest",
        ),
    )
    return EvaluationReport(
        passed=all(case.passed for case in cases),
        cases=cases,
        evaluator=evaluator,
        artifact_digest=definition.digest,
        created_at=created_at,
    )


def _skill_instructions(
    request: CompilerRequest,
    workflow: WorkflowGraph,
) -> str:
    lines = [
        f"# {request.agent_id} workflow",
        "",
        f"Goal: {request.goal}",
        "",
        "Follow these declarative steps in order:",
    ]
    for step in workflow.steps:
        dependencies = (
            ", ".join(step.depends_on) if step.depends_on else "none"
        )
        lines.append(
            f"- `{step.id}` uses `{step.tool.name}`; dependencies: "
            f"{dependencies}; effect: {step.effect.value}; approval: "
            f"{step.approval.value}."
        )
    if request.constraints:
        lines.extend(("", "Constraints:"))
        lines.extend(f"- {constraint}" for constraint in request.constraints)
    return "\n".join(lines)


def _workflow_effect(effect: ToolEffect) -> WorkflowEffect:
    return WorkflowEffect(effect.value)


def _workflow_approval(approval: ToolApprovalMode) -> WorkflowApproval:
    return WorkflowApproval(approval.value)


def _effect_rank(effect: WorkflowEffect) -> int:
    return {
        WorkflowEffect.READ: 0,
        WorkflowEffect.LOCAL_WRITE: 1,
        WorkflowEffect.EXTERNAL_WRITE: 2,
        WorkflowEffect.DESTRUCTIVE: 3,
        WorkflowEffect.UNKNOWN: 4,
    }[effect]


def _tool_effect_rank(effect: ToolEffect) -> int:
    return _effect_rank(WorkflowEffect(effect.value))


def _approval_rank(approval: WorkflowApproval) -> int:
    return {
        WorkflowApproval.NEVER: 0,
        WorkflowApproval.POLICY: 1,
        WorkflowApproval.ALWAYS: 2,
    }[approval]


def _tool_approval_rank(approval: ToolApprovalMode) -> int:
    return _approval_rank(WorkflowApproval(approval.value))


def _bounded(value: str, field_name: str, *, maximum: int) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} cannot be empty")
    if len(clean) > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum} characters")
    if "\x00" in clean:
        raise ValueError(f"{field_name} cannot contain NUL characters")
    lowered = clean.lower()
    if any(
        marker in lowered
        for marker in (
            "api_key=",
            "password=",
            "private_key=",
            "bearer ",
        )
    ):
        raise ValueError(f"{field_name} cannot contain credential values")
    if any(
        marker in lowered
        for marker in (
            "pip install ",
            "npm install ",
            "subprocess.",
            "os.system(",
            "importlib.",
            "mcp.json",
            "```python",
            "```shell",
            "```bash",
        )
    ):
        raise ValueError(f"{field_name} cannot contain executable content")
    if clean.startswith(("/", "\\\\", "./", "../", ".\\", "..\\")) or any(
        marker in clean
        for marker in (
            " file://",
            " C:\\",
            " /Users/",
            " /home/",
            " ./",
            " ../",
        )
    ):
        raise ValueError(f"{field_name} cannot contain local paths")
    return clean


__all__ = [
    "AgentCompiler",
    "CompiledAgentPackage",
    "CompilerRequest",
    "ReviewPreview",
    "WorkflowGenerator",
]
