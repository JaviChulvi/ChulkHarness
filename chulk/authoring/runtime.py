"""Resolve one published definition into local or hosted SDK construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from chulk._sdk.config import AgentConfig
from chulk._sdk.facade import Agent, AsyncHostedRuntime, HostedRuntime
from chulk.authoring.catalog import (
    AgentDefinitionRecord,
    AgentDefinitionStore,
    ArtifactCatalog,
    PromptCatalog,
    PublicationError,
    ToolCatalog,
)
from chulk.authoring.models import VersionedReference
from chulk.config import Config
from chulk.hosting import AsyncRuntimeServices, ExecutionScope, RuntimeServices
from chulk.skills.publication import PortableSkill
from chulk.skills.registry import Skill, SkillRegistry
from chulk.usage import BudgetScope, ExactCost, RunBudget


class PublishedSkillResolver(Protocol):
    """Resolve only exact immutable skill references for a run."""

    def resolve_for_run(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
    ) -> PortableSkill: ...


@dataclass(frozen=True, slots=True)
class ResolvedAgentDefinition:
    """Exact runtime inputs resolved from one published definition revision."""

    record: AgentDefinitionRecord
    system_prompt: str
    tools: tuple[object, ...]
    skills: tuple[PortableSkill, ...]
    model_profile: Mapping[str, Any]
    approval_policy: Mapping[str, Any]

    @property
    def identity(self) -> dict[str, str]:
        return self.record.identity


@dataclass(frozen=True, slots=True)
class _PortableSkillSpec:
    skill: PortableSkill

    def register(self, registry: SkillRegistry) -> str:
        runtime_skill = Skill(
            name=self.skill.name,
            description=self.skill.description,
            path=Path(f".hosted-skill-{self.skill.name}.md"),
            metadata={
                "version": self.skill.version,
                "digest": self.skill.digest,
                "source": "hosted",
            },
            keywords=[self.skill.name],
            loaded_content=self.skill.instructions,
            digest=self.skill.digest,
        )
        registry.register(runtime_skill)
        return runtime_skill.name


class AgentDefinitionRuntime:
    """Build SDK facades only after exact publication and catalog resolution."""

    def __init__(
        self,
        *,
        definitions: AgentDefinitionStore,
        tools: ToolCatalog,
        prompts: PromptCatalog,
        skills: PublishedSkillResolver,
        model_profiles: ArtifactCatalog,
        approval_policies: ArtifactCatalog,
    ) -> None:
        self.definitions = definitions
        self.tools = tools
        self.prompts = prompts
        self.skills = skills
        self.model_profiles = model_profiles
        self.approval_policies = approval_policies

    def resolve(
        self,
        scope: ExecutionScope,
        *,
        expected_digest: str | None = None,
    ) -> ResolvedAgentDefinition:
        record = self.definitions.resolve_for_run(
            scope,
            scope.agent_id,
            scope.agent_version,
        )
        if expected_digest is not None and record.definition.digest != expected_digest:
            raise PublicationError(
                "agent definition digest does not match the requested revision"
            )
        definition = record.definition
        system_prompt = self.prompts.resolve(definition.prompt)
        model_profile = self.model_profiles.resolve(definition.model_profile)
        approval_policy = self.approval_policies.resolve(
            definition.approval_policy
        )
        tools = tuple(
            self.tools.resolve(reference) for reference in definition.tools
        )
        skills = tuple(
            self.skills.resolve_for_run(scope, reference)
            for reference in definition.skills
        )
        return ResolvedAgentDefinition(
            record=record,
            system_prompt=system_prompt,
            tools=tools,
            skills=skills,
            model_profile=model_profile,
            approval_policy=approval_policy,
        )

    def create_local(
        self,
        *,
        scope: ExecutionScope,
        config: Config | AgentConfig | None = None,
        expected_digest: str | None = None,
        **kwargs: Any,
    ) -> Agent:
        resolved = self.resolve(scope, expected_digest=expected_digest)
        return Agent(
            config=_apply_runtime_policies(config, resolved),
            tools=resolved.tools,
            skills=tuple(_PortableSkillSpec(skill) for skill in resolved.skills),
            system_prompt=resolved.system_prompt,
            run_budget=_run_budget(resolved),
            conversation_metadata=_definition_metadata(
                resolved,
                kwargs.pop("conversation_metadata", None),
            ),
            runtime_metadata=_definition_metadata(
                resolved,
                kwargs.pop("runtime_metadata", None),
            ),
            **kwargs,
        )

    def create_hosted(
        self,
        *,
        scope: ExecutionScope,
        services: RuntimeServices,
        config: Config | AgentConfig | None = None,
        expected_digest: str | None = None,
        **kwargs: Any,
    ) -> HostedRuntime:
        resolved = self.resolve(scope, expected_digest=expected_digest)
        return HostedRuntime(
            services=services,
            execution_scope=scope,
            config=_apply_runtime_policies(config, resolved),
            tools=resolved.tools,
            skills=tuple(_PortableSkillSpec(skill) for skill in resolved.skills),
            system_prompt=resolved.system_prompt,
            run_budget=_run_budget(resolved),
            conversation_metadata=_definition_metadata(
                resolved,
                kwargs.pop("conversation_metadata", None),
            ),
            runtime_metadata=_definition_metadata(
                resolved,
                kwargs.pop("runtime_metadata", None),
            ),
            **kwargs,
        )

    def create_async_hosted(
        self,
        *,
        scope: ExecutionScope,
        services: AsyncRuntimeServices,
        config: Config | AgentConfig | None = None,
        expected_digest: str | None = None,
        **kwargs: Any,
    ) -> AsyncHostedRuntime:
        resolved = self.resolve(scope, expected_digest=expected_digest)
        return AsyncHostedRuntime(
            services=services,
            execution_scope=scope,
            config=_apply_runtime_policies(config, resolved),
            tools=resolved.tools,
            skills=tuple(_PortableSkillSpec(skill) for skill in resolved.skills),
            system_prompt=resolved.system_prompt,
            run_budget=_run_budget(resolved),
            conversation_metadata=_definition_metadata(
                resolved,
                kwargs.pop("conversation_metadata", None),
            ),
            runtime_metadata=_definition_metadata(
                resolved,
                kwargs.pop("runtime_metadata", None),
            ),
            **kwargs,
        )

    def capability_diff(
        self,
        before: AgentDefinitionRecord,
        after: AgentDefinitionRecord,
    ) -> dict[str, tuple[str, ...]]:
        """Return reviewable behavior changes between exact revisions."""
        before_tools = {reference.name for reference in before.definition.tools}
        after_tools = {reference.name for reference in after.definition.tools}
        before_skills = {reference.name for reference in before.definition.skills}
        after_skills = {reference.name for reference in after.definition.skills}
        before_effects = {
            step.effect.value for step in before.definition.workflow.steps
        }
        after_effects = {
            step.effect.value for step in after.definition.workflow.steps
        }
        return {
            "tools_added": tuple(sorted(after_tools - before_tools)),
            "tools_removed": tuple(sorted(before_tools - after_tools)),
            "skills_added": tuple(sorted(after_skills - before_skills)),
            "skills_removed": tuple(sorted(before_skills - after_skills)),
            "effects_added": tuple(sorted(after_effects - before_effects)),
            "effects_removed": tuple(sorted(before_effects - after_effects)),
        }

    def dry_run(
        self,
        scope: ExecutionScope,
        *,
        expected_digest: str | None = None,
    ) -> dict[str, Any]:
        """Resolve every dependency and return a side-effect-free preview."""
        resolved = self.resolve(scope, expected_digest=expected_digest)
        return {
            "definition": resolved.identity,
            "tools": [getattr(tool, "name", "") for tool in resolved.tools],
            "skills": [skill.reference.to_dict() for skill in resolved.skills],
            "model_profile": dict(resolved.model_profile),
            "approval_policy": dict(resolved.approval_policy),
            "budget": resolved.record.definition.budget.to_dict(),
        }


def _apply_runtime_policies(
    config: Config | AgentConfig | None,
    resolved: ResolvedAgentDefinition,
) -> Config | AgentConfig | None:
    model_profile = resolved.model_profile
    approval_policy = resolved.approval_policy
    overrides: dict[str, Any] = {}
    for field_name in ("provider", "model"):
        value = model_profile.get(field_name)
        if isinstance(value, str) and value.strip():
            overrides[field_name] = value.strip()
    permission_profile = approval_policy.get("permission_profile")
    if isinstance(permission_profile, str) and permission_profile.strip():
        overrides["permission_profile"] = permission_profile.strip()
    if not overrides:
        return config
    if config is None:
        return AgentConfig(**overrides)
    if isinstance(config, AgentConfig):
        return config.with_overrides(**overrides)
    return replace(
        config,
        llm_provider=overrides.get("provider", config.llm_provider),
        model=overrides.get("model", config.model),
        permission_profile=overrides.get(
            "permission_profile",
            config.permission_profile,
        ),
    )


def _run_budget(resolved: ResolvedAgentDefinition) -> RunBudget:
    budget = resolved.record.definition.budget
    max_tokens = None
    if budget.max_input_tokens is not None or budget.max_output_tokens is not None:
        max_tokens = (budget.max_input_tokens or 0) + (
            budget.max_output_tokens or 0
        )
    max_cost = (
        ExactCost(
            Decimal(budget.max_cost),
            currency=budget.currency,
            pricing_known=True,
        )
        if budget.max_cost is not None
        else None
    )
    return RunBudget(
        scope=BudgetScope.TURN,
        max_model_calls=budget.max_model_requests,
        max_tool_calls=budget.max_tool_calls,
        max_tokens=max_tokens,
        max_cost=max_cost,
    )


def _definition_metadata(
    resolved: ResolvedAgentDefinition,
    supplied: Mapping[str, object] | None,
) -> dict[str, object]:
    metadata = dict(supplied or {})
    existing = metadata.get("agent_definition")
    if existing is not None and existing != resolved.identity:
        raise PublicationError("caller metadata conflicts with agent definition")
    metadata["agent_definition"] = resolved.identity
    return metadata


__all__ = [
    "AgentDefinitionRuntime",
    "PublishedSkillResolver",
    "ResolvedAgentDefinition",
]
