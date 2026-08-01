"""Contract tests for portable definitions, publication, and compilation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chulk import (
    AgentCompiler,
    AgentConfig,
    AgentDefinition,
    AgentDefinitionRuntime,
    ArtifactCatalog,
    AsyncInMemoryAgentDefinitionStore,
    AsyncInMemorySkillPublicationStore,
    AsyncSkillPublicationManager,
    BudgetDefinition,
    CompilerRequest,
    CompiledAgentPackage,
    DefinitionProvenance,
    DefinitionStatus,
    EvaluationCaseResult,
    EvaluationReport,
    ExecutionScope,
    InMemoryAgentDefinitionStore,
    InMemorySkillPublicationStore,
    PortableSkill,
    PromptCatalog,
    PublicationError,
    SkillPublicationManager,
    Tool,
    ToolApprovalMode,
    ToolCatalog,
    ToolEffect,
    ToolIdentity,
    ToolPolicy,
    ToolReference,
    TriggerDefinition,
    ValidationReport,
    VersionedReference,
    WorkflowApproval,
    WorkflowEffect,
    WorkflowGraph,
    WorkflowStep,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.skills.lifecycle_models import SkillLifecycleStatus
from chulk.testing import ScriptedLLMClient


READ_POLICY = ToolPolicy(
    version="1.0.0",
    required_grants=frozenset({"catalog:read"}),
    effect=ToolEffect.READ,
    approval=ToolApprovalMode.NEVER,
)
WRITE_POLICY = ToolPolicy(
    version="1.0.0",
    required_grants=frozenset({"tickets:write"}),
    effect=ToolEffect.EXTERNAL_WRITE,
    approval=ToolApprovalMode.ALWAYS,
)


@Tool(policy=READ_POLICY)
def catalog_lookup(item_id: str) -> str:
    """Look up one catalog item."""
    return f"{item_id}: available"


@Tool(policy=WRITE_POLICY)
def update_ticket(ticket_id: str) -> str:
    """Update one external ticket."""
    return f"{ticket_id}: updated"


def scope(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "support",
    agent_id: str = "support-agent",
    agent_version: str = "1.0.0",
    run_id: str = "run-1",
    grants: frozenset[str] = frozenset({"catalog:read", "tickets:write"}),
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        actor_id="default",
        agent_id=agent_id,
        agent_version=agent_version,
        run_id=run_id,
        grants=grants,
    )


def catalogs() -> tuple[
    ToolCatalog,
    PromptCatalog,
    ArtifactCatalog,
    ArtifactCatalog,
]:
    tools = ToolCatalog((catalog_lookup, update_ticket))
    prompts = PromptCatalog()
    prompts.publish(
        name="support-prompt",
        version="1.0.0",
        content="You are a bounded support agent.",
    )
    models = ArtifactCatalog()
    models.publish(
        name="support-model",
        version="1.0.0",
        payload={"provider": "openai", "model": "gpt-5-mini"},
    )
    approvals = ArtifactCatalog()
    approvals.publish(
        name="support-approval",
        version="1.0.0",
        payload={"permission_profile": "read-only"},
    )
    return tools, prompts, models, approvals


def request(
    *,
    selected_tools: tuple[str, ...] = ("catalog_lookup",),
    expected_outcomes: tuple[str, ...] = ("Return catalog availability.",),
) -> CompilerRequest:
    tools, prompts, models, approvals = catalogs()
    del tools
    return CompilerRequest(
        agent_id="support-agent",
        version="1.0.0",
        goal="Answer support requests using only trusted catalog data.",
        prompt=prompts.publish(
            name="support-prompt",
            version="1.0.0",
            content="You are a bounded support agent.",
        ).reference,
        model_profile=models.publish(
            name="support-model",
            version="1.0.0",
            payload={"provider": "openai", "model": "gpt-5-mini"},
        ).reference,
        approval_policy=approvals.publish(
            name="support-approval",
            version="1.0.0",
            payload={"permission_profile": "read-only"},
        ).reference,
        selected_tools=selected_tools,
        triggers=(
            TriggerDefinition(
                kind="api.request",
                config={"route": "support"},
            ),
        ),
        constraints=("Do not invent unavailable records.",),
        sample_inputs=("Is SKU-42 available?",),
        expected_outcomes=expected_outcomes,
        budget=BudgetDefinition(
            max_model_requests=4,
            max_tool_calls=4,
            max_input_tokens=400_000,
            max_output_tokens=10_000,
        ),
    )


def compiler() -> tuple[
    AgentCompiler,
    ToolCatalog,
    PromptCatalog,
    ArtifactCatalog,
    ArtifactCatalog,
]:
    tools, prompts, models, approvals = catalogs()
    return (
        AgentCompiler(
            tools=tools,
            prompts=prompts,
            model_profiles=models,
            approval_policies=approvals,
        ),
        tools,
        prompts,
        models,
        approvals,
    )


def publish_package() -> tuple[
    CompiledAgentPackage,
    AgentDefinitionRuntime,
    InMemoryAgentDefinitionStore,
    SkillPublicationManager,
]:
    selected_compiler, tools, prompts, models, approvals = compiler()
    compiled = selected_compiler.compile(request(), caller_scope=scope())
    skill_store = InMemorySkillPublicationStore()
    skill_manager = SkillPublicationManager(skill_store, tools=tools)
    submitted = skill_manager.submit(scope(), compiled.skill)
    assert submitted.status is SkillLifecycleStatus.AWAITING_REVIEW
    skill_manager.publish(
        scope(),
        compiled.skill.name,
        compiled.skill.version,
        reviewer="reviewer",
    )
    definitions = InMemoryAgentDefinitionStore()
    definitions.save_draft(scope(), compiled.definition)
    definitions.publish(
        scope(),
        compiled.definition.agent_id,
        compiled.definition.version,
        validation_report=compiled.validation_report,
        evaluation_report=compiled.evaluation_report,
        reviewer="reviewer",
    )
    runtime = AgentDefinitionRuntime(
        definitions=definitions,
        tools=tools,
        prompts=prompts,
        skills=skill_manager,
        model_profiles=models,
        approval_policies=approvals,
    )
    return compiled, runtime, definitions, skill_manager


def test_agent_definition_is_canonical_portable_and_round_trips() -> None:
    compiled, *_ = publish_package()
    definition = compiled.definition
    reconstructed = AgentDefinition.from_json(definition.canonical_json)

    assert reconstructed == definition
    assert reconstructed.digest == definition.digest
    assert definition.canonical_json == reconstructed.canonical_json
    serialized = definition.to_dict()
    assert "project_root" not in str(serialized)
    assert "runtime_dir" not in str(serialized)
    assert "store_path" not in str(serialized)
    assert "credential" not in str(serialized).lower()

    with pytest.raises(ValueError, match="local paths"):
        TriggerDefinition(kind="api.request", config={"project_root": "/tmp/app"})
    with pytest.raises(ValueError, match="secret material"):
        TriggerDefinition(kind="api.request", config={"api_key": "secret"})
    with pytest.raises(ValueError, match="executable imports"):
        TriggerDefinition(kind="api.request", config={"module": "unsafe.module"})
    with pytest.raises(ValueError, match="local paths"):
        TriggerDefinition(kind="api.request", config={"route": "../private"})
    with pytest.raises(ValueError, match="secret material"):
        TriggerDefinition(
            kind="api.request",
            config={"header": "Bearer raw-secret"},
        )
    with pytest.raises(ValueError, match="finite"):
        BudgetDefinition(max_cost="NaN")


def test_agent_definition_schema_compatibility_and_cross_platform_digest() -> None:
    def reference(name: str, character: str) -> VersionedReference:
        return VersionedReference(
            name=name,
            version="1.0.0",
            digest="sha256:" + (character * 64),
        )

    definition = AgentDefinition(
        agent_id="portable-agent",
        version="1.0.0",
        prompt=reference("prompt", "1"),
        model_profile=reference("model", "2"),
        approval_policy=reference("approval", "3"),
        provenance=DefinitionProvenance(
            author="host",
            created_at="2026-01-02T03:04:05+00:00",
        ),
    )
    assert (
        definition.digest
        == "sha256:ef8833cf1890ca1d7182988d390c26861c68fa9124eb52242693f30f3cfb0792"
    )

    future = definition.to_dict()
    future["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        AgentDefinition.from_dict(future)

    unknown = definition.to_dict()
    unknown["future_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        AgentDefinition.from_dict(unknown)

    malformed_workflow = definition.to_dict()
    malformed_workflow["workflow"] = {"steps": ["not-an-object"]}
    with pytest.raises(ValueError, match="workflow step"):
        AgentDefinition.from_dict(malformed_workflow)


def test_definition_publication_is_immutable_scoped_and_revocable() -> None:
    compiled, _runtime, definitions, _skills = publish_package()
    current = definitions.resolve_for_run(
        scope(),
        compiled.definition.agent_id,
        compiled.definition.version,
    )
    assert current.status is DefinitionStatus.PUBLISHED

    changed = replace(
        compiled.definition,
        locale="es",
    )
    with pytest.raises(PublicationError, match="different behavior"):
        definitions.save_draft(scope(), changed)
    with pytest.raises(KeyError):
        definitions.get(
            scope(tenant_id="tenant-b"),
            compiled.definition.agent_id,
            compiled.definition.version,
        )

    historical_digest = current.definition.digest
    revoked = definitions.revoke(
        scope(),
        current.definition.agent_id,
        current.definition.version,
        revoked_by="security",
        reason="unsafe dependency",
    )
    assert revoked.definition.digest == historical_digest
    assert definitions.get(
        scope(),
        current.definition.agent_id,
        current.definition.version,
    ).definition.digest == historical_digest
    with pytest.raises(PublicationError, match="revoked"):
        definitions.resolve_for_run(
            scope(),
            current.definition.agent_id,
            current.definition.version,
        )


def test_compiler_fails_unknown_tools_and_authority_escalation() -> None:
    selected_compiler, *_ = compiler()
    with pytest.raises(KeyError, match="not in the published catalog"):
        selected_compiler.compile(
            request(selected_tools=("hidden_tool",)),
            caller_scope=scope(),
        )
    with pytest.raises(PublicationError, match="exceeds caller authority"):
        selected_compiler.compile(
            request(selected_tools=("update_ticket",)),
            caller_scope=scope(grants=frozenset({"catalog:read"})),
        )


def test_generator_cannot_invent_tools_or_downgrade_approval() -> None:
    selected_compiler, tools, prompts, models, approvals = compiler()
    hidden = ToolCatalog()

    @Tool
    def hidden_tool() -> str:
        return "hidden"

    hidden_reference = hidden.publish(hidden_tool).reference

    def invent_tool(
        _request: CompilerRequest,
        _tools: tuple[ToolReference, ...],
    ) -> WorkflowGraph:
        return WorkflowGraph(
            steps=(
                WorkflowStep(
                    id="hidden",
                    tool=hidden_reference,
                ),
            )
        )

    malicious = AgentCompiler(
        tools=tools,
        prompts=prompts,
        model_profiles=models,
        approval_policies=approvals,
        generator=invent_tool,
    )
    with pytest.raises(PublicationError, match="unselected tool"):
        malicious.compile(request(), caller_scope=scope())

    write_reference = tools.reference("update_ticket")

    def downgrade(
        _request: CompilerRequest,
        _tools: tuple[ToolReference, ...],
    ) -> WorkflowGraph:
        return WorkflowGraph(
            steps=(
                WorkflowStep(
                    id="write",
                    tool=write_reference,
                    effect=WorkflowEffect.READ,
                    approval=WorkflowApproval.NEVER,
                ),
            )
        )

    malicious = AgentCompiler(
        tools=tools,
        prompts=prompts,
        model_profiles=models,
        approval_policies=approvals,
        generator=downgrade,
    )
    with pytest.raises(PublicationError, match="reduced the declared effect"):
        malicious.compile(
            request(selected_tools=("update_ticket",)),
            caller_scope=scope(),
        )


def test_compiler_reports_all_scenarios_and_bounded_review_surface() -> None:
    selected_compiler, *_ = compiler()
    compiled = selected_compiler.compile(
        request(selected_tools=("catalog_lookup", "update_ticket")),
        caller_scope=scope(),
    )

    assert compiled.publishable
    assert {case.name for case in compiled.evaluation_report.cases} == {
        "success",
        "ambiguity",
        "unavailable_data",
        "denial",
        "timeout",
        "duplicate_trigger",
        "cancellation",
        "restart",
    }
    preview = compiled.preview.to_dict()
    assert preview["tool_names"] == ["catalog_lookup", "update_ticket"]
    assert preview["required_grants"] == ["catalog:read", "tickets:write"]
    assert preview["effects"] == ["read", "external_write"]
    assert preview["trigger_kinds"] == ["api.request"]
    assert preview["budget"]["max_tool_calls"] == 4
    assert preview["approval_steps"] == [
        "step_2_update_ticket",
    ]
    assert preview["scope"] == {
        "tenant_id": "tenant-a",
        "workspace_id": "support",
        "actor_id": "default",
    }
    assert preview["autonomy"] == "supervised"


def test_compiler_is_deterministic_for_the_same_structured_request() -> None:
    selected_compiler, *_ = compiler()
    structured_request = replace(
        request(),
        triggers=(
            TriggerDefinition(
                kind="api.request",
                config={"route": {"name": "support", "versions": [1, 2]}},
            ),
        ),
    )

    first = selected_compiler.compile(
        structured_request,
        caller_scope=scope(),
    )
    second = selected_compiler.compile(
        structured_request,
        caller_scope=scope(),
    )

    assert first == second
    assert first.definition.digest == second.definition.digest
    assert first.skill.digest == second.skill.digest


def test_failed_evaluation_cannot_publish() -> None:
    selected_compiler, *_ = compiler()
    compiled = selected_compiler.compile(
        request(expected_outcomes=()),
        caller_scope=scope(),
    )
    assert not compiled.evaluation_report.passed
    definitions = InMemoryAgentDefinitionStore()
    definitions.save_draft(scope(), compiled.definition)
    with pytest.raises(PublicationError, match="failed evaluations"):
        definitions.publish(
            scope(),
            compiled.definition.agent_id,
            compiled.definition.version,
            validation_report=compiled.validation_report,
            evaluation_report=compiled.evaluation_report,
            reviewer="reviewer",
        )


def test_skill_publication_requires_review_supports_rollback_and_revocation() -> None:
    selected_compiler, tools, *_ = compiler()
    first = selected_compiler.compile(request(), caller_scope=scope()).skill
    second = replace(
        first,
        version="1.1.0",
        instructions=first.instructions + "\n- Prefer concise answers.",
    )
    store = InMemorySkillPublicationStore()
    affected: list[tuple[str, str]] = []

    def affected_definitions(
        _scope: ExecutionScope,
        reference: VersionedReference,
    ) -> tuple[str, ...]:
        affected.append((reference.name, reference.version))
        return ("support-agent@1.0.0",)

    manager = SkillPublicationManager(
        store,
        tools=tools,
        affected_definitions=affected_definitions,
    )
    for skill in (first, second):
        submitted = manager.submit(scope(), skill)
        assert submitted.status is SkillLifecycleStatus.AWAITING_REVIEW
        manager.publish(
            scope(),
            skill.name,
            skill.version,
            reviewer="reviewer",
        )
    assert store.active(scope(), first.name).reference == second.reference
    manager.rollback(scope(), first.reference, approved_by="operator")
    assert store.active(scope(), first.name).reference == first.reference
    history = store.activation_history(scope(), first.name)
    assert [entry.reason for entry in history] == [
        "publication",
        "publication",
        "rollback",
    ]
    assert history[-1].activated_by == "operator"
    assert history[-1].previous_reference == second.reference

    revoked = manager.revoke(
        scope(),
        first.name,
        first.version,
        revoked_by="security",
        reason="superseded unsafe guidance",
    )
    assert revoked.affected_definitions == ("support-agent@1.0.0",)
    assert affected == [(first.name, first.version)]
    with pytest.raises(PublicationError, match="revoked"):
        manager.resolve_for_run(scope(), first.reference)
    assert store.get(scope(), first.name, first.version).skill.digest == first.digest


def test_skill_publication_is_scoped_and_rejects_unpublished_includes() -> None:
    selected_compiler, tools, *_ = compiler()
    compiled = selected_compiler.compile(request(), caller_scope=scope())
    store = InMemorySkillPublicationStore()
    manager = SkillPublicationManager(store, tools=tools)
    missing = VersionedReference(
        name="missing",
        version="1.0.0",
        digest="sha256:" + ("1" * 64),
    )
    with_include = replace(compiled.skill, includes=(missing,))
    submitted = manager.submit(scope(), with_include)
    assert submitted.status is SkillLifecycleStatus.VALIDATING
    assert submitted.validation_report is not None
    assert submitted.validation_report.findings[0].code == "include_unavailable"
    with pytest.raises(KeyError):
        store.get(
            scope(tenant_id="tenant-b"),
            compiled.skill.name,
            compiled.skill.version,
        )
    with pytest.raises(ValueError, match="cannot include itself"):
        replace(
            compiled.skill,
            includes=(
                VersionedReference(
                    name=compiled.skill.name,
                    version="9.9.9",
                    digest="sha256:" + ("9" * 64),
                ),
            ),
        )


@pytest.mark.asyncio
async def test_async_publication_stores_match_sync_lifecycle_contract() -> None:
    selected_compiler, tools, *_ = compiler()
    compiled = selected_compiler.compile(request(), caller_scope=scope())
    skill_store = AsyncInMemorySkillPublicationStore()
    skill_manager = AsyncSkillPublicationManager(skill_store, tools=tools)
    submitted = await skill_manager.submit(scope(), compiled.skill)
    assert submitted.status is SkillLifecycleStatus.AWAITING_REVIEW
    published = await skill_manager.publish(
        scope(),
        compiled.skill.name,
        compiled.skill.version,
        reviewer="reviewer",
    )
    assert (
        await skill_manager.resolve_for_run(scope(), published.reference)
    ).digest == compiled.skill.digest
    deprecated = await skill_manager.deprecate(
        scope(),
        compiled.skill.name,
        compiled.skill.version,
    )
    assert deprecated.status is SkillLifecycleStatus.DEPRECATED
    rolled_back = await skill_manager.rollback(
        scope(),
        published.reference,
        approved_by="operator",
    )
    assert rolled_back.reference == published.reference
    history = await skill_store.activation_history(
        scope(),
        compiled.skill.name,
    )
    assert [entry.reason for entry in history] == ["publication", "rollback"]
    with pytest.raises(KeyError):
        await skill_store.get(
            scope(tenant_id="tenant-b"),
            compiled.skill.name,
            compiled.skill.version,
        )
    await skill_manager.revoke(
        scope(),
        compiled.skill.name,
        compiled.skill.version,
        revoked_by="security",
        reason="test revocation",
    )
    with pytest.raises(PublicationError):
        await skill_manager.resolve_for_run(scope(), published.reference)

    definition_store = AsyncInMemoryAgentDefinitionStore()
    saved = await definition_store.save_draft(scope(), compiled.definition)
    assert saved.status is DefinitionStatus.DRAFT
    published_definition = await definition_store.publish(
        scope(),
        compiled.definition.agent_id,
        compiled.definition.version,
        validation_report=compiled.validation_report,
        evaluation_report=compiled.evaluation_report,
        reviewer="reviewer",
    )
    assert published_definition.status is DefinitionStatus.PUBLISHED


def test_same_definition_runs_locally_and_hosted_with_exact_identity(
    tmp_path: Path,
) -> None:
    compiled, runtime, _definitions, _skills = publish_package()
    local_llm = ScriptedLLMClient(
        [
            {
                "type": "final_answer",
                "content": "local definition ran",
            }
        ]
    )
    with runtime.create_local(
        scope=scope(run_id="local-resolution"),
        config=AgentConfig(project_root=tmp_path / "local"),
        llm=local_llm,
        expected_digest=compiled.definition.digest,
    ) as local:
        local_result = local.run_result("Check the definition.")
    assert local_result.content == "local definition ran"
    assert (
        local_result.extension_metadata["agent_definition"]["digest"]
        == compiled.definition.digest
    )

    hosted_root = tmp_path / "hosted"
    hosted_root.mkdir()
    hub = InMemoryServiceHub()
    hosted_llm = ScriptedLLMClient(
        [
            {
                "type": "final_answer",
                "content": "hosted definition ran",
            }
        ]
    )
    with runtime.create_hosted(
        scope=scope(run_id="hosted-run"),
        services=hub.services(),
        config=AgentConfig(project_root=hosted_root),
        llm=hosted_llm,
        expected_digest=compiled.definition.digest,
    ) as hosted:
        hosted_result = hosted.run_result("Check the definition.")
        assert hosted.execution_scope.agent_id == compiled.definition.agent_id
        assert (
            hosted.runtime.runtime_metadata["agent_definition"]["digest"]
            == compiled.definition.digest
        )
    assert hosted_result.content == "hosted definition ran"
    assert list(hosted_root.iterdir()) == []


@pytest.mark.asyncio
async def test_same_definition_runs_through_async_hosted_facade(
    tmp_path: Path,
) -> None:
    compiled, runtime, _definitions, _skills = publish_package()
    root = tmp_path / "async-hosted"
    root.mkdir()
    hub = InMemoryServiceHub()
    llm = ScriptedLLMClient(
        [
            {
                "type": "final_answer",
                "content": "async definition ran",
            }
        ]
    )
    async with await runtime.create_async_hosted(
        scope=scope(run_id="async-hosted-run"),
        services=hub.async_services(),
        config=AgentConfig(project_root=root),
        llm=llm,
        expected_digest=compiled.definition.digest,
    ) as hosted:
        result = await hosted.run_result("Check the definition.")
    assert result.content == "async definition ran"
    assert result.extension_metadata["agent_definition"]["version"] == "1.0.0"
    assert list(root.iterdir()) == []


def test_runtime_rejects_definition_digest_mismatch_and_dry_run_is_safe() -> None:
    compiled, runtime, _definitions, _skills = publish_package()
    with pytest.raises(PublicationError, match="digest"):
        runtime.resolve(
            scope(),
            expected_digest="sha256:" + ("f" * 64),
        )
    preview = runtime.dry_run(
        scope(),
        expected_digest=compiled.definition.digest,
    )
    assert preview["definition"]["digest"] == compiled.definition.digest
    assert preview["tools"] == ["catalog_lookup"]
    assert preview["skills"][0]["name"] == "support-agent-workflow"


def test_definition_store_requires_validation_and_evaluation_reports() -> None:
    selected_compiler, *_ = compiler()
    compiled = selected_compiler.compile(request(), caller_scope=scope())
    store = InMemoryAgentDefinitionStore()
    store.save_draft(scope(), compiled.definition)
    failed_validation = ValidationReport(valid=False)
    passed_evaluation = EvaluationReport(
        passed=True,
        cases=(
            EvaluationCaseResult(
                name="success",
                passed=True,
                detail="ok",
            ),
        ),
        evaluator="test",
    )
    with pytest.raises(PublicationError, match="invalid"):
        store.publish(
            scope(),
            compiled.definition.agent_id,
            compiled.definition.version,
            validation_report=failed_validation,
            evaluation_report=passed_evaluation,
            reviewer="reviewer",
        )

    with pytest.raises(PublicationError, match="validation report"):
        store.publish(
            scope(),
            compiled.definition.agent_id,
            compiled.definition.version,
            validation_report=replace(
                compiled.validation_report,
                artifact_digest="sha256:" + ("a" * 64),
            ),
            evaluation_report=compiled.evaluation_report,
            reviewer="reviewer",
        )


def test_catalogs_keep_exact_revisions_and_reject_portable_secrets() -> None:
    first_tool = replace(
        catalog_lookup,
        identity=ToolIdentity.from_schemas(
            "catalog_lookup",
            version="1.0.0",
            input_schema=catalog_lookup.args_schema,
            output_schema=catalog_lookup.output_schema,
        ),
    )
    second_tool = replace(
        catalog_lookup,
        identity=ToolIdentity.from_schemas(
            "catalog_lookup",
            version="2.0.0",
            input_schema=catalog_lookup.args_schema,
            output_schema=catalog_lookup.output_schema,
        ),
    )
    tools = ToolCatalog()
    first = tools.publish(first_tool)
    second = tools.publish(second_tool)

    assert tools.reference("catalog_lookup") == second.reference
    assert tools.resolve(first.reference).resolved_identity().version == "1.0.0"
    tools.revoke("catalog_lookup", "2.0.0")
    with pytest.raises(KeyError, match="active revision"):
        tools.reference("catalog_lookup")
    tools.activate(first.reference)
    assert tools.reference("catalog_lookup") == first.reference

    artifacts = ArtifactCatalog()
    with pytest.raises(ValueError, match="secret material"):
        artifacts.publish(
            name="unsafe",
            version="1.0.0",
            payload={"api_key": "raw-secret"},
        )
    with pytest.raises(ValueError, match="local paths"):
        artifacts.publish(
            name="unsafe-path",
            version="1.0.0",
            payload={"project_root": "/tmp/unsafe"},
        )


def test_portable_skill_rejects_executable_and_secret_shaped_content() -> None:
    provenance = DefinitionProvenance(author="operator")
    with pytest.raises(ValueError, match="cannot install packages"):
        PortableSkill(
            name="unsafe",
            version="1.0.0",
            description="unsafe",
            instructions="Run pip install dangerous-package",
            provenance=provenance,
        )
    selected_compiler, *_ = compiler()
    with pytest.raises(ValueError, match="credential values"):
        selected_compiler.compile(
            replace(request(), goal="Use api_key=secret"),
            caller_scope=scope(),
        )
    with pytest.raises(ValueError, match="executable content"):
        replace(request(), goal="```python\nprint('unsafe')\n```")
    with pytest.raises(ValueError, match="local paths"):
        replace(request(), goal="/Users/example/private-agent")
