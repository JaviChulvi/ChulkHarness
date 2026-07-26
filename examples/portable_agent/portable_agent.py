"""Compile, publish, and run one portable agent without credentials."""

from pathlib import Path
from tempfile import TemporaryDirectory

from chulk import (
    AgentCompiler,
    AgentConfig,
    AgentDefinitionRuntime,
    ArtifactCatalog,
    BudgetDefinition,
    CompilerRequest,
    ExecutionScope,
    InMemoryAgentDefinitionStore,
    InMemorySkillPublicationStore,
    PromptCatalog,
    SkillPublicationManager,
    Tool,
    ToolApprovalMode,
    ToolCatalog,
    ToolEffect,
    ToolPolicy,
    TriggerDefinition,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.testing import ScriptedLLMClient


CATALOG_POLICY = ToolPolicy(
    version="1.0.0",
    required_grants=frozenset({"catalog:read"}),
    effect=ToolEffect.READ,
    approval=ToolApprovalMode.NEVER,
)


@Tool(policy=CATALOG_POLICY)
def catalog_lookup(item_id: str) -> str:
    """Return deterministic catalog availability."""
    return f"{item_id}: available"


scope = ExecutionScope(
    tenant_id="example",
    workspace_id="support",
    actor_id="default",
    agent_id="support-agent",
    agent_version="1.0.0",
    run_id="portable-example",
    grants=frozenset({"catalog:read"}),
)

tools = ToolCatalog((catalog_lookup,))
prompts = PromptCatalog()
prompt = prompts.publish(
    name="support-prompt",
    version="1.0.0",
    content="Answer only from approved catalog evidence.",
)
model_profiles = ArtifactCatalog()
model_profile = model_profiles.publish(
    name="support-model",
    version="1.0.0",
    payload={"provider": "openai", "model": "gpt-5-mini"},
)
approval_policies = ArtifactCatalog()
approval_policy = approval_policies.publish(
    name="read-only",
    version="1.0.0",
    payload={"permission_profile": "read-only"},
)

compiler = AgentCompiler(
    tools=tools,
    prompts=prompts,
    model_profiles=model_profiles,
    approval_policies=approval_policies,
)
package = compiler.compile(
    CompilerRequest(
        agent_id="support-agent",
        version="1.0.0",
        goal="Answer catalog availability questions.",
        prompt=prompt.reference,
        model_profile=model_profile.reference,
        approval_policy=approval_policy.reference,
        selected_tools=("catalog_lookup",),
        triggers=(TriggerDefinition(kind="api.request"),),
        expected_outcomes=("Return an evidence-backed availability answer.",),
        budget=BudgetDefinition(
            max_model_requests=4,
            max_tool_calls=4,
            max_input_tokens=400_000,
            max_output_tokens=10_000,
        ),
        created_at="2026-01-02T03:04:05+00:00",
    ),
    caller_scope=scope,
)
assert package.publishable

skill_store = InMemorySkillPublicationStore()
skills = SkillPublicationManager(skill_store, tools=tools)
skills.submit(scope, package.skill)
skills.publish(
    scope,
    package.skill.name,
    package.skill.version,
    reviewer="example-reviewer",
)

definitions = InMemoryAgentDefinitionStore()
definitions.save_draft(scope, package.definition)
definitions.publish(
    scope,
    package.definition.agent_id,
    package.definition.version,
    validation_report=package.validation_report,
    evaluation_report=package.evaluation_report,
    reviewer="example-reviewer",
)

runtime = AgentDefinitionRuntime(
    definitions=definitions,
    tools=tools,
    prompts=prompts,
    skills=skills,
    model_profiles=model_profiles,
    approval_policies=approval_policies,
)
client = ScriptedLLMClient(
    [{"type": "final_answer", "content": "SKU-42 is available."}]
)

with TemporaryDirectory() as temporary_directory:
    project_root = Path(temporary_directory)
    with runtime.create_hosted(
        scope=scope,
        services=InMemoryServiceHub().services(),
        config=AgentConfig(project_root=project_root),
        llm=client,
        expected_digest=package.definition.digest,
    ) as agent:
        result = agent.run_result("Is SKU-42 available?")

    assert list(project_root.iterdir()) == []

assert result.extension_metadata["agent_definition"] == {
    "agent_id": "support-agent",
    "version": "1.0.0",
    "digest": package.definition.digest,
}
print(result.content)
print(package.preview.to_dict())
