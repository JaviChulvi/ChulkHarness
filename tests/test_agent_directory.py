from __future__ import annotations

from pathlib import Path

import pytest

from chulk import (
    Agent,
    AgentCompiler,
    AgentConfig,
    AgentDirectory,
    ArtifactCatalog,
    ExecutionScope,
    PromptCatalog,
    Tool,
    ToolCatalog,
    ToolEffect,
    ToolPolicy,
)
from chulk.testing import ScriptedLLMClient
from chulk.main import main


@Tool(policy=ToolPolicy(effect=ToolEffect.READ))
def catalog_lookup(item: str) -> str:
    """Read one catalog item."""
    return item


def _write_agent(path: Path, *, instructions: str = "Use the catalog.") -> None:
    path.mkdir()
    (path / "instructions.md").write_text(instructions + "\n", encoding="utf-8")
    (path / "agent.toml").write_text(
        """[agent]
id = "support-agent"
version = "1.0.0"
goal = "Answer catalog questions."
tools = ["catalog_lookup"]
expected_outcomes = ["Return catalog data."]

[prompt]
name = "support-prompt"
version = "1.0.0"

[model_profile]
name = "support-model"
version = "1.0.0"

[approval_policy]
name = "support-approval"
version = "1.0.0"
""",
        encoding="utf-8",
    )


def _compiler() -> AgentCompiler:
    prompts = PromptCatalog()
    prompts.publish(name="support-prompt", version="1.0.0", content="Use the catalog.")
    models = ArtifactCatalog()
    models.publish(name="support-model", version="1.0.0", payload={"model": "test"})
    approvals = ArtifactCatalog()
    approvals.publish(
        name="support-approval", version="1.0.0", payload={"permission_profile": "read-only"}
    )
    return AgentCompiler(
        tools=ToolCatalog((catalog_lookup,)),
        prompts=prompts,
        model_profiles=models,
        approval_policies=approvals,
    )


def _scope() -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant",
        workspace_id="workspace",
        actor_id="operator",
        agent_id="support-agent",
        agent_version="1.0.0",
        run_id="run-1",
    )


def test_directory_compiles_only_with_matching_published_prompt(tmp_path):
    path = tmp_path / "support"
    _write_agent(path)

    package = AgentDirectory.load(path).compile(_compiler(), caller_scope=_scope())

    assert package.publishable
    assert package.definition.agent_id == "support-agent"
    (path / "instructions.md").write_text("Changed locally.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        AgentDirectory.load(path).compile(_compiler(), caller_scope=_scope())


def test_local_agent_directory_requires_declared_tools(tmp_path):
    path = tmp_path / "support"
    _write_agent(path)

    with pytest.raises(ValueError, match="catalog_lookup"):
        Agent.from_directory(
            path,
            config=AgentConfig(project_root=tmp_path),
            llm=ScriptedLLMClient([{"type": "final_answer", "content": "done"}]),
            tools=[],
            skills=[],
        )

    with Agent.from_directory(
        path,
        config=AgentConfig(project_root=tmp_path),
        llm=ScriptedLLMClient([{"type": "final_answer", "content": "done"}]),
        tools=[catalog_lookup],
        skills=[],
    ) as agent:
        assert agent.run("Hello") == "done"


def test_agent_directory_cli_initializes_and_checks_without_runtime_config(tmp_path):
    path = tmp_path / "support"
    output: list[str] = []

    assert main(
        ["agent", "init", str(path), "--id", "support-agent", "--json"],
        output_func=output.append,
    ) == 0
    assert main(
        ["agent", "check", str(path), "--json"], output_func=output.append
    ) == 0
    assert '"status": "valid"' in output[-1]
