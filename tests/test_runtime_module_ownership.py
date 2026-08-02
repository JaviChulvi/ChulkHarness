"""Architecture guards for the runtime compatibility facade."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import chulk.runtime as runtime_module


def test_runtime_facade_preserves_create_agent_signature():
    assert runtime_module.create_agent.__module__ == "chulk.runtime"
    assert list(inspect.signature(runtime_module.create_agent).parameters) == [
        "config",
        "llm_client_factory",
        "conversation_id",
        "conversation_metadata",
        "llm_client",
        "tool_specs",
        "skill_specs",
        "system_prompt",
        "permission_callback",
        "mcp_servers",
        "event_sink",
        "redaction_callback",
        "redaction_fail_closed",
        "final_answer_streaming",
        "output_policy",
        "async_output_policy",
        "output_policy_failure_mode",
        "capabilities",
        "deps",
        "shell_execution_policy",
        "require_shell_containment",
        "execution_backend",
        "memory_namespace",
        "profile_id",
        "allowed_skill_names",
        "runtime_metadata",
        "run_budget",
        "additional_run_budgets",
        "usage_dimensions",
        "learning_review_policy",
        "learning_review_quota",
        "automatic_learning_approval",
        "plugin_registry",
        "goal_execution",
        "content_store",
        "media_processors",
        "services",
        "execution_scope",
        "tool_catalog_resolver",
        "async_tool_catalog_resolver",
        "tool_catalog_timeout_seconds",
    ]


def test_runtime_facade_passes_compatibility_seams_to_assembly(monkeypatch):
    captured: dict[str, object] = {}
    assembled_agent = object()

    def fake_assemble(*args, **kwargs):
        captured.update(kwargs)
        return assembled_agent

    agent_factory = object()
    bridge_factory = object()
    bridge_required = object()
    provider_path = object()
    recovery_handler = object()
    monkeypatch.setattr(runtime_module, "assemble_agent", fake_assemble)
    monkeypatch.setattr(runtime_module, "Agent", agent_factory)
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", bridge_factory)
    monkeypatch.setattr(runtime_module, "_mcp_bridge_required", bridge_required)
    monkeypatch.setattr(runtime_module, "_mcp_provider_path", provider_path)
    monkeypatch.setattr(
        runtime_module,
        "_block_unresolved_tool_intent",
        recovery_handler,
    )

    assert runtime_module.create_agent(object()) is assembled_agent  # type: ignore[arg-type]
    assert captured["agent_factory"] is agent_factory
    assert captured["bridge_tool_factory"] is bridge_factory
    assert captured["mcp_bridge_required"] is bridge_required
    assert captured["mcp_provider_path"] is provider_path
    assert captured["unresolved_tool_handler"] is recovery_handler


def test_runtime_implementation_does_not_import_compatibility_facade():
    runtime_package = Path(runtime_module.__file__).resolve().parent / "_runtime"

    offenders: list[str] = []
    for path in sorted(runtime_package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "chulk.runtime":
                offenders.append(path.name)
            elif isinstance(node, ast.Import) and any(
                alias.name == "chulk.runtime" for alias in node.names
            ):
                offenders.append(path.name)

    assert offenders == []
