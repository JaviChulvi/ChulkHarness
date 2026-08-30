"""Stable dependency guards for the focused core runtime services."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from chulk.core import Agent


CORE = Path(__file__).resolve().parents[1] / "src" / "chulk" / "core"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_transition_reducer_has_only_pure_core_dependencies() -> None:
    imports = _imports(CORE / "transitions.py")
    forbidden = (
        "chulk.core.agent",
        "chulk.llm",
        "chulk.memory",
        "chulk.storage",
        "chulk.tools",
        "chulk.tracing",
    )

    assert not any(name.startswith(forbidden) for name in imports)


def test_action_loop_uses_the_runtime_port_without_agent_coupling() -> None:
    path = CORE / "action_loop.py"
    imports = _imports(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert "chulk.core.action_runtime" in imports
    assert "chulk.core.agent" not in imports
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "agent"
        for node in ast.walk(tree)
    )


def test_action_loop_services_do_not_import_agent() -> None:
    service_files = [
        "action_runtime.py",
        "model_transport.py",
        "tool_execution.py",
        "plan_execution.py",
        "turn_effects.py",
    ]

    for filename in service_files:
        assert "chulk.core.agent" not in _imports(CORE / filename), filename


def test_agent_accepts_one_component_container() -> None:
    assert list(inspect.signature(Agent).parameters) == ["components"]


def test_agent_does_not_reclaim_runtime_owner_methods() -> None:
    forbidden = {
        "_transcript_request",
        "_resolve_external_transcript",
        "_resolve_external_transcript_async",
        "_validate_external_transcript",
        "_apply_external_transcript",
        "_revalidate_external_transcript_for_resume",
        "_revalidate_external_transcript_for_resume_async",
        "_assert_external_transcript_digest",
        "_tool_catalog_request",
        "_resolve_catalog_for_turn",
        "_resolve_catalog_for_turn_async",
        "_activate_tool_catalog",
        "_revalidate_tool_catalog",
        "_revalidate_tool_catalog_async",
        "_assert_catalog_digest",
        "_extract_long_term_memories",
        "_extract_long_term_memories_async",
        "_select_long_term_memories",
        "_select_long_term_memories_async",
        "_select_skills",
        "_select_skills_async",
        "_record_selected_skill_usage",
        "_record_selected_skill_usage_async",
        "_selected_skill_scope",
        "confirm_skill_success",
        "confirm_skill_success_async",
        "review_learning",
        "review_learning_async",
        "_record_model_accounting",
        "_record_model_accounting_async",
        "_reserve_model_accounting",
        "_reserve_model_accounting_async",
        "_release_model_accounting",
        "_release_model_accounting_async",
        "_redact_text",
        "_redact_event_payload",
        "_tool_context_for_turn",
        "_tool_context_for_turn_async",
        "_release_tool_context",
        "_release_tool_context_async",
        "_write_tool_output_artifact",
        "_write_tool_output_artifact_async",
        "_flush_async_services",
        "_flush_async_services_after_error",
        "_refresh_action_runtime",
    }

    assert forbidden.isdisjoint(Agent.__dict__)


def test_runtime_assembly_does_not_wire_agent_after_construction() -> None:
    path = CORE.parent / "_runtime" / "assembly.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assigned_agent_attributes = {
        target.attr
        for node in ast.walk(tree)
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "agent"
    }

    assert assigned_agent_attributes == set()
