"""Stable dependency guards for the focused core runtime services."""

from __future__ import annotations

import ast
from pathlib import Path


CORE = Path(__file__).parents[1] / "core"


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
