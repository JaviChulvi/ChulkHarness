"""Architecture guards for the runtime compatibility facade."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import chulk.runtime as runtime_module
import pytest
from chulk._runtime.request import AgentAssemblyRequest


def test_runtime_facade_accepts_one_assembly_request():
    assert runtime_module.create_agent.__module__ == "chulk.runtime"
    assert list(inspect.signature(runtime_module.create_agent).parameters) == ["request"]


def test_runtime_facade_passes_compatibility_seams_to_assembly(monkeypatch):
    captured: dict[str, object] = {}
    assembled_agent = object()

    def fake_assemble(*args, **kwargs):
        captured["request"] = args[0]
        captured.update(kwargs)
        return assembled_agent

    agent_factory = object()
    bridge_factory = object()
    bridge_required = object()
    provider_path = object()
    monkeypatch.setattr(runtime_module, "assemble_agent", fake_assemble)
    monkeypatch.setattr(runtime_module, "Agent", agent_factory)
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", bridge_factory)
    monkeypatch.setattr(runtime_module, "_mcp_bridge_required", bridge_required)
    monkeypatch.setattr(runtime_module, "_mcp_provider_path", provider_path)
    request = AgentAssemblyRequest(config=object())  # type: ignore[arg-type]
    assert runtime_module.create_agent(request) is assembled_agent
    assert captured["agent_factory"] is agent_factory
    assert captured["bridge_tool_factory"] is bridge_factory
    assert captured["mcp_bridge_required"] is bridge_required
    assert captured["mcp_provider_path"] is provider_path
    assert "unresolved_tool_handler" not in captured
    assert captured["request"] is request


def test_assembly_request_rejects_conflicting_model_injections():
    with pytest.raises(ValueError, match="either llm_client or llm_client_factory"):
        AgentAssemblyRequest(
            config=object(),  # type: ignore[arg-type]
            llm_client=object(),  # type: ignore[arg-type]
            llm_client_factory=object(),  # type: ignore[arg-type]
        )


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
