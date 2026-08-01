"""Tests for plugin lifecycle operations at the public SDK boundary."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from chulk import (
    Agent,
    AgentConfig,
    AsyncAgent,
    ConfigurationError,
    LocalPluginRegistry,
    PluginCategory,
    Plugins,
    inspect_plugin_directory,
    plugins,
)
from chulk.llm import LLMClient
from .test_plugin_manifests import write_plugin


class FakeLLMClient(LLMClient):
    provider = "test-provider"
    model = "test-model"

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps({"type": "final_answer", "content": "done"})


@pytest.fixture(autouse=True)
def isolate_sample_plugin_modules():
    for name in tuple(sys.modules):
        if name == "sample_plugin" or name.startswith("sample_plugin."):
            sys.modules.pop(name, None)
    yield
    for name in tuple(sys.modules):
        if name == "sample_plugin" or name.startswith("sample_plugin."):
            sys.modules.pop(name, None)


def build_agent(tmp_path, **kwargs) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient(),
        tools=[],
        skills=[],
        **kwargs,
    )


def test_plugin_types_and_static_inspection_are_public(tmp_path):
    package = write_plugin(tmp_path)

    inspected = inspect_plugin_directory(package)

    assert Plugins is plugins
    assert Plugins.LocalPluginRegistry is LocalPluginRegistry
    assert inspected.package.manifest.name == "sample-plugin"


def test_sync_facade_registers_audits_and_loads_reviewed_plugins(tmp_path):
    package = write_plugin(tmp_path)
    facade = build_agent(tmp_path)

    inspected = facade.inspect_plugin(package)
    registered = facade.register_local_plugin(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    listed = facade.list_plugins()
    report = facade.audit_plugins()
    loaded = facade.load_plugin_entry_point(
        "sample-plugin",
        PluginCategory.TOOL,
        "sample",
        available_capabilities=("files:read",),
    )

    assert inspected.compatible is True
    assert listed == (registered,)
    assert report.ok is True
    assert report.verified_plugins == ("sample-plugin",)
    assert facade.runtime.plugin_audit_report.ok is True
    assert callable(loaded.value)
    facade.close()


def test_sync_facade_exposes_managed_install_uninstall_and_rollback(
    tmp_path,
):
    package = write_plugin(tmp_path / "source")
    facade = build_agent(tmp_path)

    installed = facade.install_plugin(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    uninstalled = facade.uninstall_plugin(
        "sample-plugin",
        approved_by="operator",
    )
    restored = facade.rollback_plugin(
        "sample-plugin",
        approved_by="operator",
    )

    assert installed.action.value == "install"
    assert uninstalled.action.value == "uninstall"
    assert restored.action.value == "rollback"
    assert facade.audit_plugins().ok is True
    facade.close()


@pytest.mark.asyncio
async def test_async_facade_has_plugin_operation_parity(tmp_path):
    package = write_plugin(tmp_path)
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient(),
        tools=[],
        skills=[],
    )

    inspected = await facade.inspect_plugin(package)
    registered = await facade.register_local_plugin(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    report, listed = await asyncio.gather(
        facade.audit_plugins(),
        facade.list_plugins(),
    )

    assert inspected.compatible is True
    assert listed == (registered,)
    assert report.verified_plugins == ("sample-plugin",)
    await facade.close()


@pytest.mark.asyncio
async def test_async_facade_has_managed_lifecycle_parity(tmp_path):
    package = write_plugin(tmp_path / "source")
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient(),
        tools=[],
        skills=[],
    )

    installed = await facade.install_plugin(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    disabled = await facade.uninstall_plugin(
        "sample-plugin",
        approved_by="operator",
    )
    restored = await facade.rollback_plugin(
        "sample-plugin",
        approved_by="operator",
    )

    assert installed.action.value == "install"
    assert disabled.action.value == "uninstall"
    assert restored.action.value == "rollback"
    await facade.close()


def test_agent_startup_fails_closed_when_reviewed_plugin_changes(tmp_path):
    package = write_plugin(tmp_path)
    facade = build_agent(tmp_path)
    facade.register_local_plugin(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    facade.close()
    (package / "sample_plugin" / "tools.py").write_text(
        "def create_tool():\n    return 'changed'\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reviewed lock") as caught:
        build_agent(tmp_path)

    assert caught.value.details.extensions["operation"] == "construct"


def test_agent_rejects_registry_owned_by_another_profile(tmp_path):
    registry = LocalPluginRegistry(
        tmp_path / ".chulk",
        profile_id="another",
    )

    with pytest.raises(ConfigurationError, match="profile") as caught:
        build_agent(tmp_path, plugin_registry=registry)

    assert caught.value.details.extensions["operation"] == "construct"
