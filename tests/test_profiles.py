"""Tests for durable profile ownership and runtime isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chulk.config import load_config
from chulk._sdk.events import project_event
from chulk.core import TraceEvent
from chulk.llm import LLMClient
from chulk.profiles import (
    AgentProfile,
    CredentialRef,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    ProfileOwnershipError,
    ProfileRuntimeFactory,
    SQLiteProfileStore,
)
from tests.core_agent import create_runtime_agent as create_agent


class ProfileLLM(LLMClient):
    provider = "test"
    model = "profile"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps({"type": "final_answer", "content": "profile response"})


def _store(tmp_path: Path) -> tuple[SQLiteProfileStore, object]:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    return SQLiteProfileStore(config.runtime_dir / "control.sqlite", base_config=config), config


def test_implicit_default_profile_preserves_legacy_paths(tmp_path):
    store, config = _store(tmp_path)

    profile = store.get("default").profile

    assert profile.implicit
    assert profile.project_root == config.project_root
    assert profile.runtime_dir == config.runtime_dir
    assert profile.store_path == config.store_path
    assert profile.traces_dir == config.traces_dir
    assert profile.memory_namespace is None


def test_explicit_profiles_get_unique_owner_controlled_runtime_paths(tmp_path):
    store, config = _store(tmp_path)
    project = tmp_path / "work-project"
    project.mkdir()

    created = store.create_profile(
        "Work_Agent",
        project_root=project,
        allowed_skills=("files", "files", "shell"),
        allowed_mcp_servers=(),
        credential_refs=(CredentialRef("WORK_API_KEY"),),
    )

    profile = created.profile
    assert profile.id == "work_agent"
    assert profile.project_root == project
    assert profile.runtime_dir == config.runtime_dir / "profiles" / "work_agent"
    assert profile.store_path == profile.runtime_dir / "store.sqlite"
    assert profile.traces_dir == profile.runtime_dir / "traces"
    assert profile.memory_namespace == "profile:work_agent"
    assert profile.allowed_skills == ("files", "shell")
    assert profile.allowed_mcp_servers == ()
    assert profile.credential_refs[0].to_dict() == {
        "name": "WORK_API_KEY",
        "source": "environment",
    }


def test_profile_ids_are_immutable_and_persistent_paths_cannot_overlap(tmp_path):
    store, config = _store(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    store.create_profile("work", project_root=project)

    with pytest.raises(ProfileAlreadyExistsError):
        store.create_profile("work", project_root=project)

    existing = store.get("work").profile
    with pytest.raises(ProfileOwnershipError):
        store.create(
            AgentProfile(
                id="other",
                project_root=project,
                runtime_dir=existing.runtime_dir,
                store_path=existing.store_path,
                traces_dir=existing.traces_dir,
            )
        )

    with pytest.raises(ProfileOwnershipError):
        store.create(
            AgentProfile(
                id="outside",
                project_root=project,
                runtime_dir=tmp_path / "outside",
                store_path=tmp_path / "outside" / "store.sqlite",
                traces_dir=tmp_path / "outside" / "traces",
            )
        )
    assert config.store_path != existing.store_path


def test_cli_default_selection_is_local_and_explicit_resolution_overrides_it(tmp_path):
    store, config = _store(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    store.create_profile("work", project_root=project)

    assert store.selected_cli_profile_id() == "default"
    store.use("work")

    assert store.selected_cli_profile_id() == "work"
    assert store.resolve().profile.id == "work"
    assert store.resolve("default").profile.id == "default"
    factory = ProfileRuntimeFactory(config, profile_store=store)
    assert factory.resolve().profile.id == "default"
    assert factory.resolve_cli().profile.id == "work"
    with pytest.raises(ProfileNotFoundError):
        store.use("missing")


def test_profile_runtime_factory_isolates_databases_traces_and_project_skills(tmp_path):
    base_config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteProfileStore(
        base_config.runtime_dir / "control.sqlite",
        base_config=base_config,
    )
    project = tmp_path / "profile-project"
    project.mkdir()
    store.create_profile(
        "work",
        project_root=project,
        permission_profile="read-only",
        allowed_skills=(),
        allowed_mcp_servers=(),
        system_prompt="Profile-owned instructions.",
    )
    factory = ProfileRuntimeFactory(base_config, profile_store=store)

    resolved = factory.resolve("work")

    assert resolved.config.profile_id == "work"
    assert resolved.config.project_root == project
    assert resolved.config.store_path == base_config.runtime_dir / "profiles" / "work" / "store.sqlite"
    assert resolved.config.store_path != base_config.store_path
    assert resolved.config.traces_dir != base_config.traces_dir
    assert resolved.config.skills_dir == project / ".chulk" / "skills"
    assert resolved.config.permission_profile == "read-only"


def test_profile_runtime_records_owned_sessions_and_emits_profile_events(tmp_path):
    base_config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteProfileStore(
        base_config.runtime_dir / "control.sqlite",
        base_config=base_config,
    )
    project = tmp_path / "profile-project"
    project.mkdir()
    stored = store.create_profile(
        "work",
        project_root=project,
        allowed_skills=(),
        system_prompt="Profile-owned instructions.",
    )
    factory = ProfileRuntimeFactory(base_config, profile_store=store)
    agent = factory.create_agent(
        "work",
        llm_client=ProfileLLM(),
        tool_specs=[],
        skill_specs=None,
        allowed_skill_names=("files",),
    )
    try:
        assert agent.run_turn("hello") == "profile response"
        conversation = agent._components.session_store.get_conversation(
            agent.state.conversation_id
        )
        event = project_event(
            agent,
            TraceEvent.TURN_STARTED,
            {"turn": {"user_message": "hello"}},
        )
    finally:
        agent.close()

    assert stored.profile.store_path.exists()
    assert not base_config.store_path.exists()
    assert conversation.metadata["profile_id"] == "work"
    assert agent.memory_context.store.namespace == "profile:work"
    assert agent.system_prompt == "Profile-owned instructions."
    assert agent.skill_context.registry.list_skills() == []
    assert event is not None
    assert event.profile_id == "work"


def test_profile_runtime_rejects_an_execution_backend_mismatch(tmp_path):
    base_config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteProfileStore(
        base_config.runtime_dir / "control.sqlite",
        base_config=base_config,
    )
    project = tmp_path / "profile-project"
    project.mkdir()
    store.create_profile("work", project_root=project)
    factory = ProfileRuntimeFactory(base_config, profile_store=store)

    class WrongBackend:
        name = "docker"

    with pytest.raises(ValueError, match="does not match"):
        factory.create_agent(
            "work",
            llm_client=ProfileLLM(),
            tool_specs=[],
            skill_specs=[],
            execution_backend=WrongBackend(),
        )


def test_create_agent_rejects_conflicting_profile_metadata(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    with pytest.raises(ValueError, match="profile_id"):
        create_agent(
            config,
            llm_client=ProfileLLM(),
            tool_specs=[],
            skill_specs=[],
            profile_id="work",
            conversation_metadata={"profile_id": "personal"},
        )


def test_credential_references_validate_names_without_accepting_values():
    assert CredentialRef("OPENAI_API_KEY").source == "environment"
    with pytest.raises(ValueError):
        CredentialRef("not an environment variable")


def test_profile_project_root_must_exist(tmp_path):
    store, _config = _store(tmp_path)

    with pytest.raises(ValueError, match="existing directory"):
        store.create_profile("missing", project_root=tmp_path / "missing")
