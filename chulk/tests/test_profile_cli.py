"""CLI compatibility and selection tests for agent profiles."""

from __future__ import annotations

import json

from chulk.config import Config, load_config
from chulk.llm import LLMClient
from chulk.main import main
from chulk.profiles import SQLiteProfileStore


class CLILLM(LLMClient):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps({"type": "final_answer", "content": "ok"})


def test_profile_cli_create_list_inspect_and_use(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    project = tmp_path / "work"
    project.mkdir()

    assert main(
        [
            "profile",
            "create",
            "work",
            "--project-root",
            str(project),
            "--permission-profile",
            "read-only",
            "--skill",
            "files",
            "--credential-env",
            "WORK_API_KEY",
            "--json",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["profile"]["id"] == "work"
    assert created["profile"]["credential_refs"] == [
        {"name": "WORK_API_KEY", "source": "environment"}
    ]
    assert "secret" not in created["profile"]

    assert main(["profile", "use", "work", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["profile_id"] == "work"

    assert main(["profile", "list", "--json"]) == 0
    profiles = json.loads(capsys.readouterr().out)["profiles"]
    assert [profile["id"] for profile in profiles] == ["default", "work"]
    assert next(profile for profile in profiles if profile["id"] == "work")["selected_for_cli"]

    assert main(["profile", "inspect", "work", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)["profile"]
    assert inspected["permission_profile"] == "read-only"
    assert inspected["selected_for_cli"]


def test_selected_profile_and_one_shot_override_use_isolated_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    work_project = tmp_path / "work"
    personal_project = tmp_path / "personal"
    work_project.mkdir()
    personal_project.mkdir()
    assert main(["profile", "create", "work", "--project-root", str(work_project)]) == 0
    capsys.readouterr()
    assert main(["profile", "create", "personal", "--project-root", str(personal_project)]) == 0
    capsys.readouterr()
    assert main(["profile", "use", "work"]) == 0
    capsys.readouterr()

    seen: list[Config] = []

    def factory(config: Config) -> LLMClient:
        seen.append(config)
        return CLILLM()

    assert main(["exec", "hello", "--json"], llm_client_factory=factory) == 0
    selected_payload = json.loads(capsys.readouterr().out)
    assert selected_payload["profile_id"] == "work"
    assert seen[-1].project_root == work_project

    assert main(
        ["exec", "--profile", "personal", "hello", "--json"],
        llm_client_factory=factory,
    ) == 0
    override_payload = json.loads(capsys.readouterr().out)
    assert override_payload["profile_id"] == "personal"
    assert seen[-1].project_root == personal_project

    base_config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteProfileStore(
        tmp_path / ".chulk" / "control.sqlite",
        base_config=base_config,
    )
    assert store.selected_cli_profile_id() == "work"
