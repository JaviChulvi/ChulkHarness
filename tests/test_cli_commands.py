"""Focused regressions for interactive command routing."""

from __future__ import annotations

import json

import pytest

from chulk.llm import LLMClient
from chulk.main import main


class RecordingLLMClient(LLMClient):
    def __init__(self) -> None:
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return json.dumps({"type": "final_answer", "content": "model handled prompt"})


@pytest.mark.parametrize(
    "prompt",
    [
        "help me inspect this repo",
        "exit interview questions",
        "quit smoking tips",
    ],
)
def test_bare_alias_prefixes_remain_model_prompts(prompt, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    client = RecordingLLMClient()
    inputs = iter([prompt, "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: client,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "model handled prompt" in output
    assert len(client.requests) == 1
    assert any(message["role"] == "user" and message["content"] == prompt for message in client.requests[0])


def test_exact_skill_slash_invocation_reaches_model_with_skill_loaded(
    monkeypatch,
    tmp_path,
    capsys,
):
    skills_root = tmp_path / ".chulk" / "skills"
    for name in ("review", "tests"):
        skill_root = skills_root / name
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            f"""\
---
schema_version: 1
name: {name}
version: 1.0.0
description: {name.title()} workflow.
---
# {name.title()}

Instructions for {name}.
""",
            encoding="utf-8",
        )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    client = RecordingLLMClient()
    inputs = iter(["/review /tests inspect this change", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: client,
    )

    output = capsys.readouterr().out
    system_prompt = client.requests[0][0]["content"]

    assert exit_code == 0
    assert "model handled prompt" in output
    assert len(client.requests) == 1
    assert "<name>review</name>" in system_prompt
    assert "<name>tests</name>" in system_prompt


def test_skill_lifecycle_commands_are_handled_without_model_calls(
    monkeypatch,
    tmp_path,
    capsys,
):
    skills_root = tmp_path / ".chulk" / "skills"
    skill_root = skills_root / "review"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        """\
---
schema_version: 1
name: review
version: 1.0.0
description: Review workflow.
source: project
trust: reviewed
---
# Review
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    client = RecordingLLMClient()
    inputs = iter(["/skills list", "/learning pending", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: client,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "project:review 1.0.0 active" in output
    assert "No pending proposals." in output
    assert client.requests == []


def test_invisible_skill_slash_invocation_is_rejected_before_model(
    monkeypatch,
    tmp_path,
    capsys,
):
    skill_root = tmp_path / ".chulk" / "skills" / "restricted"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        """\
---
schema_version: 1
name: restricted
version: 1.0.0
description: Restricted workflow.
required_tools: [unavailable_tool]
---
# Restricted
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    client = RecordingLLMClient()
    inputs = iter(["/restricted do work", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: client,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Skill /restricted is unavailable: missing_tools:unavailable_tool." in output
    assert client.requests == []


def test_skill_bundle_with_missing_include_is_rejected_before_model(
    monkeypatch,
    tmp_path,
    capsys,
):
    skill_root = tmp_path / ".chulk" / "skills" / "review"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        """\
---
schema_version: 1
name: review
version: 1.0.0
description: Review workflow.
includes: [missing]
---
# Review
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    client = RecordingLLMClient()
    inputs = iter(["/review inspect this", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: client,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Skill /missing is unavailable: unknown_skill." in output
    assert client.requests == []
