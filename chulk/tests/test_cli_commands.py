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
