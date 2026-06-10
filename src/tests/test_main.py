"""Tests for the ChulkHarness CLI entrypoint."""

import json

from src import __version__
from src.llm import LLMClient
from src.main import main


class FakeLLMClient(LLMClient):
    def __init__(self, response: str = '{"type": "final_answer", "content": "mocked response"}') -> None:
        self.response = response
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.response


def fake_factory(_config):
    return FakeLLMClient('{"type": "final_answer", "content": "hello from fake llm"}')


def test_main_prints_current_status(capsys):
    inputs = iter(["/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "ChulkHarness CLI" in output
    assert "Type /exit, /quit, or /q" in output
    assert "bye" in output


def test_main_prints_version(capsys):
    exit_code = main(["--version"])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == f"ChulkHarness {__version__}"


def test_main_prints_resolved_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_MODEL", "test-model")
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    exit_code = main(["--show-config"])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "ChulkHarness configuration:" in output
    assert f"project_root: {tmp_path}" in output
    assert f"skills_dir: {tmp_path / 'skills'}" in output
    assert "llm_provider: deepseek" in output
    assert "model: test-model" in output
    assert "deepseek_api_key: set" in output


def test_main_runs_one_message_with_fake_llm(capsys):
    exit_code = main(["--once", "hello"], llm_client_factory=fake_factory)

    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "hello from fake llm"


def test_main_memory_persists_across_separate_agent_sessions(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    class MemoryAwareFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            system_prompt = messages[0]["content"]
            user_message = messages[-1]["content"]
            if user_message.startswith("What") and "separate-session memory marker is chartreuse" in system_prompt:
                return json.dumps({"type": "final_answer", "content": "The marker is chartreuse."})
            return json.dumps({"type": "final_answer", "content": "Stored."})

    def factory(_config):
        return MemoryAwareFakeLLM()

    first_exit = main(
        ["--once", "Please remember that separate-session memory marker is chartreuse."],
        llm_client_factory=factory,
    )
    first_output = capsys.readouterr().out
    second_exit = main(["--once", "What is the separate-session memory marker?"], llm_client_factory=factory)
    second_output = capsys.readouterr().out

    assert first_exit == 0
    assert first_output.strip() == "Stored."
    assert second_exit == 0
    assert second_output.strip() == "The marker is chartreuse."
    assert (tmp_path / "src" / "store.sqlite").exists()
    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 2
    assert any("memory_search_completed" in trace_file.read_text(encoding="utf-8") for trace_file in trace_files)


def test_main_reports_missing_openai_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(["--once", "hello"])

    output = capsys.readouterr().out

    assert exit_code == 1
    assert "configuration error" in output
    assert "OPENAI_API_KEY" in output
