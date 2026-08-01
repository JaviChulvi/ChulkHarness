"""CLI coverage for named model management and interactive switching."""

from __future__ import annotations

import json
from pathlib import Path

from chulk.config import load_config
from chulk.llm import LLMClient
from chulk.main import main
from chulk.model_profiles import ModelProfileStore


class _FakeCLIModel(LLMClient):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {"type": "final_answer", "content": "selected model response"}
        )


def _fake_factory(_config):
    return _FakeCLIModel()


def _run(
    argv: list[str],
    *,
    outputs: list[str],
    errors: list[str],
) -> int:
    return main(
        argv,
        output_func=outputs.append,
        error_func=errors.append,
        llm_client_factory=_fake_factory,
    )


def test_model_cli_create_list_use_inspect_health_and_reset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("FAST_MODEL_KEY", "not-persisted")
    outputs: list[str] = []
    errors: list[str] = []

    assert (
        _run(
            [
                "model",
                "create",
                "fast-coding",
                "--provider",
                "openai",
                "--model",
                "gpt-4.1-mini",
                "--credential-ref",
                "env:FAST_MODEL_KEY",
                "--max-cost-per-turn",
                "0.25",
                "--json",
            ],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    created = json.loads(outputs.pop())
    assert created["profile"]["id"] == "fast-coding"
    assert created["profile"]["credential_ref"] == "env:FAST_MODEL_KEY"
    assert created["profile"]["max_cost_per_turn"] == "0.25"

    assert (
        _run(
            ["model", "use", "fast-coding", "--channel", "cli", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    assert json.loads(outputs.pop())["channel"] == "cli"

    assert (
        _run(
            ["model", "list", "--channel", "cli", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    listed = json.loads(outputs.pop())
    assert next(item for item in listed["profiles"] if item["id"] == "fast-coding")[
        "selected"
    ]

    assert (
        _run(
            ["model", "inspect", "fast-coding", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    inspected = json.loads(outputs.pop())
    assert inspected["diagnostic"]["category"] == "ready"
    assert inspected["diagnostic"]["details"]["network_used"] is False

    assert (
        _run(
            ["model", "health", "fast-coding", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    assert json.loads(outputs.pop())["health"][0]["status"] == "healthy"

    assert (
        _run(
            ["model", "reset", "fast-coding", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    assert json.loads(outputs.pop())["status"] == "reset"
    assert errors == []


def test_exec_override_reports_model_selection_without_persisting_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("FAST_MODEL_KEY", "available")
    outputs: list[str] = []
    errors: list[str] = []
    assert (
        _run(
            [
                "model",
                "create",
                "fast",
                "--provider",
                "openai",
                "--model",
                "gpt-4.1-mini",
                "--credential-ref",
                "env:FAST_MODEL_KEY",
            ],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    outputs.clear()

    assert (
        _run(
            [
                "exec",
                "hello",
                "--model-profile",
                "fast",
                "--json",
            ],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    payload = json.loads(outputs.pop())
    assert payload["model_selection"]["requested_profile_id"] == "fast"
    assert payload["model_selection"]["selected_profile_id"] == "fast"

    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    assert store.selected_for_agent("default", default="default", channel="cli") == (
        "default"
    )


def test_model_inspect_returns_failure_for_an_unavailable_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    outputs: list[str] = []
    errors: list[str] = []
    assert (
        _run(
            [
                "model",
                "create",
                "missing-key",
                "--provider",
                "openai",
                "--model",
                "gpt-4.1-mini",
                "--credential-ref",
                "env:KEY_THAT_IS_NOT_SET",
            ],
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    outputs.clear()

    assert (
        _run(
            ["model", "inspect", "missing-key", "--json"],
            outputs=outputs,
            errors=errors,
        )
        == 2
    )
    payload = json.loads(outputs.pop())
    assert payload["ok"] is False
    assert payload["diagnostic"]["category"] == "missing_credential"


def test_interactive_model_command_switches_and_persists_channel_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("FAST_MODEL_KEY", "available")
    setup_output: list[str] = []
    assert (
        main(
            [
                "model",
                "create",
                "fast",
                "--provider",
                "openai",
                "--model",
                "gpt-4.1-mini",
                "--credential-ref",
                "env:FAST_MODEL_KEY",
            ],
            output_func=setup_output.append,
            error_func=setup_output.append,
            llm_client_factory=_fake_factory,
        )
        == 0
    )
    inputs = iter(["/model fast", "/model", "hello", "/q"])
    outputs: list[str] = []
    errors: list[str] = []

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        output_func=outputs.append,
        error_func=errors.append,
        llm_client_factory=_fake_factory,
    )

    assert exit_code == 0
    assert any("model profile fast" in output for output in outputs)
    assert any("selected model response" in output for output in outputs)
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    assert store.selected_for_agent("default", default="default", channel="cli") == (
        "fast"
    )
    assert errors == []
