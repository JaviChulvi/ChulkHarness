"""Tests for non-agent CLI maintenance commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from chulk.cli.maintenance import run_doctor
from chulk.main import main


def test_init_creates_safe_project_scaffolding_and_is_idempotent(tmp_path, capsys):
    project_root = tmp_path / "project"

    first_exit = main(["init", "--project-root", str(project_root), "--coding-agent", "--json"])
    first_payload = json.loads(capsys.readouterr().out)
    second_exit = main(["init", "--project-root", str(project_root), "--coding-agent", "--json"])
    second_payload = json.loads(capsys.readouterr().out)

    assert first_exit == 0
    assert second_exit == 0
    assert first_payload["ok"] is True
    assert (project_root / ".chulk" / "skills").is_dir()
    assert json.loads((project_root / ".chulk" / "mcp.json").read_text(encoding="utf-8")) == {"servers": []}
    env_example = (project_root / ".env.example").read_text(encoding="utf-8")
    assert "CHULK_PERMISSION_PROFILE=workspace-write" in env_example
    assert "API_KEY=\n" in env_example
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert all(
        entry in gitignore
        for entry in (".env", "!.env.example", ".chulk/", "traces/", "chulk/store.sqlite", "*.sqlite")
    )
    assert all(change["action"] == "exists" for change in second_payload["changes"])


def test_init_read_only_uses_safe_permission_default(tmp_path, capsys):
    project_root = tmp_path / "read-only-project"

    exit_code = main(["init", "--project-root", str(project_root), "--read-only"])

    output = capsys.readouterr().out
    env_example = (project_root / ".env.example").read_text(encoding="utf-8")

    assert exit_code == 0
    assert "Chulk initialized" in output
    assert "CHULK_PERMISSION_PROFILE=read-only" in env_example


def test_init_rejects_symlinked_target_before_writing(tmp_path, capsys):
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_gitignore = tmp_path / "outside.gitignore"
    outside_gitignore.write_text("keep-this-line\n", encoding="utf-8")
    (project_root / ".gitignore").symlink_to(outside_gitignore)

    exit_code = main(["init", "--project-root", str(project_root), "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["status"] == "init_error"
    assert "symlinked path" in payload["error"]
    assert outside_gitignore.read_text(encoding="utf-8") == "keep-this-line\n"
    assert not (project_root / ".chulk").exists()


def test_doctor_uses_current_directory_as_inferred_project_root(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(
        "CHULK_LLM_PROVIDER=local\nCHULK_MODEL=project-local-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    report = run_doctor(environ={})

    assert report.project_root == project_root.resolve()
    assert report.ok is True
    assert next(check for check in report.checks if check.name == "provider").detail == "local provider configured"
    assert "local/project-local-model" in next(
        check for check in report.checks if check.name == "model"
    ).detail


def test_doctor_json_reports_local_runtime_without_requiring_network(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "local")
    monkeypatch.setenv("CHULK_MODEL", "local-test-model")

    exit_code = main(["doctor", "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["project_root"] == str(tmp_path)
    assert {check["name"] for check in payload["checks"]} >= {
        "configuration",
        "provider",
        "model",
        "runtime",
        "mcp",
        "gitignore",
    }


def test_doctor_reports_effective_local_context_window(tmp_path):
    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "custom-local-model",
            "CHULK_LOCAL_CONTEXT_WINDOW_TOKENS": "32768",
        }
    )

    model_check = next(check for check in report.checks if check.name == "model")

    assert model_check.status == "pass"
    assert "32768 token context" in model_check.detail


def test_doctor_reports_invalid_configuration_without_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "invalid")

    exit_code = main(["doctor"])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "[fail] configuration" in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("config_path", [Path(".env"), Path(".chulk/mcp.json")])
def test_doctor_reports_configuration_read_errors(config_path, monkeypatch, tmp_path, capsys):
    unreadable_path = tmp_path / config_path
    unreadable_path.mkdir(parents=True)
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "local")
    monkeypatch.setenv("CHULK_MODEL", "local-test-model")

    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    configuration_check = payload["checks"][0]

    assert exit_code == 2
    assert captured.err == ""
    assert payload["ok"] is False
    assert configuration_check["name"] == "configuration"
    assert configuration_check["status"] == "fail"
    assert "Traceback" not in captured.out


def test_doctor_rejects_missing_fallback_credentials(tmp_path):
    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "local-test-model",
            "CHULK_LLM_FALLBACK_PROVIDERS": "openai:gpt-4.1-mini",
        }
    )

    provider_check = next(check for check in report.checks if check.name == "provider")

    assert report.ok is False
    assert provider_check.status == "fail"
    assert "fallback #1 openai" in provider_check.detail
    assert "OPENAI_API_KEY" in provider_check.detail


@pytest.mark.parametrize("as_fallback", [False, True])
@pytest.mark.parametrize(
    ("provider", "model", "environment", "expected_setting"),
    [
        ("openai", "gpt-4.1-mini", {}, "OPENAI_API_KEY"),
        (
            "deepseek",
            "deepseek-chat",
            {},
            "CHULK_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY",
        ),
        (
            "openai-compatible",
            "vendor/model",
            {"CHULK_OPENAI_COMPATIBLE_BASE_URL": "https://models.example/v1"},
            "CHULK_OPENAI_COMPATIBLE_API_KEY",
        ),
        (
            "openrouter",
            "vendor/model",
            {},
            "CHULK_OPENROUTER_API_KEY or OPENROUTER_API_KEY",
        ),
        (
            "anthropic",
            "claude-test",
            {},
            "CHULK_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY",
        ),
        (
            "bedrock",
            "vendor/model",
            {"CHULK_BEDROCK_BASE_URL": "https://bedrock.example/openai/v1"},
            "CHULK_BEDROCK_API_KEY, BEDROCK_API_KEY, or AWS_BEARER_TOKEN_BEDROCK",
        ),
        (
            "gemini",
            "gemini-test",
            {},
            "CHULK_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY",
        ),
    ],
)
def test_doctor_checks_credentials_for_every_provider_position(
    tmp_path,
    as_fallback,
    provider,
    model,
    environment,
    expected_setting,
):
    configured = {
        "CHULK_PROJECT_ROOT": str(tmp_path),
        "CHULK_LLM_PROVIDER": "local" if as_fallback else provider,
        "CHULK_MODEL": "local-test-model" if as_fallback else model,
        **environment,
    }
    if as_fallback:
        configured["CHULK_LLM_FALLBACK_PROVIDERS"] = f"{provider}:{model}"

    report = run_doctor(environ=configured)

    provider_check = next(check for check in report.checks if check.name == "provider")
    expected_label = "fallback #1" if as_fallback else "primary"
    assert provider_check.status == "fail"
    assert f"{expected_label} {provider}" in provider_check.detail
    assert expected_setting in provider_check.detail


@pytest.mark.parametrize("as_fallback", [False, True])
def test_doctor_checks_openai_compatible_endpoint_for_every_provider_position(
    tmp_path,
    as_fallback,
):
    configured = {
        "CHULK_PROJECT_ROOT": str(tmp_path),
        "CHULK_LLM_PROVIDER": "local" if as_fallback else "openai-compatible",
        "CHULK_MODEL": "local-test-model" if as_fallback else "vendor/model",
        "CHULK_OPENAI_COMPATIBLE_API_KEY": "test-key",
    }
    if as_fallback:
        configured["CHULK_LLM_FALLBACK_PROVIDERS"] = "openai-compatible:vendor/model"

    report = run_doctor(environ=configured)

    provider_check = next(check for check in report.checks if check.name == "provider")
    expected_label = "fallback #1" if as_fallback else "primary"
    assert provider_check.status == "fail"
    assert f"{expected_label} openai-compatible" in provider_check.detail
    assert "CHULK_OPENAI_COMPATIBLE_BASE_URL" in provider_check.detail


@pytest.mark.parametrize("as_fallback", [False, True])
@pytest.mark.parametrize(
    ("provider", "model", "environment", "expected_package", "expected_extra"),
    [
        ("openai", "gpt-4.1-mini", {"OPENAI_API_KEY": "test-key"}, "openai", "openai"),
        (
            "deepseek",
            "deepseek-chat",
            {"DEEPSEEK_API_KEY": "test-key"},
            "openai",
            "openai",
        ),
        ("local", "local-test-model", {}, "openai", "openai"),
        (
            "openai-compatible",
            "vendor/model",
            {
                "CHULK_OPENAI_COMPATIBLE_API_KEY": "test-key",
                "CHULK_OPENAI_COMPATIBLE_BASE_URL": "https://models.example/v1",
            },
            "openai",
            "openai",
        ),
        (
            "openrouter",
            "vendor/model",
            {"OPENROUTER_API_KEY": "test-key"},
            "openai",
            "openai",
        ),
        (
            "anthropic",
            "claude-test",
            {"ANTHROPIC_API_KEY": "test-key"},
            "anthropic",
            "anthropic",
        ),
        (
            "bedrock",
            "vendor/model",
            {
                "AWS_BEARER_TOKEN_BEDROCK": "test-key",
                "CHULK_BASE_URL": "https://bedrock.example/openai/v1",
            },
            "openai",
            "openai",
        ),
        (
            "gemini",
            "gemini-test",
            {"GOOGLE_API_KEY": "test-key"},
            "google-genai",
            "gemini",
        ),
    ],
)
def test_doctor_checks_sdk_package_for_every_provider_position(
    monkeypatch,
    tmp_path,
    as_fallback,
    provider,
    model,
    environment,
    expected_package,
    expected_extra,
):
    monkeypatch.setattr("chulk.cli.maintenance.find_spec", lambda _module_name: None)
    fallback_primary_provider = "anthropic" if provider == "local" else "local"
    fallback_primary_model = "claude-test" if provider == "local" else "local-test-model"
    configured = {
        "CHULK_PROJECT_ROOT": str(tmp_path),
        "CHULK_LLM_PROVIDER": fallback_primary_provider if as_fallback else provider,
        "CHULK_MODEL": fallback_primary_model if as_fallback else model,
        **environment,
    }
    if as_fallback:
        configured["CHULK_LLM_FALLBACK_PROVIDERS"] = f"{provider}:{model}"
        if provider == "local":
            configured["ANTHROPIC_API_KEY"] = "test-key"

    report = run_doctor(environ=configured)

    provider_check = next(check for check in report.checks if check.name == "provider")
    expected_label = "fallback #1" if as_fallback else "primary"
    assert report.ok is False
    assert provider_check.status == "fail"
    assert f"{expected_label} {provider}" in provider_check.detail
    assert expected_package in provider_check.detail
    assert f"python -m pip install -e '.[{expected_extra}]'" in provider_check.detail


def test_doctor_rejects_blank_provider_credentials_and_endpoints(tmp_path):
    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openai-compatible",
            "CHULK_MODEL": "vendor/model",
            "CHULK_OPENAI_COMPATIBLE_API_KEY": "   ",
            "CHULK_OPENAI_COMPATIBLE_BASE_URL": "   ",
        }
    )

    provider_check = next(check for check in report.checks if check.name == "provider")
    assert provider_check.status == "fail"
    assert "CHULK_OPENAI_COMPATIBLE_API_KEY" in provider_check.detail
    assert "CHULK_OPENAI_COMPATIBLE_BASE_URL" in provider_check.detail


@pytest.mark.parametrize(
    ("provider", "model", "environment"),
    [
        ("deepseek", "deepseek-chat", {"DEEPSEEK_API_KEY": "test-key"}),
        ("openrouter", "vendor/model", {"OPENROUTER_API_KEY": "test-key"}),
        ("anthropic", "claude-test", {"ANTHROPIC_API_KEY": "test-key"}),
        (
            "bedrock",
            "vendor/model",
            {
                "AWS_BEARER_TOKEN_BEDROCK": "test-key",
                "CHULK_BASE_URL": "https://bedrock.example/openai/v1",
            },
        ),
        ("gemini", "gemini-test", {"GOOGLE_API_KEY": "test-key"}),
    ],
)
def test_doctor_accepts_documented_provider_aliases(
    monkeypatch,
    tmp_path,
    provider,
    model,
    environment,
):
    monkeypatch.setattr("chulk.cli.maintenance.find_spec", lambda _module_name: object())
    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": provider,
            "CHULK_MODEL": model,
            **environment,
        }
    )

    provider_check = next(check for check in report.checks if check.name == "provider")
    assert provider_check.status == "pass"


@pytest.mark.parametrize("as_fallback", [False, True])
def test_doctor_reports_missing_bedrock_endpoint_as_configuration_error(
    tmp_path,
    as_fallback,
):
    configured = {
        "CHULK_PROJECT_ROOT": str(tmp_path),
        "CHULK_LLM_PROVIDER": "local" if as_fallback else "bedrock",
        "CHULK_MODEL": "local-test-model" if as_fallback else "vendor/model",
        "CHULK_BEDROCK_API_KEY": "test-key",
    }
    if as_fallback:
        configured["CHULK_LLM_FALLBACK_PROVIDERS"] = "bedrock:vendor/model"

    report = run_doctor(environ=configured)

    assert report.ok is False
    assert report.checks[0].name == "configuration"
    assert report.checks[0].status == "fail"
    assert "CHULK_BEDROCK_BASE_URL or CHULK_BASE_URL" in report.checks[0].detail


def test_doctor_rejects_unregistered_fallback_model(tmp_path):
    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "local-test-model",
            "CHULK_LLM_FALLBACK_PROVIDERS": "openai:unknown-model",
            "OPENAI_API_KEY": "test-key",
        }
    )

    model_check = next(check for check in report.checks if check.name == "model")

    assert report.ok is False
    assert model_check.status == "fail"
    assert "fallback #1 openai/unknown-model" in model_check.detail


def test_doctor_rejects_trace_destination_that_is_a_file(tmp_path):
    (tmp_path / "traces").write_text("not a directory", encoding="utf-8")

    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "local-test-model",
        }
    )

    runtime_check = next(check for check in report.checks if check.name == "runtime")

    assert report.ok is False
    assert runtime_check.status == "fail"
    assert "trace directory is not a directory" in runtime_check.detail


def test_doctor_rejects_unwritable_store_parent(monkeypatch, tmp_path):
    store_parent = tmp_path / "chulk"
    store_parent.mkdir()
    real_access = os.access

    def fake_access(path, mode):
        if Path(path) == store_parent:
            return False
        return real_access(path, mode)

    monkeypatch.setattr("chulk.cli.maintenance.os.access", fake_access)

    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "local-test-model",
        }
    )

    runtime_check = next(check for check in report.checks if check.name == "runtime")

    assert report.ok is False
    assert runtime_check.status == "fail"
    assert "SQLite store directory is not writable" in runtime_check.detail


def test_doctor_reports_ignored_runtime_state_already_tracked_by_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        ".chulk/\ntraces/\nchulk/store.sqlite\n*.sqlite\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "chulk" / "store.sqlite"
    store_path.parent.mkdir()
    store_path.write_bytes(b"sqlite state")
    subprocess.run(
        ["git", "add", ".gitignore"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "add", "-f", "chulk/store.sqlite"],
        cwd=tmp_path,
        check=True,
    )

    report = run_doctor(
        environ={
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "local-test-model",
        }
    )

    gitignore_check = next(check for check in report.checks if check.name == "gitignore")

    assert report.ok is False
    assert gitignore_check.status == "fail"
    assert "already tracked by Git" in gitignore_check.detail
    assert "chulk/store.sqlite" in gitignore_check.detail


def test_trace_inspect_and_export_are_machine_readable_and_escape_html(tmp_path, capsys):
    trace_path = tmp_path / "conversation-1.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "turn_started",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "payload": {"turn": {"turn_id": "turn-1"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "final_answer",
                        "created_at": "2026-01-01T00:00:01+00:00",
                        "payload": {"content": "<script>alert('x')</script>"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_finished",
                        "created_at": "2026-01-01T00:00:02+00:00",
                        "payload": {
                            "agent_state": {"conversation_id": "conversation-1"},
                            "turn": {"model_usage_totals": {"usage": {"total_tokens": 12}}},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inspect_exit = main(["trace", "inspect", str(trace_path), "--json"])
    summary = json.loads(capsys.readouterr().out)
    html_path = tmp_path / "report.html"
    export_exit = main(
        ["trace", "export", str(trace_path), "--output", str(html_path), "--json"]
    )
    export_payload = json.loads(capsys.readouterr().out)

    assert inspect_exit == 0
    assert export_exit == 0
    assert summary["conversation_id"] == "conversation-1"
    assert summary["event_count"] == 3
    assert summary["turn_count"] == 1
    assert export_payload["output_path"] == str(html_path)
    html = html_path.read_text(encoding="utf-8")
    assert "&lt;script&gt;alert" in html
    assert "<script>alert" not in html
    assert "Trace exports may contain sensitive runtime data" in html


def test_trace_commands_report_malformed_input_cleanly(tmp_path, capsys):
    trace_path = tmp_path / "broken.jsonl"
    trace_path.write_text("not json\n", encoding="utf-8")

    exit_code = main(["trace", "inspect", str(trace_path)])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "trace error:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["inspect", "export"])
def test_trace_commands_report_non_utf8_input_as_structured_error(command, tmp_path, capsys):
    trace_path = tmp_path / "non-utf8.jsonl"
    trace_path.write_bytes(b"\xff\xfe\x00")

    exit_code = main(["trace", command, str(trace_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["status"] == "trace_error"
    assert "not valid UTF-8" in payload["error"]


def test_trace_export_rejects_overwriting_source_with_force(tmp_path, capsys):
    trace_path = tmp_path / "conversation.jsonl"
    original = (
        json.dumps(
            {
                "type": "turn_started",
                "created_at": "2026-01-01T00:00:00+00:00",
                "payload": {"turn": {"turn_id": "turn-1"}},
            }
        )
        + "\n"
    )
    trace_path.write_text(original, encoding="utf-8")

    exit_code = main(
        [
            "trace",
            "export",
            str(trace_path),
            "--output",
            str(trace_path),
            "--force",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["status"] == "trace_error"
    assert "cannot overwrite the source trace" in payload["error"]
    assert trace_path.read_text(encoding="utf-8") == original
