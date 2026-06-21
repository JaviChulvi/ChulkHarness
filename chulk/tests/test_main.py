"""Tests for the ChulkHarness CLI entrypoint."""

import json
import re
import sqlite3
from types import SimpleNamespace

from chulk import __version__
from chulk.config import load_config
from chulk.llm import (
    DeepSeekProvider,
    FallbackChain,
    LLMCapabilities,
    LLMClient,
    LocalProvider,
    OpenAIProvider,
    OpenAIResponsesClient,
)
from chulk.llm.capabilities import LLMModelCapabilities, register_model_capabilities
from chulk.main import create_cli_llm, main
from chulk.sessions import SQLiteSessionStore


class FakeLLMClient(LLMClient):
    def __init__(self, response: str = '{"type": "final_answer", "content": "mocked response"}') -> None:
        self.response = response
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.response


def fake_factory(_config):
    return FakeLLMClient('{"type": "final_answer", "content": "hello from fake llm"}')


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_main_prints_current_status(capsys):
    inputs = iter(["/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "ChulkHarness CLI" in output
    assert "Type /exit, /quit, or /q" in output
    assert "bye" in output
    assert "/resume " in output


def test_main_exit_prints_resume_command_for_current_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["hello", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = strip_ansi(capsys.readouterr().out)
    session = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite").list_conversations()[0]

    assert exit_code == 0
    assert "Resume this session next time with:" in output
    assert f"/resume {session.id}" in output


def test_main_uses_compact_prompt(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    prompts = []
    inputs = iter(["/q"])

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    exit_code = main(
        [],
        input_func=input_func,
        llm_client_factory=fake_factory,
    )

    assert exit_code == 0
    assert [strip_ansi(prompt) for prompt in prompts] == ["> "]
    assert "chulk >" not in strip_ansi(prompts[0])


def test_main_uses_hulk_green_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "\033[1;38;2;63;255;81m" in output
    assert "ChulkHarness CLI" in output


def test_main_handles_interactive_slash_commands(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/help", "/status", "/tools", "/mcp", "/trace", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "Commands" in output
    assert "/status" in output
    assert "Status" in output
    assert "provider" in output
    assert "permissions" in output
    assert "Tools" in output
    assert "calculator" in output
    assert "MCP" in output
    assert "Trace" in output
    assert "bye" in output


def test_main_mcp_command_shows_configured_servers_with_redacted_auth(monkeypatch, tmp_path, capsys):
    mcp_dir = tmp_path / ".chulk"
    mcp_dir.mkdir()
    (mcp_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "label": "docs",
                        "transport": "streamable_http",
                        "server_url": "https://mcp.example.com",
                        "allowed_tools": ["search_docs"],
                        "authorization_env": "DOCS_MCP_TOKEN",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DOCS_MCP_TOKEN", "super-secret-token")
    inputs = iter(["/mcp", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=fake_factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "MCP" in output
    assert "docs" in output
    assert "path      hosted" in output
    assert "DOCS_MCP_TOKEN:set" in output
    assert "search_docs" in output
    assert "super-secret-token" not in output


def test_main_shows_live_progress_while_agent_works(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["what is 2 + 2?", "/q"])

    class ToolProgressFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "calculator",
                        "arguments_json": json.dumps({"expression": "2 + 2"}),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "The result is 4."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    def factory(_config):
        return ToolProgressFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert ".. starting turn" in output
    assert ".. checking memory" in output
    assert ".. loading skills" in output
    assert ".. asking model - request 1" in output
    assert ".. model chose tool - calculator" in output
    assert ".. running tool - calculator" in output
    assert ".. tool completed - calculator" in output
    assert ".. turn completed - 2 model request(s), 1 tool call(s)" in output
    assert "Turn Summary" in output
    assert "worked for" in output
    assert "model       2 request(s)" in output
    assert "tools       calculator x1" in output
    assert "The result is 4." in output


def test_main_streams_final_answer_by_default_for_streaming_provider(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["hello", "/q"])

    class StreamingFakeLLM(LLMClient):
        capabilities = LLMCapabilities(supports_streaming=True)

        def complete(self, messages: list[dict[str, str]]) -> str:
            return json.dumps({"type": "final_answer", "content": "streamed default answer"})

    def factory(_config):
        return StreamingFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "chulk\n  streamed default answer" in output
    assert output.count("streamed default answer") == 1


def test_main_shows_run_cmd_command_in_live_progress(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_PERMISSION_PROFILE", "trusted-local")
    inputs = iter(["run printf hello", "/q"])

    class ShellProgressFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "run_cmd",
                        "arguments_json": json.dumps({"command": "printf hello"}),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "The command printed hello."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    def factory(_config):
        return ShellProgressFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert ".. running tool - run_cmd - cmd: printf hello" in output
    assert ".. tool completed - run_cmd - cmd: printf hello" in output
    assert "exit 0" in output
    assert "stdout 5 chars" in output
    assert "The command printed hello." in output


def test_main_prompts_for_default_shell_permission_and_approves(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_PERMISSION_PROFILE", "workspace-write")
    prompts = []
    inputs = iter(["run printf hello", "y", "/q"])

    class ShellApprovalFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "run_cmd",
                        "arguments_json": json.dumps({"command": "printf hello"}),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "The command printed hello."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    exit_code = main(
        [],
        input_func=input_func,
        llm_client_factory=lambda _config: ShellApprovalFakeLLM(),
    )

    output = strip_ansi(capsys.readouterr().out)
    stripped_prompts = [strip_ansi(prompt) for prompt in prompts]

    assert exit_code == 0
    assert "Tool Permission" in output
    assert "tool      run_cmd" in output
    assert "level     shell" in output
    assert "permission requested - run_cmd - shell" in output
    assert "permission allow - run_cmd - shell" in output
    assert "Approve tool call? [y/N]" in " ".join(stripped_prompts)
    assert "exit 0" in output
    assert "The command printed hello." in output


def test_main_prompts_for_default_shell_permission_and_denies(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_PERMISSION_PROFILE", "workspace-write")
    inputs = iter(["run printf hello", "n", "/q"])

    class ShellDenyFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "run_cmd",
                        "arguments_json": json.dumps({"command": "printf hello"}),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "I did not run the command."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: ShellDenyFakeLLM(),
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "Tool Permission" in output
    assert "permission deny - run_cmd - shell" in output
    assert "tool failed - run_cmd - cmd: printf hello - permission_denied" in output
    assert "exit 0" not in output
    assert "I did not run the command." in output


def test_main_plan_prefix_approve_flow_with_tools(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/plan what is 2 + 2?", "/approve", "/q"])

    class PlanModeFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "plan",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": json.dumps(
                            {
                                "summary": "Calculate with a tool.",
                                "steps": [
                                    {
                                        "id": "1",
                                        "title": "Run calculator",
                                        "description": "Use the calculator for the arithmetic.",
                                        "status": "pending",
                                    }
                                ],
                            }
                        ),
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "calculator",
                        "arguments_json": json.dumps({"expression": "2 + 2"}),
                    }
                ),
                json.dumps(
                    {
                        "type": "plan_step_update",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": "{}",
                        "step_update_json": json.dumps(
                            {
                                "step_id": "1",
                                "status": "completed",
                                "evidence": "The calculator returned 4.",
                                "reason": None,
                            }
                        ),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "The result is 4."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    def factory(_config):
        return PlanModeFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert ".. model proposed plan - 1 step(s)" in output
    assert ".. plan waiting for approval - 1 step(s)" in output
    assert "Use /approve to execute this plan or /reject to cancel it." in output
    assert ".. plan approved" in output
    assert ".. plan step started - Run calculator" in output
    assert ".. plan step completed - Run calculator" in output
    assert "plan        completed" in output
    assert "The result is 4." in output


def test_main_plan_prefix_creates_one_shot_pending_plan(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/plan How would you add subagent functionality?", "/approve", "/q"])

    class OneShotPlanFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.requests: list[list[dict[str, str]]] = []
            self.responses = [
                json.dumps(
                    {
                        "type": "plan",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": json.dumps(
                            {
                                "summary": "Design subagent support.",
                                "steps": [
                                    {
                                        "id": "1",
                                        "title": "Add subagent dispatcher",
                                        "description": "Update chulk/core/agent.py with a parent-to-child delegation path.",
                                        "status": "pending",
                                    }
                                ],
                            }
                        ),
                    }
                ),
                json.dumps(
                    {
                        "type": "plan_step_update",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": "{}",
                        "step_update_json": json.dumps(
                            {
                                "step_id": "1",
                                "status": "completed",
                                "evidence": "The dispatcher design is recorded.",
                                "reason": None,
                            }
                        ),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "Subagent design approved."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.requests.append(messages)
            return self.responses.pop(0)

    fake_llm = OneShotPlanFakeLLM()

    def factory(_config):
        return fake_llm

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)
    first_prompt = fake_llm.requests[0][0]["content"]
    approved_prompt = fake_llm.requests[1][0]["content"]

    assert exit_code == 0
    assert ".. model proposed plan - 1 step(s)" in output
    assert ".. plan waiting for approval - 1 step(s)" in output
    assert "Use /approve to execute this plan or /reject to cancel it." in output
    assert "Model proposed a new plan after execution had already been approved." not in output
    assert "Before proposing the plan, you may call only these read-only reconnaissance tools" in first_prompt
    assert "Planning: approved for this turn." in approved_prompt
    assert "Subagent design approved." in output


def test_main_plan_prefix_reject_flow(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/plan inspect files", "/reject", "/q"])

    class RejectPlanFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            return json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(
                        {
                        "summary": "Inspect the files.",
                        "steps": [
                            {
                                "id": "1",
                                "title": "Add file inspection flow",
                                "description": "Implement the requested file inspection behavior using existing tools.",
                                "status": "pending",
                            }
                        ],
                        }
                    ),
                }
            )

    def factory(_config):
        return RejectPlanFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "Add file inspection flow" in output
    assert ".. plan rejected" in output
    assert "Plan rejected. No tools were run." in output
    assert "plan        rejected" in output


def test_main_pending_plan_blocks_normal_input(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/plan inspect files", "please continue anyway", "/reject", "/q"])

    class BlockingPlanFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.request_count = 0

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.request_count += 1
            return json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(
                        {
                        "summary": "Inspect the files.",
                        "steps": [
                            {
                                "id": "1",
                                "title": "Add file inspection flow",
                                "description": "Implement the requested file inspection behavior using existing tools.",
                                "status": "pending",
                            }
                        ],
                        }
                    ),
                }
            )

    fake_llm = BlockingPlanFakeLLM()

    def factory(_config):
        return fake_llm

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "A plan is waiting for approval. Use /approve to execute it or /reject to cancel it." in output
    assert fake_llm.request_count == 1


def test_main_quiet_mode_hides_live_progress(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/quiet on", "what is 2 + 2?", "/q"])

    class QuietFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            return json.dumps({"type": "final_answer", "content": "quiet answer"})

    def factory(_config):
        return QuietFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "quiet mode on" in output
    assert ".. starting turn" not in output
    assert "Turn Summary" not in output
    assert "quiet answer" in output


def test_main_verbose_mode_shows_event_names(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/verbose on", "what is 2 + 2?", "/q"])

    class VerboseFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            return json.dumps({"type": "final_answer", "content": "verbose answer"})

    def factory(_config):
        return VerboseFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "verbose mode on" in output
    assert "turn_started - starting turn" in output
    assert "model_request_started - asking model" in output
    assert "model_response_parsed - model returned final answer" in output
    assert "verbose answer" in output


def test_main_summary_mode_can_be_disabled(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["/summary off", "hello", "/q"])

    class SummaryFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            return json.dumps({"type": "final_answer", "content": "summary-free answer"})

    def factory(_config):
        return SummaryFakeLLM()

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "turn summary off" in output
    assert "Turn Summary" not in output
    assert "summary-free answer" in output


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
    monkeypatch.setenv("CHULK_LLM_FALLBACK_PROVIDERS", "openai:gpt-4.1-mini")
    monkeypatch.setenv("CHULK_PERMISSION_PROFILE", "read-only")

    exit_code = main(["--show-config"])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "ChulkHarness configuration:" in output
    assert f"project_root: {tmp_path}" in output
    assert f"skills_dir: {tmp_path / '.chulk' / 'skills'}" in output
    assert "llm_provider: deepseek" in output
    assert "model: test-model" in output
    assert "llm_fallback_providers: openai:gpt-4.1-mini" in output
    assert "permission_profile: read-only" in output
    assert "deepseek_api_key: set" in output
    assert "local_base_url: http://localhost:1234/v1" in output
    assert "trace_max_prompt_chars: 50000" in output
    assert "max_observation_chars: 12000" in output
    assert "max_tool_stdout_chars: 8000" in output
    assert "max_tool_stderr_chars: 4000" in output
    assert "max_reflection_attempts: 0" in output


def test_create_cli_llm_uses_public_fallback_provider_specs(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "deepseek",
            "CHULK_MODEL": "deepseek-v4-pro",
            "CHULK_LLM_FALLBACK_PROVIDERS": "openai:gpt-4.1-mini, deepseek:deepseek-v4-flash",
        }
    )

    chain = create_cli_llm(config)

    assert isinstance(chain, FallbackChain)
    assert chain.strategy == "first_success"
    assert isinstance(chain.providers[0], DeepSeekProvider)
    assert isinstance(chain.providers[1], OpenAIProvider)
    assert isinstance(chain.providers[2], DeepSeekProvider)
    assert [(provider.provider, provider.model) for provider in chain.providers] == [
        ("deepseek", "deepseek-v4-pro"),
        ("openai", "gpt-4.1-mini"),
        ("deepseek", "deepseek-v4-flash"),
    ]


def test_create_cli_llm_supports_local_provider_specs(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "local",
            "CHULK_MODEL": "google/gemma-4-12b-qat",
            "CHULK_LLM_FALLBACK_PROVIDERS": "openai:gpt-4.1-mini",
        }
    )

    chain = create_cli_llm(config)

    assert isinstance(chain.providers[0], LocalProvider)
    assert isinstance(chain.providers[1], OpenAIProvider)
    assert [(provider.provider, provider.model) for provider in chain.providers] == [
        ("local", "google/gemma-4-12b-qat"),
        ("openai", "gpt-4.1-mini"),
    ]


def test_main_runs_one_message_with_fake_llm(capsys):
    exit_code = main(["--once", "hello"], llm_client_factory=fake_factory)

    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "hello from fake llm"


def test_main_lists_available_skills_and_injects_selected_skill(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    skill_dir = tmp_path / ".chulk" / "skills" / "shell"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Shell Skill\n\nUse this skill when command execution is needed.\n",
        encoding="utf-8",
    )

    class SkillAwareFakeLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            system_prompt = messages[0]["content"]
            xml_sections_present = (
                system_prompt.startswith("<chulk_prompt>")
                and "<available_skills>" in system_prompt
                and "</available_skills>" in system_prompt
                and "<skills>" in system_prompt
                and "</skills>" in system_prompt
                and system_prompt.endswith("</chulk_prompt>")
            )
            catalog_visible = "- shell: Use this skill when command execution is needed." in system_prompt
            skill_loaded = "Skill: shell" in system_prompt and "# Shell Skill" in system_prompt
            if xml_sections_present and catalog_visible and skill_loaded:
                return json.dumps({"type": "final_answer", "content": "shell skill listed and loaded"})
            return json.dumps({"type": "final_answer", "content": "missing skill"})

    def factory(_config):
        return SkillAwareFakeLLM()

    exit_code = main(["--once", "run a shell command"], llm_client_factory=factory)

    output = capsys.readouterr().out

    assert exit_code == 0
    assert output.strip() == "shell skill listed and loaded"


def test_main_writes_full_model_request_trace(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_TRACE_MAX_PROMPT_CHARS", "100000")
    skill_dir = tmp_path / ".chulk" / "skills" / "shell"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Shell Skill\n\nUse this skill when command execution is needed.\n",
        encoding="utf-8",
    )

    exit_code = main(["--once", "run a shell command"], llm_client_factory=fake_factory)

    output = capsys.readouterr().out
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    request_payload = next(event["payload"] for event in events if event["type"] == "model_request_started")

    assert exit_code == 0
    assert output.strip() == "hello from fake llm"
    assert request_payload["truncated"] is False
    assert request_payload["messages"][0]["role"] == "system"
    assert "Skill: shell" in request_payload["messages"][0]["content"]
    assert request_payload["messages"][-1]["content"] == "run a shell command"


def test_main_e2e_records_turn_state_for_tool_call(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    class ToolTurnFakeLLM(LLMClient):
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "type": "tool_call",
                        "content": None,
                        "tool_name": "calculator",
                        "arguments_json": json.dumps({"expression": "2 + 2"}),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "The result is 4."}),
            ]

        def complete(self, messages: list[dict[str, str]]) -> str:
            return self.responses.pop(0)

    def factory(_config):
        return ToolTurnFakeLLM()

    exit_code = main(["--once", "what is 2 + 2?"], llm_client_factory=factory)

    output = capsys.readouterr().out
    trace_file = next((tmp_path / "traces").glob("*.jsonl"))
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    finished_payload = next(event["payload"] for event in events if event["type"] == "turn_finished")
    turn = finished_payload["turn"]

    assert exit_code == 0
    assert output.strip() == "The result is 4."
    assert "turn_started" in event_types
    assert "tool_call_completed" in event_types
    assert turn["status"] == "completed"
    assert turn["user_message"] == "what is 2 + 2?"
    assert turn["model_request_count"] == 2
    assert turn["tool_call_count"] == 1
    assert turn["tool_calls"][0]["tool_name"] == "calculator"
    assert turn["tool_calls"][0]["success"] is True
    assert turn["observations"][0]["tool_name"] == "calculator"
    assert finished_payload["agent_state"]["turn_count"] == 1


def test_main_e2e_uses_openai_native_tool_calling(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["what is 2 + 2?", "/q"])

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    output_text="",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="calculator",
                            arguments=json.dumps({"expression": "2 + 2"}),
                            call_id="call_calc",
                        )
                    ],
                    usage=SimpleNamespace(input_tokens=100, output_tokens=10, total_tokens=110),
                )
            return SimpleNamespace(
                output_text="The result is 4.",
                output=[],
                usage=SimpleNamespace(input_tokens=120, output_tokens=8, total_tokens=128),
            )

    class FakeOpenAI:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    fake_openai = FakeOpenAI()

    def factory(_config):
        return OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_openai)

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=factory,
    )

    output = strip_ansi(capsys.readouterr().out)
    first_call = fake_openai.responses.calls[0]
    tool_names = {tool["name"] for tool in first_call["tools"]}

    assert exit_code == 0
    assert "The result is 4." in output
    assert "tools       calculator x1" in output
    assert len(fake_openai.responses.calls) == 2
    assert first_call["tool_choice"] == "auto"
    assert "tools" in first_call
    assert "text" not in first_call
    assert {"calculator", "chulk_propose_plan", "chulk_plan_step_update"} <= tool_names


def test_main_e2e_compacts_context_and_resumes_with_summary(monkeypatch, tmp_path, capsys):
    model = "chulk-e2e-context-compaction"
    register_model_capabilities(
        LLMModelCapabilities(
            provider="openai",
            model=model,
            context_window_tokens=1200,
            default_response_reserve_tokens=128,
        )
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_MODEL", model)
    old_marker = "OLD_CONTEXT_E2E_MARKER"
    summary_text = "E2E compact summary: old marker established the context compaction path."

    class CompactionE2EFakeLLM(LLMClient):
        def __init__(self, action_responses: list[str]) -> None:
            self.action_responses = action_responses
            self.action_requests: list[list[dict[str, str]]] = []
            self.summary_requests: list[list[dict[str, str]]] = []

        def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
            if "You update a compact, task-local conversation summary" in messages[0]["content"]:
                self.summary_requests.append(messages)
                return summary_text
            self.action_requests.append(messages)
            return self.action_responses.pop(0)

    first_llm = CompactionE2EFakeLLM(
        [
            json.dumps({"type": "final_answer", "content": "first turn complete"}),
            json.dumps({"type": "final_answer", "content": "second turn complete"}),
        ]
    )
    inputs = iter(
        [
            old_marker + " " + ("x" * 5000),
            "continue after the old context",
            "/q",
        ]
    )

    first_exit = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: first_llm,
    )

    first_output = strip_ansi(capsys.readouterr().out)
    store = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite")
    session = next(record for record in store.list_conversations() if record.turn_count == 2)
    summary = store.load_latest_summary(session.id)
    trace_file = tmp_path / "traces" / f"{session.id}.jsonl"
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    summary_event = next(event for event in events if event["type"] == "context_summary_created")
    second_request = first_llm.action_requests[1]
    second_request_payload = json.dumps(second_request)
    model_request_payloads = [event["payload"] for event in events if event["type"] == "model_request_started"]
    summary_request_payload = next(payload for payload in model_request_payloads if payload.get("purpose") == "context_summary")
    action_request_payloads = [payload for payload in model_request_payloads if payload.get("purpose") != "context_summary"]
    with sqlite3.connect(tmp_path / "chulk" / "store.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        persisted_model_requests = [
            json.loads(row["request_json"])
            for row in conn.execute(
                "SELECT request_json FROM conversation_model_requests WHERE conversation_id = ?",
                (session.id,),
            ).fetchall()
        ]
    second_context_report = action_request_payloads[1]["context_report"]
    summary_section = next(
        section for section in second_context_report["sections"] if section["name"] == "conversation_summary"
    )

    assert first_exit == 0
    assert "first turn complete" in first_output
    assert "second turn complete" in first_output
    assert len(first_llm.summary_requests) == 1
    assert old_marker in json.dumps(first_llm.summary_requests[0])
    assert summary is not None
    assert summary.content == summary_text
    assert summary.source_message_count == 2
    assert summary_request_payload["summary_source_message_count"] == 2
    assert summary_request_payload["context_report"]["purpose"] == "context_summary"
    assert any(payload.get("purpose") == "context_summary" for payload in persisted_model_requests)
    assert summary_event["payload"]["source_message_count"] == 2
    assert summary_event["payload"]["summary"] == summary_text
    assert summary_event["payload"]["fallback"] is False
    assert summary_text in second_request[0]["content"]
    assert old_marker not in second_request_payload
    assert second_context_report["omitted_message_count"] == 0
    assert summary_section["metadata"] == {"has_summary": True, "summary_message_count": 2}

    resumed_llm = CompactionE2EFakeLLM(
        [json.dumps({"type": "final_answer", "content": "resume saw summary"})]
    )
    resumed_inputs = iter([f"/resume {session.id}", "continue from resume", "/q"])

    second_exit = main(
        [],
        input_func=lambda _prompt: next(resumed_inputs),
        llm_client_factory=lambda _config: resumed_llm,
    )

    resumed_output = strip_ansi(capsys.readouterr().out)
    resumed_request = resumed_llm.action_requests[0]
    resumed_request_payload = json.dumps(resumed_request)

    assert second_exit == 0
    assert "resumed session" in resumed_output
    assert "resume saw summary" in resumed_output
    assert summary_text in resumed_request[0]["content"]
    assert old_marker not in resumed_request_payload
    assert len(resumed_llm.summary_requests) == 1
    assert old_marker not in json.dumps(resumed_llm.summary_requests[0])


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
    assert (tmp_path / "chulk" / "store.sqlite").exists()
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
