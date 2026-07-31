"""Tests for CLI progress helpers."""

from __future__ import annotations

from io import StringIO
import re
import time
from types import SimpleNamespace

from chulk.cli import CLI_COMMANDS, Spinner, TerminalUI


class TTYBuffer(StringIO):
    """String buffer that behaves like an interactive terminal."""

    def isatty(self) -> bool:
        return True


def test_spinner_animates_for_tty_stream():
    stream = TTYBuffer()
    spinner = Spinner(
        TerminalUI(color_enabled=False),
        stream=stream,
        enabled=True,
        interval_seconds=0.001,
    )

    spinner.start("asking model request 1")
    deadline = time.monotonic() + 0.1
    while "asking model request 1" not in stream.getvalue() and time.monotonic() < deadline:
        time.sleep(0.001)
    spinner.stop()

    output = stream.getvalue()

    assert "asking model request 1" in output
    assert "\r\033[K" in output


def test_turn_summary_formats_token_usage_and_cost():
    ui = TerminalUI(color_enabled=False)

    output = ui.turn_summary(
        {
            "turn": {
                "model_request_count": 1,
                "tool_calls": [],
                "loaded_skill_names": [],
                "loaded_memory_ids": [],
                "model_usage_totals": {
                    "request_count": 1,
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                        "estimated": True,
                    },
                    "cost": {
                        "amount": "0.000004",
                        "currency": "USD",
                        "pricing_known": True,
                        "estimated": True,
                    },
                },
            }
        }
    )

    assert "usage       15 tokens (12 in, 3 out est), ~$0.000004" in output


def test_terminal_color_auto_respects_tty_no_color_and_force_modes():
    tty = TTYBuffer()

    automatic = TerminalUI.themed(stream=tty, environ={"TERM": "xterm-256color"})
    disabled = TerminalUI.themed(stream=tty, environ={"TERM": "xterm-256color", "NO_COLOR": ""})
    forced = TerminalUI.themed(stream=StringIO(), color="always", environ={"TERM": "dumb"})

    assert automatic.color_enabled is True
    assert disabled.color_enabled is False
    assert forced.color_enabled is True
    assert TerminalUI.themed(stream=StringIO()).color_enabled is False


def test_help_alignment_is_calculated_before_ansi_styling():
    ui = TerminalUI(color_enabled=True, width=100)
    output = re.sub(r"\x1b\[[0-9;]*m", "", ui.help_text(CLI_COMMANDS))
    command_lines = [line for line in output.splitlines() if line.startswith("  /")]
    status_line = next(line for line in command_lines if line.lstrip().startswith("/status"))
    context_line = next(line for line in command_lines if line.lstrip().startswith("/context"))

    assert status_line.index("show runtime") == context_line.index("show the latest")


def test_help_reflows_without_exceeding_narrow_terminal_width():
    ui = TerminalUI(color_enabled=False, width=36)

    output = ui.help_text(CLI_COMMANDS)

    assert max(len(line) for line in output.splitlines()) <= 36


def test_banner_reflows_without_exceeding_narrow_terminal_width(tmp_path):
    ui = TerminalUI(color_enabled=False, width=36)
    config = SimpleNamespace(
        llm_provider="openai",
        model="gpt-4.1-mini",
        llm_fallback_providers=(),
        permission_profile="workspace-write",
        project_root=tmp_path,
        mcp_servers=(),
    )
    agent = SimpleNamespace(
        state=SimpleNamespace(conversation_id="conversation-123"),
        tool_registry=SimpleNamespace(list_tools=lambda: [object(), object()]),
    )

    output = ui.banner(config, agent)

    assert max(len(line) for line in output.splitlines()) <= 36
