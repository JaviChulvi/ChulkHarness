"""Tests for CLI progress helpers."""

from __future__ import annotations

from io import StringIO
import time

from chulk.cli import Spinner, TerminalUI


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
