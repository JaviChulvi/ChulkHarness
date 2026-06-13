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
