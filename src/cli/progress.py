"""Live progress reporting for the interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import sys
import threading
import time
from typing import TextIO

from src.cli.terminal import TerminalUI
from src.config import Config
from src.core import Agent, TraceEvent


@dataclass
class ProgressSettings:
    """Runtime display settings toggled by slash commands."""

    quiet: bool = False
    verbose: bool = False
    summary: bool = True


class Spinner:
    """Tiny ASCII spinner for real TTYs."""

    def __init__(
        self,
        terminal: TerminalUI,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        interval_seconds: float = 0.12,
    ) -> None:
        self.terminal = terminal
        self.stream = stream or sys.stdout
        self.enabled = bool(self.stream.isatty()) if enabled is None else enabled
        self.interval_seconds = interval_seconds
        self._frames = ["-", "\\", "|", "/"]
        self._label = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, label: str) -> None:
        if not self.enabled:
            return
        self.stop()
        self._label = label
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None
        self._clear_line()

    def _spin(self) -> None:
        index = 0
        while not self._stop.is_set():
            frame = self._frames[index % len(self._frames)]
            self.stream.write("\r" + self.terminal.muted(f"{frame} {self._label}"))
            self.stream.flush()
            index += 1
            time.sleep(self.interval_seconds)

    def _clear_line(self) -> None:
        self.stream.write("\r\033[K")
        self.stream.flush()


class ProgressReporter:
    """Translate agent events into live CLI status lines."""

    def __init__(
        self,
        terminal: TerminalUI,
        output_func: Callable[[str], None],
        *,
        config: Config | None = None,
        agent: Agent | None = None,
        settings: ProgressSettings | None = None,
        spinner: Spinner | None = None,
        previous_callback: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.terminal = terminal
        self.output_func = output_func
        self.config = config
        self.agent = agent
        self.settings = settings or ProgressSettings()
        self.spinner = spinner or Spinner(terminal)
        self.previous_callback = previous_callback
        self.turn_started_at: float | None = None
        self.model_started_at: float | None = None
        self.tool_started_at: dict[int, float] = {}
        self.current_activity: str | None = None

    def callback(self, event_type: str, payload: dict) -> None:
        """Handle one agent event."""
        if self.previous_callback is not None:
            self.previous_callback(event_type, payload)

        now = time.monotonic()
        if event_type == TraceEvent.TURN_STARTED:
            self.turn_started_at = now
        elif event_type == TraceEvent.MODEL_REQUEST_STARTED:
            self.model_started_at = now
        elif event_type == TraceEvent.TOOL_CALL_STARTED:
            self.tool_started_at[_tool_key(payload)] = now

        self._stop_spinner_if_needed(event_type)

        if self.settings.quiet:
            return

        line = self.terminal.progress(
            event_type,
            payload,
            elapsed_seconds=self._elapsed(now),
            duration_seconds=self._duration(event_type, payload, now),
            verbose=self.settings.verbose,
        )
        if line is not None:
            self.output_func(line)

        if event_type == TraceEvent.TURN_FINISHED and self.settings.summary:
            self.output_func(self.terminal.turn_summary(payload, config=self.config, agent=self.agent))

        self._start_spinner_if_needed(event_type, payload)

    def close(self) -> None:
        self.spinner.stop()

    def _elapsed(self, now: float) -> float | None:
        if self.turn_started_at is None:
            return None
        return now - self.turn_started_at

    def _duration(self, event_type: str, payload: dict, now: float) -> float | None:
        if event_type == TraceEvent.MODEL_RESPONSE:
            if self.model_started_at is None:
                return None
            return now - self.model_started_at
        if event_type in {TraceEvent.TOOL_CALL_COMPLETED, TraceEvent.TOOL_CALL_FAILED}:
            started_at = self.tool_started_at.get(_tool_key(payload))
            if started_at is None:
                return None
            return now - started_at
        if event_type == TraceEvent.TURN_FINISHED:
            return self._elapsed(now)
        return None

    def _start_spinner_if_needed(self, event_type: str, payload: dict) -> None:
        if self.settings.quiet:
            return
        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            request_index = payload.get("request_index", "?")
            self.current_activity = f"asking model request {request_index}"
            self.spinner.start(self.current_activity)
        elif event_type == TraceEvent.TOOL_CALL_STARTED:
            tool_name = payload.get("tool_name", "tool")
            self.current_activity = f"running {tool_name}"
            self.spinner.start(self.current_activity)

    def _stop_spinner_if_needed(self, event_type: str) -> None:
        if event_type in {
            TraceEvent.MODEL_RESPONSE,
            TraceEvent.MODEL_RESPONSE_PARSED,
            TraceEvent.TOOL_CALL_COMPLETED,
            TraceEvent.TOOL_CALL_FAILED,
            TraceEvent.TURN_FINISHED,
            TraceEvent.TURN_FAILED,
        }:
            self.spinner.stop()


def _tool_key(payload: dict) -> int:
    iteration = payload.get("iteration")
    return iteration if isinstance(iteration, int) else -1
