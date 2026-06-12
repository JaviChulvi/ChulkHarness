"""Interactive slash-command handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.cli.progress import ProgressSettings
from src.cli.terminal import TerminalUI
from src.config import Config
from src.core import Agent


EXIT_COMMANDS = {"/exit", "/quit", "/q", "exit", "quit"}
HELP_COMMANDS = {"/help", "help", "?"}
VERBOSE_COMMANDS = {"/verbose on", "/verbose off"}
QUIET_COMMANDS = {"/quiet on", "/quiet off"}
SUMMARY_COMMANDS = {"/summary on", "/summary off"}


@dataclass
class CLICommandContext:
    """Runtime state needed by interactive slash commands."""

    agent: Agent
    config: Config | None
    terminal: TerminalUI
    progress_settings: ProgressSettings
    output_func: Callable[[str], None]


def handle_cli_command(command: str, context: CLICommandContext) -> bool:
    """Handle a CLI command. Returns True when the input was consumed."""
    raw_command = command.strip()
    normalized_command = raw_command.lower()

    if normalized_command in HELP_COMMANDS:
        context.output_func(context.terminal.help_text())
        return True
    if normalized_command == "/status":
        if context.config is None:
            context.output_func(context.terminal.warning("status unavailable: no config object"))
        else:
            context.output_func(context.terminal.status(context.config, context.agent))
        return True
    if normalized_command == "/tools":
        context.output_func(context.terminal.tools(context.agent))
        return True
    if normalized_command == "/trace":
        context.output_func(context.terminal.trace(context.agent))
        return True
    if normalized_command == "/plan":
        context.output_func(context.terminal.plan_status(context.agent))
        return True
    if normalized_command.startswith("/plan "):
        planned_message = raw_command[len("/plan ") :].strip()
        if not planned_message:
            context.output_func(context.terminal.warning("usage: /plan <request>"))
            return True
        response = context.agent.run_planned_turn(planned_message)
        context.output_func(context.terminal.assistant_message(response))
        return True
    if normalized_command == "/approve":
        response = context.agent.approve_plan()
        context.output_func(context.terminal.assistant_message(response))
        return True
    if normalized_command == "/reject":
        response = context.agent.reject_plan()
        context.output_func(context.terminal.assistant_message(response))
        return True
    if normalized_command == "/clear":
        context.output_func(context.terminal.clear())
        return True
    if normalized_command in VERBOSE_COMMANDS:
        context.progress_settings.verbose = normalized_command.endswith(" on")
        context.output_func(
            context.terminal.warning(f"verbose mode {'on' if context.progress_settings.verbose else 'off'}")
        )
        return True
    if normalized_command in QUIET_COMMANDS:
        context.progress_settings.quiet = normalized_command.endswith(" on")
        context.output_func(context.terminal.warning(f"quiet mode {'on' if context.progress_settings.quiet else 'off'}"))
        return True
    if normalized_command in SUMMARY_COMMANDS:
        context.progress_settings.summary = normalized_command.endswith(" on")
        context.output_func(
            context.terminal.warning(f"turn summary {'on' if context.progress_settings.summary else 'off'}")
        )
        return True
    return False
