"""Interactive slash-command handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chulk.cli.progress import ProgressSettings
from chulk.cli.terminal import TerminalUI
from chulk.config import Config
from chulk.core import Agent
from chulk.sessions import (
    AmbiguousSessionError,
    DEFAULT_EXPORT_FORMAT,
    SessionNotFoundError,
    SQLiteSessionStore,
    export_session,
)


EXIT_COMMANDS = {"/exit", "/quit", "/q", "exit", "quit"}
HELP_COMMANDS = {"/help", "help", "?"}
VERBOSE_COMMANDS = {"/verbose on", "/verbose off"}
QUIET_COMMANDS = {"/quiet on", "/quiet off"}
SUMMARY_COMMANDS = {"/summary on", "/summary off"}
EXPORT_USAGE = "usage: /export [session-id] [--format md|json] [--out path]"


@dataclass
class CLICommandContext:
    """Runtime state needed by interactive slash commands."""

    agent: Agent
    config: Config | None
    terminal: TerminalUI
    progress_settings: ProgressSettings
    output_func: Callable[[str], None]
    session_store: SQLiteSessionStore | None = None
    agent_factory: Callable[[str], Agent] | None = None
    switch_agent: Callable[[Agent], None] | None = None


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
    if normalized_command == "/context":
        context.output_func(context.terminal.context(context.agent))
        return True
    if normalized_command == "/tools":
        context.output_func(context.terminal.tools(context.agent))
        return True
    if normalized_command == "/mcp":
        if context.config is None:
            context.output_func(context.terminal.warning("mcp unavailable: no config object"))
        else:
            context.output_func(context.terminal.mcp(context.config, context.agent))
        return True
    if normalized_command == "/sessions":
        if context.session_store is None:
            context.output_func(context.terminal.warning("sessions unavailable: no session store"))
            return True
        context.output_func(context.terminal.sessions(context.session_store.list_conversations()))
        return True
    if normalized_command == "/history":
        if context.session_store is None:
            context.output_func(context.terminal.warning("history unavailable: no session store"))
            return True
        messages = context.session_store.list_messages(context.agent.state.conversation_id, limit=40)
        context.output_func(context.terminal.history(messages))
        return True
    if normalized_command == "/resume" or normalized_command.startswith("/resume "):
        if normalized_command == "/resume":
            context.output_func(context.terminal.warning("usage: /resume <conversation_id>"))
            return True
        if context.agent_factory is None or context.switch_agent is None:
            context.output_func(context.terminal.warning("resume unavailable: no agent factory"))
            return True
        session_id = raw_command[len("/resume") :].strip()
        try:
            next_agent = context.agent_factory(session_id)
        except SessionNotFoundError as exc:
            context.output_func(context.terminal.warning(str(exc)))
            return True
        except AmbiguousSessionError as exc:
            context.output_func(context.terminal.warning(str(exc)))
            return True
        context.switch_agent(next_agent)
        context.output_func(context.terminal.warning(f"resumed session {next_agent.state.conversation_id[:8]}"))
        return True
    if normalized_command == "/export" or normalized_command.startswith("/export "):
        if context.session_store is None:
            context.output_func(context.terminal.warning("export unavailable: no session store"))
            return True
        try:
            session_id, export_format, out_path = _parse_export_command(raw_command[len("/export") :].strip())
        except ValueError as exc:
            context.output_func(context.terminal.warning(str(exc)))
            return True
        if out_path is None and context.config is None:
            context.output_func(context.terminal.warning("export unavailable: no config object"))
            return True
        target_session_id = session_id or context.agent.state.conversation_id
        try:
            written_path = export_session(
                context.session_store,
                target_session_id,
                runtime_dir=context.config.runtime_dir if context.config is not None else Path(".chulk"),
                format=export_format,
                output_path=out_path,
            )
        except (SessionNotFoundError, AmbiguousSessionError, ValueError) as exc:
            context.output_func(context.terminal.warning(str(exc)))
            return True
        context.output_func(context.terminal.warning(f"exported session to {written_path}"))
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


def _parse_export_command(argument_text: str) -> tuple[str | None, str, Path | None]:
    """Parse `/export` arguments into (session_id, format, output_path)."""
    tokens = argument_text.split()
    session_id: str | None = None
    export_format = DEFAULT_EXPORT_FORMAT
    out_path: Path | None = None

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--format":
            index += 1
            if index >= len(tokens):
                raise ValueError(EXPORT_USAGE)
            export_format = tokens[index]
        elif token == "--out":
            index += 1
            if index >= len(tokens):
                raise ValueError(EXPORT_USAGE)
            out_path = Path(tokens[index])
        elif session_id is None and not token.startswith("--"):
            session_id = token
        else:
            raise ValueError(EXPORT_USAGE)
        index += 1

    return session_id, export_format, out_path
