"""Execution helpers for non-interactive CLI entrypoints."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from chulk.cli.maintenance import (
    TraceFormatError,
    export_trace_html,
    format_doctor_report,
    format_fixture_replay,
    format_init_changes,
    format_trace_replay,
    format_trace_summary,
    initialize_project,
    inspect_trace,
    replay_trace,
    run_doctor,
)
from chulk.core import Agent
from chulk.llm import LLMConfigurationError, LLMError
from chulk.tools.permissions import PermissionDecision
from chulk.tracing.execution import execute_replay_fixture
from chulk.tracing.fixtures import load_replay_fixture


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIGURATION_ERROR = 2
EXIT_APPROVAL_REQUIRED = 3


def run_exec_command(
    message: str,
    *,
    agent_factory: Callable[[], Agent],
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """Run one request without ever blocking for interactive approval."""
    approval_requested = False

    def deny_approval(_request: object, _record: object) -> PermissionDecision:
        nonlocal approval_requested
        approval_requested = True
        return PermissionDecision.DENY

    try:
        agent = agent_factory()
        agent.permission_callback = deny_approval
        response = agent.run_turn(message)
    except (ValueError, LLMConfigurationError) as exc:
        return _emit_error(
            "configuration_error",
            exc,
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
            exit_code=EXIT_CONFIGURATION_ERROR,
        )
    except LLMError as exc:
        return _emit_error(
            "runtime_error",
            exc,
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
            exit_code=EXIT_RUNTIME_ERROR,
        )
    except Exception as exc:
        return _emit_error(
            "runtime_error",
            exc,
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
            exit_code=EXIT_RUNTIME_ERROR,
            unexpected=True,
        )

    turn = agent.state.turns[-1]
    approval_required = turn.status == "waiting_for_approval" or approval_requested
    payload = {
        "ok": not approval_required and turn.status == "completed",
        "status": "approval_required" if approval_required else turn.status,
        "content": response,
        "conversation_id": agent.state.conversation_id,
        "profile_id": agent.profile_id,
        "trace_path": str(agent.trace_logger.path) if agent.trace_logger is not None else None,
        "usage": turn.model_usage_totals or None,
        "tool_calls": [
            {
                "tool_name": record.tool_name,
                "success": record.success,
                "error": record.error,
                "failure_kind": record.failure_kind,
            }
            for record in turn.tool_calls
        ],
    }
    if json_output:
        output_func(json_text(payload))
    elif approval_required:
        error_func(response)
        error_func("approval required: a tool call was denied in non-interactive mode")
    elif turn.status == "completed":
        output_func(response)
    else:
        error_func(response)
    if approval_required:
        return EXIT_APPROVAL_REQUIRED
    if turn.status != "completed":
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


def run_doctor_command(*, json_output: bool, output_func: Callable[[str], None]) -> int:
    report = run_doctor()
    output_func(json_text(report.to_dict()) if json_output else format_doctor_report(report))
    return EXIT_OK if report.ok else EXIT_CONFIGURATION_ERROR


def run_init_command(
    project_root: Path | str,
    *,
    mode: str,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    root = Path(project_root).expanduser().resolve()
    try:
        changes = initialize_project(root, mode=mode)
    except (OSError, ValueError) as exc:
        return _emit_error(
            "init_error",
            exc,
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
            exit_code=EXIT_RUNTIME_ERROR,
        )
    if json_output:
        output_func(
            json_text(
                {
                    "ok": True,
                    "status": "initialized",
                    "project_root": str(root),
                    "mode": mode,
                    "changes": [change.to_dict() for change in changes],
                }
            )
        )
    else:
        output_func(format_init_changes(root, changes))
    return EXIT_OK


def run_trace_command(
    command: str,
    path: Path | str | None,
    *,
    execute_fixture_path: Path | str | None = None,
    json_output: bool,
    output_path: Path | str | None,
    force: bool,
    max_bytes: int | None = None,
    max_events: int | None = None,
    unbounded: bool = False,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    try:
        if command == "replay" and execute_fixture_path is not None:
            if path is not None:
                raise ValueError(
                    "Choose either a trace path or --execute-fixture, not both"
                )
            if max_events is not None or unbounded:
                raise ValueError(
                    "Executable fixtures use a strict byte limit; "
                    "--max-events and --unbounded are not supported"
                )
            fixture = load_replay_fixture(
                execute_fixture_path,
                **({"max_bytes": max_bytes} if max_bytes is not None else {}),
            )
            report = execute_replay_fixture(fixture).to_dict()
            output_func(
                json_text(report)
                if json_output
                else format_fixture_replay(report)
            )
            return EXIT_OK if report["ok"] else EXIT_RUNTIME_ERROR
        if path is None:
            raise ValueError("A trace path is required")
        if command == "inspect":
            summary = inspect_trace(
                path,
                max_bytes=max_bytes,
                max_events=max_events,
                unbounded=unbounded,
            )
            output_func(json_text(summary) if json_output else format_trace_summary(summary))
            return EXIT_OK
        if command == "replay":
            replay = replay_trace(
                path,
                max_bytes=max_bytes,
                max_events=max_events,
                unbounded=unbounded,
            )
            output_func(json_text(replay) if json_output else format_trace_replay(replay))
            return EXIT_OK
        if command != "export":
            raise ValueError(f"Unknown trace command: {command}")
        destination = export_trace_html(
            path,
            output_path=output_path,
            force=force,
            max_bytes=max_bytes,
            max_events=max_events,
            unbounded=unbounded,
        )
        payload = {
            "ok": True,
            "status": "exported",
            "format": "html",
            "output_path": str(destination),
        }
        output_func(json_text(payload) if json_output else f"Trace exported to {destination}")
        return EXIT_OK
    except (TraceFormatError, FileExistsError, OSError, ValueError) as exc:
        return _emit_error(
            "trace_error",
            exc,
            json_output=json_output,
            output_func=output_func,
            error_func=error_func,
            exit_code=EXIT_RUNTIME_ERROR,
        )


def json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _emit_error(
    status: str,
    error: Exception,
    *,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
    exit_code: int,
    unexpected: bool = False,
) -> int:
    if json_output:
        output_func(json_text({"ok": False, "status": status, "error": str(error)}))
    else:
        prefix = "error: unexpected failure" if unexpected else status.replace("_", " ")
        error_func(f"{prefix}: {error}")
    return exit_code


__all__ = [
    "EXIT_APPROVAL_REQUIRED",
    "EXIT_CONFIGURATION_ERROR",
    "EXIT_OK",
    "EXIT_RUNTIME_ERROR",
    "json_text",
    "run_doctor_command",
    "run_exec_command",
    "run_init_command",
    "run_trace_command",
]
