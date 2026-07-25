"""Argument parser for the Chulk command-line interface."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the backwards-compatible CLI and its explicit subcommands."""
    parser = argparse.ArgumentParser(
        prog="chulk",
        description="Run the ChulkHarness agent runtime.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the ChulkHarness version and exit.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved local configuration and exit.",
    )
    parser.add_argument(
        "--once",
        metavar="MESSAGE",
        help="Compatibility alias for 'chulk exec MESSAGE'.",
    )
    parser.add_argument(
        "--profile",
        metavar="ID",
        help="Use one agent profile for this CLI invocation.",
    )
    parser.add_argument(
        "--model-profile",
        metavar="ID",
        help="Use one model profile for this CLI invocation.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Control ANSI color output (default: auto).",
    )
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--resume", metavar="ID", help="Start in a persisted interactive session."
    )
    session_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the most recently updated session.",
    )

    subparsers = parser.add_subparsers(dest="command")
    _add_exec_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_init_parser(subparsers)
    _add_profile_parser(subparsers)
    _add_model_parser(subparsers)
    _add_usage_parser(subparsers)
    _add_trace_parser(subparsers)
    return parser


def _add_exec_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "exec", help="Run one non-interactive agent request."
    )
    parser.add_argument("message", nargs="+", help="Message to send to the agent.")
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )
    parser.add_argument(
        "--profile",
        metavar="ID",
        default=argparse.SUPPRESS,
        help="Use one agent profile for this request.",
    )
    parser.add_argument(
        "--model-profile",
        metavar="ID",
        default=argparse.SUPPRESS,
        help="Use one model profile for this request.",
    )


def _add_model_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "model",
        help="Create, select, inspect, or diagnose model profiles.",
    )
    model_subparsers = parser.add_subparsers(dest="model_command", required=True)

    create = model_subparsers.add_parser("create", help="Create a model profile.")
    create.add_argument("model_profile_id")
    create.add_argument("--provider", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--credential-ref")
    create.add_argument("--endpoint-ref")
    create.add_argument("--fallback", action="append", default=[])
    create.add_argument("--require-structured-output", action="store_true")
    create.add_argument("--require-json-mode", action="store_true")
    create.add_argument("--require-streaming", action="store_true")
    create.add_argument("--require-native-tools", action="store_true")
    create.add_argument("--require-hosted-mcp", action="store_true")
    create.add_argument("--context-window-tokens", type=int)
    create.add_argument("--response-reserve-tokens", type=int)
    create.add_argument("--max-output-tokens", type=int)
    create.add_argument("--max-cost-per-turn")
    create.add_argument("--json", action="store_true", dest="json_output")

    list_parser = model_subparsers.add_parser("list", help="List model profiles.")
    list_parser.add_argument("--channel")
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    use = model_subparsers.add_parser("use", help="Select a model profile.")
    use.add_argument("model_profile_id")
    use.add_argument("--channel")
    use.add_argument("--json", action="store_true", dest="json_output")

    inspect = model_subparsers.add_parser(
        "inspect", help="Inspect and diagnose a model profile."
    )
    inspect.add_argument("model_profile_id")
    inspect.add_argument(
        "--probe",
        action="store_true",
        help="Perform a bounded live request that may consume quota.",
    )
    inspect.add_argument("--json", action="store_true", dest="json_output")

    health = model_subparsers.add_parser("health", help="Show provider health.")
    health.add_argument("model_profile_id", nargs="?")
    health.add_argument("--json", action="store_true", dest="json_output")

    reset = model_subparsers.add_parser("reset", help="Reset provider health.")
    reset.add_argument("model_profile_id")
    reset.add_argument("--json", action="store_true", dest="json_output")

    discover = model_subparsers.add_parser(
        "discover",
        help="Explicitly discover models from a supported endpoint.",
    )
    discover.add_argument("model_profile_id")
    discover.add_argument("--json", action="store_true", dest="json_output")


def _add_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "profile", help="Create, select, or inspect agent profiles."
    )
    profile_subparsers = parser.add_subparsers(dest="profile_command", required=True)

    create = profile_subparsers.add_parser(
        "create", help="Create an isolated agent profile."
    )
    create.add_argument("profile_id")
    create.add_argument("--project-root", default=".")
    create.add_argument("--permission-profile")
    create.add_argument("--model-profile", default="default")
    create.add_argument("--execution-backend", default="host")
    create.add_argument("--skill", action="append", dest="allowed_skills")
    create.add_argument("--mcp-server", action="append", dest="allowed_mcp_servers")
    create.add_argument("--credential-env", action="append", default=[])
    create.add_argument("--system-prompt")
    create.add_argument("--json", action="store_true", dest="json_output")

    list_parser = profile_subparsers.add_parser(
        "list", help="List configured profiles."
    )
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    use = profile_subparsers.add_parser(
        "use", help="Select the local CLI default profile."
    )
    use.add_argument("profile_id")
    use.add_argument("--json", action="store_true", dest="json_output")

    inspect = profile_subparsers.add_parser(
        "inspect", help="Inspect one profile without secrets."
    )
    inspect.add_argument("profile_id")
    inspect.add_argument("--json", action="store_true", dest="json_output")


def _add_usage_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "usage",
        help="Query or export profile-owned usage and exact costs.",
    )
    usage_subparsers = parser.add_subparsers(dest="usage_command", required=True)

    today = usage_subparsers.add_parser("today", help="Show today's usage.")
    _add_usage_query_options(today, dates=False)

    range_parser = usage_subparsers.add_parser(
        "range",
        help="Show usage in an inclusive ISO date range.",
    )
    _add_usage_query_options(range_parser, dates=True)
    range_parser.add_argument("--cursor")

    group = usage_subparsers.add_parser(
        "group",
        help="Group usage totals over an optional range.",
    )
    _add_usage_query_options(group, dates=True, dates_required=False)
    group.add_argument(
        "--by",
        required=True,
        choices=(
            "resource_kind",
            "model",
            "tool_service",
            "profile",
            "channel",
            "goal",
            "job",
            "child_task",
        ),
    )
    group.set_defaults(limit=10_000)

    export = usage_subparsers.add_parser(
        "export",
        help="Write a bounded credential-free usage export.",
    )
    _add_usage_query_options(export, dates=True, dates_required=False)
    export.add_argument("--format", choices=("csv", "json"), default="json")
    export.add_argument("--output", required=True)
    export.add_argument("--max-entries", type=int, default=10_000)
    export.add_argument("--force", action="store_true")


def _add_usage_query_options(
    parser: argparse.ArgumentParser,
    *,
    dates: bool,
    dates_required: bool = True,
) -> None:
    if dates:
        parser.add_argument("--from", dest="start", required=dates_required)
        parser.add_argument("--to", dest="end", required=dates_required)
    parser.add_argument(
        "--resource-kind",
        choices=("model", "tool", "media", "external_service"),
    )
    parser.add_argument("--channel")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json", action="store_true", dest="json_output")


def _add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("doctor", help="Validate local Chulk configuration.")
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )


def _add_init_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("init", help="Initialize project-local Chulk files.")
    parser.add_argument(
        "--project-root", default=".", help="Project directory to initialize."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sdk", action="store_const", const="sdk", dest="init_mode")
    mode.add_argument(
        "--coding-agent", action="store_const", const="coding-agent", dest="init_mode"
    )
    mode.add_argument(
        "--read-only", action="store_const", const="read-only", dest="init_mode"
    )
    parser.set_defaults(init_mode="coding-agent")
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )


def _add_trace_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "trace", help="Inspect, replay, or export a JSONL trace."
    )
    trace_subparsers = parser.add_subparsers(dest="trace_command", required=True)
    inspect_parser = trace_subparsers.add_parser(
        "inspect", help="Summarize a trace file."
    )
    inspect_parser.add_argument("path", help="Path to a Chulk JSONL trace.")
    inspect_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )
    _add_trace_limits(inspect_parser)
    replay_parser = trace_subparsers.add_parser(
        "replay",
        help="Reconstruct a trace or execute a deterministic replay fixture.",
    )
    replay_source = replay_parser.add_mutually_exclusive_group(required=True)
    replay_source.add_argument(
        "path",
        nargs="?",
        help="Path to a Chulk JSONL trace for read-only reconstruction.",
    )
    replay_source.add_argument(
        "--execute-fixture",
        metavar="PATH",
        help="Execute a versioned replay fixture through the offline action loop.",
    )
    replay_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )
    _add_trace_limits(replay_parser)
    export_parser = trace_subparsers.add_parser("export", help="Export a trace report.")
    export_parser.add_argument("path", help="Path to a Chulk JSONL trace.")
    export_parser.add_argument("--format", choices=("html",), default="html")
    export_parser.add_argument(
        "--output", help="Destination path (defaults beside the trace)."
    )
    export_parser.add_argument(
        "--force", action="store_true", help="Replace an existing destination."
    )
    export_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit structured JSON."
    )
    _add_trace_limits(export_parser)


def _add_trace_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="Maximum trace bytes to parse (default: 67108864).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        help="Maximum non-empty trace events to parse (default: 100000).",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Trusted-operator override that disables trace parse limits.",
    )


__all__ = ["build_parser"]
