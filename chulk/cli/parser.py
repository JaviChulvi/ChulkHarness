"""Argument parser for the Chulk command-line interface."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the backwards-compatible CLI and its explicit subcommands."""
    parser = argparse.ArgumentParser(
        prog="chulk",
        description="Run the ChulkHarness agent runtime.",
    )
    parser.add_argument("--version", action="store_true", help="Print the ChulkHarness version and exit.")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved local configuration and exit.",
    )
    parser.add_argument("--once", metavar="MESSAGE", help="Compatibility alias for 'chulk exec MESSAGE'.")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Control ANSI color output (default: auto).",
    )
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--resume", metavar="ID", help="Start in a persisted interactive session.")
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
    _add_trace_parser(subparsers)
    return parser


def _add_exec_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("exec", help="Run one non-interactive agent request.")
    parser.add_argument("message", nargs="+", help="Message to send to the agent.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")


def _add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("doctor", help="Validate local Chulk configuration.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")


def _add_init_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("init", help="Initialize project-local Chulk files.")
    parser.add_argument("--project-root", default=".", help="Project directory to initialize.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sdk", action="store_const", const="sdk", dest="init_mode")
    mode.add_argument("--coding-agent", action="store_const", const="coding-agent", dest="init_mode")
    mode.add_argument("--read-only", action="store_const", const="read-only", dest="init_mode")
    parser.set_defaults(init_mode="coding-agent")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")


def _add_trace_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("trace", help="Inspect, replay, or export a JSONL trace.")
    trace_subparsers = parser.add_subparsers(dest="trace_command", required=True)
    inspect_parser = trace_subparsers.add_parser("inspect", help="Summarize a trace file.")
    inspect_parser.add_argument("path", help="Path to a Chulk JSONL trace.")
    inspect_parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")
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
    replay_parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")
    _add_trace_limits(replay_parser)
    export_parser = trace_subparsers.add_parser("export", help="Export a trace report.")
    export_parser.add_argument("path", help="Path to a Chulk JSONL trace.")
    export_parser.add_argument("--format", choices=("html",), default="html")
    export_parser.add_argument("--output", help="Destination path (defaults beside the trace).")
    export_parser.add_argument("--force", action="store_true", help="Replace an existing destination.")
    export_parser.add_argument("--json", action="store_true", dest="json_output", help="Emit structured JSON.")
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
