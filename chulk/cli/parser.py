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
    _add_plugins_parser(subparsers)
    _add_usage_parser(subparsers)
    _add_goal_parser(subparsers)
    _add_child_parser(subparsers)
    _add_automation_parser(subparsers)
    _add_session_parser(subparsers)
    _add_gateway_parser(subparsers)
    _add_server_parser(subparsers)
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


def _add_gateway_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gateway",
        help="Run or administer the shared channel gateway.",
    )
    commands = parser.add_subparsers(dest="gateway_command", required=True)

    start = commands.add_parser("start", help="Run the configured gateway.")
    _add_gateway_identity(start)

    status = commands.add_parser("status", help="Show durable adapter status.")
    _add_gateway_identity(status)
    status.add_argument("--json", action="store_true", dest="json_output")

    stop = commands.add_parser("stop", help="Request a running adapter to stop.")
    _add_gateway_identity(stop)
    stop.add_argument("--json", action="store_true", dest="json_output")

    routes = commands.add_parser("routes", help="List or change identity routes.")
    route_commands = routes.add_subparsers(dest="route_command", required=True)
    route_list = route_commands.add_parser("list", help="List gateway routes.")
    route_list.add_argument("--include-disabled", action="store_true")
    route_list.add_argument("--json", action="store_true", dest="json_output")
    route_add = route_commands.add_parser("add", help="Add an owner-approved route.")
    _add_gateway_identity(route_add)
    route_add.add_argument("--profile", required=True)
    route_add.add_argument("--principal")
    route_add.add_argument("--destination")
    route_add.add_argument("--thread")
    route_add.add_argument("--json", action="store_true", dest="json_output")
    route_remove = route_commands.add_parser("remove", help="Disable a route.")
    route_remove.add_argument("route_id")
    route_remove.add_argument("--json", action="store_true", dest="json_output")

    pair = commands.add_parser("pair", help="Create a one-time pairing code.")
    _add_gateway_identity(pair)
    pair.add_argument("--profile", required=True)
    pair.add_argument("--principal")
    pair.add_argument("--ttl-seconds", type=int, default=600)
    pair.add_argument("--json", action="store_true", dest="json_output")


def _add_gateway_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", default="telegram")
    parser.add_argument("--account", default="primary")


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "server",
        help="Run or administer the authenticated local control server.",
    )
    commands = parser.add_subparsers(dest="server_command", required=True)
    start = commands.add_parser("start", help="Run the local control server.")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)
    start.add_argument(
        "--allow-remote",
        action="store_true",
        help="Explicitly permit a non-loopback bind.",
    )
    status = commands.add_parser("status", help="Show durable server status.")
    status.add_argument("--json", action="store_true", dest="json_output")
    stop = commands.add_parser("stop", help="Request the running server to stop.")
    stop.add_argument("--json", action="store_true", dest="json_output")
    rotate = commands.add_parser(
        "rotate-token",
        help="Rotate the owner-local control credential.",
    )
    rotate.add_argument("--json", action="store_true", dest="json_output")


def _add_plugins_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "plugins",
        help="Inspect, register, list, or audit local plugins.",
    )
    plugin_subparsers = parser.add_subparsers(
        dest="plugin_command",
        required=True,
    )

    inspect = plugin_subparsers.add_parser(
        "inspect",
        help="Statically inspect a local package without importing it.",
    )
    inspect.add_argument("path")
    inspect.add_argument("--json", action="store_true", dest="json_output")

    register = plugin_subparsers.add_parser(
        "register",
        help="Register an exact local package after explicit review.",
    )
    register.add_argument("path")
    register.add_argument("--approved-by", required=True)
    register.add_argument(
        "--acknowledge-host-authority",
        action="store_true",
        help="Acknowledge that importing Python code has host-process authority.",
    )
    register.add_argument(
        "--grant-capability",
        action="append",
        default=[],
        dest="granted_capabilities",
    )
    register.add_argument("--json", action="store_true", dest="json_output")

    list_parser = plugin_subparsers.add_parser(
        "list",
        help="List reviewed profile-local registrations.",
    )
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    audit = plugin_subparsers.add_parser(
        "audit",
        help="Verify exact locks and packages without importing code.",
    )
    audit.add_argument("--json", action="store_true", dest="json_output")


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


def _add_goal_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "goal",
        help="Promote, inspect, and control durable goals.",
    )
    commands = parser.add_subparsers(dest="goal_command", required=True)

    list_parser = commands.add_parser("list", help="List profile-owned goals.")
    list_parser.add_argument(
        "--status",
        choices=(
            "draft",
            "approved",
            "running",
            "paused",
            "blocked",
            "completed",
            "cancelled",
            "failed",
        ),
    )
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    inspect = commands.add_parser("inspect", help="Inspect one durable goal.")
    inspect.add_argument("goal_id")
    inspect.add_argument("--json", action="store_true", dest="json_output")

    export = commands.add_parser(
        "export",
        help="Write one bounded redacted goal and its event history.",
    )
    export.add_argument("goal_id")
    export.add_argument("--output", required=True)
    export.add_argument("--force", action="store_true")
    export.add_argument("--json", action="store_true", dest="json_output")

    promote = commands.add_parser(
        "promote",
        help="Copy a persisted conversation plan into a durable goal.",
    )
    promote.add_argument("conversation_id")
    promote.add_argument("--turn", dest="turn_id")
    promote.add_argument("--max-model-calls", type=int)
    promote.add_argument("--max-tool-calls", type=int)
    promote.add_argument("--max-tokens", type=int)
    promote.add_argument("--max-cost")
    promote.add_argument("--deadline")
    promote.add_argument("--actor", default="cli")
    promote.add_argument("--json", action="store_true", dest="json_output")

    for name, help_text in (
        ("approve", "Approve a draft goal."),
        ("run", "Move an approved goal into running state."),
        ("pause", "Pause a goal between actions."),
        ("resume", "Resume a paused or blocked goal."),
        ("cancel", "Request durable goal cancellation."),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_goal_mutation_options(command)
        if name == "approve":
            command.add_argument("--reason")

    steer = commands.add_parser("steer", help="Append operator steering.")
    steer.add_argument("goal_id")
    steer.add_argument("instruction", nargs="+")
    steer.add_argument("--revision", type=int, required=True)
    steer.add_argument("--actor", default="cli")
    steer.add_argument("--json", action="store_true", dest="json_output")

    approve_step = commands.add_parser(
        "approve-step",
        help="Approve one selected high-risk step.",
    )
    _add_goal_step_mutation_options(approve_step)
    approve_step.add_argument("--reason")

    skip_step = commands.add_parser(
        "skip-step",
        help="Skip one step with an auditable operator reason.",
    )
    _add_goal_step_mutation_options(skip_step)
    skip_step.add_argument("--reason", required=True)

    retry_step = commands.add_parser(
        "retry-step",
        help="Retry a blocked, failed, or uncertain step.",
    )
    _add_goal_step_mutation_options(retry_step)


def _add_child_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "child",
        help="Inspect and control durable child tasks.",
    )
    commands = parser.add_subparsers(dest="child_command", required=True)

    list_parser = commands.add_parser(
        "list",
        help="List profile-owned child tasks.",
    )
    list_parser.add_argument(
        "--status",
        choices=(
            "pending",
            "ready",
            "running",
            "waiting",
            "completed",
            "failed",
            "blocked",
            "cancelled",
            "budget_exhausted",
            "unknown",
        ),
    )
    list_parser.add_argument("--goal")
    list_parser.add_argument("--parent")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    inspect = commands.add_parser("inspect", help="Inspect one child task.")
    inspect.add_argument("task_id")
    inspect.add_argument("--json", action="store_true", dest="json_output")

    for name, help_text in (
        ("cancel", "Cancel a child task and its descendants."),
        ("retry", "Explicitly retry a failed or unknown child task."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("task_id")
        command.add_argument("--revision", type=int, required=True)
        command.add_argument("--actor", default="cli")
        command.add_argument("--json", action="store_true", dest="json_output")
        if name == "cancel":
            command.add_argument("--reason")

    recover = commands.add_parser(
        "recover",
        help="Mark expired in-flight child attempts unknown.",
    )
    recover.add_argument("--actor", default="cli-recovery")
    recover.add_argument("--json", action="store_true", dest="json_output")

    deliveries = commands.add_parser(
        "deliveries",
        help="List durable parent-completion deliveries.",
    )
    deliveries.add_argument(
        "--status",
        choices=("pending", "claimed", "delivered", "failed", "unknown"),
    )
    deliveries.add_argument("--limit", type=int, default=100)
    deliveries.add_argument("--json", action="store_true", dest="json_output")


def _add_automation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "automation",
        help="Inspect and control profile-owned automation.",
    )
    commands = parser.add_subparsers(dest="automation_command", required=True)
    list_parser = commands.add_parser("list", help="List automation definitions.")
    list_parser.add_argument(
        "--status",
        choices=(
            "pending_approval",
            "active",
            "running",
            "paused",
            "completed",
            "cancelled",
            "expired",
        ),
    )
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--json", action="store_true", dest="json_output")
    for name in ("inspect", "history", "triggers"):
        command = commands.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--limit", type=int, default=100)
        command.add_argument("--json", action="store_true", dest="json_output")
    for name in ("pause", "resume", "approve", "run-now", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--revision", type=int, required=True)
        command.add_argument("--idempotency-key")
        command.add_argument("--actor", default="cli")
        command.add_argument("--json", action="store_true", dest="json_output")
    update = commands.add_parser("update")
    update.add_argument("job_id")
    update.add_argument("--revision", type=int, required=True)
    update.add_argument("--idempotency-key")
    update.add_argument("--actor", default="cli")
    update.add_argument("--prompt")
    update.add_argument("--run-at")
    update.add_argument("--timezone", default="UTC")
    recurrence = update.add_mutually_exclusive_group()
    recurrence.add_argument("--interval-seconds", type=int)
    recurrence.add_argument("--cron")
    recurrence.add_argument("--rrule")
    update.add_argument("--json", action="store_true", dest="json_output")
    recover = commands.add_parser("recover")
    recover.add_argument("--actor", default="cli-recovery")
    recover.add_argument("--json", action="store_true", dest="json_output")
    webhook = commands.add_parser("webhook")
    webhook.add_argument("job_id")
    webhook.add_argument("--json", action="store_true", dest="json_output")
    completion = commands.add_parser("completion-trigger")
    completion.add_argument("job_id")
    completion.add_argument(
        "--kind",
        required=True,
        choices=("job_completion", "goal_completion", "child_completion"),
    )
    completion.add_argument("--source", required=True)
    completion.add_argument("--json", action="store_true", dest="json_output")


def _add_goal_mutation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("goal_id")
    parser.add_argument("--revision", type=int, required=True)
    parser.add_argument("--actor", default="cli")
    parser.add_argument("--json", action="store_true", dest="json_output")


def _add_goal_step_mutation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("goal_id")
    parser.add_argument("step_id")
    parser.add_argument("--revision", type=int, required=True)
    parser.add_argument("--actor", default="cli")
    parser.add_argument("--json", action="store_true", dest="json_output")


def _add_session_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "session",
        help="Search, read, or rebuild profile-owned session evidence.",
    )
    session_subparsers = parser.add_subparsers(
        dest="session_command",
        required=True,
    )

    search = session_subparsers.add_parser(
        "search",
        help="Search eligible prior-session messages.",
    )
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--cursor")
    search.add_argument("--json", action="store_true", dest="json_output")

    read = session_subparsers.add_parser(
        "read",
        help="Read a bounded redacted message window.",
    )
    read.add_argument("conversation_id")
    read.add_argument("ordinal", type=int)
    read.add_argument("--before", type=int, default=3)
    read.add_argument("--after", type=int, default=3)
    read.add_argument("--limit", type=int, default=20)
    read.add_argument("--cursor")
    read.add_argument("--json", action="store_true", dest="json_output")

    rebuild = session_subparsers.add_parser(
        "rebuild-index",
        help="Deterministically rebuild the eligible session FTS index.",
    )
    rebuild.add_argument("--json", action="store_true", dest="json_output")


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
