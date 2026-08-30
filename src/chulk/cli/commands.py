"""Registry-backed interactive slash-command handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from typing import cast
from uuid import uuid4

from chulk.cli.progress import ProgressSettings
from chulk.cli.terminal import TerminalUI
from chulk.config import Config
from chulk.core import Agent
from chulk.llm import LLMError
from chulk.skills import SkillScope, SkillUsageKind, explicit_skill_names
from chulk.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SQLiteSessionStore,
)


CommandHandler = Callable[[str, "CLICommandContext"], None]


@dataclass(frozen=True)
class CLICommand:
    """One discoverable interactive command."""

    name: str
    usage: str
    description: str
    category: str
    handler: CommandHandler | None
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass
class CLICommandContext:
    """Runtime state needed by interactive slash commands."""

    agent: Agent
    config: Config | None
    terminal: TerminalUI
    progress_settings: ProgressSettings
    output_func: Callable[[str], None]
    response_func: Callable[[str], None] | None = None
    session_store: SQLiteSessionStore | None = None
    agent_factory: Callable[[str], Agent] | None = None
    switch_agent: Callable[[Agent], None] | None = None
    model_profile_id: str | None = None
    active_model_profile_id: str | None = None
    model_selector: Callable[[str, str], tuple[Agent, Config, str]] | None = None
    switch_runtime: Callable[[Agent, Config], None] | None = None


def _help(_arguments: str, context: CLICommandContext) -> None:
    context.output_func(context.terminal.help_text(CLI_COMMANDS))


def _status(_arguments: str, context: CLICommandContext) -> None:
    if context.config is None:
        context.output_func(
            context.terminal.warning("status unavailable: no config object")
        )
        return
    context.output_func(context.terminal.status(context.config, context.agent))


def _context(_arguments: str, context: CLICommandContext) -> None:
    context.output_func(context.terminal.context(context.agent))


def _tools(_arguments: str, context: CLICommandContext) -> None:
    context.output_func(context.terminal.tools(context.agent))


def _mcp(_arguments: str, context: CLICommandContext) -> None:
    if context.config is None:
        context.output_func(
            context.terminal.warning("mcp unavailable: no config object")
        )
        return
    context.output_func(context.terminal.mcp(context.config, context.agent))


def _sessions(_arguments: str, context: CLICommandContext) -> None:
    if context.session_store is None:
        context.output_func(
            context.terminal.warning("sessions unavailable: no session store")
        )
        return
    context.output_func(
        context.terminal.sessions(context.session_store.list_conversations())
    )


def _history(_arguments: str, context: CLICommandContext) -> None:
    if context.session_store is None:
        context.output_func(
            context.terminal.warning("history unavailable: no session store")
        )
        return
    messages = context.session_store.list_messages(
        context.agent.state.conversation_id, limit=40
    )
    context.output_func(context.terminal.history(messages))


def _resume(arguments: str, context: CLICommandContext) -> None:
    if not arguments:
        context.output_func(
            context.terminal.warning("usage: /resume <conversation_id>")
        )
        return
    if context.agent_factory is None or context.switch_agent is None:
        context.output_func(
            context.terminal.warning("resume unavailable: no agent factory")
        )
        return
    try:
        next_agent = context.agent_factory(arguments)
    except (SessionNotFoundError, AmbiguousSessionError) as exc:
        context.output_func(context.terminal.warning(str(exc)))
        return
    context.switch_agent(next_agent)
    context.output_func(
        context.terminal.warning(
            f"resumed session {next_agent.state.conversation_id[:8]}"
        )
    )


def _trace(_arguments: str, context: CLICommandContext) -> None:
    context.output_func(context.terminal.trace(context.agent))


def _model(arguments: str, context: CLICommandContext) -> None:
    if not arguments:
        requested = context.model_profile_id or "unknown"
        active = context.active_model_profile_id or requested
        suffix = f" (active fallback: {active})" if active != requested else ""
        context.output_func(
            context.terminal.warning(f"model profile {requested}{suffix}")
        )
        return
    if context.model_selector is None or context.switch_runtime is None:
        context.output_func(
            context.terminal.warning("model switching unavailable in this host")
        )
        return
    try:
        next_agent, next_config, active_id = context.model_selector(
            arguments,
            context.agent.state.conversation_id,
        )
    except (LLMError, LookupError, OSError, ValueError) as exc:
        context.output_func(context.terminal.warning(str(exc)))
        return
    context.switch_runtime(next_agent, next_config)
    context.model_profile_id = arguments.strip().lower()
    context.active_model_profile_id = active_id
    suffix = (
        f" (active fallback: {active_id})"
        if active_id != context.model_profile_id
        else ""
    )
    context.output_func(
        context.terminal.warning(f"model profile {context.model_profile_id}{suffix}")
    )


def _plan(arguments: str, context: CLICommandContext) -> None:
    if not arguments:
        context.output_func(context.terminal.plan_status(context.agent))
        return
    response = context.agent.run_planned_turn(arguments)
    _respond(context, response)


def _approve(_arguments: str, context: CLICommandContext) -> None:
    response = context.agent.approve_plan()
    _respond(context, response)


def _reject(_arguments: str, context: CLICommandContext) -> None:
    response = context.agent.reject_plan()
    _respond(context, response)


def _display(arguments: str, context: CLICommandContext) -> None:
    mode = arguments.strip().lower()
    if not mode:
        context.output_func(
            context.terminal.warning(f"display mode {context.progress_settings.mode}")
        )
        return
    try:
        context.progress_settings.set_mode(mode)
    except ValueError:
        context.output_func(
            context.terminal.warning("usage: /display compact|verbose|quiet")
        )
        return
    context.output_func(context.terminal.warning(f"display mode {mode}"))


def _learning(arguments: str, context: CLICommandContext) -> None:
    service = context.agent.learning.proposals
    if service is None:
        context.output_func(
            context.terminal.warning("learning proposals are unavailable")
        )
        return
    action, _, proposal_id = arguments.strip().partition(" ")
    action = action or "pending"
    try:
        if action == "pending":
            proposals = service.list()
            lines = [
                (
                    f"{item.id} {item.kind.value} {item.status.value}"
                    + (
                        f" target={item.target_name}"
                        if item.target_name is not None
                        else ""
                    )
                )
                for item in proposals
            ]
            context.output_func("\n".join(lines) if lines else "No pending proposals.")
            return
        if action == "review":
            outcome = context.agent.learning.review(
                turn_id=proposal_id or None,
            )
            if outcome.proposal_ids:
                context.output_func(
                    "\n".join(
                        f"created learning proposal {item}"
                        for item in outcome.proposal_ids
                    )
                )
            else:
                context.output_func(
                    f"No proposal created: {outcome.rationale}"
                )
            return
        if action == "approve" and proposal_id:
            proposal = service.approve(
                proposal_id,
                approved_by="cli-host",
            )
        elif action == "reject" and proposal_id:
            proposal = service.reject(
                proposal_id,
                rejected_by="cli-host",
            )
        else:
            raise ValueError(
                "usage: /learning review [turn]|pending|approve <id>|"
                "reject <id>"
            )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        context.output_func(context.terminal.warning(str(exc)))
        return
    context.output_func(
        context.terminal.warning(
            f"{proposal.kind.value} proposal {proposal.id} "
            f"is {proposal.status.value}"
        )
    )


def _skills(arguments: str, context: CLICommandContext) -> None:
    store = context.agent.skill_context.lifecycle_store
    lifecycle = context.agent.skill_context.lifecycle
    proposals = context.agent.learning.proposals
    if store is None or lifecycle is None or proposals is None:
        context.output_func(
            context.terminal.warning("skill lifecycle is unavailable")
        )
        return
    action, _, remainder = arguments.strip().partition(" ")
    action = action or "list"
    try:
        if action == "list":
            event_id = f"cli-view:{uuid4()}"
            records = tuple(
                store.record_usage(
                    name=item.name,
                    scope=item.scope,
                    version=item.version,
                    digest=item.digest,
                    kind=SkillUsageKind.VIEW,
                    source_event_id=event_id,
                )
                for item in store.list_skills()
            )
            lines = [
                (
                    f"{item.scope}:{item.name} {item.version} "
                    f"{item.status.value} uses={item.use_count} "
                    f"successes={item.success_count}"
                )
                for item in records
            ]
            context.output_func("\n".join(lines) if lines else "No governed skills.")
            return
        if action == "pending":
            items = [
                item
                for item in proposals.list()
                if item.kind.value.startswith("skill_")
            ]
            lines = [
                f"{item.id} {item.kind.value} target={item.target_name}"
                for item in items
            ]
            context.output_func(
                "\n".join(lines) if lines else "No pending skill proposals."
            )
            return
        if action == "diff" and remainder:
            proposal = proposals.get(remainder)
            if not proposal.kind.value.startswith("skill_"):
                raise ValueError("proposal is not a skill change")
            context.output_func(proposal.diff or "No diff recorded.")
            return
        if action == "history" and remainder:
            name, _, scope = remainder.partition(" ")
            selected_scope = scope.strip() or "project"
            revisions = store.list_revisions(
                name,
                scope=selected_scope,
            )
            lines = [
                (
                    f"{item.id} {item.scope}:{item.name} "
                    f"{item.version} {item.digest}"
                )
                for item in revisions
            ]
            context.output_func(
                "\n".join(lines) if lines else "No skill revisions."
            )
            return
        if action == "approve" and remainder:
            proposal = proposals.approve(
                remainder,
                approved_by="cli-host",
            )
            context.output_func(
                context.terminal.warning(
                    f"skill proposal {proposal.id} is {proposal.status.value}"
                )
            )
            return
        if action == "reject" and remainder:
            proposal = proposals.reject(
                remainder,
                rejected_by="cli-host",
            )
            context.output_func(
                context.terminal.warning(
                    f"skill proposal {proposal.id} is {proposal.status.value}"
                )
            )
            return
        if action == "rollback" and remainder:
            revision_id, _, scope = remainder.partition(" ")
            selected_scope = scope.strip() or "project"
            if selected_scope not in {"project", "profile"}:
                raise ValueError("scope must be project or profile")
            record = lifecycle.rollback(
                revision_id,
                scope=cast(SkillScope, selected_scope),
                approved_by="cli-host",
            )
            context.output_func(
                context.terminal.warning(
                    f"restored {record.scope}:{record.name} {record.version}"
                )
            )
            return
        raise ValueError(
            "usage: /skills list|pending|diff <id>|approve <id>|"
            "reject <id>|history <name> [scope]|"
            "rollback <revision> [project|profile]"
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        context.output_func(context.terminal.warning(str(exc)))


def _clear(_arguments: str, context: CLICommandContext) -> None:
    context.output_func(context.terminal.clear())


def _respond(context: CLICommandContext, response: str) -> None:
    if context.response_func is not None:
        context.response_func(response)
        return
    context.output_func(context.terminal.assistant_message(response))


CLI_COMMANDS: tuple[CLICommand, ...] = (
    CLICommand(
        "/help",
        "/help",
        "show this command list",
        "General",
        _help,
        aliases=("help", "?"),
    ),
    CLICommand(
        "/exit",
        "/exit",
        "end the session",
        "General",
        None,
        aliases=("/quit", "/q", "exit", "quit"),
    ),
    CLICommand(
        "/plan", "/plan [request]", "show or propose an approval plan", "Run", _plan
    ),
    CLICommand("/approve", "/approve", "approve the pending plan", "Run", _approve),
    CLICommand("/reject", "/reject", "cancel the active plan", "Run", _reject),
    CLICommand("/status", "/status", "show runtime status", "Inspect", _status),
    CLICommand(
        "/context", "/context", "show the latest prompt context", "Inspect", _context
    ),
    CLICommand("/tools", "/tools", "list registered tools", "Inspect", _tools),
    CLICommand("/mcp", "/mcp", "show configured MCP servers", "Inspect", _mcp),
    CLICommand("/trace", "/trace", "show the current trace file", "Inspect", _trace),
    CLICommand(
        "/learning",
        "/learning review [turn]|pending|approve <id>|reject <id>",
        "review durable learning proposals",
        "Learning",
        _learning,
    ),
    CLICommand(
        "/skills",
        "/skills list|pending|diff|approve|reject|history|rollback",
        "inspect and manage governed skills",
        "Learning",
        _skills,
    ),
    CLICommand(
        "/model", "/model [profile]", "show or switch model profile", "Run", _model
    ),
    CLICommand(
        "/sessions",
        "/sessions",
        "list recent persisted sessions",
        "Sessions",
        _sessions,
    ),
    CLICommand(
        "/resume", "/resume <id>", "resume a persisted session", "Sessions", _resume
    ),
    CLICommand(
        "/history", "/history", "show recent persisted messages", "Sessions", _history
    ),
    CLICommand(
        "/display",
        "/display compact|verbose|quiet",
        "choose transcript detail",
        "Display",
        _display,
    ),
    CLICommand("/clear", "/clear", "clear the terminal screen", "Display", _clear),
)


EXIT_COMMANDS = frozenset(
    name.lower()
    for command in CLI_COMMANDS
    if command.name == "/exit"
    for name in command.names
)


def command_completion_candidates() -> tuple[str, ...]:
    """Return canonical slash-command names for readline completion."""
    return tuple(command.name for command in CLI_COMMANDS)


def handle_cli_command(command: str, context: CLICommandContext) -> bool:
    """Handle a CLI command and consume unknown slash-prefixed input."""
    raw_command = command.strip()
    if raw_command.startswith("/"):
        command_name, _, raw_arguments = raw_command.partition(" ")
    else:
        command_name = raw_command
        raw_arguments = ""
    normalized_name = command_name.lower()
    arguments = raw_arguments.strip()

    for command_spec in CLI_COMMANDS:
        if normalized_name not in {name.lower() for name in command_spec.names}:
            continue
        if command_spec.handler is not None:
            command_spec.handler(arguments, context)
        return True

    if raw_command.startswith("/"):
        explicit_names = explicit_skill_names(raw_command)
        registry = context.agent.skill_context.registry
        if explicit_names and registry is not None:
            if all(
                registry.get_skill(name) is not None
                for name in explicit_names
            ):
                routing = registry.route_skills(
                    raw_command,
                    pinned_names=context.agent.pinned_skill_names,
                    limit=context.agent.max_skills_per_turn,
                )
                if routing.errors:
                    error = routing.errors[0]
                    context.output_func(
                        context.terminal.warning(
                            f"Skill /{error.skill_name} is unavailable: "
                            f"{error.reason}."
                        )
                    )
                    return True
                return False
        canonical_names = [command_spec.name for command_spec in CLI_COMMANDS]
        suggestion = get_close_matches(
            normalized_name, canonical_names, n=1, cutoff=0.5
        )
        suffix = (
            f" Did you mean {suggestion[0]}?"
            if suggestion
            else " Type /help for commands."
        )
        context.output_func(
            context.terminal.warning(f"Unknown command: {command_name}.{suffix}")
        )
        return True
    return False


__all__ = [
    "CLICommand",
    "CLICommandContext",
    "CLI_COMMANDS",
    "EXIT_COMMANDS",
    "command_completion_candidates",
    "handle_cli_command",
]
