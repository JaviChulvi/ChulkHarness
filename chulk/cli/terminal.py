"""Small ANSI terminal skin for the CLI.

The CLI intentionally avoids a heavy dependency here. This gives Chulk a nicer
terminal presence while keeping the harness easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from chulk.config import Config
from chulk.core import Agent, TraceEvent
from chulk.sessions import ConversationRecord, MessageRecord


HULK_GREEN = (63, 255, 81)
DIM_GREEN = (83, 176, 99)
MUTED = (142, 155, 148)
ERROR_RED = (255, 96, 96)


@dataclass(frozen=True)
class TerminalUI:
    """Format CLI text with an optional Hulk-green ANSI theme."""

    color_enabled: bool = True
    width: int = 80

    @classmethod
    def themed(cls) -> "TerminalUI":
        """Create the default Hulk-green terminal formatter."""
        width = shutil.get_terminal_size((80, 24)).columns
        return cls(color_enabled=True, width=max(64, min(width, 110)))

    def prompt(self) -> str:
        """Return the interactive input prompt."""
        return f"{self.accent('>')} "

    def banner(self, config: Config, agent: Agent) -> str:
        """Return the startup banner."""
        title = "ChulkHarness CLI"
        top = _rule(title, self.width)
        bottom = "+" + "-" * (len(top) - 2) + "+"
        trace_path = agent.trace_logger.path if agent.trace_logger else None
        rows = [
            self._row("mode", "interactive agent harness"),
            self._row("provider", _provider_text(config)),
            self._row("session", _short_id(agent.state.conversation_id)),
            self._row("project", _short_path(config.project_root)),
            self._row("tools", str(len(agent.tool_registry.list_tools()))),
            self._row("trace", _short_path(trace_path) if trace_path else "disabled"),
        ]
        return "\n".join([self.accent(top), *rows, self.accent(bottom)])

    def help_text(self) -> str:
        """Return slash-command help."""
        commands = [
            ("/help", "show this command list"),
            ("/status", "show provider, model, project, tools, and trace"),
            ("/context", "show latest prompt context report"),
            ("/tools", "list registered tools"),
            ("/sessions", "list recent persisted sessions"),
            ("/resume <id>", "resume a persisted session"),
            ("/history", "show recent persisted messages"),
            ("/trace", "show the current trace file"),
            ("/plan <request>", "propose a plan for one request"),
            ("/plan", "show current plan status"),
            ("/approve", "approve a pending plan"),
            ("/reject", "reject a pending plan"),
            ("/verbose on|off", "show or hide extra progress details"),
            ("/quiet on|off", "show or hide live progress lines"),
            ("/summary on|off", "show or hide the end-of-turn summary"),
            ("/clear", "clear the terminal screen"),
            ("/q", "exit the session"),
        ]
        lines = [self.heading("Commands")]
        lines.extend(f"  {self.accent(command):<12} {description}" for command, description in commands)
        return "\n".join(lines)

    def status(self, config: Config, agent: Agent) -> str:
        """Return a compact runtime status block."""
        trace_path = agent.trace_logger.path if agent.trace_logger else None
        lines = [
            self.heading("Status"),
            f"  provider  {_provider_text(config)}",
            f"  session   {agent.state.conversation_id}",
            f"  project   {_short_path(config.project_root)}",
            f"  memory    {_short_path(config.store_path)}",
            f"  trace     {_short_path(trace_path) if trace_path else 'disabled'}",
            f"  tools     {len(agent.tool_registry.list_tools())}",
            f"  turns     {len(agent.state.turns)}",
            f"  plan      {'pending' if agent.has_pending_plan() else 'none'}",
            f"  context   {_context_status(agent.state.last_context_report, getattr(agent, 'context_budget', None))}",
        ]
        return "\n".join(lines)

    def context(self, agent: Agent) -> str:
        """Return the latest prompt context report."""
        report = agent.state.last_context_report
        if not isinstance(report, dict):
            return f"{self.heading('Context')}\n  no model request context recorded yet"

        budget = report.get("budget", {})
        sections = [section for section in report.get("sections", []) if isinstance(section, dict)]
        largest_sections = sorted(
            sections,
            key=lambda section: int(section.get("estimated_tokens") or 0),
            reverse=True,
        )[:8]
        lines = [
            self.heading("Context"),
            f"  estimated {_format_count(int(report.get('estimated_tokens') or 0))} tokens, "
            f"{_format_count(int(report.get('total_char_count') or 0))} chars",
            f"  budget    {_context_budget_text(budget, int(report.get('over_budget_tokens') or 0))}",
            f"  messages  {report.get('included_message_count', 0)} included, "
            f"{report.get('omitted_message_count', 0)} omitted",
            f"  obs omit  {report.get('omitted_observation_count', 0)} observation message(s)",
            "  sections",
        ]
        if not largest_sections:
            lines.append("    none")
            return "\n".join(lines)
        for section in largest_sections:
            name = str(section.get("label") or section.get("name") or "section")
            tokens = _format_count(int(section.get("estimated_tokens") or 0))
            chars = _format_count(int(section.get("char_count") or 0))
            items = section.get("item_count", 0)
            detail = _context_section_detail(section)
            suffix = f" - {detail}" if detail else ""
            lines.append(f"    {name:<22} {tokens:>6} tok  {chars:>6} chars  items {items}{suffix}")
        return "\n".join(lines)

    def plan_status(self, agent: Agent) -> str:
        """Return current plan-mode and pending-plan status."""
        return f"{self.heading('Plan')}\n  {agent.describe_plan_status().replace(chr(10), chr(10) + '  ')}"

    def tools(self, agent: Agent) -> str:
        """Return registered tool names and descriptions."""
        lines = [self.heading("Tools")]
        for tool in agent.tool_registry.list_tools():
            lines.append(f"  {self.accent(tool.name):<24} {tool.description}")
        return "\n".join(lines)

    def sessions(self, records: list[ConversationRecord]) -> str:
        """Return recent persisted sessions."""
        lines = [self.heading("Sessions")]
        if not records:
            lines.append("  none")
            return "\n".join(lines)
        for record in records:
            title = record.title or "(untitled)"
            lines.append(
                f"  {self.accent(_short_id(record.id)):<8} "
                f"{record.status:<20} turns {record.turn_count:<3} "
                f"{_short_timestamp(record.updated_at):<16} {_compact(title, 56)}"
            )
        return "\n".join(lines)

    def history(self, messages: list[MessageRecord]) -> str:
        """Return recent persisted conversation messages."""
        lines = [self.heading("History")]
        if not messages:
            lines.append("  none")
            return "\n".join(lines)
        for message in messages:
            lines.append(f"  {message.role:<11} {_compact(message.content, 86)}")
        return "\n".join(lines)

    def trace(self, agent: Agent) -> str:
        """Return the current trace path."""
        if agent.trace_logger is None:
            return self.warning("trace disabled")
        return f"{self.heading('Trace')}\n  {agent.trace_logger.path}"

    def assistant_message(self, content: str) -> str:
        """Format an assistant response."""
        return _labeled_block(self.accent("chulk"), content)

    def progress(
        self,
        event_type: str,
        payload: dict,
        *,
        elapsed_seconds: float | None = None,
        duration_seconds: float | None = None,
        verbose: bool = False,
    ) -> str | None:
        """Return a compact status line for a trace event."""
        message = _progress_message(
            event_type,
            payload,
            elapsed_seconds=elapsed_seconds,
            duration_seconds=duration_seconds,
            verbose=verbose,
        )
        if message is None:
            return None
        return f"{self.muted('..')} {message}"

    def turn_summary(self, payload: dict, *, config: Config | None = None, agent: Agent | None = None) -> str:
        """Return a compact end-of-turn summary."""
        turn = payload.get("turn", {})
        tool_counts = _tool_counts(turn.get("tool_calls", []))
        skills = turn.get("loaded_skill_names", [])
        loaded_memories = turn.get("loaded_memory_ids", [])
        plan = turn.get("active_plan")
        plan_text = "none"
        if isinstance(plan, dict):
            plan_text = plan.get("status", "unknown")
        lines = [
            self.heading("Turn Summary"),
            f"  worked for  {_format_duration(_turn_duration(turn))}",
            f"  model       {turn.get('model_request_count', 0)} request(s)",
            f"  tools       {_format_tool_counts(tool_counts)}",
            f"  memory      {len(loaded_memories)} loaded",
            f"  skills      {', '.join(skills) if skills else 'none'}",
            f"  plan        {plan_text}",
        ]
        context_report = turn.get("context_reports", [])
        if isinstance(context_report, list) and context_report:
            lines.append(f"  context     {_context_summary_text(context_report[-1])}")
        trace_path = agent.trace_logger.path if agent and agent.trace_logger else None
        if trace_path is not None:
            lines.append(f"  trace       {_short_path(trace_path)}")
        elif config is not None:
            lines.append(f"  traces dir  {_short_path(config.traces_dir)}")
        return "\n".join(lines)

    def bye(self, agent: Agent | None = None) -> str:
        if agent is None:
            return self.muted("bye")
        command = f"/resume {agent.state.conversation_id}"
        return "\n".join(
            [
                self.muted("bye"),
                self.muted("Resume this session next time with:"),
                f"  {self.accent(command)}",
            ]
        )

    def hint(self) -> str:
        return self.muted("Type /help for commands. Type /exit, /quit, or /q to end the session.")

    def clear(self) -> str:
        return "\033[2J\033[H" if self.color_enabled else "[screen cleared]"

    def heading(self, text: str) -> str:
        return self.accent(f"> {text}")

    def warning(self, text: str) -> str:
        return self._paint(text, DIM_GREEN)

    def error(self, text: str) -> str:
        return self._paint(text, ERROR_RED)

    def accent(self, text: str) -> str:
        return self._paint(text, HULK_GREEN, bold=True)

    def muted(self, text: str) -> str:
        return self._paint(text, MUTED)

    def _row(self, label: str, value: str) -> str:
        return f"| {self.muted(label):<10} {value}"

    def _paint(self, text: str, rgb: tuple[int, int, int], *, bold: bool = False) -> str:
        if not self.color_enabled:
            return text
        prefix = "1;" if bold else ""
        red, green, blue = rgb
        return f"\033[{prefix}38;2;{red};{green};{blue}m{text}\033[0m"


def _context_status(report, budget=None) -> str:
    if not isinstance(report, dict):
        if budget is not None and getattr(budget, "enabled", False):
            input_budget = getattr(budget, "input_token_budget", None)
            context_window = getattr(budget, "max_prompt_tokens", 0)
            reserve = getattr(budget, "response_reserve_tokens", 0)
            return (
                f"{_format_count(int(input_budget or 0))} input budget "
                f"({_format_count(int(context_window or 0))} context, {_format_count(int(reserve or 0))} reserve)"
            )
        return "not recorded"
    tokens = int(report.get("estimated_tokens") or 0)
    omitted = int(report.get("omitted_message_count") or 0)
    suffix = f", {omitted} omitted" if omitted else ""
    return f"{_format_count(tokens)} est tokens{suffix}"


def _context_summary_text(report) -> str:
    if not isinstance(report, dict):
        return "not recorded"
    tokens = _format_count(int(report.get("estimated_tokens") or 0))
    omitted = int(report.get("omitted_message_count") or 0)
    budget = report.get("budget", {})
    budget_text = "budget off"
    if isinstance(budget, dict) and budget.get("enabled"):
        input_budget = budget.get("input_token_budget")
        budget_text = f"budget {_format_count(int(input_budget or 0))}"
    return f"{tokens} est tokens, {omitted} omitted, {budget_text}"


def _context_budget_text(budget, over_budget_tokens: int) -> str:
    if not isinstance(budget, dict) or not budget.get("enabled"):
        return "off"
    input_budget = int(budget.get("input_token_budget") or 0)
    context_window = int(budget.get("context_window_tokens") or budget.get("max_prompt_tokens") or 0)
    reserve = int(budget.get("response_reserve_tokens") or 0)
    over_text = f", over by {_format_count(over_budget_tokens)}" if over_budget_tokens else ""
    return (
        f"{_format_count(input_budget)} input tokens "
        f"({_format_count(context_window)} context, {_format_count(reserve)} output reserve{over_text})"
    )


def _context_section_detail(section: dict) -> str:
    metadata = section.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    if "profile_memory_ids" in metadata or "relevant_memory_ids" in metadata:
        profile = metadata.get("profile_memory_ids", [])
        relevant = metadata.get("relevant_memory_ids", [])
        return f"profile {len(profile)}, relevant {len(relevant)}"
    if "skill_names" in metadata:
        skills = metadata.get("skill_names", [])
        return ", ".join(skills) if skills else "none"
    if "tool_names" in metadata:
        tools = metadata.get("tool_names", [])
        return f"{len(tools)} tool(s)"
    if "roles" in metadata:
        roles = metadata.get("roles", {})
        if isinstance(roles, dict) and roles:
            return ", ".join(f"{role} {count}" for role, count in sorted(roles.items()))
    return ""


def _rule(title: str, width: int) -> str:
    label = f"+-- {title} "
    return label + "-" * max(2, width - len(label) - 1) + "+"


def _labeled_block(label: str, content: str) -> str:
    lines = content.splitlines() or [""]
    return "\n".join([label, *[f"  {line}" for line in lines]])


def _short_path(path: Path | str | None) -> str:
    if path is None:
        return ""
    raw = str(path)
    home = str(Path.home())
    if raw == home:
        return "~"
    if raw.startswith(home + os.sep):
        return "~" + raw[len(home) :]
    return raw


def _short_id(value: str) -> str:
    return value[:8]


def _short_timestamp(value: str) -> str:
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value[:16]
    return parsed.strftime("%Y-%m-%d %H:%M")


def _progress_message(
    event_type: str,
    payload: dict,
    *,
    elapsed_seconds: float | None = None,
    duration_seconds: float | None = None,
    verbose: bool = False,
) -> str | None:
    label = f"{event_type} - " if verbose else ""
    if event_type == TraceEvent.TURN_STARTED:
        turn = payload.get("turn", {})
        tools = len(turn.get("available_tool_names", []))
        return label + f"starting turn - {tools} tools available{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.MEMORY_SEARCH_STARTED:
        return label + f"checking memory{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.MEMORY_SEARCH_COMPLETED:
        profile = payload.get("profile_memory_ids", [])
        relevant = payload.get("relevant_memory_ids", [])
        loaded = payload.get("loaded_memory_ids", [])
        return (
            label
            + f"memory loaded - {len(loaded)} selected - profile: {len(profile)}, relevant: {len(relevant)}"
            + _elapsed_suffix(elapsed_seconds)
        )
    if event_type == TraceEvent.SKILL_SELECTION_STARTED:
        return label + f"loading skills{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.SKILL_SELECTION_COMPLETED:
        skills = payload.get("loaded_skill_names", [])
        if not skills:
            return label + f"skills loaded - none{_elapsed_suffix(elapsed_seconds)}"
        matches = _skill_matches(payload)
        match_text = f" - matched: {matches}" if matches else ""
        return label + "skills loaded - " + ", ".join(skills) + match_text + _elapsed_suffix(elapsed_seconds)
    if event_type == TraceEvent.MODEL_REQUEST_STARTED:
        request_index = payload.get("request_index", "?")
        return label + f"asking model - request {request_index}{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.MODEL_RESPONSE:
        return label + f"model responded{_duration_suffix(duration_seconds)}"
    if event_type == TraceEvent.MODEL_RESPONSE_PARSED:
        action_type = payload.get("type")
        if action_type == "tool_call":
            return label + f"model chose tool - {payload.get('tool_name')}{_elapsed_suffix(elapsed_seconds)}"
        if action_type == "plan":
            plan = payload.get("plan", {})
            steps = plan.get("steps", []) if isinstance(plan, dict) else []
            return label + f"model proposed plan - {len(steps)} step(s){_elapsed_suffix(elapsed_seconds)}"
        if action_type == "final_answer":
            return label + f"model returned final answer{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_CREATED:
        plan = payload.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        return label + f"plan waiting for approval - {len(steps)} step(s){_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_APPROVED:
        return label + f"plan approved{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_REJECTED:
        return label + f"plan rejected{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_REVISION_REQUESTED:
        revision_count = payload.get("revision_count", "?")
        return label + f"plan needs implementation details - revision {revision_count}{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_STEP_STARTED:
        step = payload.get("step", {})
        title = step.get("title", "step") if isinstance(step, dict) else "step"
        return label + f"plan step started - {_compact(str(title))}{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_STEP_COMPLETED:
        step = payload.get("step", {})
        title = step.get("title", "step") if isinstance(step, dict) else "step"
        return label + f"plan step completed - {_compact(str(title))}{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.PLAN_STEP_BLOCKED:
        step = payload.get("step", {})
        title = step.get("title", "step") if isinstance(step, dict) else "step"
        return label + f"plan step blocked - {_compact(str(title))}{_elapsed_suffix(elapsed_seconds)}"
    if event_type == TraceEvent.TOOL_CALL_STARTED:
        return label + _format_tool_progress("running tool", payload, elapsed_seconds=elapsed_seconds)
    if event_type == TraceEvent.TOOL_CALL_COMPLETED:
        return label + _format_tool_progress(
            "tool completed",
            payload,
            duration_seconds=duration_seconds,
            include_result=True,
        )
    if event_type == TraceEvent.TOOL_CALL_FAILED:
        error = payload.get("error") or "failed"
        return label + _format_tool_progress(
            "tool failed",
            payload,
            suffix=str(error),
            duration_seconds=duration_seconds,
            include_result=True,
        )
    if event_type == TraceEvent.TURN_FINISHED:
        turn = payload.get("turn", {})
        status = turn.get("status", "done")
        requests = turn.get("model_request_count", 0)
        tools = turn.get("tool_call_count", 0)
        return label + (
            f"turn {status} - {requests} model request(s), {tools} tool call(s)"
            + _duration_suffix(duration_seconds)
        )
    if event_type == TraceEvent.TURN_FAILED:
        return label + f"turn failed{_elapsed_suffix(elapsed_seconds)}"
    return None


def _format_tool_progress(
    prefix: str,
    payload: dict,
    *,
    suffix: str | None = None,
    elapsed_seconds: float | None = None,
    duration_seconds: float | None = None,
    include_result: bool = False,
) -> str:
    tool_name = payload.get("tool_name")
    detail = _tool_detail(payload)
    parts = [prefix, str(tool_name)]
    if detail:
        parts.append(detail)
    if include_result:
        result = _tool_result_detail(payload)
        if result:
            parts.append(result)
    if suffix:
        parts.append(suffix)
    message = " - ".join(parts)
    if duration_seconds is not None:
        return message + _duration_suffix(duration_seconds)
    return message + _elapsed_suffix(elapsed_seconds)


def _tool_detail(payload: dict) -> str | None:
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    tool_name = payload.get("tool_name")
    if tool_name == "run_cmd":
        command = arguments.get("command")
        if isinstance(command, str) and command:
            return f"cmd: {_compact(command)}"
    return None


def _tool_result_detail(payload: dict) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    tool_name = payload.get("tool_name")
    if tool_name == "run_cmd":
        parts = []
        if "exit_code" in metadata:
            parts.append(f"exit {metadata['exit_code']}")
        stdout_length = metadata.get("stdout_length")
        stderr_length = metadata.get("stderr_length")
        if isinstance(stdout_length, int) and stdout_length:
            parts.append(f"stdout {_format_count(stdout_length)} chars")
        if isinstance(stderr_length, int) and stderr_length:
            parts.append(f"stderr {_format_count(stderr_length)} chars")
        return " - ".join(parts) if parts else None
    return None


def _skill_matches(payload: dict) -> str:
    skills = payload.get("skills", [])
    if not isinstance(skills, list):
        return ""
    matches: list[str] = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        for keyword in skill.get("matched_keywords", []):
            if isinstance(keyword, str) and keyword not in matches:
                matches.append(keyword)
            if len(matches) >= 4:
                return ", ".join(matches)
    return ", ".join(matches)


def _tool_counts(tool_calls: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = tool_call.get("tool_name")
        if not isinstance(name, str):
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _format_tool_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))


def _turn_duration(turn: dict) -> float | None:
    started_at = _parse_iso_timestamp(turn.get("started_at"))
    ended_at = _parse_iso_timestamp(turn.get("ended_at"))
    if started_at is None or ended_at is None:
        return None
    return max(0.0, ended_at - started_at)


def _parse_iso_timestamp(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _duration_suffix(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f" - {_format_duration(seconds)}"


def _elapsed_suffix(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f" - elapsed {_format_duration(seconds)}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def _format_count(value: int) -> str:
    if value >= 1000:
        compact = value / 1000
        return f"{compact:.1f}k"
    return str(value)


def _provider_text(config: Config) -> str:
    primary = f"{config.llm_provider} / {config.model}"
    if not config.llm_fallback_providers:
        return primary
    fallback = " -> ".join(f"{item.provider} / {item.model}" for item in config.llm_fallback_providers)
    return f"{primary} -> {fallback}"


def _compact(text: str, limit: int = 96) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."
