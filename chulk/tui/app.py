"""Keyboard-first Textual control room for Chulk operators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Static
from textual.widgets import TabbedContent, TabPane
from textual.widgets import TextArea

from chulk.tui.data import OperatorDataSource
from chulk.tui.models import OperatorSnapshot, Record


class OperatorApp(App[None]):
    """A compact operations bench backed only by the public control API."""

    TITLE = "Chulk Operator"
    SUB_TITLE = "profile-owned control room"
    CSS = """
    $ink: #e8ecef;
    $muted: #8c99a5;
    $bench: #11171c;
    $panel: #182127;
    $line: #33424c;
    $signal: #e6b450;
    $safe: #78c091;
    $danger: #e06c75;
    $cold: #67b0e8;

    Screen {
        background: $bench;
        color: $ink;
    }

    Header {
        background: #0c1115;
        color: $ink;
    }

    #workspace {
        height: 1fr;
        overflow-x: auto;
    }

    #rail {
        width: 25;
        min-width: 20;
        border-right: solid $line;
        background: #10171b;
    }

    #conversation-stage {
        width: 2fr;
        min-width: 38;
        padding: 0 1;
    }

    #drawers {
        width: 3fr;
        min-width: 45;
        border-left: solid $line;
        background: #141d22;
    }

    .eyebrow {
        height: 2;
        padding: 1 1 0 1;
        color: $signal;
        text-style: bold;
    }

    #profile {
        height: 2;
        padding: 0 1 1 1;
        color: $muted;
    }

    #profiles {
        height: 8;
        border-top: solid $line;
        background: transparent;
    }

    #conversations {
        height: 1fr;
        border-top: solid $line;
        background: transparent;
    }

    #evidence-pulse {
        height: 5;
        margin: 1 0 0 0;
        padding: 1 2;
        border: tall $signal;
        background: #1e2627;
    }

    #timeline {
        height: 1fr;
        margin-top: 1;
        padding: 0 1;
        border: tall $line;
        background: #0f1519;
        scrollbar-color: $signal;
    }

    #prompt {
        height: 6;
        margin-top: 1;
        border: tall $cold;
        background: #10171b;
    }

    #slash-help {
        display: none;
        height: auto;
        max-height: 4;
        padding: 0 1;
        background: #202b31;
        color: $signal;
    }

    #slash-help.visible {
        display: block;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 0;
    }

    DataTable {
        height: 1fr;
        background: transparent;
    }

    DataTable > .datatable--header {
        background: #202b31;
        color: $signal;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #30414a;
        color: $ink;
    }

    #status-line {
        height: 2;
        padding: 0 1;
        background: #0c1115;
        color: $muted;
    }

    .has-attention {
        color: $danger;
    }

    .healthy {
        color: $safe;
    }

    Footer {
        background: #0c1115;
    }

    .dialog {
        width: 64;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: tall $signal;
        background: #182127;
    }

    .dialog-title {
        height: 2;
        color: $signal;
        text-style: bold;
    }

    .dialog-actions {
        height: 3;
        align-horizontal: right;
    }

    .dialog-actions Button {
        margin-left: 1;
    }

    ConfirmScreen, SteerScreen {
        align: center middle;
        background: #000000 55%;
    }

    .no-color {
        color: white;
        background: black;
    }

    .no-color #rail, .no-color #drawers, .no-color #conversation-stage,
    .no-color DataTable, .no-color #timeline, .no-color #evidence-pulse,
    .no-color #prompt {
        color: white;
        background: black;
        border: solid white;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "reload", "Refresh", show=True),
        Binding("ctrl+p", "focus_prompt", "Prompt", show=True),
        Binding("n", "new_conversation", "New", show=True),
        Binding("m", "toggle_mode", "Run/Plan", show=True),
        Binding("ctrl+enter", "submit_prompt", "Send", show=True),
        Binding("a", "drawer('attention')", "Attention", show=False),
        Binding("g", "drawer('goals')", "Goals", show=False),
        Binding("t", "drawer('tasks')", "Tasks", show=False),
        Binding("j", "drawer('jobs')", "Jobs", show=False),
        Binding("u", "drawer('usage')", "Usage", show=False),
        Binding("e", "drawer('evidence')", "Evidence", show=False),
        Binding("space", "approve", "Approve", show=True),
        Binding("d", "deny", "Deny", show=True),
        Binding("s", "advance_work", "Advance", show=True),
        Binding("i", "steer", "Steer", show=True),
        Binding("x", "cancel_selected", "Cancel", show=True),
        Binding("ctrl+x", "cancel_conversation", "Stop chat", show=False),
    ]

    def __init__(
        self,
        data_source: OperatorDataSource,
        *,
        refresh_seconds: float = 2.0,
        no_color: bool = False,
    ) -> None:
        super().__init__()
        if refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be greater than zero")
        self.data_source = data_source
        self.refresh_seconds = refresh_seconds
        self.no_color = no_color
        self.snapshot = OperatorSnapshot(profile_id=data_source.profile_id)
        self.message_mode = "run"
        self._records: dict[tuple[str, str], tuple[str, Record]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with HorizontalScroll(id="workspace"):
            with Vertical(id="rail"):
                yield Static("PROFILE", classes="eyebrow")
                yield Static(self.data_source.profile_id, id="profile")
                yield DataTable(id="profiles")
                yield DataTable(id="conversations")
            with Vertical(id="conversation-stage"):
                yield Static("Waiting for the first signal…", id="evidence-pulse")
                yield RichLog(id="timeline", wrap=True, markup=True)
                yield TextArea(
                    "",
                    id="prompt",
                )
                yield Static("", id="slash-help")
            with Vertical(id="drawers"):
                with TabbedContent(id="operator-tabs"):
                    with TabPane("Attention", id="attention"):
                        yield DataTable(id="attention-table")
                    with TabPane("Goals", id="goals"):
                        yield DataTable(id="goals-table")
                    with TabPane("Tasks", id="tasks"):
                        yield DataTable(id="tasks-table")
                    with TabPane("Jobs", id="jobs"):
                        yield DataTable(id="jobs-table")
                    with TabPane("Usage", id="usage"):
                        yield DataTable(id="usage-table")
                    with TabPane("Evidence", id="evidence"):
                        yield DataTable(id="evidence-table")
        yield Static("connecting to control plane", id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        if self.no_color:
            self.screen.add_class("no-color")
        self._configure_tables()
        self.set_interval(self.refresh_seconds, self.load_snapshot)
        self.load_snapshot()

    def _configure_tables(self) -> None:
        profiles = self.query_one("#profiles", DataTable)
        profiles.cursor_type = "row"
        profiles.add_columns("Profile", "Model")
        conversations = self.query_one("#conversations", DataTable)
        conversations.cursor_type = "row"
        conversations.add_columns("Conversation", "State", "Turns")
        attention = self.query_one("#attention-table", DataTable)
        attention.cursor_type = "row"
        attention.add_columns("Type", "Subject", "State")
        for table_id, noun in (
            ("#goals-table", "Goal"),
            ("#tasks-table", "Task"),
            ("#jobs-table", "Job"),
        ):
            table = self.query_one(table_id, DataTable)
            table.cursor_type = "row"
            table.add_columns(noun, "State", "Rev")
        usage = self.query_one("#usage-table", DataTable)
        usage.cursor_type = "row"
        usage.add_columns("Resource", "Units", "Cost")
        evidence = self.query_one("#evidence-table", DataTable)
        evidence.cursor_type = "row"
        evidence.add_columns("Evidence", "State", "Artifacts")

    @work(exclusive=True, group="snapshot")
    async def load_snapshot(self) -> None:
        try:
            snapshot = await asyncio.to_thread(self.data_source.refresh)
        except Exception as exc:
            self.query_one("#status-line", Static).update(
                f"[red]control plane unavailable[/red] · {exc}"
            )
            return
        self.snapshot = snapshot
        self._render_snapshot(snapshot)

    def _render_snapshot(self, snapshot: OperatorSnapshot) -> None:
        self._records.clear()
        self._render_profiles(snapshot)
        self._render_conversations(snapshot)
        self._render_attention(snapshot)
        self._render_work("goals", "goal", snapshot.goals)
        self._render_work("tasks", "task", snapshot.tasks)
        self._render_work("jobs", "job", snapshot.jobs)
        self._render_usage(snapshot)
        self._render_evidence(snapshot)
        self._render_timeline(snapshot)
        self._render_pulse(snapshot)

        status = (
            f"profile {snapshot.profile_id} · "
            f"{snapshot.attention_count} attention · "
            f"{snapshot.active_work_count} active"
        )
        if snapshot.errors:
            status += " · degraded: " + ", ".join(sorted(snapshot.errors))
        self.query_one("#status-line", Static).update(status)

    def _render_profiles(self, snapshot: OperatorSnapshot) -> None:
        table = self.query_one("#profiles", DataTable)
        table.clear(columns=False)
        for item in snapshot.profiles:
            profile_id = str(item.get("id") or "")
            if not profile_id:
                continue
            marker = "●" if profile_id == snapshot.profile_id else " "
            table.add_row(
                f"{marker} {profile_id}",
                str(item.get("model_profile_id") or "default"),
                key=profile_id,
            )
            self._records[("profiles", profile_id)] = ("profile", item)

    def _render_conversations(self, snapshot: OperatorSnapshot) -> None:
        table = self.query_one("#conversations", DataTable)
        table.clear(columns=False)
        for item in snapshot.conversations:
            conversation_id = str(item.get("id") or "")
            if not conversation_id:
                continue
            marker = "●" if conversation_id == snapshot.selected_conversation_id else " "
            title = str(item.get("title") or conversation_id[:8])
            table.add_row(
                f"{marker} {title}",
                str(item.get("status") or "unknown"),
                str(item.get("turn_count") or 0),
                key=conversation_id,
            )
            self._records[("conversations", conversation_id)] = (
                "conversation",
                item,
            )

    def _render_attention(self, snapshot: OperatorSnapshot) -> None:
        table = self.query_one("#attention-table", DataTable)
        table.clear(columns=False)
        for item in snapshot.permissions:
            key = f"permission:{item.get('id')}"
            table.add_row(
                "permission",
                str(item.get("tool_name") or item.get("reason") or "tool"),
                str(item.get("status") or "pending"),
                key=key,
            )
            self._records[("attention-table", key)] = ("permission", item)
        for item in snapshot.proposals:
            key = f"proposal:{item.get('id')}"
            table.add_row(
                "learning",
                str(
                    item.get("target_name")
                    or item.get("kind")
                    or item.get("id")
                    or "proposal"
                ),
                str(item.get("status") or "pending"),
                key=key,
            )
            self._records[("attention-table", key)] = ("proposal", item)

    def _render_work(
        self,
        plural: str,
        kind: str,
        items: tuple[Record, ...],
    ) -> None:
        table_id = f"{plural}-table"
        table = self.query_one(f"#{table_id}", DataTable)
        table.clear(columns=False)
        for item in items:
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            label = str(
                item.get("objective")
                or item.get("title")
                or item.get("name")
                or item.get("prompt_preview")
                or item_id
            )
            table.add_row(
                _clip(label, 34),
                str(item.get("status") or "unknown"),
                str(item.get("revision") or 0),
                key=item_id,
            )
            self._records[(table_id, item_id)] = (kind, item)

    def _render_usage(self, snapshot: OperatorSnapshot) -> None:
        table = self.query_one("#usage-table", DataTable)
        table.clear(columns=False)
        for index, item in enumerate(snapshot.usage):
            cost = item.get("cost")
            if isinstance(cost, Mapping):
                amount = cost.get("amount")
                currency = cost.get("currency") or "USD"
                cost_text = f"{amount or 'unknown'} {currency}"
            else:
                cost_text = str(item.get("total_cost") or "—")
            units = item.get("units")
            units_text = (
                ", ".join(f"{key}={value}" for key, value in units.items())
                if isinstance(units, Mapping)
                else str(item.get("total_units") or "—")
            )
            table.add_row(
                str(
                    item.get("key")
                    or item.get("turn_id")
                    or item.get("resource_kind")
                    or item.get("group")
                    or "usage"
                ),
                _clip(units_text, 28),
                cost_text,
                key=f"usage:{index}",
            )

    def _render_evidence(self, snapshot: OperatorSnapshot) -> None:
        table = self.query_one("#evidence-table", DataTable)
        table.clear(columns=False)
        for item in snapshot.traces:
            conversation_id = str(item.get("conversation_id") or "")
            table.add_row(
                f"trace {conversation_id[:8]}",
                str(item.get("status") or "unknown"),
                str(item.get("artifact_count") or 0),
                key=f"trace:{conversation_id}",
            )
        for item in snapshot.artifacts:
            artifact_id = str(item.get("artifact_id") or "")
            table.add_row(
                _clip(str(item.get("name") or artifact_id or "artifact"), 34),
                str(item.get("kind") or "artifact"),
                str(item.get("size_bytes") or "—"),
                key=f"artifact:{artifact_id}",
            )

    def _render_timeline(self, snapshot: OperatorSnapshot) -> None:
        log = self.query_one("#timeline", RichLog)
        log.clear()
        if not snapshot.timeline:
            log.write(
                "[dim]No local activity yet. Select a conversation or start a new one.[/dim]"
            )
            return
        colors = {
            "you": "bright_cyan",
            "chulk": "bright_white",
            "permission": "yellow",
            "learning": "magenta",
            "control": "orange3",
            "system": "dim",
        }
        for entry in snapshot.timeline:
            color = colors.get(entry.kind, "white")
            log.write(f"[{color}]{entry.kind.upper():>10}[/]  {entry.text}")

    def _render_pulse(self, snapshot: OperatorSnapshot) -> None:
        conversation = next(
            (
                item
                for item in snapshot.conversations
                if item.get("id") == snapshot.selected_conversation_id
            ),
            {},
        )
        selected = (
            str(conversation.get("title") or snapshot.selected_conversation_id or "none")
        )
        signal = (
            "[red]ATTENTION[/red]"
            if snapshot.attention_count
            else "[green]CLEAR[/green]"
        )
        evidence = next(
            (
                item
                for item in snapshot.traces
                if item.get("conversation_id") == snapshot.selected_conversation_id
            ),
            None,
        )
        evidence_text = (
            f"{evidence.get('artifact_count', 0)} artifacts · "
            f"trace {'ready' if evidence.get('available') else 'pending'}"
            if evidence
            else "no trace evidence yet"
        )
        self.query_one("#evidence-pulse", Static).update(
            f"[b]{signal}[/b]  {_clip(selected, 48)}\n"
            f"{snapshot.attention_count} decisions · "
            f"{snapshot.active_work_count} active work items\n"
            f"[dim]{evidence_text}[/dim]"
        )

    @on(DataTable.RowSelected, "#conversations")
    def select_conversation(self, event: DataTable.RowSelected) -> None:
        conversation_id = str(event.row_key.value)
        self.data_source.select_conversation(conversation_id)
        self.load_snapshot()

    @on(DataTable.RowSelected, "#profiles")
    def select_profile(self, event: DataTable.RowSelected) -> None:
        profile_id = str(event.row_key.value)
        self.data_source.select_profile(profile_id)
        self.load_snapshot()

    @on(TextArea.Changed, "#prompt")
    def update_slash_help(self, event: TextArea.Changed) -> None:
        text = event.text_area.text.lstrip()
        help_line = self.query_one("#slash-help", Static)
        if not text.startswith("/"):
            help_line.remove_class("visible")
            help_line.update("")
            return
        commands = ("/run", "/plan", "/new", "/stop", "/refresh")
        prefix = text.splitlines()[0].split(" ", 1)[0].lower()
        matches = tuple(command for command in commands if command.startswith(prefix))
        help_line.update("  ".join(matches) if matches else "Unknown local command")
        help_line.add_class("visible")

    def action_submit_prompt(self) -> None:
        editor = self.query_one("#prompt", TextArea)
        message = editor.text.strip()
        if not message:
            return
        editor.text = ""
        if self._handle_local_command(message):
            return
        self._mutate(
            lambda: self.data_source.send_message(
                message,
                mode=self.message_mode,
            ),
            f"{self.message_mode} submitted",
        )

    def action_reload(self) -> None:
        self.load_snapshot()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", TextArea).focus()

    def action_new_conversation(self) -> None:
        self._mutate(self.data_source.create_conversation, "conversation created")

    def action_toggle_mode(self) -> None:
        self.message_mode = "plan" if self.message_mode == "run" else "run"
        self.notify(f"Message mode: {self.message_mode}")

    def action_drawer(self, drawer: str) -> None:
        self.query_one("#operator-tabs", TabbedContent).active = drawer
        table = self.query_one(f"#{drawer}-table", DataTable)
        table.focus()

    def action_approve(self) -> None:
        selection = self._selected_record("attention-table")
        if selection is None:
            self.notify("Select an attention item first.", severity="warning")
            return
        kind, item = selection
        item_id = str(item.get("id") or "")
        self.push_screen(
            ConfirmScreen(
                f"Approve {kind} {item_id[:8]}?",
                confirm_label="Approve",
            ),
            lambda confirmed: self._confirm_attention(
                confirmed,
                kind,
                item_id,
                positive=True,
            ),
        )

    def action_deny(self) -> None:
        selection = self._selected_record("attention-table")
        if selection is None:
            self.notify("Select an attention item first.", severity="warning")
            return
        kind, item = selection
        item_id = str(item.get("id") or "")
        self.push_screen(
            ConfirmScreen(
                f"Reject {kind} {item_id[:8]}?",
                confirm_label="Reject",
            ),
            lambda confirmed: self._confirm_attention(
                confirmed,
                kind,
                item_id,
                positive=False,
            ),
        )

    def action_advance_work(self) -> None:
        selection = self._selected_work()
        if selection is None:
            self.notify("Select a goal, task, or job first.", severity="warning")
            return
        kind, item = selection
        status = str(item.get("status") or "")
        actions = {
            "goal": {
                "draft": "approve",
                "approved": "run",
                "running": "pause",
                "paused": "resume",
                "blocked": "resume",
            },
            "task": {
                "failed": "retry",
                "budget_exhausted": "retry",
            },
            "job": {
                "pending_approval": "approve",
                "active": "pause",
                "running": "pause",
                "paused": "resume",
            },
        }
        action = actions.get(kind, {}).get(status)
        if action is None:
            self.notify(
                f"No advance action is available for {kind} in {status or 'unknown'} state.",
                severity="warning",
            )
            return
        self._mutate(
            lambda: self.data_source.control_work(kind, item, action),
            f"{action} requested",
        )

    def action_cancel_selected(self) -> None:
        selection = self._selected_work()
        if selection is None:
            self.notify("Select a goal, task, or job first.", severity="warning")
            return
        kind, item = selection
        self.push_screen(
            ConfirmScreen(
                f"Cancel {kind} {str(item.get('id') or '')[:8]}?",
                confirm_label="Cancel work",
            ),
            lambda confirmed: self._confirm_work_cancel(confirmed, kind, item),
        )

    def action_cancel_conversation(self) -> None:
        self.push_screen(
            ConfirmScreen(
                "Stop the active conversation?",
                confirm_label="Stop",
            ),
            lambda confirmed: (
                self._mutate(
                    self.data_source.cancel_conversation,
                    "conversation stop requested",
                )
                if confirmed
                else None
            ),
        )

    def action_steer(self) -> None:
        selection = self._selected_work()
        if selection is None or selection[0] != "goal":
            self.notify("Select a goal to steer.", severity="warning")
            return
        kind, item = selection
        self.push_screen(
            SteerScreen(str(item.get("objective") or item.get("id") or "goal")),
            lambda instruction: (
                self._mutate(
                    lambda: self._steer_goal(item, instruction),
                    "goal steering submitted",
                )
                if instruction
                else None
            ),
        )

    def _selected_work(self) -> tuple[str, Record] | None:
        active = self.query_one("#operator-tabs", TabbedContent).active
        if active not in {"goals", "tasks", "jobs"}:
            return None
        return self._selected_record(f"{active}-table")

    def _selected_record(self, table_id: str) -> tuple[str, Record] | None:
        table = self.query_one(f"#{table_id}", DataTable)
        if table.row_count == 0:
            return None
        row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        return self._records.get((table_id, str(row_key.value)))

    def _decide_attention(
        self,
        kind: str,
        item_id: str,
        *,
        positive: bool,
    ) -> None:
        if kind == "permission":
            self.data_source.decide_permission(
                item_id,
                "allow" if positive else "deny",
            )
            return
        self.data_source.decide_proposal(
            item_id,
            "approve" if positive else "reject",
        )

    def _confirm_attention(
        self,
        confirmed: bool | None,
        kind: str,
        item_id: str,
        *,
        positive: bool,
    ) -> None:
        if not confirmed:
            return
        self._mutate(
            lambda: self._decide_attention(kind, item_id, positive=positive),
            f"{kind} {'approved' if positive else 'rejected'}",
        )

    def _confirm_work_cancel(
        self,
        confirmed: bool | None,
        kind: str,
        item: Record,
    ) -> None:
        if not confirmed:
            return
        self._mutate(
            lambda: self.data_source.control_work(kind, item, "cancel"),
            f"{kind} cancellation requested",
        )

    def _steer_goal(self, item: Record, instruction: str) -> None:
        steer = getattr(self.data_source, "steer_goal", None)
        if steer is None:
            raise ValueError("goal steering is unavailable")
        steer(item, instruction)

    def _handle_local_command(self, message: str) -> bool:
        command, _, arguments = message.partition(" ")
        normalized = command.lower()
        if normalized == "/new" and not arguments:
            self.action_new_conversation()
            return True
        if normalized == "/stop" and not arguments:
            self.action_cancel_conversation()
            return True
        if normalized == "/refresh" and not arguments:
            self.action_reload()
            return True
        if normalized in {"/run", "/plan"} and arguments.strip():
            mode = normalized[1:]
            self._mutate(
                lambda: self.data_source.send_message(
                    arguments.strip(),
                    mode=mode,
                ),
                f"{mode} submitted",
            )
            return True
        return False

    def _mutate(self, operation: Callable[[], Any], success: str) -> None:
        self.run_worker(
            self._perform_mutation(operation, success),
            exclusive=True,
            group="mutation",
        )

    async def _perform_mutation(
        self,
        operation: Callable[[], Any],
        success: str,
    ) -> None:
        try:
            await asyncio.to_thread(operation)
        except Exception as exc:
            self.notify(str(exc), title="Control action failed", severity="error")
            return
        self.notify(success)
        self.load_snapshot()


def _clip(value: str, length: int) -> str:
    clean = " ".join(value.split())
    return clean if len(clean) <= length else f"{clean[: length - 1]}…"


__all__ = ["OperatorApp"]


class ConfirmScreen(ModalScreen[bool]):
    """Keyboard-readable confirmation for consequential control actions."""

    BINDINGS = [
        Binding("y", "confirm", "Confirm"),
        Binding("n", "cancel", "Cancel"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, prompt: str, *, confirm_label: str) -> None:
        super().__init__()
        self.prompt = prompt
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static("CONFIRM CONTROL ACTION", classes="dialog-title")
            yield Static(self.prompt)
            with Horizontal(classes="dialog-actions"):
                yield Button("Back", id="cancel")
                yield Button(self.confirm_label, id="confirm", variant="warning")

    @on(Button.Pressed)
    def press_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class SteerScreen(ModalScreen[str | None]):
    """Collect one explicit steering instruction for a selected goal."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, goal: str) -> None:
        super().__init__()
        self.goal = goal

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static("STEER ACTIVE GOAL", classes="dialog-title")
            yield Static(_clip(self.goal, 58))
            yield Input(placeholder="New instruction", id="steer-instruction")
            with Horizontal(classes="dialog-actions"):
                yield Button("Back", id="cancel")
                yield Button("Steer", id="confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#steer-instruction", Input).focus()

    @on(Input.Submitted, "#steer-instruction")
    def submit_instruction(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    @on(Button.Pressed)
    def press_button(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            value = self.query_one("#steer-instruction", Input).value.strip()
            self.dismiss(value or None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
