"""Headless interaction tests for the Textual operator control room."""

from __future__ import annotations

import pytest
from textual.widgets import DataTable, Static, TabbedContent, TextArea

from chulk.tui.app import OperatorApp
from chulk.tui.models import OperatorSnapshot, TimelineEntry


class FakeOperatorSource:
    profile_id = "default"
    selected_conversation_id = "conv-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def refresh(self) -> OperatorSnapshot:
        return OperatorSnapshot(
            profile_id=self.profile_id,
            selected_conversation_id=self.selected_conversation_id,
            profiles=(
                {"id": "default", "model_profile_id": "balanced"},
                {"id": "research", "model_profile_id": "deep"},
            ),
            conversations=(
                {
                    "id": "conv-1",
                    "title": "Release investigation",
                    "status": "active",
                    "turn_count": 3,
                },
            ),
            permissions=(
                {"id": "perm-1", "tool_name": "shell", "status": "pending"},
            ),
            proposals=(
                {"id": "proposal-1", "kind": "skill", "status": "pending"},
            ),
            goals=(
                {
                    "id": "goal-1",
                    "objective": "Ship operator controls",
                    "status": "running",
                    "revision": 5,
                },
            ),
            tasks=(
                {
                    "id": "task-1",
                    "title": "Verify controls",
                    "status": "failed",
                    "revision": 2,
                },
            ),
            jobs=(
                {
                    "id": "job-1",
                    "prompt_preview": "Run release checks",
                    "status": "paused",
                    "revision": 4,
                },
            ),
            usage=({"key": "model", "total_cost": "0.02"},),
            traces=(
                {
                    "conversation_id": "conv-1",
                    "status": "active",
                    "available": True,
                    "artifact_count": 2,
                },
            ),
            artifacts=(
                {
                    "artifact_id": "artifact-1",
                    "name": "shell-output.txt",
                    "kind": "tool-output",
                    "size_bytes": 128,
                },
            ),
            timeline=(TimelineEntry("system", "Connected."),),
        )

    def select_conversation(self, conversation_id: str) -> None:
        self.selected_conversation_id = conversation_id
        self.calls.append(("select", conversation_id))

    def select_profile(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self.selected_conversation_id = None
        self.calls.append(("profile", profile_id))

    def create_conversation(self) -> str:
        self.calls.append(("create",))
        return "conv-2"

    def send_message(self, message: str, *, mode: str = "run") -> str:
        self.calls.append(("send", message, mode))
        return "command-1"

    def decide_permission(self, request_id: str, decision: str) -> None:
        self.calls.append(("permission", request_id, decision))

    def decide_proposal(self, proposal_id: str, action: str) -> None:
        self.calls.append(("proposal", proposal_id, action))

    def control_work(self, kind: str, item, action: str) -> None:
        self.calls.append(("work", kind, str(item["id"]), action))

    def steer_goal(self, item, instruction: str) -> None:
        self.calls.append(("steer", str(item["id"]), instruction))

    def cancel_conversation(self) -> None:
        self.calls.append(("cancel_conversation",))


@pytest.mark.asyncio
async def test_operator_app_renders_all_control_surfaces_and_submits_prompt() -> None:
    source = FakeOperatorSource()
    app = OperatorApp(source, refresh_seconds=60)

    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        assert app.query_one("#conversations", DataTable).row_count == 1
        assert app.query_one("#attention-table", DataTable).row_count == 2
        assert app.query_one("#goals-table", DataTable).row_count == 1
        assert app.query_one("#usage-table", DataTable).row_count == 1
        assert app.query_one("#evidence-table", DataTable).row_count == 2
        assert "ATTENTION" in str(
            app.query_one("#evidence-pulse", Static).render()
        )

        app.action_toggle_mode()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Draft a safe plan"
        app.action_submit_prompt()
        await pilot.pause()

        assert ("send", "Draft a safe plan", "plan") in source.calls


@pytest.mark.asyncio
async def test_operator_app_keyboard_actions_approve_and_control_work() -> None:
    source = FakeOperatorSource()
    app = OperatorApp(source, refresh_seconds=60)

    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.query_one("#operator-tabs", TabbedContent).active = "attention"
        app.query_one("#attention-table", DataTable).focus()
        app.action_approve()
        await pilot.press("y")
        await pilot.pause()

        app.action_drawer("goals")
        app.action_advance_work()
        await pilot.pause()

        app.action_drawer("tasks")
        app.action_advance_work()
        await pilot.pause()

        app.action_drawer("jobs")
        app.action_advance_work()
        await pilot.pause()

        assert ("permission", "perm-1", "allow") in source.calls
        assert ("work", "goal", "goal-1", "pause") in source.calls
        assert ("work", "task", "task-1", "retry") in source.calls
        assert ("work", "job", "job-1", "resume") in source.calls


@pytest.mark.asyncio
async def test_operator_app_remains_usable_narrow_monochrome_and_reconnects() -> None:
    source = FakeOperatorSource()
    app = OperatorApp(source, refresh_seconds=60, no_color=True)

    async with app.run_test(size=(76, 30)) as pilot:
        await pilot.pause()
        assert app.screen.has_class("no-color")
        await pilot.resize_terminal(64, 24)
        await pilot.pause()
        assert app.query_one("#conversations", DataTable).row_count == 1

        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/pl"
        await pilot.pause()
        assert app.query_one("#slash-help", Static).has_class("visible")
        prompt.text = "/new"
        app.action_submit_prompt()
        await pilot.pause()
        assert ("create",) in source.calls

        source.profile_id = "recovered"
        app.action_reload()
        await pilot.pause()
        assert "profile recovered" in str(
            app.query_one("#status-line", Static).render()
        )
