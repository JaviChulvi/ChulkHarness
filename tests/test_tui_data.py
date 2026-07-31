"""Tests for the operator TUI control-plane adapter."""

from __future__ import annotations

from typing import Any

from chulk.tui.data import ControlApiDataSource


class FakeControlClient:
    def __init__(self) -> None:
        self.command_status = "running"
        self.calls: list[tuple[Any, ...]] = []

    def list_profiles(self):
        return {
            "profiles": [
                {"id": "default", "model_profile_id": "balanced"}
            ]
        }

    def list_conversations(self, profile_id, *, limit):
        return {
            "conversations": [
                {
                    "id": "conv-1",
                    "title": "Operator test",
                    "status": "active",
                    "turn_count": 2,
                }
            ]
        }

    def list_permissions(self, profile_id, conversation_id, *, status, limit):
        return {
            "permissions": [
                {
                    "id": "perm-1",
                    "tool_name": "shell",
                    "status": "pending",
                }
            ]
        }

    def list_artifacts(self, profile_id, conversation_id, *, limit):
        return {"artifacts": [{"artifact_id": "artifact-1", "size_bytes": 12}]}

    def list_events(self, profile_id, conversation_id, *, after=None):
        if after is not None:
            return {"events": [], "next_cursor": after}
        return {
            "events": [
                {
                    "id": "event-1",
                    "event": {
                        "name": "model.delta",
                        "payload": {"text": "Streaming evidence"},
                    },
                }
            ],
            "next_cursor": "event-1",
        }

    def list_proposals(self, profile_id, *, status, limit):
        return {"proposals": [{"id": "proposal-1", "status": "pending"}]}

    def list_goals(self, profile_id, *, limit):
        return {"goals": [{"id": "goal-1", "status": "running", "revision": 3}]}

    def list_tasks(self, profile_id, *, limit):
        return {"tasks": [{"id": "task-1", "status": "running", "revision": 2}]}

    def list_jobs(self, profile_id, *, include_terminal, limit):
        return {"jobs": [{"id": "job-1", "status": "queued", "revision": 4}]}

    def query_usage(
        self,
        profile_id,
        *,
        conversation_id=None,
        group_by=None,
        limit,
    ):
        if conversation_id:
            return {
                "entries": [
                    {
                        "resource_kind": "model",
                        "turn_id": "turn-1",
                        "cost": {"amount": "0.12", "currency": "USD"},
                    }
                ]
            }
        return {"groups": [{"key": "model", "total_cost": "0.12"}]}

    def list_traces(self, profile_id, *, limit):
        return {
            "traces": [
                {
                    "conversation_id": "conv-1",
                    "available": True,
                    "artifact_count": 1,
                }
            ]
        }

    def create_conversation(self, profile_id, *, metadata):
        self.calls.append(("create", profile_id, metadata))
        return {"conversation": {"id": "conv-new"}}

    def send_message(self, profile_id, conversation_id, message, *, mode):
        self.calls.append(("send", profile_id, conversation_id, message, mode))
        return {"command": {"id": "command-1"}}

    def get_command(self, profile_id, conversation_id, command_id):
        return {
            "command": {
                "id": command_id,
                "status": self.command_status,
                "result": (
                    {"content": "Finished cleanly."}
                    if self.command_status == "completed"
                    else None
                ),
            }
        }

    def decide_permission(
        self,
        profile_id,
        conversation_id,
        request_id,
        *,
        decision,
    ):
        self.calls.append(("permission", request_id, decision))

    def decide_proposal(self, profile_id, proposal_id, *, action):
        self.calls.append(("proposal", proposal_id, action))

    def control_goal(
        self,
        profile_id,
        item_id,
        *,
        action,
        revision,
        instruction=None,
    ):
        self.calls.append(("goal", item_id, action, revision))

    def control_task(self, profile_id, item_id, *, action, revision):
        self.calls.append(("task", item_id, action, revision))

    def control_job(self, profile_id, item_id, *, action, revision):
        self.calls.append(("job", item_id, action, revision))

    def cancel_conversation(self, profile_id, conversation_id):
        self.calls.append(("cancel_conversation", conversation_id))


def test_control_api_data_source_builds_complete_operator_snapshot() -> None:
    client = FakeControlClient()
    source = ControlApiDataSource(client, profile_id="default")  # type: ignore[arg-type]

    snapshot = source.refresh()

    assert snapshot.selected_conversation_id == "conv-1"
    assert snapshot.attention_count == 2
    assert snapshot.active_work_count == 3
    assert snapshot.usage[0]["turn_id"] == "turn-1"
    assert snapshot.artifacts[0]["artifact_id"] == "artifact-1"
    assert snapshot.timeline[-1].text == "Streaming evidence"


def test_control_api_data_source_runs_conversation_and_supervision_actions() -> None:
    client = FakeControlClient()
    source = ControlApiDataSource(client, profile_id="default")  # type: ignore[arg-type]
    source.refresh()

    assert source.send_message("Investigate this", mode="plan") == "command-1"
    client.command_status = "completed"
    snapshot = source.refresh()
    source.decide_permission("perm-1", "allow")
    source.decide_proposal("proposal-1", "reject")
    source.control_work(
        "goal",
        {"id": "goal-1", "revision": 3},
        "pause",
    )
    source.cancel_conversation()
    source.steer_goal(
        {"id": "goal-1", "revision": 3},
        "Focus on the failing lane",
    )

    assert snapshot.timeline[-1].text == "Finished cleanly."
    assert ("send", "default", "conv-1", "Investigate this", "plan") in client.calls
    assert ("permission", "perm-1", "allow") in client.calls
    assert ("proposal", "proposal-1", "reject") in client.calls
    assert ("goal", "goal-1", "pause", 3) in client.calls
    assert ("cancel_conversation", "conv-1") in client.calls
    assert ("goal", "goal-1", "steer", 3) in client.calls


def test_control_api_data_source_isolates_partial_read_failures() -> None:
    client = FakeControlClient()

    def fail(*_args, **_kwargs):
        raise RuntimeError("usage store unavailable")

    client.query_usage = fail  # type: ignore[method-assign]
    source = ControlApiDataSource(client, profile_id="default")  # type: ignore[arg-type]

    snapshot = source.refresh()

    assert snapshot.conversations
    assert snapshot.usage == ()
    assert snapshot.errors == {"usage": "usage store unavailable"}
