"""Control-API data boundary for the operator terminal."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from chulk.server import ControlApiClient, ControlApiError
from chulk.tui.models import OperatorSnapshot, Record, TimelineEntry


class OperatorDataSource(Protocol):
    """Operations the Textual application needs from the control plane."""

    profile_id: str
    selected_conversation_id: str | None

    def refresh(self) -> OperatorSnapshot: ...

    def select_profile(self, profile_id: str) -> None: ...

    def select_conversation(self, conversation_id: str) -> None: ...

    def create_conversation(self) -> str: ...

    def send_message(self, message: str, *, mode: str = "run") -> str: ...

    def decide_permission(self, request_id: str, decision: str) -> None: ...

    def decide_proposal(self, proposal_id: str, action: str) -> None: ...

    def control_work(self, kind: str, item: Record, action: str) -> None: ...

    def steer_goal(self, item: Record, instruction: str) -> None: ...

    def cancel_conversation(self) -> None: ...


class ControlApiDataSource:
    """Stateful adapter that keeps UI concerns outside the HTTP client."""

    def __init__(self, client: ControlApiClient, *, profile_id: str) -> None:
        self.client = client
        self.profile_id = profile_id
        self.selected_conversation_id: str | None = None
        self._timeline: list[TimelineEntry] = []
        self._pending_commands: dict[str, str] = {}
        self._event_cursors: dict[tuple[str, str], str] = {}

    def refresh(self) -> OperatorSnapshot:
        errors: dict[str, str] = {}
        profiles = self._read(
            "profiles",
            self.client.list_profiles,
            errors,
        )
        conversations = self._read(
            "conversations",
            lambda: self.client.list_conversations(self.profile_id, limit=50),
            errors,
        )
        if self.selected_conversation_id is None and conversations:
            self.selected_conversation_id = str(conversations[0].get("id") or "") or None
        elif (
            self.selected_conversation_id is not None
            and not any(
                item.get("id") == self.selected_conversation_id
                for item in conversations
            )
        ):
            self.selected_conversation_id = (
                str(conversations[0].get("id")) if conversations else None
            )

        conversation_id = self.selected_conversation_id
        if conversation_id:
            self._poll_events(conversation_id, errors)
        self._poll_command(errors)
        permissions = (
            self._read(
                "permissions",
                lambda: self.client.list_permissions(
                    self.profile_id,
                    conversation_id,
                    status="pending",
                    limit=100,
                ),
                errors,
            )
            if conversation_id
            else ()
        )
        artifacts = (
            self._read(
                "artifacts",
                lambda: self.client.list_artifacts(
                    self.profile_id,
                    conversation_id,
                    limit=50,
                ),
                errors,
            )
            if conversation_id
            else ()
        )
        return OperatorSnapshot(
            profile_id=self.profile_id,
            selected_conversation_id=conversation_id,
            profiles=profiles,
            conversations=conversations,
            permissions=permissions,
            proposals=self._read(
                "proposals",
                lambda: self.client.list_proposals(
                    self.profile_id,
                    status="pending",
                    limit=50,
                ),
                errors,
            ),
            goals=self._read(
                "goals",
                lambda: self.client.list_goals(self.profile_id, limit=50),
                errors,
            ),
            tasks=self._read(
                "tasks",
                lambda: self.client.list_tasks(self.profile_id, limit=50),
                errors,
            ),
            jobs=self._read(
                "jobs",
                lambda: self.client.list_jobs(
                    self.profile_id,
                    include_terminal=False,
                    limit=50,
                ),
                errors,
            ),
            usage=self._read(
                "usage",
                lambda: self._usage_payload(conversation_id),
                errors,
            ),
            traces=self._read(
                "traces",
                lambda: self.client.list_traces(self.profile_id, limit=50),
                errors,
            ),
            artifacts=artifacts,
            timeline=tuple(self._timeline[-100:]),
            errors=errors,
        )

    def select_conversation(self, conversation_id: str) -> None:
        self.selected_conversation_id = conversation_id

    def select_profile(self, profile_id: str) -> None:
        clean = profile_id.strip()
        if not clean:
            raise ValueError("profile id cannot be empty")
        if clean == self.profile_id:
            return
        self.profile_id = clean
        self.selected_conversation_id = None
        self._timeline.clear()
        self._pending_commands.clear()
        self._event_cursors.clear()
        self._timeline.append(
            TimelineEntry("system", f"Switched to profile {clean}.")
        )

    def create_conversation(self) -> str:
        payload = self.client.create_conversation(
            self.profile_id,
            metadata={"source": "operator_tui"},
        )
        conversation = _mapping(payload.get("conversation"))
        conversation_id = str(conversation.get("id") or "")
        if not conversation_id:
            raise ValueError("control API did not return a conversation id")
        self.selected_conversation_id = conversation_id
        self._timeline.append(
            TimelineEntry("system", f"Created conversation {conversation_id[:8]}.")
        )
        return conversation_id

    def send_message(self, message: str, *, mode: str = "run") -> str:
        clean = message.strip()
        if not clean:
            raise ValueError("message cannot be empty")
        conversation_id = self.selected_conversation_id or self.create_conversation()
        payload = self.client.send_message(
            self.profile_id,
            conversation_id,
            clean,
            mode=mode,
        )
        command = _mapping(payload.get("command"))
        command_id = str(command.get("id") or "")
        if not command_id:
            raise ValueError("control API did not return a command id")
        self._pending_commands[conversation_id] = command_id
        self._timeline.append(TimelineEntry("you", clean, status="submitted"))
        return command_id

    def decide_permission(self, request_id: str, decision: str) -> None:
        conversation_id = self._require_conversation()
        self.client.decide_permission(
            self.profile_id,
            conversation_id,
            request_id,
            decision=decision,
        )
        self._timeline.append(
            TimelineEntry(
                "permission",
                f"{decision.title()}ed permission {request_id[:8]}.",
                status=decision,
            )
        )

    def decide_proposal(self, proposal_id: str, action: str) -> None:
        self.client.decide_proposal(
            self.profile_id,
            proposal_id,
            action=action,
        )
        self._timeline.append(
            TimelineEntry(
                "learning",
                f"{'Approved' if action == 'approve' else 'Rejected'} "
                f"proposal {proposal_id[:8]}.",
                status=action,
            )
        )

    def control_work(self, kind: str, item: Record, action: str) -> None:
        item_id = str(item.get("id") or "")
        revision = item.get("revision")
        if not item_id or isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError(f"{kind} is missing an id or revision")
        if kind == "goal":
            self.client.control_goal(
                self.profile_id,
                item_id,
                action=action,
                revision=revision,
            )
        elif kind == "task":
            self.client.control_task(
                self.profile_id,
                item_id,
                action=action,
                revision=revision,
            )
        elif kind == "job":
            self.client.control_job(
                self.profile_id,
                item_id,
                action=action,
                revision=revision,
            )
        else:
            raise ValueError(f"unsupported work kind: {kind}")
        self._timeline.append(
            TimelineEntry(
                "control",
                f"{action.title()} requested for {kind} {item_id[:8]}.",
                status=action,
            )
        )

    def steer_goal(self, item: Record, instruction: str) -> None:
        goal_id = str(item.get("id") or "")
        revision = item.get("revision")
        clean = instruction.strip()
        if (
            not goal_id
            or isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise ValueError("goal is missing an id or revision")
        if not clean:
            raise ValueError("steering instruction cannot be empty")
        self.client.control_goal(
            self.profile_id,
            goal_id,
            action="steer",
            revision=revision,
            instruction=clean,
        )
        self._timeline.append(
            TimelineEntry(
                "control",
                f"Steered goal {goal_id[:8]}: {clean}",
                status="steer",
            )
        )

    def cancel_conversation(self) -> None:
        conversation_id = self._require_conversation()
        self.client.cancel_conversation(self.profile_id, conversation_id)
        self._timeline.append(
            TimelineEntry(
                "control",
                f"Stop requested for conversation {conversation_id[:8]}.",
                status="cancel",
            )
        )

    def _poll_command(self, errors: dict[str, str]) -> None:
        conversation_id = self.selected_conversation_id
        if not conversation_id:
            return
        command_id = self._pending_commands.get(conversation_id)
        if command_id is None:
            return
        try:
            payload = self.client.get_command(
                self.profile_id,
                conversation_id,
                command_id,
            )
            command = _mapping(payload.get("command"))
            status = str(command.get("status") or "unknown")
            if status not in {"completed", "failed", "cancelled", "uncertain"}:
                return
            result = _mapping(command.get("result"))
            content = str(
                result.get("content")
                or command.get("error")
                or f"Command finished with status {status}."
            )
            if not self._timeline or self._timeline[-1].text != content:
                self._timeline.append(
                    TimelineEntry("chulk", content, status=status)
                )
            self._pending_commands.pop(conversation_id, None)
        except Exception as exc:
            errors["command"] = str(exc)

    def _poll_events(
        self,
        conversation_id: str,
        errors: dict[str, str],
    ) -> None:
        key = (self.profile_id, conversation_id)
        cursor = self._event_cursors.get(key)
        try:
            payload = self.client.list_events(
                self.profile_id,
                conversation_id,
                after=cursor,
            )
        except ControlApiError as exc:
            if exc.code != "event_cursor_expired":
                errors["events"] = str(exc)
                return
            self._timeline.append(
                TimelineEntry(
                    "system",
                    "Event history rotated; reconnecting from retained evidence.",
                    status="reconnect",
                )
            )
            self._event_cursors.pop(key, None)
            try:
                payload = self.client.list_events(
                    self.profile_id,
                    conversation_id,
                )
            except Exception as retry_exc:
                errors["events"] = str(retry_exc)
                return
        except Exception as exc:
            errors["events"] = str(exc)
            return

        values = payload.get("events", ())
        if isinstance(values, list):
            for value in values:
                record = _mapping(value)
                self._append_event(record)
        next_cursor = payload.get("next_cursor")
        if isinstance(next_cursor, str) and next_cursor:
            self._event_cursors[key] = next_cursor

    def _append_event(self, record: Record) -> None:
        event = _mapping(record.get("event"))
        name = str(event.get("name") or "event")
        payload = _mapping(event.get("payload"))
        if name == "model.delta":
            text = str(payload.get("text") or "")
            if not text:
                return
            if self._timeline and self._timeline[-1].kind == "stream":
                previous = self._timeline[-1]
                self._timeline[-1] = TimelineEntry(
                    "stream",
                    f"{previous.text}{text}",
                    status="streaming",
                )
            else:
                self._timeline.append(
                    TimelineEntry("stream", text, status="streaming")
                )
            return
        if name == "run.started":
            text = "Run started."
        elif name == "run.completed":
            result = _mapping(payload.get("result"))
            text = str(result.get("content") or "Run completed.")
            if self._timeline and self._timeline[-1].kind == "stream":
                self._timeline[-1] = TimelineEntry(
                    "chulk",
                    text,
                    status=name,
                )
                return
        elif name == "run.failed":
            error = _mapping(payload.get("error"))
            text = str(error.get("message") or "Run failed.")
        elif name == "plan.created":
            plan = _mapping(payload.get("plan"))
            text = f"Plan ready: {plan.get('summary') or 'review required'}"
        elif name.startswith("tool.call."):
            text = f"{payload.get('tool_name') or 'tool'} · {name.rsplit('.', 1)[-1]}"
        elif name == "permission.requested":
            text = f"Permission requested for {payload.get('tool_name') or 'tool'}."
        else:
            text = name.replace(".", " ")
        self._timeline.append(
            TimelineEntry("event", text, status=name)
        )

    @staticmethod
    def _read(
        key: str,
        operation: Any,
        errors: dict[str, str],
    ) -> tuple[Record, ...]:
        try:
            payload = operation()
            values = payload.get(key, ())
            if not isinstance(values, list):
                return ()
            return tuple(_mapping(item) for item in values)
        except Exception as exc:
            errors[key] = str(exc)
            return ()

    def _require_conversation(self) -> str:
        if self.selected_conversation_id is None:
            raise ValueError("select or create a conversation first")
        return self.selected_conversation_id

    def _usage_payload(self, conversation_id: str | None) -> dict[str, Any]:
        if conversation_id is not None:
            payload = self.client.query_usage(
                self.profile_id,
                conversation_id=conversation_id,
                limit=50,
            )
            return {"usage": payload.get("entries", [])}
        payload = self.client.query_usage(
            self.profile_id,
            group_by="resource_kind",
            limit=50,
        )
        return {"usage": payload.get("groups", [])}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["ControlApiDataSource", "OperatorDataSource"]
