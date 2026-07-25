"""Versioned, redacted fixtures for deterministic action-loop replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal, TYPE_CHECKING, cast

from chulk.core.actions import (
    ActionParseError,
    AgentAction,
    parse_model_response,
)
from chulk.core.trace_format import format_action_trace
from chulk.errors import ErrorDetails, TraceError
from chulk.redaction import redact_data
from chulk.storage.private_files import write_private_text

if TYPE_CHECKING:
    from chulk.tracing.reader import Trace, TraceRecord


REPLAY_FIXTURE_SCHEMA_VERSION = 1
SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS = frozenset(
    {REPLAY_FIXTURE_SCHEMA_VERSION}
)
DEFAULT_REPLAY_FIXTURE_MAX_BYTES = 16 * 1024 * 1024

_ACTION_FIELDS = {
    "final_answer": frozenset({"type", "content"}),
    "tool_call": frozenset({"type", "tool_name", "arguments"}),
    "plan": frozenset({"type", "plan"}),
    "plan_step_update": frozenset(
        {"type", "step_id", "status", "evidence", "reason"}
    ),
}
_GENERATED_ID_KEYS = frozenset(
    {
        "artifact_id",
        "conversation_id",
        "goal_id",
        "job_id",
        "permission_id",
        "plan_step_id",
        "proposal_id",
        "request_id",
        "response_id",
        "run_id",
        "session_id",
        "task_id",
        "tool_call_id",
        "trace_id",
        "turn_id",
    }
)
_UNORDERED_SEQUENCE_KEYS = frozenset(
    {
        "available_tool_names",
        "hosted_mcp_server_labels",
        "loaded_memory_ids",
        "loaded_skill_names",
        "native_tool_names",
    }
)
_PERMISSION_EVENT_TYPES = frozenset(
    {
        "mcp_approval_decided",
        "mcp_approval_requested",
        "tool_permission_decided",
        "tool_permission_requested",
    }
)
_PLAN_EVENT_TYPES = frozenset(
    {
        "plan_approved",
        "plan_created",
        "plan_rejected",
        "plan_revision_requested",
        "plan_step_blocked",
        "plan_step_completed",
        "plan_step_started",
    }
)


class ReplayFixtureError(TraceError, ValueError):
    """Raised when a replay fixture cannot be built or parsed safely."""

    def __init__(
        self,
        message: str,
        *,
        fixture_path: Path | str | None = None,
    ) -> None:
        super().__init__(
            message,
            details=ErrorDetails(
                trace_path=str(fixture_path) if fixture_path is not None else None
            ),
        )


@dataclass(frozen=True)
class ReplayFixtureSource:
    """Stable provenance retained without embedding a host-specific trace path."""

    trace_schema_versions: tuple[int, ...]
    event_count: int

    def __post_init__(self) -> None:
        versions = tuple(self.trace_schema_versions)
        if not versions or any(
            isinstance(version, bool) or not isinstance(version, int)
            for version in versions
        ):
            raise ReplayFixtureError(
                "source.trace_schema_versions must contain integers"
            )
        if isinstance(self.event_count, bool) or not isinstance(self.event_count, int):
            raise ReplayFixtureError("source.event_count must be an integer")
        if self.event_count < 1:
            raise ReplayFixtureError("source.event_count must be greater than zero")
        object.__setattr__(self, "trace_schema_versions", tuple(sorted(set(versions))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema_versions": list(self.trace_schema_versions),
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReplayFixtureSource":
        payload = _strict_mapping(
            value,
            field="source",
            required={"trace_schema_versions", "event_count"},
        )
        versions = payload["trace_schema_versions"]
        if not isinstance(versions, list):
            raise ReplayFixtureError("source.trace_schema_versions must be a list")
        return cls(
            trace_schema_versions=tuple(versions),
            event_count=payload["event_count"],
        )


@dataclass(frozen=True)
class RecordedModelAction:
    """One validated provider-neutral action in replay order."""

    sequence: int
    action: dict[str, Any]
    request_index: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.sequence, field="model_actions.sequence")
        if self.request_index is not None:
            _positive_integer(
                self.request_index,
                field="model_actions.request_index",
            )
        canonical = _canonical_action_payload(redact_data(self.action))
        object.__setattr__(self, "action", canonical)

    def to_action(self) -> AgentAction:
        """Return the shared action dataclass consumed by the reducer."""
        return _parse_canonical_action(self.action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "request_index": self.request_index,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecordedModelAction":
        payload = _strict_mapping(
            value,
            field="model_actions[]",
            required={"sequence", "request_index", "action"},
        )
        return cls(
            sequence=payload["sequence"],
            request_index=payload["request_index"],
            action=_mapping(payload["action"], field="model_actions[].action"),
        )


@dataclass(frozen=True)
class RecordedToolResult:
    """One model-visible tool outcome in replay order."""

    sequence: int
    tool_name: str
    iteration: int
    phase: Literal["planning", "execution"]
    success: bool
    observation: str
    exit_code: int | None = None
    error: str | None = None
    failure_kind: str | None = None
    metadata: dict[str, Any] | None = None
    output_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.sequence, field="tool_results.sequence")
        _positive_integer(self.iteration, field="tool_results.iteration")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ReplayFixtureError(
                "tool_results.tool_name must be a non-empty string"
            )
        if self.phase not in {"planning", "execution"}:
            raise ReplayFixtureError(
                "tool_results.phase must be planning or execution"
            )
        if not isinstance(self.success, bool):
            raise ReplayFixtureError("tool_results.success must be a boolean")
        if not isinstance(self.observation, str):
            raise ReplayFixtureError("tool_results.observation must be a string")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ReplayFixtureError(
                "tool_results.exit_code must be an integer or null"
            )
        for field_name, item in (
            ("error", self.error),
            ("failure_kind", self.failure_kind),
        ):
            if item is not None and not isinstance(item, str):
                raise ReplayFixtureError(
                    f"tool_results.{field_name} must be a string or null"
                )
        object.__setattr__(self, "tool_name", self.tool_name.strip())
        object.__setattr__(
            self,
            "observation",
            str(redact_data(self.observation)),
        )
        object.__setattr__(
            self,
            "error",
            redact_data(self.error) if self.error is not None else None,
        )
        object.__setattr__(
            self,
            "metadata",
            _redacted_mapping(self.metadata, field="tool_results.metadata"),
        )
        object.__setattr__(
            self,
            "output_metadata",
            _redacted_mapping(
                self.output_metadata,
                field="tool_results.output_metadata",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tool_name": self.tool_name,
            "iteration": self.iteration,
            "phase": self.phase,
            "success": self.success,
            "observation": self.observation,
            "exit_code": self.exit_code,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "metadata": self.metadata,
            "output_metadata": self.output_metadata,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RecordedToolResult":
        payload = _strict_mapping(
            value,
            field="tool_results[]",
            required={
                "sequence",
                "tool_name",
                "iteration",
                "phase",
                "success",
                "observation",
                "exit_code",
                "error",
                "failure_kind",
                "metadata",
                "output_metadata",
            },
        )
        return cls(**payload)


@dataclass(frozen=True)
class ExpectedReplayOutcome:
    """Normalized evidence compared after executable replay."""

    state: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    permissions: tuple[dict[str, Any], ...]
    plans: tuple[dict[str, Any], ...]
    usage: tuple[dict[str, Any], ...]
    costs: tuple[dict[str, Any], ...]
    result: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            _redacted_mapping(self.state, field="expected.state"),
        )
        object.__setattr__(
            self,
            "events",
            _mapping_tuple(self.events, field="expected.events"),
        )
        object.__setattr__(
            self,
            "permissions",
            _mapping_tuple(self.permissions, field="expected.permissions"),
        )
        object.__setattr__(
            self,
            "plans",
            _mapping_tuple(self.plans, field="expected.plans"),
        )
        object.__setattr__(
            self,
            "usage",
            _mapping_tuple(self.usage, field="expected.usage"),
        )
        object.__setattr__(
            self,
            "costs",
            _mapping_tuple(self.costs, field="expected.costs"),
        )
        object.__setattr__(
            self,
            "result",
            _redacted_mapping(self.result, field="expected.result"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "events": list(self.events),
            "permissions": list(self.permissions),
            "plans": list(self.plans),
            "usage": list(self.usage),
            "costs": list(self.costs),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExpectedReplayOutcome":
        payload = _strict_mapping(
            value,
            field="expected",
            required={
                "state",
                "events",
                "permissions",
                "plans",
                "usage",
                "costs",
                "result",
            },
        )
        sequence_fields = ("events", "permissions", "plans", "usage", "costs")
        for field_name in sequence_fields:
            if not isinstance(payload[field_name], list):
                raise ReplayFixtureError(
                    f"expected.{field_name} must be a list"
                )
        return cls(
            state=_mapping(payload["state"], field="expected.state"),
            events=tuple(payload["events"]),
            permissions=tuple(payload["permissions"]),
            plans=tuple(payload["plans"]),
            usage=tuple(payload["usage"]),
            costs=tuple(payload["costs"]),
            result=_mapping(payload["result"], field="expected.result"),
        )


@dataclass(frozen=True)
class ReplayFixture:
    """A complete, versioned input and expected-output replay contract."""

    schema_version: int
    source: ReplayFixtureSource
    model_actions: tuple[RecordedModelAction, ...]
    tool_results: tuple[RecordedToolResult, ...]
    expected: ExpectedReplayOutcome

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version,
            int,
        ):
            raise ReplayFixtureError(
                "Replay fixture schema_version must be an integer"
            )
        if self.schema_version not in SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(version)
                for version in sorted(SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS)
            )
            raise ReplayFixtureError(
                f"Replay fixture schema version {self.schema_version!r} is not "
                f"supported; supported versions: {supported}"
            )
        object.__setattr__(self, "model_actions", tuple(self.model_actions))
        object.__setattr__(self, "tool_results", tuple(self.tool_results))
        _unique_sequences(self.model_actions, field="model_actions")
        _unique_sequences(self.tool_results, field="tool_results")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "model_actions": [item.to_dict() for item in self.model_actions],
            "tool_results": [item.to_dict() for item in self.tool_results],
            "expected": self.expected.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    @classmethod
    def from_dict(cls, value: object) -> "ReplayFixture":
        payload = _strict_mapping(
            value,
            field="fixture",
            required={
                "schema_version",
                "source",
                "model_actions",
                "tool_results",
                "expected",
            },
        )
        for field_name in ("model_actions", "tool_results"):
            if not isinstance(payload[field_name], list):
                raise ReplayFixtureError(f"{field_name} must be a list")
        return cls(
            schema_version=payload["schema_version"],
            source=ReplayFixtureSource.from_dict(payload["source"]),
            model_actions=tuple(
                RecordedModelAction.from_dict(item)
                for item in payload["model_actions"]
            ),
            tool_results=tuple(
                RecordedToolResult.from_dict(item)
                for item in payload["tool_results"]
            ),
            expected=ExpectedReplayOutcome.from_dict(payload["expected"]),
        )

    @classmethod
    def from_trace(
        cls,
        trace: "Trace",
        *,
        acknowledge_sensitive_data: bool,
    ) -> "ReplayFixture":
        if acknowledge_sensitive_data is not True:
            raise ReplayFixtureError(
                "Trace-to-fixture export requires explicit sensitive-data "
                "acknowledgement"
            )
        actions = _recorded_model_actions(trace.events)
        tool_results = _recorded_tool_results(trace.events)
        expected = _expected_outcome(trace.events)
        return cls(
            schema_version=REPLAY_FIXTURE_SCHEMA_VERSION,
            source=ReplayFixtureSource(
                trace_schema_versions=tuple(
                    sorted({event.schema_version for event in trace.events})
                ),
                event_count=len(trace.events),
            ),
            model_actions=actions,
            tool_results=tool_results,
            expected=expected,
        )


def normalize_replay_value(value: Any) -> Any:
    """Return deterministic JSON-safe data for replay comparison."""
    return _ReplayNormalizer().normalize(redact_data(value))


def load_replay_fixture(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_REPLAY_FIXTURE_MAX_BYTES,
) -> ReplayFixture:
    """Load one bounded fixture without following symlink or special targets."""
    fixture_path = Path(path).expanduser().absolute()
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("replay fixture max_bytes must be greater than zero")
    descriptor = _open_fixture(fixture_path, max_bytes=max_bytes)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            try:
                text = stream.read(max_bytes + 1)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReplayFixtureError(
                    f"Replay fixture is not valid UTF-8 JSON: {fixture_path}",
                    fixture_path=fixture_path,
                ) from exc
            if len(text.encode("utf-8")) > max_bytes:
                raise ReplayFixtureError(
                    f"Replay fixture exceeds the {max_bytes}-byte parse limit",
                    fixture_path=fixture_path,
                )
            try:
                value = json.loads(
                    text,
                    object_pairs_hook=_reject_duplicate_object_pairs,
                )
            except ReplayFixtureError as exc:
                raise ReplayFixtureError(
                    str(exc),
                    fixture_path=fixture_path,
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReplayFixtureError(
                    f"Replay fixture is not valid UTF-8 JSON: {fixture_path}",
                    fixture_path=fixture_path,
                ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        return ReplayFixture.from_dict(value)
    except ReplayFixtureError as exc:
        if exc.details.trace_path is not None:
            raise
        raise ReplayFixtureError(
            str(exc),
            fixture_path=fixture_path,
        ) from exc


def export_replay_fixture(
    trace: "Trace",
    output_path: Path | str,
    *,
    acknowledge_sensitive_data: bool,
    overwrite: bool = False,
) -> Path:
    """Write a redacted fixture after explicit acknowledgement."""
    fixture = ReplayFixture.from_trace(
        trace,
        acknowledge_sensitive_data=acknowledge_sensitive_data,
    )
    destination = Path(output_path).expanduser().absolute()
    if destination.resolve() == trace.path.resolve():
        raise ReplayFixtureError(
            "Replay fixture output cannot overwrite the source trace",
            fixture_path=destination,
        )
    try:
        return write_private_text(
            destination,
            fixture.to_json(),
            overwrite=overwrite,
            private_parent=False,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        raise ReplayFixtureError(
            str(exc),
            fixture_path=destination,
        ) from exc


class _ReplayNormalizer:
    def __init__(self) -> None:
        self._ids: dict[tuple[str, str], str] = {}
        self._id_counts: dict[str, int] = {}

    def normalize(self, value: Any, *, key: str | None = None) -> Any:
        if key is not None and _is_timestamp_key(key) and value is not None:
            return "<timestamp>"
        if key is not None and _is_duration_key(key) and value is not None:
            return "<duration>"
        if key in _GENERATED_ID_KEYS and isinstance(value, str) and value:
            return self._normalize_id(key, value)
        if key == "depends_on" and isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [
                self._normalize_id("plan_step_id", item)
                if isinstance(item, str) and item
                else self.normalize(item)
                for item in value
            ]
        if isinstance(value, Mapping):
            plan_step = _looks_like_plan_step(value)
            normalized = {
                str(item_key): self.normalize(
                    item,
                    key=(
                        "plan_step_id"
                        if plan_step and str(item_key) == "id"
                        else str(item_key)
                    ),
                )
                for item_key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            }
            return normalized
        if isinstance(value, (set, frozenset)):
            items = [self.normalize(item) for item in value]
            return sorted(items, key=_stable_json)
        if isinstance(value, (list, tuple)):
            items = [self.normalize(item) for item in value]
            if key in _UNORDERED_SEQUENCE_KEYS:
                return sorted(items, key=_stable_json)
            return items
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def _normalize_id(self, category: str, value: str) -> str:
        identity = (category, value)
        existing = self._ids.get(identity)
        if existing is not None:
            return existing
        number = self._id_counts.get(category, 0) + 1
        self._id_counts[category] = number
        placeholder = f"<{category}:{number}>"
        self._ids[identity] = placeholder
        return placeholder


def _recorded_model_actions(
    events: tuple["TraceRecord", ...],
) -> tuple[RecordedModelAction, ...]:
    preferred = [event for event in events if event.type == "parsed_action"]
    action_events = preferred or [
        event for event in events if event.type == "model_response_parsed"
    ]
    final_answers: dict[str | None, list[str]] = {}
    for event in events:
        content = event.payload.get("content")
        if event.type == "final_answer" and isinstance(content, str):
            final_answers.setdefault(event.turn_id, []).append(content)

    actions: list[RecordedModelAction] = []
    for sequence, event in enumerate(action_events, start=1):
        payload = dict(event.payload)
        payload.pop("turn_id", None)
        request_index = payload.pop("request_index", None)
        if payload.get("type") == "final_answer" and not isinstance(
            payload.get("content"), str
        ):
            candidates = final_answers.get(event.turn_id, [])
            if not candidates:
                raise ReplayFixtureError(
                    "Trace final_answer action is missing replayable content; "
                    "capture a new trace with the current trace schema"
                )
            payload["content"] = candidates.pop(0)
        actions.append(
            RecordedModelAction(
                sequence=sequence,
                request_index=(
                    request_index if isinstance(request_index, int) else None
                ),
                action=payload,
            )
        )
    if not actions:
        raise ReplayFixtureError(
            "Trace contains no parsed model actions to export"
        )
    return tuple(actions)


def _recorded_tool_results(
    events: tuple["TraceRecord", ...],
) -> tuple[RecordedToolResult, ...]:
    observations: dict[tuple[str | None, int], "TraceRecord"] = {}
    for event in events:
        if event.type != "tool_observation":
            continue
        output_metadata = event.payload.get("output_metadata")
        metadata = output_metadata if isinstance(output_metadata, Mapping) else {}
        identity = metadata.get("tool_call_identity")
        identity_mapping = identity if isinstance(identity, Mapping) else {}
        iteration = identity_mapping.get("iteration")
        if isinstance(iteration, int):
            observations[(event.turn_id, iteration)] = event

    records: list[RecordedToolResult] = []
    completed_types = {"tool_call_completed", "tool_call_failed"}
    for event in events:
        if event.type not in completed_types:
            continue
        iteration = event.payload.get("iteration")
        if not isinstance(iteration, int):
            raise ReplayFixtureError(
                "Trace tool result is missing an integer iteration"
            )
        observation_event = observations.get((event.turn_id, iteration))
        if observation_event is None:
            raise ReplayFixtureError(
                f"Trace tool result {iteration} is missing its model observation"
            )
        observation = observation_event.payload.get("observation")
        if not isinstance(observation, str):
            raise ReplayFixtureError(
                f"Trace tool result {iteration} has no string observation"
            )
        success = event.payload.get("success")
        if not isinstance(success, bool):
            success = event.type == "tool_call_completed"
        exit_code = event.payload.get("exit_code")
        raw_phase = event.payload.get("phase")
        phase = cast(
            Literal["planning", "execution"],
            raw_phase if raw_phase in {"planning", "execution"} else "execution",
        )
        records.append(
            RecordedToolResult(
                sequence=len(records) + 1,
                tool_name=str(event.payload.get("tool_name") or "unknown"),
                iteration=iteration,
                phase=phase,
                success=success,
                observation=observation,
                exit_code=exit_code if isinstance(exit_code, int) else None,
                error=(
                    event.payload.get("error")
                    if isinstance(event.payload.get("error"), str)
                    else None
                ),
                failure_kind=(
                    event.payload.get("failure_kind")
                    if isinstance(event.payload.get("failure_kind"), str)
                    else None
                ),
                metadata=(
                    dict(event.payload["metadata"])
                    if isinstance(event.payload.get("metadata"), Mapping)
                    else {}
                ),
                output_metadata=(
                    dict(observation_event.payload["output_metadata"])
                    if isinstance(
                        observation_event.payload.get("output_metadata"),
                        Mapping,
                    )
                    else {}
                ),
            )
        )
    return tuple(records)


def _expected_outcome(
    events: tuple["TraceRecord", ...],
) -> ExpectedReplayOutcome:
    event_values = [
        {
            "type": event.type,
            "turn_id": event.turn_id,
            "payload": event.payload,
        }
        for event in events
    ]
    permission_values = [
        value for value in event_values if value["type"] in _PERMISSION_EVENT_TYPES
    ]
    plan_values = [
        value for value in event_values if value["type"] in _PLAN_EVENT_TYPES
    ]
    usage_values: list[dict[str, Any]] = []
    cost_values: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    result: dict[str, Any] = {}
    for event in events:
        if event.type == "model_response":
            request_index = event.payload.get("request_index")
            if isinstance(event.payload.get("usage"), Mapping):
                usage_values.append(
                    {
                        "request_index": request_index,
                        "usage": event.payload["usage"],
                    }
                )
            if isinstance(event.payload.get("cost"), Mapping):
                cost_values.append(
                    {
                        "request_index": request_index,
                        "cost": event.payload["cost"],
                    }
                )
        if event.type == "final_answer" and isinstance(
            event.payload.get("content"), str
        ):
            result = {
                "status": "completed",
                "content": event.payload["content"],
            }
        if event.type == "turn_finished":
            if isinstance(event.payload.get("agent_state"), Mapping):
                state = dict(event.payload["agent_state"])
            if isinstance(event.payload.get("turn"), Mapping):
                turn = dict(event.payload["turn"])
                result = {
                    "status": turn.get("status"),
                    "content": turn.get("final_answer"),
                    "errors": turn.get("errors", []),
                    "plan": turn.get("active_plan"),
                }

    normalized = normalize_replay_value(
        {
            "state": state,
            "events": event_values,
            "permissions": permission_values,
            "plans": plan_values,
            "usage": usage_values,
            "costs": cost_values,
            "result": result,
        }
    )
    return ExpectedReplayOutcome(
        state=normalized["state"],
        events=tuple(normalized["events"]),
        permissions=tuple(normalized["permissions"]),
        plans=tuple(normalized["plans"]),
        usage=tuple(normalized["usage"]),
        costs=tuple(normalized["costs"]),
        result=normalized["result"],
    )


def _canonical_action_payload(value: object) -> dict[str, Any]:
    payload = _mapping(value, field="model_actions[].action")
    action_type = payload.get("type")
    if action_type not in _ACTION_FIELDS:
        allowed = ", ".join(sorted(_ACTION_FIELDS))
        raise ReplayFixtureError(
            f"model action type must be one of: {allowed}"
        )
    unexpected = set(payload) - _ACTION_FIELDS[action_type]
    missing = _ACTION_FIELDS[action_type] - set(payload)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ReplayFixtureError(
            f"model action {action_type} has unexpected field(s): {names}"
        )
    if missing:
        names = ", ".join(sorted(missing))
        raise ReplayFixtureError(
            f"model action {action_type} is missing field(s): {names}"
        )
    action = _parse_canonical_action(payload)
    return _action_to_payload(action)


def _parse_canonical_action(payload: Mapping[str, Any]) -> AgentAction:
    transport = dict(payload)
    if transport.get("type") == "plan_step_update":
        transport = {
            "type": "plan_step_update",
            "step_update": {
                "step_id": transport.get("step_id"),
                "status": transport.get("status"),
                "evidence": transport.get("evidence"),
                "reason": transport.get("reason"),
            },
        }
    try:
        return parse_model_response(transport)
    except ActionParseError as exc:
        raise ReplayFixtureError(f"Invalid recorded model action: {exc}") from exc


def _action_to_payload(action: AgentAction) -> dict[str, Any]:
    return dict(redact_data(format_action_trace(action)))


def _open_fixture(path: Path, *, max_bytes: int) -> int:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ReplayFixtureError(
            f"Replay fixture does not exist: {path}",
            fixture_path=path,
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ReplayFixtureError(
            f"Replay fixture path is not a regular file: {path}",
            fixture_path=path,
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayFixtureError(
            f"Replay fixture could not be opened safely: {path}",
            fixture_path=path,
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ReplayFixtureError(
            f"Replay fixture path is not a regular file: {path}",
            fixture_path=path,
        )
    if opened.st_size > max_bytes:
        os.close(descriptor)
        raise ReplayFixtureError(
            f"Replay fixture exceeds the {max_bytes}-byte parse limit",
            fixture_path=path,
        )
    return descriptor


def _strict_mapping(
    value: object,
    *,
    field: str,
    required: set[str],
) -> dict[str, Any]:
    payload = _mapping(value, field=field)
    unexpected = set(payload) - required
    missing = required - set(payload)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ReplayFixtureError(f"{field} has unexpected field(s): {names}")
    if missing:
        names = ", ".join(sorted(missing))
        raise ReplayFixtureError(f"{field} is missing field(s): {names}")
    return payload


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayFixtureError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _redacted_mapping(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    payload = {} if value is None else _mapping(value, field=field)
    return dict(redact_data(payload))


def _mapping_tuple(
    values: object,
    *,
    field: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, Sequence) or isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise ReplayFixtureError(f"{field} must be a sequence")
    return tuple(
        _redacted_mapping(item, field=f"{field}[]")
        for item in values
    )


def _positive_integer(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReplayFixtureError(f"{field} must be a positive integer")


def _unique_sequences(values: Sequence[object], *, field: str) -> None:
    sequences: list[int] = []
    for item in values:
        sequence = getattr(item, "sequence", None)
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ReplayFixtureError(f"{field} contains an invalid sequence")
        sequences.append(sequence)
    if len(set(sequences)) != len(sequences):
        raise ReplayFixtureError(f"{field} contains duplicate sequence values")
    if sequences != sorted(sequences):
        raise ReplayFixtureError(f"{field} must be ordered by sequence")


def _is_timestamp_key(key: str) -> bool:
    return key == "timestamp" or key.endswith("_at")


def _is_duration_key(key: str) -> bool:
    lowered = key.lower()
    return "duration" in lowered or "latency" in lowered or "elapsed" in lowered


def _looks_like_plan_step(value: Mapping[object, object]) -> bool:
    keys = {str(key) for key in value}
    return {"id", "title", "description", "status"} <= keys


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayFixtureError(
                f"Replay fixture JSON contains duplicate field: {key}"
            )
        result[key] = value
    return result


__all__ = [
    "DEFAULT_REPLAY_FIXTURE_MAX_BYTES",
    "ExpectedReplayOutcome",
    "REPLAY_FIXTURE_SCHEMA_VERSION",
    "RecordedModelAction",
    "RecordedToolResult",
    "ReplayFixture",
    "ReplayFixtureError",
    "ReplayFixtureSource",
    "SUPPORTED_REPLAY_FIXTURE_SCHEMA_VERSIONS",
    "export_replay_fixture",
    "load_replay_fixture",
    "normalize_replay_value",
]
