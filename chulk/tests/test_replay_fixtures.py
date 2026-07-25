"""Versioned deterministic replay fixture contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from chulk import Agent, AgentConfig
from chulk.core.actions import (
    FinalAnswerAction,
    PlanAction,
    PlanStepUpdateAction,
    ToolCallAction,
)
from chulk.core.trace_format import format_action_trace
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolResult
from chulk.tracing import (
    REPLAY_FIXTURE_SCHEMA_VERSION,
    RecordedModelAction,
    ReplayFixture,
    ReplayFixtureError,
    Trace,
    export_replay_fixture,
    load_replay_fixture,
    normalize_replay_value,
)


def _event(
    event_type: str,
    payload: dict,
    *,
    turn_id: str | None = "turn-sensitive-123",
) -> dict:
    value = {
        "schema_version": 1,
        "conversation_id": "conversation-sensitive-456",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "type": event_type,
        "payload": payload,
    }
    if turn_id is not None:
        value["turn_id"] = turn_id
    return value


def _write_trace(path: Path, events: list[dict]) -> Trace:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return Trace.from_jsonl(path)


def _fixture_payload() -> dict:
    return {
        "schema_version": REPLAY_FIXTURE_SCHEMA_VERSION,
        "source": {
            "trace_schema_versions": [1],
            "event_count": 2,
        },
        "model_actions": [
            {
                "sequence": 1,
                "request_index": 1,
                "action": {
                    "type": "tool_call",
                    "tool_name": "lookup",
                    "arguments": {"query": "stable"},
                },
            },
            {
                "sequence": 2,
                "request_index": 2,
                "action": {
                    "type": "final_answer",
                    "content": "done",
                },
            },
        ],
        "tool_results": [],
        "expected": {
            "state": {},
            "events": [],
            "permissions": [],
            "plans": [],
            "usage": [],
            "costs": [],
            "result": {"status": "completed", "content": "done"},
        },
    }


def test_fixture_round_trips_validated_action_dataclasses() -> None:
    fixture = ReplayFixture.from_dict(_fixture_payload())

    tool_action = fixture.model_actions[0].to_action()
    final_action = fixture.model_actions[1].to_action()

    assert fixture.to_dict() == _fixture_payload()
    assert isinstance(tool_action, ToolCallAction)
    assert tool_action.arguments == {"query": "stable"}
    assert isinstance(final_action, FinalAnswerAction)
    assert final_action.content == "done"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(schema_version=999),
            "schema version 999",
        ),
        (
            lambda payload: payload.update(schema_version=True),
            "schema_version must be an integer",
        ),
        (
            lambda payload: payload.update(unexpected=True),
            "unexpected field",
        ),
        (
            lambda payload: payload["model_actions"][0]["action"].update(
                unexpected=True
            ),
            "unexpected field",
        ),
        (
            lambda payload: payload["model_actions"][0].update(sequence=2),
            "duplicate sequence",
        ),
        (
            lambda payload: payload["expected"].update(state=[]),
            "expected.state must be an object",
        ),
    ],
)
def test_fixture_rejects_schema_drift(mutate, message: str) -> None:
    payload = _fixture_payload()
    mutate(payload)

    with pytest.raises(ReplayFixtureError, match=message):
        ReplayFixture.from_dict(payload)


def test_recorded_action_redacts_sensitive_arguments_before_validation() -> None:
    record = RecordedModelAction(
        sequence=1,
        request_index=1,
        action={
            "type": "tool_call",
            "tool_name": "lookup",
            "arguments": {
                "api_key": "sk-sensitive-123456789",
                "query": "Authorization: Bearer secret-value",
            },
        },
    )

    serialized = json.dumps(record.to_dict())

    assert "sk-sensitive" not in serialized
    assert "secret-value" not in serialized
    assert serialized.count("[redacted]") >= 2


@pytest.mark.parametrize(
    ("payload", "action_type"),
    [
        (
            {
                "type": "plan",
                "plan": {
                    "summary": "Make the change.",
                    "steps": [
                        {
                            "id": "implementation",
                            "title": "Implement",
                            "description": "Implement and verify.",
                        }
                    ],
                },
            },
            PlanAction,
        ),
        (
            {
                "type": "plan_step_update",
                "step_id": "implementation",
                "status": "completed",
                "evidence": "Focused tests pass.",
                "reason": None,
            },
            PlanStepUpdateAction,
        ),
    ],
)
def test_recorded_action_supports_plan_transitions(
    payload: dict,
    action_type: type,
) -> None:
    record = RecordedModelAction(
        sequence=1,
        request_index=1,
        action=payload,
    )

    assert isinstance(record.to_action(), action_type)
    assert ReplayFixture.from_dict(
        {
            **_fixture_payload(),
            "model_actions": [record.to_dict()],
        }
    ).model_actions[0].to_dict() == record.to_dict()


def test_normalizer_removes_runtime_variance_but_preserves_event_order() -> None:
    first = {
        "conversation_id": "conversation-a",
        "turn_id": "turn-a",
        "started_at": "2026-01-01T00:00:00+00:00",
        "duration_ms": 12.5,
        "available_tool_names": ["write_file", "read_file"],
        "events": [{"type": "started"}, {"type": "finished"}],
        "plan": {
            "steps": [
                {
                    "id": "generated-first",
                    "title": "First",
                    "description": "Start",
                    "status": "completed",
                    "depends_on": [],
                },
                {
                    "id": "generated-second",
                    "title": "Second",
                    "description": "Finish",
                    "status": "pending",
                    "depends_on": ["generated-first"],
                },
            ]
        },
    }
    second = {
        **first,
        "conversation_id": "conversation-b",
        "turn_id": "turn-b",
        "started_at": "2030-04-05T10:20:30+00:00",
        "duration_ms": 999,
        "available_tool_names": ["read_file", "write_file"],
        "plan": {
            "steps": [
                {
                    **first["plan"]["steps"][0],
                    "id": "other-first",
                },
                {
                    **first["plan"]["steps"][1],
                    "id": "other-second",
                    "depends_on": ["other-first"],
                },
            ]
        },
    }

    normalized = normalize_replay_value(first)

    assert normalized == normalize_replay_value(second)
    assert normalized["conversation_id"] == "<conversation_id:1>"
    assert normalized["turn_id"] == "<turn_id:1>"
    assert normalized["started_at"] == "<timestamp>"
    assert normalized["duration_ms"] == "<duration>"
    assert normalized["available_tool_names"] == ["read_file", "write_file"]
    assert [event["type"] for event in normalized["events"]] == [
        "started",
        "finished",
    ]
    steps = normalized["plan"]["steps"]
    assert steps[1]["depends_on"] == [steps[0]["id"]]


def test_trace_export_requires_acknowledgement_and_redacts_fixture(
    tmp_path: Path,
) -> None:
    secret = "sk-fixture-secret-123456789"
    events = [
        _event("session_started", {}, turn_id=None),
        _event(
            "turn_started",
            {
                "turn": {
                    "turn_id": "turn-sensitive-123",
                    "started_at": "2026-01-01T00:00:00+00:00",
                }
            },
        ),
        _event(
            "parsed_action",
            {
                "type": "tool_call",
                "tool_name": "lookup",
                "arguments": {"api_key": secret, "query": "safe"},
                "request_index": 1,
            },
        ),
        _event(
            "tool_call_completed",
            {
                "tool_name": "lookup",
                "iteration": 1,
                "phase": "execution",
                "success": True,
                "exit_code": 0,
                "error": None,
                "failure_kind": None,
                "metadata": {"authorization": secret},
            },
        ),
        _event(
            "tool_observation",
            {
                "tool_name": "lookup",
                "observation": f"lookup result token={secret}",
                "output_metadata": {
                    "tool_call_identity": {
                        "iteration": 1,
                        "phase": "execution",
                    },
                    "cookie": secret,
                },
            },
        ),
        _event(
            "parsed_action",
            {
                "type": "final_answer",
                "content": "lookup complete",
                "request_index": 2,
            },
        ),
        _event("final_answer", {"content": "lookup complete"}),
        _event(
            "turn_finished",
            {
                "agent_state": {
                    "conversation_id": "conversation-sensitive-456",
                    "current_turn_id": "turn-sensitive-123",
                },
                "turn": {
                    "turn_id": "turn-sensitive-123",
                    "status": "completed",
                    "final_answer": "lookup complete",
                    "ended_at": "2026-01-01T00:00:01+00:00",
                    "errors": [],
                },
            },
        ),
    ]
    trace = _write_trace(tmp_path / "source.jsonl", events)

    with pytest.raises(ReplayFixtureError, match="explicit sensitive-data"):
        ReplayFixture.from_trace(
            trace,
            acknowledge_sensitive_data=False,
        )

    fixture = ReplayFixture.from_trace(
        trace,
        acknowledge_sensitive_data=True,
    )
    serialized = fixture.to_json()

    assert secret not in serialized
    assert "[redacted]" in serialized
    assert fixture.source.event_count == len(events)
    assert fixture.model_actions[0].request_index == 1
    assert fixture.model_actions[-1].to_action() == FinalAnswerAction(
        type="final_answer",
        content="lookup complete",
    )
    assert fixture.tool_results[0].exit_code == 0
    assert fixture.tool_results[0].observation.endswith("[redacted]")
    assert fixture.expected.state["conversation_id"] == "<conversation_id:1>"
    assert "path" not in fixture.source.to_dict()


def test_fixture_export_consumes_real_action_loop_trace(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    tool = Tool(
        name="lookup",
        description="Return a deterministic value.",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        callable=lambda arguments: ToolResult(
            tool_name="lookup",
            success=True,
            observation=f"Found {arguments['query']}.",
            exit_code=0,
        ),
    )
    client = ScriptedLLMClient(
        [
            ToolCallAction(
                type="tool_call",
                tool_name="lookup",
                arguments={"query": "alpha"},
            ),
            FinalAnswerAction(
                type="final_answer",
                content="Found alpha.",
            ),
        ]
    )

    with Agent(
        config=AgentConfig(project_root=project_root),
        llm=client,
        tools=[tool],
        skills=[],
    ) as agent:
        assert agent.run("Look up alpha.") == "Found alpha."
        trace_path = agent.trace_path

    trace = Trace.from_jsonl(trace_path)
    fixture = ReplayFixture.from_trace(
        trace,
        acknowledge_sensitive_data=True,
    )

    assert [
        record.to_action() for record in fixture.model_actions
    ] == [
        ToolCallAction(
            type="tool_call",
            tool_name="lookup",
            arguments={"query": "alpha"},
        ),
        FinalAnswerAction(
            type="final_answer",
            content="Found alpha.",
        ),
    ]
    assert [record.request_index for record in fixture.model_actions] == [1, 2]
    assert len(fixture.tool_results) == 1
    assert fixture.tool_results[0].tool_name == "lookup"
    assert fixture.tool_results[0].exit_code == 0
    assert "Found alpha." in fixture.tool_results[0].observation
    assert fixture.expected.result["content"] == "Found alpha."


def test_trace_export_writes_private_round_trippable_fixture(
    tmp_path: Path,
) -> None:
    trace = _write_trace(
        tmp_path / "source.jsonl",
        [
            _event(
                "parsed_action",
                {"type": "final_answer", "request_index": 1},
            ),
            _event("final_answer", {"content": "legacy-compatible answer"}),
        ],
    )
    output = tmp_path / "fixture.json"

    result = export_replay_fixture(
        trace,
        output,
        acknowledge_sensitive_data=True,
    )

    assert result == output
    assert load_replay_fixture(output).to_dict() == json.loads(
        output.read_text(encoding="utf-8")
    )
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(ReplayFixtureError, match="already exists"):
        export_replay_fixture(
            trace,
            output,
            acknowledge_sensitive_data=True,
        )


@pytest.mark.skipif(os.name != "posix", reason="symlink behavior")
def test_fixture_loader_rejects_symlink_target(tmp_path: Path) -> None:
    target = tmp_path / "fixture.json"
    target.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)

    with pytest.raises(ReplayFixtureError, match="not a regular file"):
        load_replay_fixture(linked)


def test_fixture_loader_rejects_duplicate_fields_and_oversized_input(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_fixture_payload()), encoding="utf-8")

    with pytest.raises(ReplayFixtureError, match="duplicate field"):
        load_replay_fixture(duplicate)
    with pytest.raises(ReplayFixtureError, match="byte parse limit"):
        load_replay_fixture(valid, max_bytes=10)


def test_final_answer_trace_retains_replayable_content() -> None:
    payload = format_action_trace(
        FinalAnswerAction(type="final_answer", content="stable answer")
    )

    assert payload == {
        "type": "final_answer",
        "content": "stable answer",
    }
