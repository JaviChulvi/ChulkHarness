"""Versioned internal trace schema and replay tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from chulk.main import main
from chulk.tracing import JSONLTraceLogger, TRACE_SCHEMA_VERSION, Trace, TraceFormatError


def _write_legacy_trace(path: Path) -> str:
    lines = [
        {
            "type": "turn_started",
            "created_at": "2026-01-01T00:00:00+00:00",
            "payload": {"turn": {"turn_id": "turn-legacy", "status": "running"}},
        },
        {
            "type": "user_message",
            "created_at": "2026-01-01T00:00:01+00:00",
            "payload": {"content": "inspect the project"},
        },
        {
            "type": "model_request_started",
            "created_at": "2026-01-01T00:00:02+00:00",
            "payload": {"request_index": 1},
        },
        {
            "type": "tool_call_started",
            "created_at": "2026-01-01T00:00:03+00:00",
            "payload": {"tool_name": "read_file", "iteration": 1},
        },
        {
            "type": "tool_call_completed",
            "created_at": "2026-01-01T00:00:04+00:00",
            "payload": {"tool_name": "read_file", "iteration": 1, "success": True},
        },
        {
            "type": "final_answer",
            "created_at": "2026-01-01T00:00:05+00:00",
            "payload": {"content": "inspection complete"},
        },
        {
            "type": "turn_finished",
            "created_at": "2026-01-01T00:00:06+00:00",
            "payload": {"turn": {"turn_id": "turn-legacy", "status": "completed"}},
        },
    ]
    text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    path.write_text(text, encoding="utf-8")
    return text


def test_logger_writes_v1_lifecycle_turn_ids_timing_and_redacted_payloads(tmp_path):
    secret = "sk-trace-secret-123456789"
    logger = JSONLTraceLogger(tmp_path, "conversation-1")

    logger.log(
        "turn_started",
        {"turn": {"turn_id": "turn-1"}, "api_key": secret},
    )
    logger.log("model_request_started", {"message": f"Authorization: Bearer {secret}"})
    logger.log("turn_finished", {"turn": {"turn_id": "turn-1", "status": "completed"}})
    logger.close()
    logger.close()

    trace_text = logger.path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace_text.splitlines()]

    assert [event["type"] for event in events] == [
        "session_started",
        "turn_started",
        "model_request_started",
        "turn_finished",
        "session_finished",
    ]
    assert all(event["schema_version"] == TRACE_SCHEMA_VERSION for event in events)
    assert all(event["conversation_id"] == "conversation-1" for event in events)
    assert all(
        {"schema_version", "conversation_id", "timestamp", "type", "payload"} <= event.keys()
        for event in events
    )
    assert all("created_at" not in event for event in events)
    assert all(event["turn_id"] == "turn-1" for event in events[1:4])
    assert events[3]["payload"]["timing"]["duration_ms"] >= 0
    assert events[-1]["payload"]["duration_ms"] >= 0
    assert events[-1]["payload"]["event_count"] == len(events)
    assert secret not in trace_text
    assert "[redacted]" in trace_text


def test_deferred_logger_close_preserves_lazy_no_trace_behavior(tmp_path):
    logger = JSONLTraceLogger(tmp_path, "unused", defer_until_event="turn_started")

    logger.close()

    assert logger.path.exists() is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode assertions")
def test_logger_repairs_private_trace_and_artifact_modes(tmp_path):
    tmp_path.chmod(0o755)
    trace_path = tmp_path / "conversation-1.jsonl"
    trace_path.write_text("", encoding="utf-8")
    trace_path.chmod(0o644)
    logger = JSONLTraceLogger(tmp_path, "conversation-1")

    artifact = logger.write_artifact("full output", "sensitive")
    logger.close()
    artifact_path = logger.artifacts_dir / f"{artifact['artifact_id']}.txt"

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(trace_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(logger.artifacts_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(logger.artifact_store.manifest_path.stat().st_mode) == 0o600
    assert "path" not in artifact


@pytest.mark.skipif(os.name != "posix", reason="symlink behavior")
def test_logger_rejects_symlink_trace_and_artifact_targets(tmp_path):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("preserved", encoding="utf-8")
    (traces_dir / "linked.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="not a regular file"):
        JSONLTraceLogger(traces_dir, "linked")
    assert outside.read_text(encoding="utf-8") == "preserved"

    logger = JSONLTraceLogger(traces_dir, "artifact-owner")
    outside_directory = tmp_path / "outside-artifacts"
    outside_directory.mkdir()
    logger.artifacts_dir.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="not a regular directory"):
        logger.write_artifact("output", "sensitive")


def test_logger_rejects_non_regular_trace_target(tmp_path):
    (tmp_path / "not-a-file.jsonl").mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        JSONLTraceLogger(tmp_path, "not-a-file")


def test_deferred_logger_checks_target_only_when_activated(tmp_path):
    logger = JSONLTraceLogger(
        tmp_path,
        "lazy",
        defer_until_event="turn_started",
    )
    logger.path.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        logger.log("turn_started", {"turn_id": "turn-1"})


def test_reader_normalizes_v0_and_replay_is_deterministic_and_read_only(tmp_path, capsys):
    trace_path = tmp_path / "legacy-conversation.jsonl"
    original = _write_legacy_trace(trace_path)
    before_entries = sorted(tmp_path.iterdir())
    trace = Trace.from_jsonl(trace_path)

    assert {event.schema_version for event in trace.events} == {0}
    assert {event.turn_id for event in trace.events} == {"turn-legacy"}
    assert trace.events[0].timestamp == trace.events[0].created_at
    assert trace.summary()["legacy_event_count"] == len(trace.events)

    first_exit = main(["trace", "replay", str(trace_path), "--json"])
    first = json.loads(capsys.readouterr().out)
    second_exit = main(["trace", "replay", str(trace_path), "--json"])
    second = json.loads(capsys.readouterr().out)

    assert first_exit == second_exit == 0
    assert first == second
    assert first["mode"] == "read_only"
    assert first["executed"] is False
    assert first["schema_versions"] == [0]
    assert first["sessions"][0]["explicit"] is False
    assert first["turns"] == [
        {
            "duration_ms": 6000.0,
            "ended_at": "2026-01-01T00:00:06+00:00",
            "failures": [],
            "final_answer": "inspection complete",
            "model_request_count": 1,
            "started_at": "2026-01-01T00:00:00+00:00",
            "status": "completed",
            "tool_calls": [
                {
                    "error": None,
                    "failure_kind": None,
                    "iteration": 1,
                    "status": "completed",
                    "success": True,
                    "tool_name": "read_file",
                }
            ],
            "turn_id": "turn-legacy",
            "user_message": "inspect the project",
        }
    ]
    assert trace_path.read_text(encoding="utf-8") == original
    assert sorted(tmp_path.iterdir()) == before_entries


def test_reader_accepts_mixed_legacy_and_v1_events_from_an_upgraded_trace(tmp_path):
    trace_path = tmp_path / "conversation-1.jsonl"
    legacy = {
        "type": "turn_started",
        "created_at": "2026-01-01T00:00:00+00:00",
        "payload": {"turn_id": "turn-old"},
    }
    versioned = {
        "schema_version": 1,
        "conversation_id": "conversation-1",
        "turn_id": "turn-old",
        "timestamp": "2026-01-01T00:00:01+00:00",
        "type": "turn_finished",
        "payload": {"turn_id": "turn-old"},
    }
    trace_path.write_text(
        json.dumps(legacy) + "\n" + json.dumps(versioned) + "\n",
        encoding="utf-8",
    )

    trace = Trace.from_jsonl(trace_path)

    assert trace.summary()["schema_versions"] == [0, 1]
    assert [event.conversation_id for event in trace.events] == ["conversation-1"] * 2
    assert [event.turn_id for event in trace.events] == ["turn-old"] * 2


def test_reader_preserves_turnless_session_event_while_turn_is_active(tmp_path):
    trace_path = tmp_path / "conversation-1.jsonl"
    events = [
        {
            "schema_version": 1,
            "conversation_id": "conversation-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "session_started",
            "payload": {},
        },
        {
            "schema_version": 1,
            "conversation_id": "conversation-1",
            "turn_id": "turn-open",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "type": "turn_started",
            "payload": {},
        },
        {
            "schema_version": 1,
            "conversation_id": "conversation-1",
            "timestamp": "2026-01-01T00:00:02+00:00",
            "type": "model_request_started",
            "payload": {},
        },
        {
            "schema_version": 1,
            "conversation_id": "conversation-1",
            "timestamp": "2026-01-01T00:00:03+00:00",
            "type": "session_finished",
            "payload": {"duration_ms": 3000},
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    trace = Trace.from_jsonl(trace_path)

    assert [event.turn_id for event in trace.events] == [None, "turn-open", "turn-open", None]
    assert "turn_id" not in trace.events[-1].to_dict()


@pytest.mark.parametrize(
    "event",
    [
        {
            "schema_version": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "session_started",
            "payload": {},
        },
        {
            "schema_version": 2,
            "conversation_id": "conversation-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "session_started",
            "payload": {},
        },
    ],
)
def test_reader_rejects_incomplete_or_unknown_versioned_envelopes(tmp_path, event):
    trace_path = tmp_path / "invalid.jsonl"
    trace_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(TraceFormatError):
        Trace.from_jsonl(trace_path)


def test_trace_replay_text_makes_non_execution_boundary_explicit(tmp_path, capsys):
    trace_path = tmp_path / "legacy.jsonl"
    _write_legacy_trace(trace_path)

    exit_code = main(["trace", "replay", str(trace_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Chulk trace replay" in output
    assert "read-only; no model, tool, or network execution" in output
    assert "read_file [completed]" in output
