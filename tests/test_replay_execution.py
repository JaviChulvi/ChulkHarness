"""Executable deterministic replay through the real action-loop boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import subprocess
from typing import Any

import pytest

from chulk.capabilities import ToolRetryPolicy
from tests.core_agent import build_core_agent as Agent
from chulk.llm import LLMClient
from chulk.main import main
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolFailureKind, ToolRegistry, ToolResult
from chulk.tracing import JSONLTraceLogger, ReplayFixture, Trace
from chulk.tracing.execution import (
    execute_replay_fixture,
    execute_replay_fixture_async,
)


def _capture_fixture(
    tmp_path: Path,
    client: LLMClient,
    *,
    registry: ToolRegistry | None = None,
    max_reflection_attempts: int = 0,
    planned: bool = False,
) -> ReplayFixture:
    logger = JSONLTraceLogger(tmp_path / "traces", "golden")
    agent = Agent(
        client,
        trace_logger=logger,
        tool_registry=registry,
        max_reflection_attempts=max_reflection_attempts,
    )
    if planned:
        agent.run_planned_turn("Perform the recorded work.")
        agent.approve_plan()
    else:
        agent.run_turn("Perform the recorded work.")
    agent.close()
    return ReplayFixture.from_trace(
        Trace.from_jsonl(logger.path),
        acknowledge_sensitive_data=True,
    )


def _tool_registry(
    result_factory,
    *,
    retry_policy: ToolRetryPolicy | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Return deterministic recorded evidence.",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            callable=result_factory,
            retry_policy=retry_policy,
            idempotent=retry_policy is not None,
        )
    )
    return registry


def _tool_action() -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_name": "lookup",
        "arguments": {"query": "alpha"},
    }


def _plan_action(*, retry_limit: int = 0) -> dict[str, Any]:
    return {
        "type": "plan",
        "plan": {
            "summary": "Perform and verify the change.",
            "steps": [
                {
                    "id": "implementation",
                    "title": "Implement",
                    "description": "Perform the requested work.",
                    "acceptance_criteria": ["Recorded evidence is available."],
                    "retry_limit": retry_limit,
                }
            ],
        },
    }


def _complete_step() -> dict[str, Any]:
    return {
        "type": "plan_step_update",
        "step_update": {
            "step_id": "implementation",
            "status": "completed",
            "evidence": "The recorded evidence is available.",
        },
    }


@pytest.mark.asyncio
async def test_direct_answer_fixture_has_sync_async_parity(tmp_path: Path) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [{"type": "final_answer", "content": "Recorded answer."}]
        ),
    )

    sync_report = execute_replay_fixture(fixture)
    async_report = await execute_replay_fixture_async(fixture)

    assert sync_report.ok is True
    assert async_report.ok is True
    assert async_report.actual == sync_report.actual
    assert async_report.expected == sync_report.expected
    assert sync_report.model_actions_consumed == 1
    assert sync_report.tool_results_consumed == 0


def test_tool_fixture_uses_recorded_observation_without_real_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [
                _tool_action(),
                {"type": "final_answer", "content": "Found alpha."},
            ]
        ),
        registry=_tool_registry(
            lambda arguments: ToolResult(
                tool_name="lookup",
                success=True,
                observation=f"Found {arguments['query']}.",
                exit_code=0,
            )
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a real side-effect boundary was invoked")

    monkeypatch.setattr(ToolRegistry, "run", forbidden)
    monkeypatch.setattr(ToolRegistry, "run_async", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    report = execute_replay_fixture(fixture)

    assert report.ok is True
    assert report.tool_results_consumed == 1
    observation = report.actual["result"]["content"]
    assert observation == "Found alpha."


def test_retry_fixture_replays_final_transport_result_and_attempt_evidence(
    tmp_path: Path,
) -> None:
    attempts = 0

    def flaky(_arguments: dict[str, Any]) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(
                tool_name="lookup",
                success=False,
                observation="Temporary failure.",
                error="temporary_failure",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            )
        return ToolResult(
            tool_name="lookup",
            success=True,
            observation="Retry succeeded.",
        )

    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [
                _tool_action(),
                {"type": "final_answer", "content": "Retry completed."},
            ]
        ),
        registry=_tool_registry(
            flaky,
            retry_policy=ToolRetryPolicy(max_attempts=2),
        ),
    )

    report = execute_replay_fixture(fixture)

    assert attempts == 2
    assert (
        fixture.tool_results[0].metadata["attempt_history"][0]["retry_scheduled"]
        is True
    )
    assert report.ok is True
    assert report.tool_results_consumed == 1


def test_plan_fixture_replays_approval_steps_and_final_result(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [
                _plan_action(),
                _complete_step(),
                {"type": "final_answer", "content": "Plan complete."},
            ]
        ),
        planned=True,
    )

    report = execute_replay_fixture(fixture)

    assert report.ok is True
    assert [item["type"] for item in report.actual["plans"]] == [
        "plan_created",
        "plan_approved",
        "plan_step_started",
        "plan_step_completed",
    ]
    assert report.actual["result"]["status"] == "completed"


def test_reflection_fixture_replays_revision_without_a_model_provider(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [
                {"type": "final_answer", "content": "First answer."},
                {
                    "approved": False,
                    "reason": "The answer needs evidence.",
                    "feedback": "State the evidence limitation.",
                },
                {
                    "type": "final_answer",
                    "content": "No external evidence was gathered.",
                },
            ]
        ),
        max_reflection_attempts=1,
    )

    report = execute_replay_fixture(fixture)

    assert report.ok is True
    assert report.model_actions_consumed == 2
    assert "reflection_revision_requested" in report.actual["events"]


def test_fatal_safety_fixture_stops_before_another_model_action(
    tmp_path: Path,
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient([_tool_action()]),
        registry=_tool_registry(
            lambda _arguments: ToolResult(
                tool_name="lookup",
                success=False,
                observation="Blocked destructive request.",
                error="destructive_command",
                failure_kind=ToolFailureKind.FATAL_SAFETY,
            )
        ),
    )

    report = execute_replay_fixture(fixture)

    assert report.ok is True
    assert report.actual["result"]["status"] == "failed"
    assert report.model_actions_consumed == 1
    assert report.tool_results_consumed == 1


@pytest.mark.asyncio
async def test_cancelled_fixture_terminalizes_without_running_a_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def wait_forever(_arguments: dict[str, Any]) -> None:
        started.set()
        await asyncio.Event().wait()

    logger = JSONLTraceLogger(tmp_path / "traces", "cancelled")
    agent = Agent(
        ScriptedLLMClient([_tool_action()]),
        trace_logger=logger,
        tool_registry=_tool_registry(wait_forever),
    )
    task = asyncio.create_task(agent.run_turn_async("Perform the recorded work."))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    agent.close()
    fixture = ReplayFixture.from_trace(
        Trace.from_jsonl(logger.path),
        acknowledge_sensitive_data=True,
    )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("the original async tool was invoked")

    monkeypatch.setattr(ToolRegistry, "run_async", forbidden)
    report = await execute_replay_fixture_async(fixture)

    assert report.ok is True
    assert report.actual["result"]["status"] == "cancelled"
    assert report.tool_results_consumed == 0


def test_fallback_evidence_does_not_trigger_a_provider_call(
    tmp_path: Path,
) -> None:
    class FallbackRecordedClient(ScriptedLLMClient):
        @property
        def last_attempts(self) -> list[dict[str, object]]:
            return [
                {
                    "provider": "primary",
                    "status": "failed",
                    "error": "unavailable",
                },
                {
                    "provider": "secondary",
                    "status": "succeeded",
                },
            ]

    fixture = _capture_fixture(
        tmp_path,
        FallbackRecordedClient(
            [{"type": "final_answer", "content": "Fallback answer."}]
        ),
    )

    assert any(
        event["type"] == "llm_fallback_attempts"
        for event in fixture.expected.events
    )
    report = execute_replay_fixture(fixture)

    assert report.ok is True
    assert report.model_actions_consumed == 1
    assert report.actual["result"]["content"] == "Fallback answer."


def test_cli_executes_fixture_while_diagnostic_replay_remains_separate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [{"type": "final_answer", "content": "CLI replay answer."}]
        ),
    )
    fixture_path = tmp_path / "answer.replay.json"
    fixture_path.write_text(fixture.to_json(), encoding="utf-8")

    exit_code = main(
        [
            "trace",
            "replay",
            "--execute-fixture",
            str(fixture_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["mode"] == "executable_fixture"
    assert payload["offline"] is True
    assert payload["executed"] is True


def test_cli_rejects_unbounded_fixture_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [{"type": "final_answer", "content": "Bounded replay."}]
        ),
    )
    fixture_path = tmp_path / "bounded.replay.json"
    fixture_path.write_text(fixture.to_json(), encoding="utf-8")

    exit_code = main(
        [
            "trace",
            "replay",
            "--execute-fixture",
            str(fixture_path),
            "--unbounded",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "trace_error"
    assert "strict byte limit" in payload["error"]


def test_replay_reports_a_tampered_expected_result(tmp_path: Path) -> None:
    fixture = _capture_fixture(
        tmp_path,
        ScriptedLLMClient(
            [{"type": "final_answer", "content": "Recorded answer."}]
        ),
    )
    payload = fixture.to_dict()
    payload["expected"]["result"]["content"] = "Tampered answer."

    report = execute_replay_fixture(ReplayFixture.from_dict(payload))

    assert report.ok is False
    assert report.mismatches == ("result",)
    assert report.actual["result"]["content"] == "Recorded answer."
    assert report.expected["result"]["content"] == "Tampered answer."
