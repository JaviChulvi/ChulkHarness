"""Runner isolation, cleanup, timeout, replay, live, and safety tests."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
import threading
import time

import pytest

import chulk.evals.runner as eval_runner
from chulk import Agent, AgentConfig
from chulk.evals import (
    AsyncEvalRunner,
    EvalCase,
    EvalDataset,
    EvalReference,
    EvalRunner,
    EvalSafetyPolicy,
    EvalSuite,
    EvalTarget,
    EvalTurn,
    EvaluationMode,
    ToolCallGrader,
)
from chulk.results import Cost, RunResult, RunStatus
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolPermissionLevel
from chulk.tracing import Trace, export_replay_fixture


def _result(content: str = "done") -> RunResult:
    return RunResult(content, RunStatus.COMPLETED, None, "conversation", None)


def test_each_trial_gets_a_fresh_workspace_and_agent_reused_across_turns() -> None:
    agents = []
    workspaces: list[Path] = []

    class AgentDouble:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace
            self.calls = 0
            self.closed = False

        def run_result(self, _message, **_kwargs):
            assert self.workspace.exists()
            self.calls += 1
            return _result()

        def close(self):
            self.closed = True

    def factory(context):
        workspaces.append(context.workspace)
        agent = AgentDouble(context.workspace)
        agents.append(agent)
        return agent

    suite = EvalSuite(
        "isolation",
        EvalDataset((EvalCase("case", (EvalTurn("one"), EvalTurn("two"))),)),
        (EvalTarget("target", factory),),
        trials=2,
    )

    report = EvalRunner().run(suite)

    assert not report.operational_errors
    assert len(agents) == 2
    assert all(agent.calls == 2 and agent.closed for agent in agents)
    assert workspaces[0] != workspaces[1]
    assert all(not workspace.exists() for workspace in workspaces)


def test_cleanup_failures_are_operational_and_other_resources_still_close() -> None:
    fixture_closed = False

    class Fixture:
        deps = object()

        def close(self):
            nonlocal fixture_closed
            fixture_closed = True

    class AgentDouble:
        def run_result(self, _message, **_kwargs):
            return _result()

        def close(self):
            raise RuntimeError("agent close failed")

    suite = EvalSuite(
        "cleanup",
        EvalDataset((EvalCase("case", (EvalTurn("run"),), fixture="fixture"),)),
        (EvalTarget("target", lambda _context: AgentDouble()),),
        fixtures={"fixture": lambda _context: Fixture()},
    )

    report = EvalRunner().run(suite)

    assert fixture_closed
    assert report.operational_errors
    assert "agent cleanup failed" in report.operational_errors[0]


def test_sync_timeout_cooperatively_cancels_before_cleanup() -> None:
    release = threading.Event()
    cancelled = threading.Event()
    closed = threading.Event()
    workspaces: list[Path] = []

    class BlockingAgent:
        def run_result(self, _message, **_kwargs):
            release.wait(timeout=2)
            return _result()

        def cancel(self):
            cancelled.set()
            release.set()

        def close(self):
            closed.set()

    def factory(context):
        workspaces.append(context.workspace)
        return BlockingAgent()

    suite = EvalSuite(
        "timeout",
        EvalDataset((EvalCase("case", (EvalTurn("wait"),)),)),
        (EvalTarget("target", factory),),
        timeout_seconds=0.03,
    )

    started = time.monotonic()
    report = EvalRunner().run(suite)
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert report.operational_errors and "turn exceeded" in report.operational_errors[0]
    assert cancelled.is_set()
    assert closed.is_set()
    assert not workspaces[0].exists()


def test_sync_timeout_halts_suite_when_agent_does_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_runner, "_SYNC_CANCELLATION_GRACE_SECONDS", 0.03)
    release = threading.Event()
    workspaces: list[Path] = []
    calls = 0

    class BlockingAgent:
        def run_result(self, _message, **_kwargs):
            nonlocal calls
            calls += 1
            release.wait(timeout=2)
            return _result()

        def close(self):
            return None

    def factory(context):
        workspaces.append(context.workspace)
        return BlockingAgent()

    suite = EvalSuite(
        "unstoppable-timeout",
        EvalDataset(
            (
                EvalCase("case-1", (EvalTurn("wait"),)),
                EvalCase("case-2", (EvalTurn("must-not-run"),)),
            )
        ),
        (EvalTarget("target", factory),),
        timeout_seconds=0.03,
    )

    started = time.monotonic()
    report = EvalRunner().run(suite)
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert calls == 1
    assert len(report.cases) == 1
    assert "timed-out turn did not stop" in report.operational_errors[0]
    assert workspaces[0].exists()

    release.set()
    for _ in range(100):
        if not workspaces[0].exists():
            break
        time.sleep(0.01)
    assert not workspaces[0].exists()


@pytest.mark.asyncio
async def test_async_runner_bounds_concurrency_and_closes_every_agent() -> None:
    active = 0
    peak = 0
    closed = 0
    lock = asyncio.Lock()

    class AsyncAgentDouble:
        async def run_result(self, _message, **_kwargs):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
            return _result()

        async def aclose(self):
            nonlocal closed
            closed += 1

    dataset = EvalDataset(
        tuple(EvalCase(f"case-{index}", (EvalTurn("run"),)) for index in range(4))
    )
    suite = EvalSuite(
        "async-concurrency",
        dataset,
        (EvalTarget("target", lambda _context: AsyncAgentDouble()),),
        concurrency=2,
    )

    report = await AsyncEvalRunner().run(suite)

    assert not report.operational_errors
    assert peak == 2
    assert closed == 4


def test_sync_runner_stops_when_cost_cap_is_exactly_exhausted() -> None:
    calls = 0

    class MeteredAgent:
        def run_result(self, _message, **_kwargs):
            nonlocal calls
            calls += 1
            return RunResult(
                "done",
                RunStatus.COMPLETED,
                None,
                "conversation",
                None,
                cost=Cost(Decimal("1.00"), pricing_known=True),
            )

        def close(self):
            return None

    suite = _exact_cost_cap_suite(lambda _context: MeteredAgent())

    report = EvalRunner().run(suite)

    assert calls == 1
    assert len(report.cases) == 1
    assert report.metrics["total_cost"] == 1.0
    assert report.passed is False
    assert "exhausted before all trials completed" in report.operational_errors[0]
    assert "exceeded" not in report.operational_errors[0]

    complete = EvalRunner().run(
        _exact_cost_cap_suite(lambda _context: MeteredAgent(), case_count=1)
    )
    assert calls == 2
    assert complete.passed is True


@pytest.mark.asyncio
async def test_async_runner_stops_when_cost_cap_is_exactly_exhausted() -> None:
    calls = 0

    class AsyncMeteredAgent:
        async def run_result(self, _message, **_kwargs):
            nonlocal calls
            calls += 1
            return RunResult(
                "done",
                RunStatus.COMPLETED,
                None,
                "conversation",
                None,
                cost=Cost(Decimal("1.00"), pricing_known=True),
            )

        async def aclose(self):
            return None

    suite = _exact_cost_cap_suite(lambda _context: AsyncMeteredAgent())

    report = await AsyncEvalRunner().run(suite)

    assert calls == 1
    assert len(report.cases) == 1
    assert report.metrics["total_cost"] == 1.0
    assert report.passed is False
    assert "exhausted before all trials completed" in report.operational_errors[0]
    assert "exceeded" not in report.operational_errors[0]

    complete = await AsyncEvalRunner().run(
        _exact_cost_cap_suite(lambda _context: AsyncMeteredAgent(), case_count=1)
    )
    assert calls == 2
    assert complete.passed is True


def _exact_cost_cap_suite(factory, *, case_count: int = 3) -> EvalSuite:
    cases = tuple(
        EvalCase(f"case-{index}", (EvalTurn("run"),))
        for index in range(case_count)
    )
    return EvalSuite(
        "exact-cost-cap",
        EvalDataset(cases),
        (EvalTarget("target", factory, provider="fake", model="fake"),),
        mode=EvaluationMode.LIVE,
        max_total_cost=1.0,
    )


@pytest.mark.asyncio
async def test_async_cancellation_closes_agent_and_propagates_cancellation() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    class BlockingAsyncAgent:
        async def run_result(self, _message, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    suite = EvalSuite(
        "cancel",
        EvalDataset((EvalCase("case", (EvalTurn("wait"),)),)),
        (EvalTarget("target", lambda _context: BlockingAsyncAgent()),),
    )
    task = asyncio.create_task(AsyncEvalRunner().run(suite))
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


def test_scripted_exhaustion_is_an_operational_failure() -> None:
    def factory(context):
        return Agent(
            config=AgentConfig(project_root=context.workspace),
            llm=context.llm,
            tools=[],
            skills=[],
        )

    suite = EvalSuite(
        "exhaustion",
        EvalDataset((EvalCase("case", (EvalTurn("answer"),)),)),
        (EvalTarget("target", factory),),
    )

    report = EvalRunner().run(suite)

    assert report.operational_errors
    assert "exhausted" in report.operational_errors[0].lower()


def test_replay_and_live_fake_modes_execute_without_credentials(tmp_path: Path) -> None:
    lookup = Tool(
        name="lookup",
        description="Return deterministic replay evidence.",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        callable=lambda arguments: '{"value":"' + arguments["query"] + '"}',
        permission_level=ToolPermissionLevel.READ,
    )
    captured = Agent(
        config=AgentConfig(project_root=tmp_path / "captured"),
        llm=ScriptedLLMClient(
            [
                {
                    "type": "tool_call",
                    "tool_name": "lookup",
                    "arguments": {"query": "alpha"},
                },
                {"type": "final_answer", "content": "recorded"},
            ]
        ),
        tools=[lookup],
        skills=[],
    )
    captured.run("record this")
    trace_path = captured.trace_path
    captured.close()
    export_replay_fixture(
        Trace.from_jsonl(trace_path),
        tmp_path / "replay.json",
        acknowledge_sensitive_data=True,
    )
    replay_case = EvalCase(
        "replay",
        (EvalTurn("ignored"),),
        EvalReference(
            tool_sequence=("lookup",),
            tool_arguments={"lookup": {"query": "alpha"}},
            tool_results={"lookup": {"value": "alpha"}},
        ),
        replay_fixture="replay.json",
    )
    replay_suite = EvalSuite(
        "replay",
        EvalDataset((replay_case,), source=tmp_path / "cases.jsonl"),
        (EvalTarget("target", lambda _context: object()),),
        (ToolCallGrader(),),
        mode=EvaluationMode.REPLAY,
        required_graders=("tools.calls",),
    )
    replay_report = EvalRunner().run(replay_suite)
    assert not replay_report.operational_errors
    replay_trial = replay_report.cases[0].trials[0]
    replay_result = replay_trial.final_result
    assert replay_result is not None
    assert replay_result.content == "recorded"
    assert replay_result.usage is not None
    assert replay_result.cost is not None
    assert replay_result.tool_calls[0].arguments == {"query": "alpha"}
    assert replay_result.observations[0].content.endswith('{"value":"alpha"}')
    assert replay_trial.grades[0].passed
    assert any(event.to_dict()["payload"] for event in replay_trial.turns[0].events)

    def live_factory(context):
        del context
        return Agent(
            config=AgentConfig(project_root=tmp_path / "live"),
            llm=ScriptedLLMClient([{"type": "final_answer", "content": "live-fake"}]),
            tools=[],
            skills=[],
        )

    live_suite = EvalSuite(
        "live-fake",
        EvalDataset((EvalCase("live", (EvalTurn("run"),)),)),
        (EvalTarget("target", live_factory, provider="fake", model="fake-model"),),
        mode=EvaluationMode.LIVE,
        max_total_cost=1,
        safety=EvalSafetyPolicy(allow_unknown_cost=True),
    )
    live_report = EvalRunner().run(live_suite)
    assert not live_report.operational_errors
    assert live_report.cases[0].trials[0].final_result.content == "live-fake"

    unknown_cost_suite = EvalSuite(
        "live-unknown-cost",
        EvalDataset((EvalCase("live", (EvalTurn("run"),)),)),
        (EvalTarget("target", live_factory, provider="fake", model="fake-model"),),
        mode=EvaluationMode.LIVE,
        max_total_cost=1,
    )
    unknown_cost_report = EvalRunner().run(unknown_cost_suite)
    assert "unknown cost" in unknown_cost_report.operational_errors[0]


def test_denied_tool_never_runs_and_allowlisted_tool_uses_fixture_double() -> None:
    write_calls = 0
    blocked_calls = 0
    fixture_closed = False

    class FakeWriter:
        def write(self):
            nonlocal write_calls
            write_calls += 1
            return "fake write"

        def blocked(self):
            nonlocal blocked_calls
            blocked_calls += 1
            return "should never run"

    class Fixture:
        deps = FakeWriter()

        def close(self):
            nonlocal fixture_closed
            fixture_closed = True

    def write(_arguments, context):
        return context.require_deps().write()

    def blocked(_arguments, context):
        return context.require_deps().blocked()

    tool = Tool(
        name="write_record",
        description="Write through an injected dependency.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        callable=write,
        accepts_context=True,
        permission_level=ToolPermissionLevel.WRITE,
    )
    blocked_tool = Tool(
        name="blocked_write",
        description="A production-like write that must remain denied.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        callable=blocked,
        accepts_context=True,
        permission_level=ToolPermissionLevel.WRITE,
    )

    def factory(context):
        return Agent(
            config=AgentConfig(project_root=context.workspace),
            llm=context.llm,
            tools=[tool, blocked_tool],
            skills=[],
        )

    case = EvalCase(
        "write",
        (
            EvalTurn(
                "write",
                (
                    {"type": "tool_call", "tool_name": "write_record", "arguments": {}},
                    {"type": "final_answer", "content": "done"},
                ),
            ),
        ),
        fixture="writer",
    )
    base = dict(
        name="safety",
        dataset=EvalDataset((case,)),
        targets=(EvalTarget("target", factory),),
        fixtures={"writer": lambda _context: Fixture()},
    )

    denied = EvalRunner().run(EvalSuite(**base))
    assert not denied.operational_errors
    assert write_calls == 0

    allowed = EvalRunner().run(
        EvalSuite(
            **base,
            safety=EvalSafetyPolicy(allowed_tool_names=("write_record",)),
        )
    )
    assert not allowed.operational_errors
    assert write_calls == 1
    assert fixture_closed

    blocked_case = EvalCase(
        "blocked",
        (
            EvalTurn(
                "blocked write",
                (
                    {"type": "tool_call", "tool_name": "blocked_write", "arguments": {}},
                    {"type": "final_answer", "content": "done"},
                ),
            ),
        ),
        fixture="writer",
    )
    blocked_report = EvalRunner().run(
        EvalSuite(
            "exact-safety",
            EvalDataset((blocked_case,)),
            (EvalTarget("target", factory),),
            fixtures={"writer": lambda _context: Fixture()},
            safety=EvalSafetyPolicy(allowed_tool_names=("write_record",)),
        )
    )
    assert not blocked_report.operational_errors
    assert blocked_calls == 0
