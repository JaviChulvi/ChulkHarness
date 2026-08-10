"""Evaluation execution through the public Chulk SDK boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, TypeAlias, cast
from uuid import uuid4

from chulk.core import Agent as CoreAgent
from chulk.events import AgentEvent, SerializedEventPayload
from chulk.hosting import ExecutionScope
from chulk.results import RunResult, RunStatus
from chulk.testing import ScriptedLLMClient, ScriptedResponse
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
    ToolPermissionPolicy,
)

from .models import (
    CaseResult,
    EvalCase,
    EvalContext,
    EvalReport,
    EvalSuite,
    EvalTarget,
    EvalTurnResult,
    EvaluationMode,
    GradeResult,
    TrialResult,
)


# Compatibility contracts retained from the original deterministic harness.
EvalAgentFactory: TypeAlias = Callable[[ScriptedLLMClient], object]


@dataclass(frozen=True)
class EvalExpectations:
    answer: str | None = None
    status: str | None = "completed"
    tool_sequence: tuple[str, ...] | None = None
    trace_event_sequence: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tool_sequence is not None:
            object.__setattr__(self, "tool_sequence", tuple(self.tool_sequence))
        if self.trace_event_sequence is not None:
            object.__setattr__(self, "trace_event_sequence", tuple(self.trace_event_sequence))


@dataclass(frozen=True)
class EvalScenario:
    name: str
    user_message: str
    scripted_responses: tuple[ScriptedResponse, ...]
    expectations: EvalExpectations

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("EvalScenario.name cannot be empty")
        if not self.user_message.strip():
            raise ValueError("EvalScenario.user_message cannot be empty")
        object.__setattr__(self, "scripted_responses", tuple(self.scripted_responses))


@dataclass(frozen=True)
class EvalResult:
    scenario_name: str
    passed: bool
    answer: str | None
    status: str | None
    tool_sequence: tuple[str, ...]
    trace_event_sequence: tuple[str, ...]
    responses_remaining: int
    failures: tuple[str, ...] = ()
    exception: str | None = None

    def assert_passed(self) -> None:
        if not self.passed:
            details = "\n".join(f"- {failure}" for failure in self.failures)
            raise AssertionError(f"Eval scenario {self.scenario_name!r} failed:\n{details}")


class EvalRunner:
    """Run complete suites or legacy deterministic scenarios."""

    def __init__(self, agent_factory: EvalAgentFactory | None = None) -> None:
        self.agent_factory = agent_factory

    def run(
        self,
        evaluation: EvalSuite | EvalScenario,
        *,
        agent: object | None = None,
        tags: tuple[str, ...] = (),
    ) -> EvalReport | EvalResult:
        if isinstance(evaluation, EvalScenario):
            return self._run_legacy(evaluation, agent=agent)
        if agent is not None or self.agent_factory is not None:
            raise ValueError("suite execution uses EvalTarget factories, not EvalRunner.agent_factory")
        return self._run_suite(evaluation, tags=tags)

    def run_many(self, scenarios: Iterable[EvalScenario]) -> tuple[EvalResult, ...]:
        return tuple(self._run_legacy(scenario) for scenario in scenarios)

    def _run_suite(self, suite: EvalSuite, *, tags: tuple[str, ...]) -> EvalReport:
        started = datetime.now(timezone.utc)
        run_id = uuid4().hex
        case_results: list[CaseResult] = []
        operational_errors: list[str] = []
        dataset = suite.dataset.filtered(tags=tags)
        total_cost = 0.0
        unknown_cost = False
        cost_exhausted = False

        if suite.concurrency > 1 and suite.mode is not EvaluationMode.LIVE and not suite.fail_fast:
            combinations = [
                (target, case, trial_number)
                for target in suite.targets
                for case in dataset.cases
                for trial_number in range(1, suite.trials + 1)
            ]
            with ThreadPoolExecutor(max_workers=suite.concurrency, thread_name_prefix="chulk-eval") as executor:
                futures = [
                    executor.submit(_execute_trial_sync, suite, target, case, trial_number)
                    for target, case, trial_number in combinations
                ]
                completed = [future.result() for future in futures]
            grouped: dict[tuple[str, str], list[TrialResult]] = {}
            for trial in completed:
                grouped.setdefault((trial.target_name, trial.case_id), []).append(trial)
                operational_errors.extend(_required_grader_errors(suite, trial))
                cost, known = _trial_cost(trial)
                total_cost += cost
                unknown_cost = unknown_cost or not known
                if trial.exception:
                    operational_errors.append(
                        f"{trial.target_name}/{trial.case_id}/trial-{trial.trial}: {trial.exception}"
                    )
            for target in suite.targets:
                for case in dataset.cases:
                    trials = sorted(grouped.get((target.name, case.id), ()), key=lambda item: item.trial)
                    case_results.append(
                        CaseResult(
                            case.id,
                            target.name,
                            tuple(trials),
                            bool(trials) and all(_trial_required_passed(suite, trial) for trial in trials),
                        )
                    )
            if suite.max_total_cost is not None and total_cost > suite.max_total_cost:
                operational_errors.append(
                    f"evaluation cost ${total_cost:.6f} exceeded cap ${suite.max_total_cost:.6f}"
                )
        else:
            for target in suite.targets:
                for case in dataset.cases:
                    trials = []
                    for trial_number in range(1, suite.trials + 1):
                        trial = _execute_trial_sync(suite, target, case, trial_number)
                        trials.append(trial)
                        operational_errors.extend(_required_grader_errors(suite, trial))
                        cost, known = _trial_cost(trial)
                        total_cost += cost
                        unknown_cost = unknown_cost or not known
                        if trial.exception:
                            operational_errors.append(
                                f"{target.name}/{case.id}/trial-{trial_number}: {trial.exception}"
                            )
                        if suite.max_total_cost is not None and total_cost > suite.max_total_cost:
                            operational_errors.append(
                                f"evaluation cost ${total_cost:.6f} exceeded cap ${suite.max_total_cost:.6f}"
                            )
                            cost_exhausted = True
                            break
                        if suite.fail_fast and (trial.exception or not _trial_required_passed(suite, trial)):
                            break
                    case_results.append(
                        CaseResult(
                            case.id,
                            target.name,
                            tuple(trials),
                            bool(trials) and all(_trial_required_passed(suite, trial) for trial in trials),
                        )
                    )
                    if operational_errors and suite.fail_fast:
                        break
                    if cost_exhausted:
                        break
                if (operational_errors and suite.fail_fast) or cost_exhausted:
                    break

        if suite.mode is EvaluationMode.LIVE and unknown_cost and not suite.safety.allow_unknown_cost:
            operational_errors.append("live evaluation produced unknown cost; opt in with allow_unknown_cost")
        report = _build_report(run_id, suite, dataset.digest, started, case_results, operational_errors)
        if suite.store is not None:
            try:
                cast(Any, suite.store).save_report(report)
            except Exception as exc:
                operational_errors.append(f"store failed: {_format_exception(exc)}")
                report = _build_report(run_id, suite, dataset.digest, started, case_results, operational_errors)
        return report

    def _run_legacy(self, scenario: EvalScenario, *, agent: object | None = None) -> EvalResult:
        if agent is not None and self.agent_factory is not None:
            raise ValueError("Pass an agent or configure an agent_factory, not both")
        client = ScriptedLLMClient(scenario.scripted_responses)
        instance = agent
        runtime: Any | None = None
        owned = agent is None
        original_client = None
        original_callback = None
        turn_count = 0
        answer = None
        execution_exception = None
        cleanup_exception = None
        events: list[str] = []
        try:
            if instance is None:
                instance = (self.agent_factory or CoreAgent)(client)
            runtime = _runtime_from_agent(instance)
            turn_count = len(runtime.state.turns)
            original_client = runtime.llm_client
            original_callback = runtime.event_callback
            runtime.llm_client = client

            def capture(event_type: str, payload: dict[str, Any]) -> None:
                events.append(event_type)
                if callable(original_callback):
                    original_callback(event_type, payload)

            runtime.event_callback = capture
            answer = _run_legacy_agent(instance, scenario.user_message)
        except Exception as exc:
            execution_exception = exc
        finally:
            if runtime is not None:
                runtime.event_callback = original_callback
                runtime.llm_client = original_client
            if owned and instance is not None:
                close = getattr(instance, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        cleanup_exception = exc
        turn = _newest_turn(runtime, turn_count)
        status = getattr(turn, "status", None)
        tools = _legacy_tool_sequence(turn)
        failures = _legacy_evaluate(scenario.expectations, answer, status, tools, tuple(events))
        if execution_exception:
            failures.insert(0, f"execution raised {_format_exception(execution_exception)}")
        if cleanup_exception:
            failures.append(f"agent cleanup raised {_format_exception(cleanup_exception)}")
        return EvalResult(
            scenario.name, not failures, answer, status, tools, tuple(events), client.remaining,
            tuple(failures), _format_exception(execution_exception) if execution_exception else None,
        )


class AsyncEvalRunner:
    """Run suites natively against AsyncAgent factories."""

    async def run(self, suite: EvalSuite, *, tags: tuple[str, ...] = ()) -> EvalReport:
        started = datetime.now(timezone.utc)
        run_id = uuid4().hex
        dataset = suite.dataset.filtered(tags=tags)
        semaphore = asyncio.Semaphore(suite.concurrency)

        async def execute(target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
            async with semaphore:
                trial = await _run_trial_async(suite, target, case, trial_number)
                return await _grade_async(suite, case, trial)

        combinations = [
            (target, case, trial_number)
            for target in suite.targets
            for case in dataset.cases
            for trial_number in range(1, suite.trials + 1)
        ]
        if suite.mode is EvaluationMode.LIVE or suite.fail_fast:
            trials = []
            spent = 0.0
            for target, case, trial_number in combinations:
                trial = await execute(target, case, trial_number)
                trials.append(trial)
                spent += _trial_cost(trial)[0]
                if suite.max_total_cost is not None and spent > suite.max_total_cost:
                    break
                if suite.fail_fast and (trial.exception or not _trial_required_passed(suite, trial)):
                    break
        else:
            tasks = [
                asyncio.create_task(execute(target, case, trial_number))
                for target, case, trial_number in combinations
            ]
            trials = list(await asyncio.gather(*tasks))
        grouped: dict[tuple[str, str], list[TrialResult]] = {}
        operational_errors: list[str] = []
        for trial in trials:
            grouped.setdefault((trial.target_name, trial.case_id), []).append(trial)
            if trial.exception:
                operational_errors.append(
                    f"{trial.target_name}/{trial.case_id}/trial-{trial.trial}: {trial.exception}"
                )
            operational_errors.extend(_required_grader_errors(suite, trial))
        case_results = [
            CaseResult(case_id, target_name, tuple(sorted(items, key=lambda item: item.trial)), all(_trial_required_passed(suite, item) for item in items))
            for (target_name, case_id), items in grouped.items()
        ]
        total_cost = sum(_trial_cost(trial)[0] for trial in trials)
        known = all(_trial_cost(trial)[1] for trial in trials)
        if suite.max_total_cost is not None and total_cost > suite.max_total_cost:
            operational_errors.append(
                f"evaluation cost ${total_cost:.6f} exceeded cap ${suite.max_total_cost:.6f}"
            )
        if suite.mode is EvaluationMode.LIVE and not known and not suite.safety.allow_unknown_cost:
            operational_errors.append("live evaluation produced unknown cost; opt in with allow_unknown_cost")
        report = _build_report(run_id, suite, dataset.digest, started, case_results, operational_errors)
        if suite.store is not None:
            save_async = getattr(suite.store, "save_report_async", None)
            try:
                if callable(save_async):
                    await save_async(report)
                else:
                    await asyncio.to_thread(cast(Any, suite.store).save_report, report)
            except Exception as exc:
                operational_errors.append(f"store failed: {_format_exception(exc)}")
                report = _build_report(run_id, suite, dataset.digest, started, case_results, operational_errors)
        return report


def run_eval(
    scenario: EvalScenario,
    *,
    agent: object | None = None,
    agent_factory: EvalAgentFactory | None = None,
) -> EvalResult:
    return EvalRunner(agent_factory).run(scenario, agent=agent)  # type: ignore[return-value]


def _execute_trial_sync(
    suite: EvalSuite,
    target: EvalTarget,
    case: EvalCase,
    trial_number: int,
) -> TrialResult:
    return _grade_sync(
        suite,
        case,
        _run_trial_sync(suite, target, case, trial_number),
    )


def _run_trial_sync(suite: EvalSuite, target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
    if suite.mode is EvaluationMode.REPLAY:
        return _run_replay_sync(suite, target, case, trial_number)
    started = time.monotonic()
    turns: list[EvalTurnResult] = []
    exception: str | None = None
    temporary = tempfile.TemporaryDirectory(prefix="chulk-eval-")
    workspace = Path(temporary.name).resolve()
    deferred_cleanup = False
    try:
        client = _scripted_client(case) if suite.mode is EvaluationMode.SCRIPTED else None
        scope = ExecutionScope.local(
            profile_id=f"eval-{target.name}",
            run_id=f"eval-{uuid4().hex}",
        )
        context = EvalContext(
            suite.name, target.name, case.id, trial_number, workspace, suite.mode, scope,
            llm=client, provider=target.provider, model=target.model, safety=suite.safety,
        )
        fixture = None
        agent = None
        try:
            fixture, context = _open_fixture_sync(suite, case, context)
            agent = target.agent_factory(context)
            if inspect.isawaitable(agent):
                raise TypeError("async agent factory requires AsyncEvalRunner")
            _validate_agent_safety(agent, suite)
            for index, turn in enumerate(case.turns):
                events: list[Any] = []
                turn_started = time.monotonic()
                result = _run_sync_with_timeout(
                    lambda: cast(Any, agent).run_result(
                        turn.input,
                        on_event=events.append,
                        deps=context.deps,
                    ),
                    suite.timeout_seconds,
                )
                if not isinstance(result, RunResult):
                    raise TypeError("eval agent run_result() must return RunResult")
                turns.append(EvalTurnResult(index, result, tuple(events), time.monotonic() - turn_started))
        except _SyncTurnTimeout as exc:
            deferred_cleanup = True
            exc.defer(
                lambda: _cleanup_sync_resources(agent, fixture, temporary)
            )
            exception = _format_exception(
                TimeoutError(f"turn exceeded {suite.timeout_seconds:g}s timeout")
            )
        except Exception as exc:
            exception = _format_exception(exc)
        finally:
            if not deferred_cleanup:
                cleanup_errors = _cleanup_sync_resources(agent, fixture, temporary)
                exception = _merge_exceptions(exception, cleanup_errors)
        return TrialResult(case.id, target.name, trial_number, tuple(turns), time.monotonic() - started, exception=exception, workspace=workspace)
    except Exception:
        if not deferred_cleanup:
            temporary.cleanup()
        raise


async def _run_trial_async(suite: EvalSuite, target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
    if suite.mode is EvaluationMode.REPLAY:
        return await _run_replay_async(suite, target, case, trial_number)
    started = time.monotonic()
    turns: list[EvalTurnResult] = []
    exception: str | None = None
    with tempfile.TemporaryDirectory(prefix="chulk-eval-") as temporary:
        workspace = Path(temporary).resolve()
        client = _scripted_client(case) if suite.mode is EvaluationMode.SCRIPTED else None
        scope = ExecutionScope.local(profile_id=f"eval-{target.name}", run_id=f"eval-{uuid4().hex}")
        context = EvalContext(suite.name, target.name, case.id, trial_number, workspace, suite.mode, scope, llm=client, provider=target.provider, model=target.model, safety=suite.safety)
        fixture = None
        agent = None
        try:
            fixture, context = await _open_fixture_async(suite, case, context)
            agent = target.agent_factory(context)
            if inspect.isawaitable(agent):
                agent = await agent
            _validate_agent_safety(agent, suite)
            for index, turn in enumerate(case.turns):
                events: list[Any] = []
                turn_started = time.monotonic()
                result = await asyncio.wait_for(
                    cast(Any, agent).run_result(turn.input, on_event=events.append, deps=context.deps),
                    timeout=suite.timeout_seconds,
                )
                if not isinstance(result, RunResult):
                    raise TypeError("eval agent run_result() must return RunResult")
                turns.append(EvalTurnResult(index, result, tuple(events), time.monotonic() - turn_started))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exception = _format_exception(exc)
        finally:
            cleanup_errors = await _cleanup_async_resources(agent, fixture)
            exception = _merge_exceptions(exception, cleanup_errors)
        return TrialResult(case.id, target.name, trial_number, tuple(turns), time.monotonic() - started, exception=exception, workspace=workspace)


def _scripted_client(case: EvalCase) -> ScriptedLLMClient:
    return ScriptedLLMClient(response for turn in case.turns for response in turn.scripted_responses)


def _run_replay_sync(suite: EvalSuite, target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
    from chulk.tracing.execution import execute_replay_fixture
    from chulk.tracing.fixtures import load_replay_fixture

    started = time.monotonic()
    try:
        fixture_path = _replay_path(suite, case)
        replay = execute_replay_fixture(load_replay_fixture(fixture_path))
        turn = _replay_turn_result(case, replay, time.monotonic() - started)
        return TrialResult(case.id, target.name, trial_number, (turn,), time.monotonic() - started)
    except Exception as exc:
        return TrialResult(case.id, target.name, trial_number, (), time.monotonic() - started, exception=_format_exception(exc))


async def _run_replay_async(suite: EvalSuite, target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
    from chulk.tracing.execution import execute_replay_fixture_async
    from chulk.tracing.fixtures import load_replay_fixture

    started = time.monotonic()
    try:
        fixture_path = _replay_path(suite, case)
        replay = await asyncio.wait_for(
            execute_replay_fixture_async(load_replay_fixture(fixture_path)),
            timeout=suite.timeout_seconds,
        )
        turn = _replay_turn_result(case, replay, time.monotonic() - started)
        return TrialResult(case.id, target.name, trial_number, (turn,), time.monotonic() - started)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return TrialResult(case.id, target.name, trial_number, (), time.monotonic() - started, exception=_format_exception(exc))


def _replay_path(suite: EvalSuite, case: EvalCase) -> Path:
    if case.replay_fixture is None:
        raise ValueError("replay eval cases require replay_fixture")
    root = suite.dataset.source.parent if suite.dataset.source is not None else Path.cwd().resolve()
    path = (root / case.replay_fixture).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("replay fixture resolves outside the dataset directory") from exc
    return path


def _replay_turn_result(case: EvalCase, replay: Any, duration: float) -> EvalTurnResult:
    actual = replay.actual
    result_payload = actual.get("result", {})
    state = actual.get("state", {})
    content = result_payload.get("content")
    status = result_payload.get("status") or "unknown"
    errors = list(result_payload.get("errors") or [])
    if not replay.ok:
        errors.append("replay mismatch: " + ", ".join(replay.mismatches))
    conversation_id = str(state.get("conversation_id") or f"replay-{case.id}")
    result = RunResult(
        content=str(content or ""),
        status=RunStatus(status) if status in {item.value for item in RunStatus} else RunStatus.UNKNOWN,
        turn_id=None,
        conversation_id=conversation_id,
        trace_path=None,
        loaded_skill_names=tuple(state.get("loaded_skill_names") or ()),
        loaded_memory_ids=tuple(state.get("loaded_memory_ids") or ()),
        errors=tuple(str(error) for error in errors),
        extension_metadata={"replay": replay.to_dict()},
    )
    events = tuple(
        AgentEvent(
            name=str(event),
            conversation_id=conversation_id,
            payload=SerializedEventPayload(data={"replay": True}),
        )
        for event in actual.get("events", ())
    )
    return EvalTurnResult(0, result, events, duration)


def _open_fixture_sync(suite: EvalSuite, case: EvalCase, context: EvalContext) -> tuple[object | None, EvalContext]:
    if case.fixture is None:
        return None, context
    factory = suite.fixtures.get(case.fixture)
    if factory is None:
        raise ValueError(f"unknown eval fixture {case.fixture!r}")
    fixture = factory(context)
    if inspect.isawaitable(fixture):
        raise TypeError("async fixture requires AsyncEvalRunner")
    deps = getattr(fixture, "deps", fixture)
    return fixture, _replace_context_deps(context, deps)


async def _open_fixture_async(suite: EvalSuite, case: EvalCase, context: EvalContext) -> tuple[object | None, EvalContext]:
    if case.fixture is None:
        return None, context
    factory = suite.fixtures.get(case.fixture)
    if factory is None:
        raise ValueError(f"unknown eval fixture {case.fixture!r}")
    fixture = factory(context)
    if inspect.isawaitable(fixture):
        fixture = await fixture
    return fixture, _replace_context_deps(context, getattr(fixture, "deps", fixture))


def _replace_context_deps(context: EvalContext, deps: object) -> EvalContext:
    return EvalContext(
        context.suite_name, context.target_name, context.case_id, context.trial,
        context.workspace, context.mode, context.scope, context.llm, deps,
        context.provider, context.model, context.safety,
    )


def _validate_agent_safety(agent: object, suite: EvalSuite) -> None:
    registry = getattr(agent, "tool_registry", None)
    if registry is None or not callable(getattr(registry, "list_tools", None)):
        return
    tools = registry.list_tools()
    has_side_effects = any(
        tool.normalized_permission_level() is not ToolPermissionLevel.READ
        for tool in tools
    )
    runtime = getattr(agent, "runtime", agent)
    if not hasattr(runtime, "permission_policy"):
        if has_side_effects:
            raise TypeError("eval agent cannot apply the required tool permission policy")
        return
    runtime.permission_policy = _EvalToolPermissionPolicy(
        suite.safety.allowed_tool_names
    )


class _EvalToolPermissionPolicy(ToolPermissionPolicy):
    """Allow READ tools and only the explicitly named side-effecting tools."""

    def __init__(self, allowed_tool_names: tuple[str, ...]) -> None:
        super().__init__(
            name="evaluation",
            default_decision=PermissionDecision.DENY,
            confirmation_decision=PermissionDecision.ALLOW,
            level_decisions={
                level: PermissionDecision.ALLOW
                for level in ToolPermissionLevel
            },
        )
        self.allowed_tool_names = frozenset(allowed_tool_names)

    def decide(self, request: PermissionRequest) -> PermissionDecisionRecord:
        record = super().decide(request)
        if (
            request.permission_level is ToolPermissionLevel.READ
            or request.tool_name in self.allowed_tool_names
        ):
            return record
        return replace(
            record,
            decision=PermissionDecision.DENY,
            reason="tool is not allowlisted by the evaluation suite",
        )


def _grade_sync(suite: EvalSuite, case: EvalCase, trial: TrialResult) -> TrialResult:
    grades: list[GradeResult] = []
    for grader in suite.graders:
        try:
            grade = cast(Any, grader).grade(case, trial)
            if inspect.isawaitable(grade):
                raise TypeError("async grader requires AsyncEvalRunner")
            if not isinstance(grade, GradeResult):
                raise TypeError("grader must return GradeResult")
        except Exception as exc:
            name = str(getattr(grader, "name", type(grader).__name__))
            grade = GradeResult(name, 0.0, False, f"grader failed: {exc}", error=_format_exception(exc))
        grades.append(_mark_required(grade, suite))
    return TrialResult(trial.case_id, trial.target_name, trial.trial, trial.turns, trial.duration_seconds, tuple(grades), trial.exception, trial.workspace)


async def _grade_async(suite: EvalSuite, case: EvalCase, trial: TrialResult) -> TrialResult:
    grades: list[GradeResult] = []
    for grader in suite.graders:
        try:
            grade_async = getattr(grader, "grade_async", None)
            if callable(grade_async):
                grade = await grade_async(case, trial)
            else:
                grade = await asyncio.to_thread(cast(Any, grader).grade, case, trial)
            if not isinstance(grade, GradeResult):
                raise TypeError("grader must return GradeResult")
        except Exception as exc:
            name = str(getattr(grader, "name", type(grader).__name__))
            grade = GradeResult(name, 0.0, False, f"grader failed: {exc}", error=_format_exception(exc))
        grades.append(_mark_required(grade, suite))
    return TrialResult(trial.case_id, trial.target_name, trial.trial, trial.turns, trial.duration_seconds, tuple(grades), trial.exception, trial.workspace)


def _trial_required_passed(suite: EvalSuite, trial: TrialResult) -> bool:
    del suite
    return trial.passed


def _mark_required(grade: GradeResult, suite: EvalSuite) -> GradeResult:
    details = dict(grade.details)
    details["required"] = grade.grader in suite.required_graders
    return GradeResult(
        grade.grader,
        grade.score,
        grade.passed,
        grade.reason,
        details,
        grade.error,
    )


def _required_grader_errors(suite: EvalSuite, trial: TrialResult) -> list[str]:
    required = set(suite.required_graders)
    return [
        f"{trial.target_name}/{trial.case_id}/trial-{trial.trial}/{grade.grader}: {grade.error}"
        for grade in trial.grades
        if grade.grader in required and grade.error is not None
    ]


def _build_report(run_id: str, suite: EvalSuite, digest: str, started: datetime, cases: list[CaseResult], errors: list[str]) -> EvalReport:
    trials = [trial for case in cases for trial in case.trials]
    agent_tokens = sum(
        turn.result.usage.total_tokens
        for trial in trials
        for turn in trial.turns
        if turn.result.usage
    )
    judge_tokens = sum(_trial_judge_tokens(trial) for trial in trials)
    judge_cost = sum(_trial_judge_cost(trial) for trial in trials)
    metrics: dict[str, float] = {
        "case_count": float(len(cases)),
        "trial_count": float(len(trials)),
        "pass_rate": sum(case.passed for case in cases) / len(cases) if cases else 1.0,
        "pass_at_k": sum(any(_trial_required_passed(suite, trial) for trial in case.trials) for case in cases) / len(cases) if cases else 1.0,
        "mean_latency_seconds": sum(trial.duration_seconds for trial in trials) / len(trials) if trials else 0.0,
        "agent_tokens": float(agent_tokens),
        "judge_tokens": float(judge_tokens),
        "total_tokens": float(agent_tokens + judge_tokens),
        "judge_cost": judge_cost,
        "total_cost": sum(_trial_cost(trial)[0] for trial in trials),
        "error_rate": sum(trial.exception is not None for trial in trials) / len(trials) if trials else 0.0,
    }
    durations = sorted(trial.duration_seconds for trial in trials)
    metrics["p50_latency_seconds"] = _percentile(durations, 0.50)
    metrics["p95_latency_seconds"] = _percentile(durations, 0.95)
    for target in suite.targets:
        selected = [case for case in cases if case.target_name == target.name]
        metrics[f"target.{target.name}.pass_rate"] = (
            sum(case.passed for case in selected) / len(selected) if selected else 1.0
        )
    for dimension in ("provider", "model"):
        values = sorted(
            {
                str(getattr(target, dimension))
                for target in suite.targets
                if getattr(target, dimension) is not None
            }
        )
        for value in values:
            target_names = {
                target.name for target in suite.targets
                if getattr(target, dimension) == value
            }
            selected = [case for case in cases if case.target_name in target_names]
            metrics[f"{dimension}.{value}.pass_rate"] = (
                sum(case.passed for case in selected) / len(selected) if selected else 1.0
            )
    tags_by_case = {case.id: case.tags for case in suite.dataset.cases}
    for tag in sorted({tag for values in tags_by_case.values() for tag in values}):
        selected = [case for case in cases if tag in tags_by_case.get(case.case_id, ())]
        metrics[f"tag.{tag}.pass_rate"] = (
            sum(case.passed for case in selected) / len(selected) if selected else 1.0
        )
    grader_names = sorted({grade.grader for trial in trials for grade in trial.grades})
    for name in grader_names:
        grades = [grade for trial in trials for grade in trial.grades if grade.grader == name and not grade.details.get("skipped")]
        if grades:
            metrics[f"grader.{name}.score"] = sum(grade.score for grade in grades) / len(grades)
            metrics[f"grader.{name}.pass_rate"] = sum(grade.passed for grade in grades) / len(grades)
    threshold_failures = tuple(
        f"{name}: value {metrics.get(name)!r} did not satisfy {threshold}"
        for name, threshold in suite.thresholds.items()
        if not threshold.accepts(metrics.get(name))
    )
    return EvalReport(
        run_id, suite.name, digest, started.isoformat(), datetime.now(timezone.utc).isoformat(),
        tuple(cases), metrics, threshold_failures, tuple(errors),
        _report_metadata(suite),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int((len(values) - 1) * quantile + 0.5)))
    return values[index]


def _report_metadata(suite: EvalSuite) -> dict[str, Any]:
    from chulk._version import __version__

    targets = [
        {"name": target.name, "provider": target.provider, "model": target.model}
        for target in suite.targets
    ]
    graders = [str(getattr(grader, "name", type(grader).__name__)) for grader in suite.graders]
    fingerprint_payload = {
        "suite": suite.name,
        "dataset": suite.dataset.digest,
        "mode": suite.mode.value,
        "targets": targets,
        "graders": graders,
        "required_graders": list(suite.required_graders),
        "trials": suite.trials,
    }
    return {
        "mode": suite.mode.value,
        "trials": suite.trials,
        "concurrency": suite.concurrency,
        "chulk_version": __version__,
        "git_revision": _git_revision(),
        "targets": targets,
        "graders": graders,
        "suite_fingerprint": sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _trial_cost(trial: TrialResult) -> tuple[float, bool]:
    total = 0.0
    known = True
    for turn in trial.turns:
        cost = turn.result.cost
        if cost is None or cost.amount is None:
            known = False
        else:
            total += float(cost.amount)
    for grade in trial.grades:
        if not grade.details.get("judge"):
            continue
        cost = grade.details.get("cost")
        if not isinstance(cost, Mapping) or cost.get("amount") is None:
            known = False
        else:
            total += float(cost["amount"])
    return total, known


def _trial_judge_tokens(trial: TrialResult) -> int:
    total = 0
    for grade in trial.grades:
        if not grade.details.get("judge"):
            continue
        usage = grade.details.get("usage")
        if isinstance(usage, Mapping) and isinstance(usage.get("total_tokens"), int):
            total += int(usage["total_tokens"])
    return total


def _trial_judge_cost(trial: TrialResult) -> float:
    total = 0.0
    for grade in trial.grades:
        if not grade.details.get("judge"):
            continue
        cost = grade.details.get("cost")
        if isinstance(cost, Mapping) and cost.get("amount") is not None:
            total += float(cost["amount"])
    return total


class _SyncTurnTimeout(TimeoutError):
    def __init__(self, future: Future[Any], executor: ThreadPoolExecutor) -> None:
        super().__init__("evaluation turn timed out")
        self.future = future
        self.executor = executor

    def defer(self, cleanup: Callable[[], object]) -> None:
        def complete(_future: Future[Any]) -> None:
            try:
                cleanup()
            finally:
                self.executor.shutdown(wait=False, cancel_futures=True)

        self.future.add_done_callback(complete)


def _cleanup_sync_resources(
    agent: object | None,
    fixture: object | None,
    temporary: tempfile.TemporaryDirectory[str],
) -> list[str]:
    errors: list[str] = []
    for label, value in (("agent", agent), ("fixture", fixture)):
        if value is None:
            continue
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                errors.append(f"{label} cleanup failed: {_format_exception(exc)}")
    try:
        temporary.cleanup()
    except Exception as exc:
        errors.append(f"workspace cleanup failed: {_format_exception(exc)}")
    return errors


def _run_sync_with_timeout(operation: Callable[[], Any], timeout_seconds: float) -> Any:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chulk-eval-turn")
    future = executor.submit(operation)
    timed_out = False
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        timed_out = True
        raise _SyncTurnTimeout(future, executor) from exc
    finally:
        if not timed_out:
            executor.shutdown(wait=True, cancel_futures=True)


async def _cleanup_async_resources(
    agent: object | None,
    fixture: object | None,
) -> list[str]:
    errors: list[str] = []
    for label, value in (("agent", agent), ("fixture", fixture)):
        if value is None:
            continue
        try:
            aclose = getattr(value, "aclose", None)
            if callable(aclose):
                result = aclose()
                if inspect.isawaitable(result):
                    await result
                continue
            close = getattr(value, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            errors.append(f"{label} cleanup failed: {_format_exception(exc)}")
    return errors


def _merge_exceptions(exception: str | None, cleanup_errors: list[str]) -> str | None:
    if not cleanup_errors:
        return exception
    cleanup = "; ".join(cleanup_errors)
    return f"{exception}; {cleanup}" if exception else cleanup


def _runtime_from_agent(agent: object) -> Any:
    runtime = getattr(agent, "runtime", agent)
    missing = [name for name in ("state", "llm_client", "event_callback") if not hasattr(runtime, name)]
    if missing:
        raise TypeError(f"Eval agent runtime is missing: {', '.join(missing)}")
    return runtime


def _run_legacy_agent(agent: object, message: str) -> str:
    method = getattr(agent, "run_turn", None) or getattr(agent, "run", None)
    if not callable(method):
        raise TypeError("Eval agent must expose run_turn(message) or run(message)")
    answer = method(message)
    if not isinstance(answer, str):
        raise TypeError("Eval agent must return an answer string")
    return answer


def _newest_turn(runtime: Any | None, count: int) -> Any | None:
    return runtime.state.turns[-1] if runtime is not None and len(runtime.state.turns) > count else None


def _legacy_tool_sequence(turn: Any | None) -> tuple[str, ...]:
    return () if turn is None else tuple(record.resolved_tool_name or record.tool_name for record in turn.tool_calls)


def _legacy_evaluate(expected: EvalExpectations, answer: str | None, status: str | None, tools: tuple[str, ...], events: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    if expected.answer is not None and answer != expected.answer:
        failures.append(f"expected answer {expected.answer!r}, got {answer!r}")
    if expected.status is not None and status != expected.status:
        failures.append(f"expected status {expected.status!r}, got {status!r}")
    if expected.tool_sequence is not None and tools != expected.tool_sequence:
        failures.append(f"expected tool sequence {expected.tool_sequence!r}, got {tools!r}")
    if expected.trace_event_sequence is not None and not _ordered(expected.trace_event_sequence, events):
        failures.append(f"expected trace-event subsequence {expected.trace_event_sequence!r}, got {events!r}")
    return failures


def _ordered(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    index = 0
    for item in actual:
        if index < len(expected) and item == expected[index]:
            index += 1
    return index == len(expected)


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "AsyncEvalRunner", "EvalAgentFactory", "EvalExpectations", "EvalResult",
    "EvalRunner", "EvalScenario", "run_eval",
]
