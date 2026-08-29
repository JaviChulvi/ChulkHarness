"""Evaluation execution through the public Chulk SDK boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass, fields, is_dataclass, replace
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import inspect
import json
import marshal
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, TypeAlias, cast
from uuid import uuid4

from chulk._sdk.results import (
    cost_snapshot,
    observation_snapshot,
    plan_snapshot,
    tool_call_snapshot,
    usage_snapshot,
)
from chulk.core import Agent as CoreAgent
from chulk.events import AgentEvent, SerializedEventPayload
from chulk.hosting import ExecutionScope
from chulk.llm.usage import (
    aggregate_cost,
    aggregate_usage,
    cost_from_dict,
    usage_from_dict,
)
from chulk.results import Cost, Observation, RunResult, RunStatus, ToolCall, Usage, plain_data
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
    EvalRunStatus,
    EvalSuite,
    EvalTarget,
    EvalTurnResult,
    EvaluationMode,
    GradeResult,
    TrialResult,
)


# Compatibility contracts retained from the original deterministic harness.
EvalAgentFactory: TypeAlias = Callable[[ScriptedLLMClient], object]
_SYNC_CANCELLATION_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class _SyncTrialExecution:
    trial: TrialResult
    halt: bool = False
    cost_unknown: bool = False


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
        resume_from: str | None = None,
    ) -> EvalReport | EvalResult:
        if isinstance(evaluation, EvalScenario):
            if resume_from is not None:
                raise ValueError("legacy eval scenarios cannot be resumed")
            return self._run_legacy(evaluation, agent=agent)
        if agent is not None or self.agent_factory is not None:
            raise ValueError("suite execution uses EvalTarget factories, not EvalRunner.agent_factory")
        return self._run_suite(evaluation, tags=tags, resume_from=resume_from)

    def run_many(self, scenarios: Iterable[EvalScenario]) -> tuple[EvalResult, ...]:
        return tuple(self._run_legacy(scenario) for scenario in scenarios)

    def _run_suite(
        self,
        suite: EvalSuite,
        *,
        tags: tuple[str, ...],
        resume_from: str | None,
    ) -> EvalReport:
        dataset = suite.dataset.filtered(tags=tags)
        started, run_id, completed, operational_errors = _load_resume_sync(
            suite, dataset.digest, resume_from
        )
        baseline = _load_baseline_sync(suite, operational_errors)
        total_cost = sum(_trial_cost(trial)[0] for trial in completed.values())
        unknown_cost = any(not _trial_cost(trial)[1] for trial in completed.values())

        def checkpoint(status: EvalRunStatus = EvalRunStatus.RUNNING) -> EvalReport:
            report = _build_report(
                run_id,
                suite,
                dataset.digest,
                started,
                _case_results_from_trials(suite, dataset.cases, completed),
                operational_errors,
                status=status,
                baseline=baseline,
            )
            if suite.store is not None:
                try:
                    cast(Any, suite.store).save_report(report)
                except Exception as exc:
                    message = f"store failed: {_format_exception(exc)}"
                    if message not in operational_errors:
                        operational_errors.append(message)
                    report = _build_report(
                        run_id,
                        suite,
                        dataset.digest,
                        started,
                        _case_results_from_trials(suite, dataset.cases, completed),
                        operational_errors,
                        status=status,
                        baseline=baseline,
                    )
            return report

        checkpoint()
        combinations = [
            (target, case, trial_number)
            for target in suite.targets
            for case in dataset.cases
            for trial_number in range(1, suite.trials + 1)
            if (target.name, case.id, trial_number) not in completed
        ]
        try:
            if (
                suite.concurrency > 1
                and suite.max_total_cost is None
                and not suite.fail_fast
            ):
                with ThreadPoolExecutor(
                    max_workers=suite.concurrency,
                    thread_name_prefix="chulk-eval",
                ) as executor:
                    futures = [
                        executor.submit(
                            _execute_trial_sync,
                            suite,
                            target,
                            case,
                            trial_number,
                        )
                        for target, case, trial_number in combinations
                    ]
                    for future in as_completed(futures):
                        execution = future.result()
                        trial = execution.trial
                        _record_trial(suite, completed, operational_errors, trial)
                        cost, known = _trial_cost(trial)
                        total_cost += cost
                        unknown_cost = (
                            unknown_cost or execution.cost_unknown or not known
                        )
                        checkpoint()
                if suite.max_total_cost is not None and total_cost > suite.max_total_cost:
                    operational_errors.append(
                        f"evaluation cost ${total_cost:.6f} exceeded cap ${suite.max_total_cost:.6f}"
                    )
            else:
                for target, case, trial_number in combinations:
                    if (
                        suite.max_total_cost is not None
                        and total_cost >= suite.max_total_cost
                    ):
                        break
                    execution = _execute_trial_sync(
                        suite, target, case, trial_number
                    )
                    trial = execution.trial
                    _record_trial(suite, completed, operational_errors, trial)
                    cost, known = _trial_cost(trial)
                    total_cost += cost
                    unknown_cost = (
                        unknown_cost or execution.cost_unknown or not known
                    )
                    checkpoint()
                    if execution.halt:
                        break
                    if (
                        suite.max_total_cost is not None
                        and total_cost >= suite.max_total_cost
                    ):
                        break
                    if suite.fail_fast and (
                        trial.exception or not _trial_required_passed(suite, trial)
                    ):
                        break
        except BaseException:
            checkpoint(EvalRunStatus.INTERRUPTED)
            raise

        if suite.max_total_cost is not None:
            if total_cost > suite.max_total_cost:
                operational_errors.append(
                    f"evaluation cost ${total_cost:.6f} exceeded cap ${suite.max_total_cost:.6f}"
                )
            elif total_cost >= suite.max_total_cost and any(
                (target.name, case.id, trial_number) not in completed
                for target, case, trial_number in combinations
            ):
                operational_errors.append(
                    f"evaluation cost cap ${suite.max_total_cost:.6f} exhausted before all trials completed"
                )

        if _requires_known_cost(suite) and unknown_cost and not suite.safety.allow_unknown_cost:
            operational_errors.append(
                "metered evaluation produced unknown cost; opt in with allow_unknown_cost"
            )
        return checkpoint(EvalRunStatus.COMPLETED)

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

    async def run(
        self,
        suite: EvalSuite,
        *,
        tags: tuple[str, ...] = (),
        resume_from: str | None = None,
    ) -> EvalReport:
        dataset = suite.dataset.filtered(tags=tags)
        started, run_id, completed, operational_errors = await _load_resume_async(
            suite, dataset.digest, resume_from
        )
        baseline = await _load_baseline_async(suite, operational_errors)
        semaphore = asyncio.Semaphore(suite.concurrency)

        async def execute(target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
            async with semaphore:
                trial = await _run_trial_async(suite, target, case, trial_number)
                return await _grade_async(suite, case, trial)

        async def checkpoint(
            status: EvalRunStatus = EvalRunStatus.RUNNING,
        ) -> EvalReport:
            report = _build_report(
                run_id,
                suite,
                dataset.digest,
                started,
                _case_results_from_trials(suite, dataset.cases, completed),
                operational_errors,
                status=status,
                baseline=baseline,
            )
            if suite.store is None:
                return report
            save_async = getattr(suite.store, "save_report_async", None)
            try:
                if callable(save_async):
                    await save_async(report)
                else:
                    await asyncio.to_thread(
                        cast(Any, suite.store).save_report,
                        report,
                    )
            except Exception as exc:
                message = f"store failed: {_format_exception(exc)}"
                if message not in operational_errors:
                    operational_errors.append(message)
                report = _build_report(
                    run_id,
                    suite,
                    dataset.digest,
                    started,
                    _case_results_from_trials(suite, dataset.cases, completed),
                    operational_errors,
                    status=status,
                    baseline=baseline,
                )
            return report

        await checkpoint()
        combinations = [
            (target, case, trial_number)
            for target in suite.targets
            for case in dataset.cases
            for trial_number in range(1, suite.trials + 1)
            if (target.name, case.id, trial_number) not in completed
        ]
        spent = sum(_trial_cost(trial)[0] for trial in completed.values())
        known = all(_trial_cost(trial)[1] for trial in completed.values())
        tasks: list[asyncio.Task[TrialResult]] = []
        try:
            if suite.max_total_cost is not None or suite.fail_fast:
                for target, case, trial_number in combinations:
                    if (
                        suite.max_total_cost is not None
                        and spent >= suite.max_total_cost
                    ):
                        break
                    trial = await execute(target, case, trial_number)
                    _record_trial(suite, completed, operational_errors, trial)
                    cost, cost_known = _trial_cost(trial)
                    spent += cost
                    known = known and cost_known
                    await checkpoint()
                    if (
                        suite.max_total_cost is not None
                        and spent >= suite.max_total_cost
                    ):
                        break
                    if suite.fail_fast and (
                        trial.exception or not _trial_required_passed(suite, trial)
                    ):
                        break
            else:
                tasks = [
                    asyncio.create_task(execute(target, case, trial_number))
                    for target, case, trial_number in combinations
                ]
                for future in asyncio.as_completed(tasks):
                    trial = await future
                    _record_trial(suite, completed, operational_errors, trial)
                    cost, cost_known = _trial_cost(trial)
                    spent += cost
                    known = known and cost_known
                    await checkpoint()
        except BaseException:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.shield(checkpoint(EvalRunStatus.INTERRUPTED))
            raise
        if suite.max_total_cost is not None:
            if spent > suite.max_total_cost:
                operational_errors.append(
                    f"evaluation cost ${spent:.6f} exceeded cap ${suite.max_total_cost:.6f}"
                )
            elif spent >= suite.max_total_cost and any(
                (target.name, case.id, trial_number) not in completed
                for target, case, trial_number in combinations
            ):
                operational_errors.append(
                    f"evaluation cost cap ${suite.max_total_cost:.6f} exhausted before all trials completed"
                )
        if _requires_known_cost(suite) and not known and not suite.safety.allow_unknown_cost:
            operational_errors.append(
                "metered evaluation produced unknown cost; opt in with allow_unknown_cost"
            )
        return await checkpoint(EvalRunStatus.COMPLETED)


def run_eval(
    scenario: EvalScenario,
    *,
    agent: object | None = None,
    agent_factory: EvalAgentFactory | None = None,
) -> EvalResult:
    return EvalRunner(agent_factory).run(scenario, agent=agent)  # type: ignore[return-value]


def _load_resume_sync(
    suite: EvalSuite,
    dataset_digest: str,
    resume_from: str | None,
) -> tuple[
    datetime,
    str,
    dict[tuple[str, str, int], TrialResult],
    list[str],
]:
    if resume_from is None:
        return datetime.now(timezone.utc), uuid4().hex, {}, []
    if suite.store is None:
        raise ValueError("resuming an evaluation requires a configured store")
    payload = cast(Any, suite.store).get_report(resume_from)
    return _resume_state(suite, dataset_digest, resume_from, payload)


async def _load_resume_async(
    suite: EvalSuite,
    dataset_digest: str,
    resume_from: str | None,
) -> tuple[
    datetime,
    str,
    dict[tuple[str, str, int], TrialResult],
    list[str],
]:
    if resume_from is None:
        return datetime.now(timezone.utc), uuid4().hex, {}, []
    if suite.store is None:
        raise ValueError("resuming an evaluation requires a configured store")
    get_async = getattr(suite.store, "get_report_async", None)
    if callable(get_async):
        payload = await get_async(resume_from)
    else:
        payload = await asyncio.to_thread(
            cast(Any, suite.store).get_report,
            resume_from,
        )
    return _resume_state(suite, dataset_digest, resume_from, payload)


def _load_baseline_sync(
    suite: EvalSuite,
    operational_errors: list[str],
) -> Mapping[str, Any] | None:
    if suite.store is None:
        return None
    try:
        return cast(Any, suite.store).get_baseline(suite.name)
    except Exception as exc:
        message = f"baseline store failed: {_format_exception(exc)}"
        if message not in operational_errors:
            operational_errors.append(message)
        return None


async def _load_baseline_async(
    suite: EvalSuite,
    operational_errors: list[str],
) -> Mapping[str, Any] | None:
    if suite.store is None:
        return None
    get_async = getattr(suite.store, "get_baseline_async", None)
    try:
        if callable(get_async):
            return await get_async(suite.name)
        return await asyncio.to_thread(
            cast(Any, suite.store).get_baseline,
            suite.name,
        )
    except Exception as exc:
        message = f"baseline store failed: {_format_exception(exc)}"
        if message not in operational_errors:
            operational_errors.append(message)
        return None


def _resume_state(
    suite: EvalSuite,
    dataset_digest: str,
    resume_from: str,
    payload: Mapping[str, Any],
) -> tuple[
    datetime,
    str,
    dict[tuple[str, str, int], TrialResult],
    list[str],
]:
    report = EvalReport.from_dict(payload)
    if report.id != resume_from:
        raise ValueError("stored evaluation id does not match the requested resume id")
    if report.status is EvalRunStatus.COMPLETED:
        raise ValueError("completed evaluation runs cannot be resumed")
    if report.suite_name != suite.name:
        raise ValueError("resume run belongs to a different evaluation suite")
    if report.dataset_digest != dataset_digest:
        raise ValueError("resume run uses a different evaluation dataset")
    expected_fingerprint = _report_metadata(suite, dataset_digest)["suite_fingerprint"]
    if report.metadata.get("suite_fingerprint") != expected_fingerprint:
        raise ValueError("resume run uses a different suite configuration")
    try:
        started = datetime.fromisoformat(report.started_at)
    except ValueError as exc:
        raise ValueError("resume run has an invalid start timestamp") from exc
    completed = {
        (trial.target_name, trial.case_id, trial.trial): trial
        for case in report.cases
        for trial in case.trials
    }
    return started, report.id, completed, list(report.operational_errors)


def _record_trial(
    suite: EvalSuite,
    completed: dict[tuple[str, str, int], TrialResult],
    operational_errors: list[str],
    trial: TrialResult,
) -> None:
    completed[(trial.target_name, trial.case_id, trial.trial)] = trial
    for message in _required_grader_errors(suite, trial):
        if message not in operational_errors:
            operational_errors.append(message)
    if trial.exception:
        message = (
            f"{trial.target_name}/{trial.case_id}/trial-{trial.trial}: "
            f"{trial.exception}"
        )
        if message not in operational_errors:
            operational_errors.append(message)


def _case_results_from_trials(
    suite: EvalSuite,
    cases: tuple[EvalCase, ...],
    completed: Mapping[tuple[str, str, int], TrialResult],
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for target in suite.targets:
        for case in cases:
            trials = tuple(
                sorted(
                    (
                        trial
                        for (target_name, case_id, _), trial in completed.items()
                        if target_name == target.name and case_id == case.id
                    ),
                    key=lambda item: item.trial,
                )
            )
            if not trials:
                continue
            results.append(
                CaseResult(
                    case.id,
                    target.name,
                    trials,
                    len(trials) == suite.trials
                    and all(_trial_required_passed(suite, trial) for trial in trials),
                )
            )
    return results


def _execute_trial_sync(
    suite: EvalSuite,
    target: EvalTarget,
    case: EvalCase,
    trial_number: int,
) -> _SyncTrialExecution:
    execution = _run_trial_sync(suite, target, case, trial_number)
    if execution.halt:
        return execution
    return replace(execution, trial=_grade_sync(suite, case, execution.trial))


def _run_trial_sync(
    suite: EvalSuite,
    target: EvalTarget,
    case: EvalCase,
    trial_number: int,
) -> _SyncTrialExecution:
    if suite.mode is EvaluationMode.REPLAY:
        return _SyncTrialExecution(
            _run_replay_sync(suite, target, case, trial_number)
        )
    started = time.monotonic()
    turns: list[EvalTurnResult] = []
    exception: str | None = None
    temporary = tempfile.TemporaryDirectory(prefix="chulk-eval-")
    workspace = Path(temporary.name).resolve()
    deferred_cleanup = False
    halt = False
    cost_unknown = False
    try:
        client = _scripted_client(case) if suite.mode is EvaluationMode.SCRIPTED else None
        scope = ExecutionScope.local(
            profile_id=f"eval-{target.name}",
            run_id=f"eval-{uuid4().hex}",
        )
        context = EvalContext(
            suite.name, target.name, case.id, trial_number, workspace, suite.mode, scope,
            llm=client, provider=target.provider, model=target.model, safety=suite.safety,
            sampling=suite.sampling,
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
                    lambda: _cancel_sync_agent(agent),
                )
                if not isinstance(result, RunResult):
                    raise TypeError("eval agent run_result() must return RunResult")
                turns.append(EvalTurnResult(index, result, tuple(events), time.monotonic() - turn_started))
        except _SyncTurnTimeout as exc:
            halt = True
            cost_unknown = True
            if not exc.stopped:
                deferred_cleanup = True
                exc.defer(
                    lambda: _cleanup_sync_resources(agent, fixture, temporary)
                )
            exception = _format_exception(
                TimeoutError(f"turn exceeded {suite.timeout_seconds:g}s timeout")
            )
            if exc.cancellation_error is not None:
                exception = (
                    f"{exception}; cancellation failed: "
                    f"{_format_exception(exc.cancellation_error)}"
                )
            if not exc.stopped:
                exception = f"{exception}; timed-out turn did not stop"
        except Exception as exc:
            exception = _format_exception(exc)
        finally:
            if not deferred_cleanup:
                cleanup_errors = _cleanup_sync_resources(agent, fixture, temporary)
                exception = _merge_exceptions(exception, cleanup_errors)
        return _SyncTrialExecution(
            TrialResult(
                case.id,
                target.name,
                trial_number,
                tuple(turns),
                time.monotonic() - started,
                exception=exception,
                workspace=workspace,
            ),
            halt=halt,
            cost_unknown=cost_unknown,
        )
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
        context = EvalContext(suite.name, target.name, case.id, trial_number, workspace, suite.mode, scope, llm=client, provider=target.provider, model=target.model, safety=suite.safety, sampling=suite.sampling)
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
        fixture = load_replay_fixture(fixture_path)
        replay = execute_replay_fixture(fixture)
        turn = _replay_turn_result(
            case,
            replay,
            fixture,
            time.monotonic() - started,
        )
        return TrialResult(case.id, target.name, trial_number, (turn,), time.monotonic() - started)
    except Exception as exc:
        return TrialResult(case.id, target.name, trial_number, (), time.monotonic() - started, exception=_format_exception(exc))


async def _run_replay_async(suite: EvalSuite, target: EvalTarget, case: EvalCase, trial_number: int) -> TrialResult:
    from chulk.tracing.execution import execute_replay_fixture_async
    from chulk.tracing.fixtures import load_replay_fixture

    started = time.monotonic()
    try:
        fixture_path = _replay_path(suite, case)
        fixture = load_replay_fixture(fixture_path)
        replay = await asyncio.wait_for(
            execute_replay_fixture_async(fixture),
            timeout=suite.timeout_seconds,
        )
        turn = _replay_turn_result(
            case,
            replay,
            fixture,
            time.monotonic() - started,
        )
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


def _replay_turn_result(
    case: EvalCase,
    replay: Any,
    fixture: Any,
    duration: float,
) -> EvalTurnResult:
    actual = replay.actual
    result_payload = actual.get("result", {})
    state = actual.get("state", {})
    recorded_turn = _replay_recorded_turn(fixture)
    content = result_payload.get("content")
    status = result_payload.get("status") or "unknown"
    errors = list(result_payload.get("errors") or [])
    if not replay.ok:
        errors.append("replay mismatch: " + ", ".join(replay.mismatches))
    conversation_id = str(state.get("conversation_id") or f"replay-{case.id}")
    result = RunResult(
        content=str(content or ""),
        status=RunStatus(status) if status in {item.value for item in RunStatus} else RunStatus.UNKNOWN,
        turn_id=(
            str(recorded_turn.get("turn_id"))
            if recorded_turn.get("turn_id") is not None
            else None
        ),
        conversation_id=conversation_id,
        trace_path=None,
        usage=_replay_usage(actual),
        cost=_replay_cost(actual),
        tool_calls=_replay_tool_calls(fixture, recorded_turn),
        observations=_replay_observations(fixture, recorded_turn),
        loaded_skill_names=tuple(state.get("loaded_skill_names") or ()),
        loaded_memory_ids=tuple(state.get("loaded_memory_ids") or ()),
        errors=tuple(str(error) for error in errors),
        plan=plan_snapshot(result_payload.get("plan")),
        extension_metadata={"replay": replay.to_dict()},
    )
    events = _replay_events(actual, fixture, conversation_id)
    return EvalTurnResult(0, result, events, duration)


def _replay_recorded_turn(fixture: Any) -> Mapping[str, Any]:
    for event in reversed(fixture.expected.events):
        if event.get("type") != "turn_finished":
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("turn"), Mapping):
            return cast(Mapping[str, Any], payload["turn"])
    return {}


def _replay_tool_calls(
    fixture: Any,
    recorded_turn: Mapping[str, Any],
) -> tuple[ToolCall, ...]:
    recorded = recorded_turn.get("tool_calls")
    if isinstance(recorded, list | tuple):
        return tuple(tool_call_snapshot(item) for item in recorded)
    results = iter(fixture.tool_results)
    calls = []
    for index, action_record in enumerate(fixture.model_actions, 1):
        action = action_record.action
        if action.get("type") != "tool_call":
            continue
        result = next(results, None)
        calls.append(
            tool_call_snapshot(
                {
                    "tool_name": action.get("tool_name"),
                    "arguments": action.get("arguments", {}),
                    "iteration": getattr(result, "iteration", index),
                    "phase": getattr(result, "phase", "execution"),
                    "resolved_tool_name": getattr(result, "tool_name", None),
                    "success": getattr(result, "success", None),
                    "error": getattr(result, "error", None),
                    "failure_kind": getattr(result, "failure_kind", None),
                    "metadata": getattr(result, "metadata", None) or {},
                }
            )
        )
    return tuple(calls)


def _replay_observations(
    fixture: Any,
    recorded_turn: Mapping[str, Any],
) -> tuple[Observation, ...]:
    recorded = recorded_turn.get("observations")
    if isinstance(recorded, list | tuple):
        return tuple(observation_snapshot(item) for item in recorded)
    return tuple(
        observation_snapshot(
            {
                "tool_name": result.tool_name,
                "content": result.observation,
                "output_metadata": result.output_metadata or {},
            }
        )
        for result in fixture.tool_results
    )


def _replay_usage(actual: Mapping[str, Any]) -> Usage | None:
    records = actual.get("usage")
    if not isinstance(records, list | tuple):
        return None
    aggregate = aggregate_usage(
        [
            usage_from_dict(dict(payload))
            for record in records
            if isinstance(record, Mapping)
            and isinstance((payload := record.get("usage")), Mapping)
        ],
        source="replay",
    )
    return usage_snapshot(aggregate)


def _replay_cost(actual: Mapping[str, Any]) -> Cost | None:
    records = actual.get("costs")
    if not isinstance(records, list | tuple):
        return None
    aggregate = aggregate_cost(
        [
            cost_from_dict(dict(payload))
            for record in records
            if isinstance(record, Mapping)
            and isinstance((payload := record.get("cost")), Mapping)
        ]
    )
    return cost_snapshot(aggregate)


def _replay_events(
    actual: Mapping[str, Any],
    fixture: Any,
    conversation_id: str,
) -> tuple[AgentEvent, ...]:
    expected_events = fixture.expected.events
    events: list[AgentEvent] = []
    expected_index = 0
    for event_name in actual.get("events", ()):
        data: Mapping[str, Any] = {"replay": True}
        while expected_index < len(expected_events):
            expected = expected_events[expected_index]
            expected_index += 1
            if expected.get("type") != event_name:
                continue
            payload = expected.get("payload")
            if isinstance(payload, Mapping):
                data = payload
            break
        events.append(
            AgentEvent(
                name=str(event_name),
                conversation_id=conversation_id,
                payload=SerializedEventPayload(data=data),
            )
        )
    return tuple(events)


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
        context.provider, context.model, context.safety, context.sampling,
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
        name = str(getattr(grader, "name", type(grader).__name__))
        try:
            grade = cast(Any, grader).grade(case, trial)
            if inspect.isawaitable(grade):
                raise TypeError("async grader requires AsyncEvalRunner")
            if not isinstance(grade, GradeResult):
                raise TypeError("grader must return GradeResult")
            if grade.grader != name:
                raise ValueError(
                    f"grader {name!r} returned GradeResult for {grade.grader!r}"
                )
        except Exception as exc:
            grade = GradeResult(name, 0.0, False, f"grader failed: {exc}", error=_format_exception(exc))
        grades.append(_mark_required(grade, suite))
    return TrialResult(trial.case_id, trial.target_name, trial.trial, trial.turns, trial.duration_seconds, tuple(grades), trial.exception, trial.workspace)


async def _grade_async(suite: EvalSuite, case: EvalCase, trial: TrialResult) -> TrialResult:
    grades: list[GradeResult] = []
    for grader in suite.graders:
        name = str(getattr(grader, "name", type(grader).__name__))
        try:
            grade_async = getattr(grader, "grade_async", None)
            if callable(grade_async):
                grade = await grade_async(case, trial)
            else:
                grade = await asyncio.to_thread(cast(Any, grader).grade, case, trial)
            if not isinstance(grade, GradeResult):
                raise TypeError("grader must return GradeResult")
            if grade.grader != name:
                raise ValueError(
                    f"grader {name!r} returned GradeResult for {grade.grader!r}"
                )
        except Exception as exc:
            grade = GradeResult(name, 0.0, False, f"grader failed: {exc}", error=_format_exception(exc))
        grades.append(_mark_required(grade, suite))
    return TrialResult(trial.case_id, trial.target_name, trial.trial, trial.turns, trial.duration_seconds, tuple(grades), trial.exception, trial.workspace)


def _trial_required_passed(suite: EvalSuite, trial: TrialResult) -> bool:
    del suite
    return trial.passed


def _mark_required(grade: GradeResult, suite: EvalSuite) -> GradeResult:
    details = dict(grade.details)
    details["required"] = grade.grader in suite.required_graders
    grader = next(
        (
            item
            for item in suite.graders
            if str(getattr(item, "name", type(item).__name__)) == grade.grader
        ),
        None,
    )
    if grader is not None:
        contract = _grader_contract(grader)
        details["grader_version"] = contract["version"]
        details["grader_identity"] = contract["identity"]
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


def _build_report(
    run_id: str,
    suite: EvalSuite,
    digest: str,
    started: datetime,
    cases: list[CaseResult],
    errors: list[str],
    *,
    status: EvalRunStatus = EvalRunStatus.COMPLETED,
    baseline: Mapping[str, Any] | None = None,
) -> EvalReport:
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
        "pass_all_k": sum(case.passed for case in cases) / len(cases) if cases else 1.0,
        "mean_latency_seconds": sum(trial.duration_seconds for trial in trials) / len(trials) if trials else 0.0,
        "agent_tokens": float(agent_tokens),
        "judge_tokens": float(judge_tokens),
        "total_tokens": float(agent_tokens + judge_tokens),
        "judge_cost": judge_cost,
        "total_cost": sum(_trial_cost(trial)[0] for trial in trials),
        "exception_count": float(sum(trial.exception is not None for trial in trials)),
        "error_rate": sum(trial.exception is not None for trial in trials) / len(trials) if trials else 0.0,
    }
    durations = sorted(trial.duration_seconds for trial in trials)
    metrics["p50_latency_seconds"] = _percentile(durations, 0.50)
    metrics["p95_latency_seconds"] = _percentile(durations, 0.95)
    for target in suite.targets:
        selected = [case for case in cases if case.target_name == target.name]
        _add_group_metrics(metrics, f"target.{target.name}", selected)
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
            _add_group_metrics(metrics, f"{dimension}.{value}", selected)
    tags_by_case = {case.id: case.tags for case in suite.dataset.cases}
    for tag in sorted({tag for values in tags_by_case.values() for tag in values}):
        selected = [case for case in cases if tag in tags_by_case.get(case.case_id, ())]
        _add_group_metrics(metrics, f"tag.{tag}", selected)
    grader_names = sorted({grade.grader for trial in trials for grade in trial.grades})
    for name in grader_names:
        grades = [grade for trial in trials for grade in trial.grades if grade.grader == name and not grade.details.get("skipped")]
        if grades:
            metrics[f"grader.{name}.score"] = sum(grade.score for grade in grades) / len(grades)
            metrics[f"grader.{name}.pass_rate"] = sum(grade.passed for grade in grades) / len(grades)
    metadata = _report_metadata(suite, digest)
    if baseline is not None:
        from .reporting import compare_reports

        comparison = compare_reports(
            {
                "id": run_id,
                "suite_name": suite.name,
                "cases": [case.to_dict() for case in cases],
                "metrics": metrics,
                "metadata": metadata,
            },
            baseline,
        )
        metrics["baseline_coverage"] = comparison.baseline_coverage
        metadata["baseline_comparison"] = comparison.to_dict()
    elif "baseline_coverage" in suite.thresholds:
        metrics["baseline_coverage"] = 0.0
        metadata["baseline_comparison"] = None
    threshold_failures = tuple(
        f"{name}: value {metrics.get(name)!r} did not satisfy {threshold}"
        for name, threshold in suite.thresholds.items()
        if not threshold.accepts(metrics.get(name))
    )
    return EvalReport(
        run_id, suite.name, digest, started.isoformat(), datetime.now(timezone.utc).isoformat(),
        tuple(cases), metrics, threshold_failures, tuple(errors),
        metadata, status,
    )


def _add_group_metrics(
    metrics: dict[str, float],
    prefix: str,
    cases: list[CaseResult],
) -> None:
    trials = [trial for case in cases for trial in case.trials]
    durations = sorted(trial.duration_seconds for trial in trials)
    agent_tokens = sum(
        turn.result.usage.total_tokens
        for trial in trials
        for turn in trial.turns
        if turn.result.usage is not None
    )
    judge_tokens = sum(_trial_judge_tokens(trial) for trial in trials)
    exceptions = sum(trial.exception is not None for trial in trials)
    values = {
        "case_count": float(len(cases)),
        "trial_count": float(len(trials)),
        "pass_rate": (
            sum(case.passed for case in cases) / len(cases) if cases else 1.0
        ),
        "pass_at_k": (
            sum(any(trial.passed for trial in case.trials) for case in cases)
            / len(cases)
            if cases
            else 1.0
        ),
        "pass_all_k": (
            sum(case.passed for case in cases) / len(cases) if cases else 1.0
        ),
        "mean_latency_seconds": (
            sum(durations) / len(durations) if durations else 0.0
        ),
        "p50_latency_seconds": _percentile(durations, 0.50),
        "p95_latency_seconds": _percentile(durations, 0.95),
        "total_tokens": float(agent_tokens + judge_tokens),
        "total_cost": sum(_trial_cost(trial)[0] for trial in trials),
        "exception_count": float(exceptions),
        "error_rate": exceptions / len(trials) if trials else 0.0,
    }
    metrics.update({f"{prefix}.{name}": value for name, value in values.items()})


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int((len(values) - 1) * quantile + 0.5)))
    return values[index]


def _report_metadata(suite: EvalSuite, dataset_digest: str) -> dict[str, Any]:
    from chulk._version import __version__

    targets = [
        {
            "name": target.name,
            "provider": target.provider,
            "model": target.model,
            "fingerprint": target.fingerprint,
        }
        for target in suite.targets
    ]
    graders = [_grader_contract(grader) for grader in suite.graders]
    fingerprint_payload = {
        "suite": suite.name,
        "dataset": dataset_digest,
        "mode": suite.mode.value,
        "targets": targets,
        "graders": graders,
        "required_graders": list(suite.required_graders),
        "trials": suite.trials,
        "concurrency": suite.concurrency,
        "timeout_seconds": suite.timeout_seconds,
        "thresholds": plain_data(suite.thresholds),
        "safety": plain_data(suite.safety),
        "sampling": plain_data(suite.sampling),
        "max_total_cost": suite.max_total_cost,
        "fail_fast": suite.fail_fast,
    }
    return {
        "mode": suite.mode.value,
        "trials": suite.trials,
        "concurrency": suite.concurrency,
        "chulk_version": __version__,
        "git_revision": _git_revision(),
        "targets": targets,
        "graders": graders,
        "sampling": plain_data(suite.sampling),
        "suite_fingerprint": sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _grader_contract(grader: object) -> dict[str, str]:
    name = str(getattr(grader, "name", type(grader).__name__))
    version = str(getattr(grader, "version", "1"))
    type_name = f"{type(grader).__module__}:{type(grader).__qualname__}"
    configuration = _grader_configuration(grader)
    configuration_fingerprint = sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    identity = sha256(
        json.dumps(
            {
                "name": name,
                "version": version,
                "type": type_name,
                "configuration": configuration_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "name": name,
        "version": version,
        "type": type_name,
        "configuration_fingerprint": configuration_fingerprint,
        "identity": identity,
    }


def _grader_configuration(grader: object) -> Any:
    if is_dataclass(grader) and not isinstance(grader, type):
        return {
            item.name: _contract_value(getattr(grader, item.name))
            for item in fields(grader)
            if item.name not in {"name", "version"}
        }
    explicit = getattr(grader, "configuration", None)
    return _contract_value(explicit) if explicit is not None else {}


def _contract_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _contract_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_contract_value(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_contract_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _callable_contract(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {
                item.name: _contract_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    return {
        "type": f"{type(value).__module__}:{type(value).__qualname__}",
        "provider": getattr(value, "provider", None),
        "model": getattr(value, "model", None),
    }


def _callable_contract(value: Callable[..., Any]) -> dict[str, Any]:
    code = getattr(value, "__code__", None)
    closure = getattr(value, "__closure__", None) or ()
    payload: dict[str, Any] = {
        "module": getattr(value, "__module__", ""),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        "defaults": _contract_value(getattr(value, "__defaults__", None)),
        "kwdefaults": _contract_value(getattr(value, "__kwdefaults__", None)),
        "closure": [
            _contract_value(_closure_cell_value(cell))
            for cell in closure
        ],
    }
    if code is not None:
        payload["code"] = sha256(marshal.dumps(code)).hexdigest()
    return payload


def _closure_cell_value(cell: Any) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return "<empty>"


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


def _requires_known_cost(suite: EvalSuite) -> bool:
    return suite.mode is EvaluationMode.LIVE or any(
        bool(getattr(grader, "requires_cost_cap", False))
        for grader in suite.graders
    )


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
    def __init__(
        self,
        future: Future[Any],
        worker: threading.Thread,
        *,
        stopped: bool,
        cancellation_error: BaseException | None,
    ) -> None:
        super().__init__("evaluation turn timed out")
        self.future = future
        self.worker = worker
        self.stopped = stopped
        self.cancellation_error = cancellation_error

    def defer(self, cleanup: Callable[[], object]) -> None:
        def complete(_future: Future[Any]) -> None:
            cleanup()

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


def _run_sync_with_timeout(
    operation: Callable[[], Any],
    timeout_seconds: float,
    cancel: Callable[[], object],
) -> Any:
    future, worker = _submit_daemon(operation, name="chulk-eval-turn")
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        if future.done():
            return future.result()
        cancellation_error: BaseException | None = None
        cancel_future, cancel_worker = _submit_daemon(
            cancel,
            name="chulk-eval-cancel",
        )
        deadline = time.monotonic() + _SYNC_CANCELLATION_GRACE_SECONDS
        try:
            cancel_future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError:
            cancellation_error = TimeoutError(
                "agent cancellation hook did not return within "
                f"{_SYNC_CANCELLATION_GRACE_SECONDS:g}s"
            )
        except BaseException as cancel_exc:
            cancellation_error = cancel_exc
        if cancel_future.done():
            cancel_worker.join()

        stopped = future.done()
        if not stopped:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeoutError:
                stopped = future.done()
            except BaseException:
                stopped = True
            else:
                stopped = True
        if stopped:
            worker.join()
        raise _SyncTurnTimeout(
            future,
            worker,
            stopped=stopped,
            cancellation_error=cancellation_error,
        ) from exc
    finally:
        if future.done():
            worker.join()


def _submit_daemon(
    operation: Callable[[], Any],
    *,
    name: str,
) -> tuple[Future[Any], threading.Thread]:
    future: Future[Any] = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = operation()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    return future, worker


def _cancel_sync_agent(agent: object | None) -> None:
    if agent is None:
        raise RuntimeError("timed-out agent was not constructed")
    cancel = getattr(agent, "cancel", None)
    if callable(cancel):
        result = cancel()
        if result is not False:
            return
    close = getattr(agent, "close", None)
    if callable(close):
        close()
        return
    raise RuntimeError("sync eval agent must expose cancel() or close()")


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
