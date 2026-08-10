"""Durable evaluation checkpoints, resume, provenance, and baseline tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3

import pytest

from chulk.evals import (
    AsyncEvalRunner,
    AsyncEvalStore,
    AsyncSQLiteEvalStore,
    CallableGrader,
    EvalCase,
    EvalDataset,
    EvalReference,
    EvalReport,
    EvalRunStatus,
    EvalRunner,
    EvalStore,
    EvalSuite,
    EvalTarget,
    EvalTurn,
    ExactAnswerGrader,
    GradeResult,
    LatencyGrader,
    MetricThreshold,
    SQLiteEvalStore,
    compare_reports,
)
from chulk.hosting import ExecutionScope
from chulk.results import RunResult, RunStatus
from chulk.storage import SQLITE_MIGRATIONS, initialize_sqlite_database, sqlite_connection


def _result(content: str = "done") -> RunResult:
    return RunResult(content, RunStatus.COMPLETED, None, "conversation", None)


def _case(case_id: str, answer: str = "done") -> EvalCase:
    return EvalCase(
        case_id,
        (EvalTurn("run", ({"type": "final_answer", "content": answer},)),),
        EvalReference(answer=answer),
    )


def test_report_json_round_trip_preserves_public_results_and_events() -> None:
    class AgentDouble:
        def run_result(self, _message, *, on_event, **_kwargs):
            from chulk.events import AgentEvent, SerializedEventPayload

            on_event(
                AgentEvent(
                    name="run.completed",
                    conversation_id="conversation",
                    profile_id="default",
                    payload=SerializedEventPayload(data={"safe": True}),
                )
            )
            return _result()

        def close(self):
            return None

    suite = EvalSuite(
        "round-trip",
        EvalDataset((_case("one"),)),
        (EvalTarget("agent", lambda _context: AgentDouble()),),
        (ExactAnswerGrader(),),
        required_graders=("answer.exact",),
        sampling={"temperature": 0, "seed": 42},
    )

    report = EvalRunner().run(suite)
    assert isinstance(report, EvalReport)

    restored = EvalReport.from_dict(json.loads(json.dumps(report.to_dict())))

    assert restored.to_dict() == report.to_dict()
    assert restored.metadata["sampling"] == {"temperature": 0, "seed": 42}
    assert restored.cases[0].trials[0].grades[0].details["grader_version"] == "1"


def test_interrupted_run_is_checkpointed_and_resumes_without_repeating_trials(
    tmp_path: Path,
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.sqlite")
    assert isinstance(store, EvalStore)
    interrupt = True
    calls: list[str] = []

    class AgentDouble:
        def __init__(self, case_id: str) -> None:
            self.case_id = case_id

        def run_result(self, _message, **_kwargs):
            calls.append(self.case_id)
            if self.case_id == "two" and interrupt:
                raise KeyboardInterrupt
            return _result()

        def close(self):
            return None

    def factory(context):
        return AgentDouble(context.case_id)

    suite = EvalSuite(
        "resume",
        EvalDataset((_case("one"), _case("two"))),
        (EvalTarget("agent", factory),),
        (ExactAnswerGrader(),),
        required_graders=("answer.exact",),
        store=store,
    )

    with pytest.raises(KeyboardInterrupt):
        EvalRunner().run(suite)

    summary = store.list_reports()[0]
    interrupted = EvalReport.from_dict(store.get_report(summary.id))
    assert interrupted.status is EvalRunStatus.INTERRUPTED
    assert [case.case_id for case in interrupted.cases] == ["one"]
    with pytest.raises(ValueError, match="completed"):
        store.set_baseline("resume", interrupted.id)

    interrupt = False
    resumed = EvalRunner().run(suite, resume_from=interrupted.id)
    assert isinstance(resumed, EvalReport)

    assert resumed.id == interrupted.id
    assert resumed.status is EvalRunStatus.COMPLETED
    assert resumed.passed
    assert calls == ["one", "two", "two"]
    assert store.list_reports()[0].status == "completed"


@pytest.mark.asyncio
async def test_async_runner_resumes_an_interrupted_checkpoint(tmp_path: Path) -> None:
    store = AsyncSQLiteEvalStore(tmp_path / "async-evals.sqlite")
    assert isinstance(store, AsyncEvalStore)
    interrupted = True
    calls: list[str] = []
    second_started = asyncio.Event()

    class AsyncAgentDouble:
        def __init__(self, case_id: str) -> None:
            self.case_id = case_id

        async def run_result(self, _message, **_kwargs):
            calls.append(self.case_id)
            if self.case_id == "two" and interrupted:
                second_started.set()
                await asyncio.Event().wait()
            return _result()

        async def aclose(self):
            return None

    suite = EvalSuite(
        "async-resume",
        EvalDataset((_case("one"), _case("two"))),
        (EvalTarget("agent", lambda context: AsyncAgentDouble(context.case_id)),),
        (ExactAnswerGrader(),),
        required_graders=("answer.exact",),
        store=store,
    )

    task = asyncio.create_task(AsyncEvalRunner().run(suite))
    await second_started.wait()
    for _ in range(100):
        summaries = store.store.list_reports()
        if summaries and store.store.get_report(summaries[0].id).get("cases"):
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    run_id = store.store.list_reports()[0].id
    assert store.store.list_reports()[0].status == "interrupted"

    interrupted = False
    report = await AsyncEvalRunner().run(suite, resume_from=run_id)

    assert report.id == run_id
    assert report.status is EvalRunStatus.COMPLETED
    assert calls.count("one") == 1
    assert calls.count("two") == 2


def test_v20_store_migrates_provenance_and_idempotent_normalized_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "upgrade.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:20])
    with sqlite_connection(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 20

    store = SQLiteEvalStore(path)
    report = EvalRunner().run(
        EvalSuite(
            "migration",
            EvalDataset((_case("one"),)),
            (EvalTarget("agent", lambda _context: _Agent()),),
            (ExactAnswerGrader(),),
            required_graders=("answer.exact",),
            sampling={"top_p": 0.8},
        )
    )
    assert isinstance(report, EvalReport)

    store.save_report(report)
    store.save_report(report)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        run = connection.execute(
            "SELECT status, suite_fingerprint, target_fingerprints_json, "
            "grader_versions_json, sampling_json FROM eval_runs WHERE id = ?",
            (report.id,),
        ).fetchone()
        assert run is not None
        assert run["status"] == "completed"
        assert len(run["suite_fingerprint"]) == 64
        assert json.loads(run["sampling_json"]) == {"top_p": 0.8}
        assert connection.execute(
            "SELECT COUNT(*) FROM eval_trials WHERE run_id = ?", (report.id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM eval_grades WHERE run_id = ?", (report.id,)
        ).fetchone()[0] == 1


def test_store_rejects_report_id_collisions_across_execution_scopes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scoped.sqlite"
    first_scope = ExecutionScope(
        "tenant-a", "workspace-a", None, "agent", "1", "run-a"
    )
    second_scope = ExecutionScope(
        "tenant-b", "workspace-b", None, "agent", "1", "run-b"
    )
    first = SQLiteEvalStore(path, scope=first_scope)
    second = SQLiteEvalStore(path, scope=second_scope)
    report = EvalRunner().run(
        EvalSuite(
            "tenant-a-suite",
            EvalDataset((_case("one"),)),
            (EvalTarget("agent", lambda _context: _Agent()),),
        )
    )
    assert isinstance(report, EvalReport)
    first.save_report(report)

    with pytest.raises(PermissionError, match="different execution scope"):
        second.save_report(replace(report, suite_name="tenant-b-suite"))

    assert first.get_report(report.id)["suite_name"] == "tenant-a-suite"
    with pytest.raises(KeyError):
        second.get_report(report.id)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM eval_trials WHERE run_id = ?", (report.id,)
        ).fetchone()[0] == 1


def test_store_redacts_report_and_normalized_payloads(tmp_path: Path) -> None:
    secret = "sk-testsecret123456"

    class SecretAgent:
        def run_result(self, _message, **_kwargs):
            return RunResult(
                f"OPENAI_API_KEY={secret}",
                RunStatus.COMPLETED,
                None,
                "conversation",
                None,
                extension_metadata={"api_key": secret},
            )

        def close(self):
            return None

    path = tmp_path / "redacted.sqlite"
    store = SQLiteEvalStore(path)
    report = EvalRunner().run(
        EvalSuite(
            "redaction",
            EvalDataset((_case("one"),)),
            (EvalTarget("agent", lambda _context: SecretAgent()),),
            store=store,
        )
    )
    assert isinstance(report, EvalReport)
    trial = report.cases[0].trials[0]
    secret_grade = GradeResult(
        "secret.error",
        0.0,
        False,
        "provider failed",
        error=f"provider failed with {secret}",
    )
    trial = replace(
        trial,
        grades=(secret_grade,),
        exception=f"request failed with {secret}",
    )
    report = replace(
        report,
        cases=(replace(report.cases[0], trials=(trial,), passed=False),),
    )
    store.save_report(report)

    with sqlite3.connect(path) as connection:
        payloads = [
            connection.execute(
                "SELECT report_json FROM eval_runs WHERE id = ?", (report.id,)
            ).fetchone()[0]
        ]
        payloads.extend(
            row[0]
            for table in ("eval_trials", "eval_turns")
            for row in connection.execute(
                f"SELECT payload_json FROM {table} WHERE run_id = ?", (report.id,)
            )
        )
        payloads.extend(
            str(item)
            for item in connection.execute(
                "SELECT exception FROM eval_trials WHERE run_id = ?", (report.id,)
            ).fetchone()
            if item is not None
        )
        payloads.extend(
            str(item)
            for item in connection.execute(
                "SELECT error FROM eval_grades WHERE run_id = ?", (report.id,)
            ).fetchone()
            if item is not None
        )
    persisted = "\n".join(payloads)
    assert secret not in persisted
    assert "[redacted]" in persisted


def test_baseline_identity_score_deltas_and_coverage_gate(tmp_path: Path) -> None:
    store = SQLiteEvalStore(tmp_path / "baseline.sqlite")
    target = EvalTarget("agent", lambda _context: _Agent(), provider="fake", model="v1")
    baseline_suite = EvalSuite(
        "coverage",
        EvalDataset((_case("one"), _case("two"))),
        (target,),
        (VersionedExactGrader(),),
        required_graders=("answer.exact",),
        store=store,
    )
    baseline = EvalRunner().run(baseline_suite)
    assert isinstance(baseline, EvalReport)
    store.set_baseline("coverage", baseline.id)

    current_suite = EvalSuite(
        "coverage",
        EvalDataset((_case("one"),)),
        (target,),
        (VersionedExactGrader(),),
        required_graders=("answer.exact",),
        thresholds={"baseline_coverage": MetricThreshold(min=1.0)},
        store=store,
    )
    current = EvalRunner().run(current_suite)
    assert isinstance(current, EvalReport)
    comparison = compare_reports(current.to_dict(), baseline.to_dict())

    assert current.metrics["baseline_coverage"] == 0.5
    assert current.threshold_failures
    assert comparison.baseline_coverage == 0.5
    assert comparison.matched_graders == ("agent:one:answer.exact",)
    assert comparison.grade_score_deltas == {"agent:one:answer.exact": 0.0}
    assert comparison.removed_cases

    changed = dict(current.to_dict())
    changed_metadata = dict(changed["metadata"])
    targets = [dict(item) for item in changed_metadata["targets"]]
    targets[0]["fingerprint"] = "different-target"
    changed_metadata["targets"] = targets
    changed["metadata"] = changed_metadata
    identity_comparison = compare_reports(changed, baseline.to_dict())
    assert not identity_comparison.matched_cases
    assert identity_comparison.new_cases
    assert identity_comparison.removed_cases

    changed_grader = dict(current.to_dict())
    changed_grader_metadata = dict(changed_grader["metadata"])
    graders = [dict(item) for item in changed_grader_metadata["graders"]]
    graders[0]["identity"] = "different-grader"
    changed_grader_metadata["graders"] = graders
    changed_grader["metadata"] = changed_grader_metadata
    changed_cases = [dict(item) for item in changed_grader["cases"]]
    changed_trials = [dict(item) for item in changed_cases[0]["trials"]]
    changed_grades = [dict(item) for item in changed_trials[0]["grades"]]
    changed_details = dict(changed_grades[0]["details"])
    changed_details["grader_identity"] = "different-grader"
    changed_grades[0]["details"] = changed_details
    changed_trials[0]["grades"] = changed_grades
    changed_cases[0]["trials"] = changed_trials
    changed_grader["cases"] = changed_cases
    grader_comparison = compare_reports(changed_grader, baseline.to_dict())
    assert grader_comparison.matched_cases == ("agent:one",)
    assert not grader_comparison.matched_graders
    assert grader_comparison.new_graders
    assert grader_comparison.removed_graders


def test_grader_identity_tracks_configuration_and_callable_behavior() -> None:
    dataset = EvalDataset((_case("one"),))
    target = EvalTarget("agent", lambda _context: _Agent())

    def contract(grader: object) -> tuple[str, str]:
        report = EvalRunner().run(
            EvalSuite("identity", dataset, (target,), (grader,))
        )
        assert isinstance(report, EvalReport)
        grader_metadata = report.metadata["graders"][0]
        return (
            str(grader_metadata["identity"]),
            str(report.metadata["suite_fingerprint"]),
        )

    assert contract(LatencyGrader(1.0)) != contract(LatencyGrader(10.0))

    def passing(_case, _trial):
        return True

    def failing(_case, _trial):
        return False

    assert contract(CallableGrader("custom", passing)) != contract(
        CallableGrader("custom", failing)
    )


class _Agent:
    def run_result(self, _message, **_kwargs):
        return _result()

    def close(self):
        return None


@dataclass(frozen=True)
class VersionedExactGrader(ExactAnswerGrader):
    version: str = "2026-08-10"
