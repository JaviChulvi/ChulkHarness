"""Durable evaluation report protocols and SQLite reference store."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from chulk.hosting import ExecutionScope
from chulk.redaction import redact_data
from chulk.storage import initialize_sqlite_database, sqlite_connection

from .models import EvalReport


@dataclass(frozen=True)
class StoredEvalSummary:
    id: str
    suite_name: str
    started_at: str
    ended_at: str
    passed: bool
    mode: str
    case_count: int
    pass_rate: float
    total_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "suite_name": self.suite_name, "started_at": self.started_at,
            "ended_at": self.ended_at, "passed": self.passed, "mode": self.mode,
            "case_count": self.case_count, "pass_rate": self.pass_rate,
            "total_cost": self.total_cost,
        }


@runtime_checkable
class EvalStore(Protocol):
    def save_report(self, report: EvalReport) -> None: ...
    def get_report(self, report_id: str) -> Mapping[str, Any]: ...
    def list_reports(self, *, suite_name: str | None = None, limit: int = 100, offset: int = 0) -> tuple[StoredEvalSummary, ...]: ...
    def set_baseline(self, suite_name: str, report_id: str) -> None: ...
    def get_baseline(self, suite_name: str) -> Mapping[str, Any] | None: ...


@runtime_checkable
class AsyncEvalStore(Protocol):
    async def save_report_async(self, report: EvalReport) -> None: ...
    async def get_report_async(self, report_id: str) -> Mapping[str, Any]: ...
    async def list_reports_async(self, *, suite_name: str | None = None, limit: int = 100, offset: int = 0) -> tuple[StoredEvalSummary, ...]: ...
    async def set_baseline_async(self, suite_name: str, report_id: str) -> None: ...
    async def get_baseline_async(self, suite_name: str) -> Mapping[str, Any] | None: ...


class SQLiteEvalStore:
    """Evaluation store in Chulk's shared forward-migrated SQLite database."""

    def __init__(self, path: Path | str, *, scope: ExecutionScope | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.scope = scope or ExecutionScope.local(profile_id="evals")
        initialize_sqlite_database(self.path)

    @contextmanager
    def _connect(self):
        with sqlite_connection(self.path) as conn:
            yield conn

    def save_report(self, report: EvalReport) -> None:
        payload = redact_data(report.to_dict())
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO eval_runs (
                    id, tenant_id, workspace_id, suite_name, dataset_digest, mode,
                    started_at, ended_at, passed, case_count, pass_rate, total_cost,
                    report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    ended_at = excluded.ended_at,
                    passed = excluded.passed,
                    case_count = excluded.case_count,
                    pass_rate = excluded.pass_rate,
                    total_cost = excluded.total_cost,
                    report_json = excluded.report_json
                """,
                (
                    report.id, self.scope.tenant_id, self.scope.workspace_id,
                    report.suite_name, report.dataset_digest, str(report.metadata.get("mode", "unknown")),
                    report.started_at, report.ended_at, int(report.passed), len(report.cases),
                    float(report.metrics.get("pass_rate", 0.0)), float(report.metrics.get("total_cost", 0.0)), encoded,
                ),
            )
            conn.execute("DELETE FROM eval_trials WHERE run_id = ?", (report.id,))
            conn.execute("DELETE FROM eval_turns WHERE run_id = ?", (report.id,))
            conn.execute("DELETE FROM eval_grades WHERE run_id = ?", (report.id,))
            for case in report.cases:
                for trial in case.trials:
                    conn.execute(
                        """INSERT INTO eval_trials
                        (run_id, target_name, case_id, trial_number, passed, duration_seconds, exception, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            report.id, case.target_name, case.case_id, trial.trial,
                            int(trial.exception is None and all(grade.passed for grade in trial.grades)),
                            trial.duration_seconds, trial.exception,
                            json.dumps(redact_data(trial.to_dict()), sort_keys=True, default=str),
                        ),
                    )
                    for turn in trial.turns:
                        conn.execute(
                            """INSERT INTO eval_turns
                            (run_id, target_name, case_id, trial_number, turn_index, status, duration_seconds, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                report.id, case.target_name, case.case_id, trial.trial, turn.index,
                                turn.result.status.value, turn.duration_seconds,
                                json.dumps(redact_data(turn.to_dict()), sort_keys=True, default=str),
                            ),
                        )
                    for grade in trial.grades:
                        conn.execute(
                            """INSERT INTO eval_grades
                            (run_id, target_name, case_id, trial_number, grader, score, passed, error, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                report.id, case.target_name, case.case_id, trial.trial, grade.grader,
                                grade.score, int(grade.passed), grade.error,
                                json.dumps(redact_data(grade.to_dict()), sort_keys=True, default=str),
                            ),
                        )

    def get_report(self, report_id: str) -> Mapping[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM eval_runs WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (report_id, self.scope.tenant_id, self.scope.workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown eval report: {report_id}")
        payload = json.loads(row["report_json"])
        if not isinstance(payload, Mapping):
            raise ValueError("stored eval report is invalid")
        return payload

    def list_reports(self, *, suite_name: str | None = None, limit: int = 100, offset: int = 0) -> tuple[StoredEvalSummary, ...]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        clauses = ["tenant_id = ?", "workspace_id = ?"]
        params: list[object] = [self.scope.tenant_id, self.scope.workspace_id]
        if suite_name is not None:
            clauses.append("suite_name = ?")
            params.append(suite_name)
        params.extend((limit, offset))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, suite_name, started_at, ended_at, passed, mode, case_count, pass_rate, total_cost "
                f"FROM eval_runs WHERE {' AND '.join(clauses)} ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        return tuple(
            StoredEvalSummary(
                row["id"], row["suite_name"], row["started_at"], row["ended_at"],
                bool(row["passed"]), row["mode"], int(row["case_count"]),
                float(row["pass_rate"]), float(row["total_cost"]),
            )
            for row in rows
        )

    def set_baseline(self, suite_name: str, report_id: str) -> None:
        report = self.get_report(report_id)
        if report.get("suite_name") != suite_name:
            raise ValueError("baseline report belongs to a different suite")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO eval_baselines
                (tenant_id, workspace_id, suite_name, report_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, workspace_id, suite_name) DO UPDATE SET
                    report_id = excluded.report_id, updated_at = excluded.updated_at""",
                (
                    self.scope.tenant_id, self.scope.workspace_id, suite_name, report_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_baseline(self, suite_name: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_id FROM eval_baselines WHERE tenant_id = ? AND workspace_id = ? AND suite_name = ?",
                (self.scope.tenant_id, self.scope.workspace_id, suite_name),
            ).fetchone()
        return None if row is None else self.get_report(row["report_id"])


class AsyncSQLiteEvalStore:
    """Non-blocking adapter for the SQLite evaluation store."""

    def __init__(self, path: Path | str, *, scope: ExecutionScope | None = None) -> None:
        self.store = SQLiteEvalStore(path, scope=scope)

    async def save_report_async(self, report: EvalReport) -> None:
        await asyncio.to_thread(self.store.save_report, report)

    async def get_report_async(self, report_id: str) -> Mapping[str, Any]:
        return await asyncio.to_thread(self.store.get_report, report_id)

    async def list_reports_async(self, *, suite_name: str | None = None, limit: int = 100, offset: int = 0) -> tuple[StoredEvalSummary, ...]:
        return await asyncio.to_thread(self.store.list_reports, suite_name=suite_name, limit=limit, offset=offset)

    async def set_baseline_async(self, suite_name: str, report_id: str) -> None:
        await asyncio.to_thread(self.store.set_baseline, suite_name, report_id)

    async def get_baseline_async(self, suite_name: str) -> Mapping[str, Any] | None:
        return await asyncio.to_thread(self.store.get_baseline, suite_name)


__all__ = [
    "AsyncEvalStore", "AsyncSQLiteEvalStore", "EvalStore", "SQLiteEvalStore",
    "StoredEvalSummary",
]
