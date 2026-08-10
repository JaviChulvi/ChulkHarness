"""PostgreSQL evaluation-store adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from sqlalchemy.engine import Engine

from chulk.evals.models import EvalReport
from chulk.evals.storage import SQLiteEvalStore, StoredEvalSummary
from chulk.hosting import ExecutionScope
from chulk.postgres._compat import PostgreSQLConnectionOwner


class PostgreSQLEvalStore(PostgreSQLConnectionOwner, SQLiteEvalStore):
    """Evaluation report store backed by a migrated PostgreSQL database."""

    def __init__(self, engine: Engine, *, scope: ExecutionScope | None = None) -> None:
        self._initialize_postgres(engine)
        self.scope = scope or ExecutionScope.local(profile_id="evals")


class AsyncPostgreSQLEvalStore:
    """Async-callable adapter around a synchronous PostgreSQL eval store."""

    def __init__(self, engine: Engine, *, scope: ExecutionScope | None = None) -> None:
        self.store = PostgreSQLEvalStore(engine, scope=scope)

    async def save_report_async(self, report: EvalReport) -> None:
        await asyncio.to_thread(self.store.save_report, report)

    async def get_report_async(self, report_id: str) -> Mapping[str, Any]:
        return await asyncio.to_thread(self.store.get_report, report_id)

    async def list_reports_async(self, *, suite_name: str | None = None, limit: int = 100, offset: int = 0) -> tuple[StoredEvalSummary, ...]:
        return await asyncio.to_thread(
            self.store.list_reports,
            suite_name=suite_name,
            limit=limit,
            offset=offset,
        )

    async def set_baseline_async(self, suite_name: str, report_id: str) -> None:
        await asyncio.to_thread(self.store.set_baseline, suite_name, report_id)

    async def get_baseline_async(self, suite_name: str) -> Mapping[str, Any] | None:
        return await asyncio.to_thread(self.store.get_baseline, suite_name)


__all__ = ["AsyncPostgreSQLEvalStore", "PostgreSQLEvalStore"]
