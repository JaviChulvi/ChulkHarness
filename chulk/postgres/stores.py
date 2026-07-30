"""PostgreSQL implementations of the stable hosted persistence contracts."""

from __future__ import annotations

from hashlib import blake2b
import sqlite3
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from chulk.approvals.async_store import AsyncApprovalStoreAdapter
from chulk.approvals.store import SQLiteApprovalStore
from chulk.gateway.ledger import SQLiteGatewayLedger
from chulk.gateway.models import InboundEnvelope
from chulk.gateway.stores import GatewayRunTarget
from chulk.hosting import ExecutionScope
from chulk.runs.async_store import AsyncRunStoreAdapter
from chulk.runs.models import RunRecord, RunSubmission
from chulk.runs.store import RunConflictError, SQLiteRunStore
from chulk.scheduling.recurrence import RecurrenceCalculator
from chulk.scheduling.store import SQLiteScheduleStore

from ._compat import PostgreSQLConnectionOwner


_GATEWAY_ADMISSION_LOCK_ID = 0x4348554C4B


class PostgreSQLRunStore(PostgreSQLConnectionOwner, SQLiteRunStore):
    """Durable-run store backed by a synchronous SQLAlchemy engine."""

    def __init__(self, engine: Engine) -> None:
        self._initialize_postgres(engine)

    def _recovery_lock_clause(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord:
        try:
            return super().submit(scope, submission, actor=actor)
        except RunConflictError:
            if self._connection_is_bound():
                raise
            # A concurrent identical insert may win after our initial lookup.
            # Re-read through the owner's normal idempotency validation.
            return super().submit(scope, submission, actor=actor)


class PostgreSQLApprovalStore(PostgreSQLConnectionOwner, SQLiteApprovalStore):
    """Durable approval store backed by a synchronous SQLAlchemy engine."""

    def __init__(self, engine: Engine) -> None:
        self._initialize_postgres(engine)

    def _serialize_creation(self, conn: Any, run_id: str) -> None:
        conn._lock_run(run_id)

    def _serialize_request_mutation(
        self,
        conn: Any,
        approval_id: str,
    ) -> None:
        conn.execute(
            "SELECT id FROM durable_approval_requests WHERE id = ? FOR UPDATE",
            (approval_id,),
        )


class PostgreSQLGatewayStore(PostgreSQLConnectionOwner, SQLiteGatewayLedger):
    """Gateway inbox/outbox store backed by a synchronous SQLAlchemy engine."""

    def __init__(self, engine: Engine) -> None:
        self._initialize_postgres(engine)

    def _serialize_execution_claim(self, conn: Any) -> None:
        conn.execute(
            "SELECT pg_advisory_xact_lock(?)",
            (_GATEWAY_ADMISSION_LOCK_ID,),
        )

    def _serialize_pending_admission(self, conn: Any) -> None:
        conn.execute(
            "SELECT pg_advisory_xact_lock(?)",
            (_GATEWAY_ADMISSION_LOCK_ID,),
        )

    def _serialize_inbox_mutation(self, conn: Any, inbox_id: str) -> None:
        conn.execute(
            "SELECT id FROM gateway_inbox WHERE id = ? FOR UPDATE",
            (inbox_id,),
        )

    def _serialize_outbox_mutation(self, conn: Any, outbox_id: str) -> None:
        conn.execute(
            "SELECT id FROM gateway_outbox WHERE id = ? FOR UPDATE",
            (outbox_id,),
        )

    def _execution_recovery_lock_clause(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    def ingest(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        conversation_key: str | None = None,
        max_pending: int | None = None,
        run_target: GatewayRunTarget | None = None,
    ) -> Any:
        try:
            return super().ingest(
                envelope,
                profile_id=profile_id,
                conversation_key=conversation_key,
                max_pending=max_pending,
                run_target=run_target,
            )
        except sqlite3.IntegrityError:
            if self._connection_is_bound():
                raise
            # Resolve a concurrent duplicate through the established collision
            # and run-target validation path.
            return super().ingest(
                envelope,
                profile_id=profile_id,
                conversation_key=conversation_key,
                max_pending=max_pending,
                run_target=run_target,
            )

    def ignore(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().ignore(*args, **kwargs)
        except sqlite3.IntegrityError:
            if self._connection_is_bound():
                raise
            return super().ignore(*args, **kwargs)


class PostgreSQLScheduleStore(PostgreSQLConnectionOwner, SQLiteScheduleStore):
    """Profile-owned schedule execution store backed by PostgreSQL."""

    def __init__(
        self,
        engine: Engine,
        *,
        profile_id: str = "default",
        recurrence_calculator: RecurrenceCalculator | None = None,
    ) -> None:
        selected = profile_id.strip()
        if not selected:
            raise ValueError("profile_id is required")
        self.profile_id = selected
        self.recurrence = recurrence_calculator or RecurrenceCalculator()
        self._initialize_postgres(engine)

    def create(self, **kwargs: Any) -> Any:
        try:
            return super().create(**kwargs)
        except sqlite3.IntegrityError:
            if self._connection_is_bound():
                raise
            # A concurrent create can win on the control-action idempotency key.
            # Re-read it through the owner's fingerprint validation path.
            return super().create(**kwargs)

    def _serialize_job_mutation(self, conn: Any, job_id: str) -> None:
        conn.execute(
            """
            SELECT id FROM automation_jobs
            WHERE profile_id = ? AND id = ? FOR UPDATE
            """,
            (self.profile_id, job_id),
        )

    def _serialize_control_action(
        self,
        conn: Any,
        idempotency_key: str,
    ) -> None:
        digest = blake2b(
            f"{self.profile_id}\0{idempotency_key.strip()}".encode(),
            digest_size=8,
            person=b"chulk-schedule",
        ).digest()
        lock_id = int.from_bytes(digest, byteorder="big", signed=True)
        conn.execute("SELECT pg_advisory_xact_lock(?)", (lock_id,))

    def _serialize_trigger_ingest(self, conn: Any, trigger_id: str) -> None:
        conn.execute(
            """
            SELECT id FROM automation_triggers
            WHERE profile_id = ? AND id = ? FOR UPDATE
            """,
            (self.profile_id, trigger_id),
        )

    def _claim_lock_clause(self) -> str:
        return "FOR UPDATE OF j SKIP LOCKED"

    def _claim_candidate_limit(self, limit: int) -> int:
        return limit

    def _recovery_lock_clause(self) -> str:
        return "FOR UPDATE SKIP LOCKED"


class _AsyncPostgreSQLStore:
    _idempotent_retry_methods: frozenset[str] = frozenset()

    def __init__(
        self,
        engine: AsyncEngine,
        store: Any,
    ) -> None:
        self.async_engine = engine
        self.store = store

    async def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(2):
            try:
                async with self.async_engine.begin() as connection:
                    def invoke(sync_connection: Any) -> Any:
                        with self.store._using_connection(sync_connection):
                            return getattr(self.store, method)(*args, **kwargs)

                    return await connection.run_sync(invoke)
            except (RunConflictError, sqlite3.IntegrityError):
                if method not in self._idempotent_retry_methods or attempt:
                    raise
        raise AssertionError("unreachable")

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_") or not hasattr(self.store, method):
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {method!r}"
            )

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            return await self.call(method, *args, **kwargs)

        return invoke


class AsyncPostgreSQLRunStore(_AsyncPostgreSQLStore, AsyncRunStoreAdapter):
    """Native-async durable-run store using SQLAlchemy's async engine."""

    _idempotent_retry_methods = frozenset({"submit"})

    def __init__(self, engine: AsyncEngine) -> None:
        store = PostgreSQLRunStore(engine.sync_engine)
        _AsyncPostgreSQLStore.__init__(self, engine, store)


class AsyncPostgreSQLApprovalStore(
    _AsyncPostgreSQLStore,
    AsyncApprovalStoreAdapter,
):
    """Native-async durable approval store."""

    def __init__(self, engine: AsyncEngine) -> None:
        store = PostgreSQLApprovalStore(engine.sync_engine)
        _AsyncPostgreSQLStore.__init__(self, engine, store)


class AsyncPostgreSQLGatewayStore(_AsyncPostgreSQLStore):
    """Native-async gateway store exposing the complete gateway protocol."""

    _idempotent_retry_methods = frozenset({"ignore", "ingest"})

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine, PostgreSQLGatewayStore(engine.sync_engine))


class AsyncPostgreSQLScheduleStore(_AsyncPostgreSQLStore):
    """Native-async schedule store for host worker loops."""

    _idempotent_retry_methods = frozenset({"create"})

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        profile_id: str = "default",
        recurrence_calculator: RecurrenceCalculator | None = None,
    ) -> None:
        super().__init__(
            engine,
            PostgreSQLScheduleStore(
                engine.sync_engine,
                profile_id=profile_id,
                recurrence_calculator=recurrence_calculator,
            ),
        )


__all__ = [
    "AsyncPostgreSQLApprovalStore",
    "AsyncPostgreSQLGatewayStore",
    "AsyncPostgreSQLRunStore",
    "AsyncPostgreSQLScheduleStore",
    "PostgreSQLApprovalStore",
    "PostgreSQLGatewayStore",
    "PostgreSQLRunStore",
    "PostgreSQLScheduleStore",
]
