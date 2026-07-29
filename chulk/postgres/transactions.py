"""Cross-ledger PostgreSQL transactions for hosted ownership transfers."""

from __future__ import annotations

from collections.abc import Mapping
import sqlite3
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from chulk.gateway import GatewayRunTarget, InboundEnvelope, OutboundEnvelope
from chulk.hosting import ExecutionScope
from chulk.runs import RunClaim, RunConflictError, RunRecord, RunSubmission

from chulk.postgres.stores import (
    AsyncPostgreSQLGatewayStore,
    AsyncPostgreSQLRunStore,
    PostgreSQLGatewayStore,
    PostgreSQLRunStore,
)


class PostgreSQLTransactionError(RuntimeError):
    """Raised when one side of an ownership transfer rejects the transaction."""


def ingest_and_submit_run(
    engine: Engine,
    gateway: PostgreSQLGatewayStore,
    runs: PostgreSQLRunStore,
    envelope: InboundEnvelope,
    *,
    profile_id: str,
    target: GatewayRunTarget,
    submission: RunSubmission,
    conversation_key: str | None = None,
    max_pending: int | None = None,
) -> tuple[Any, RunRecord]:
    """Atomically accept an inbound event and submit its immutable run."""

    _require_sync_engine(engine, gateway, runs)
    for attempt in range(2):
        try:
            with engine.begin() as connection:
                with (
                    gateway._using_connection(connection),
                    runs._using_connection(connection),
                ):
                    ingested = gateway.ingest(
                        envelope,
                        profile_id=profile_id,
                        conversation_key=conversation_key,
                        max_pending=max_pending,
                        run_target=target,
                    )
                    record = runs.submit(
                        target.scope,
                        submission,
                        actor="gateway",
                    )
                    return ingested, record
        except (RunConflictError, sqlite3.IntegrityError):
            if attempt:
                raise
    raise AssertionError("unreachable")


async def async_ingest_and_submit_run(
    engine: AsyncEngine,
    gateway: AsyncPostgreSQLGatewayStore,
    runs: AsyncPostgreSQLRunStore,
    envelope: InboundEnvelope,
    *,
    profile_id: str,
    target: GatewayRunTarget,
    submission: RunSubmission,
    conversation_key: str | None = None,
    max_pending: int | None = None,
) -> tuple[Any, RunRecord]:
    """Native-async equivalent of :func:`ingest_and_submit_run`."""

    _require_async_engine(engine, gateway, runs)
    for attempt in range(2):
        try:
            async with engine.begin() as connection:
                def invoke(sync_connection: Any) -> tuple[Any, RunRecord]:
                    sync_gateway = gateway.store
                    sync_runs = runs.store
                    with (
                        sync_gateway._using_connection(sync_connection),
                        sync_runs._using_connection(sync_connection),
                    ):
                        ingested = sync_gateway.ingest(
                            envelope,
                            profile_id=profile_id,
                            conversation_key=conversation_key,
                            max_pending=max_pending,
                            run_target=target,
                        )
                        record = sync_runs.submit(
                            target.scope,
                            submission,
                            actor="gateway",
                        )
                        return ingested, record

                return await connection.run_sync(invoke)
        except (RunConflictError, sqlite3.IntegrityError):
            if attempt:
                raise
    raise AssertionError("unreachable")


def complete_run_and_enqueue(
    engine: Engine,
    gateway: PostgreSQLGatewayStore,
    runs: PostgreSQLRunStore,
    scope: ExecutionScope,
    claim: RunClaim,
    *,
    inbox_id: str,
    execution_token: str,
    result: Mapping[str, Any],
    responses: tuple[OutboundEnvelope, ...],
) -> RunRecord:
    """Atomically complete a run and transfer its responses to the outbox."""

    _require_sync_engine(engine, gateway, runs)
    with engine.begin() as connection:
        with (
            gateway._using_connection(connection),
            runs._using_connection(connection),
        ):
            _validate_execution_ownership(
                gateway,
                scope,
                claim,
                inbox_id=inbox_id,
                execution_token=execution_token,
            )
            record = runs.complete(scope, claim, result=result)
            if not gateway.complete_execution(
                inbox_id,
                execution_token,
                responses,
            ):
                raise PostgreSQLTransactionError(
                    "gateway execution lease changed before outbox enqueue"
                )
            return record


async def async_complete_run_and_enqueue(
    engine: AsyncEngine,
    gateway: AsyncPostgreSQLGatewayStore,
    runs: AsyncPostgreSQLRunStore,
    scope: ExecutionScope,
    claim: RunClaim,
    *,
    inbox_id: str,
    execution_token: str,
    result: Mapping[str, Any],
    responses: tuple[OutboundEnvelope, ...],
) -> RunRecord:
    """Native-async terminal run/outbox ownership transfer."""

    _require_async_engine(engine, gateway, runs)
    async with engine.begin() as connection:
        def invoke(sync_connection: Any) -> RunRecord:
            sync_gateway = gateway.store
            sync_runs = runs.store
            with (
                sync_gateway._using_connection(sync_connection),
                sync_runs._using_connection(sync_connection),
            ):
                _validate_execution_ownership(
                    sync_gateway,
                    scope,
                    claim,
                    inbox_id=inbox_id,
                    execution_token=execution_token,
                )
                record = sync_runs.complete(scope, claim, result=result)
                if not sync_gateway.complete_execution(
                    inbox_id,
                    execution_token,
                    responses,
                ):
                    raise PostgreSQLTransactionError(
                        "gateway execution lease changed before outbox enqueue"
                    )
                return record

        return await connection.run_sync(invoke)


def _validate_execution_ownership(
    gateway: PostgreSQLGatewayStore,
    scope: ExecutionScope,
    claim: RunClaim,
    *,
    inbox_id: str,
    execution_token: str,
) -> None:
    with gateway._connect() as conn:
        gateway._serialize_inbox_mutation(conn, inbox_id)
        inbox = gateway.get_inbox(inbox_id)
    if (
        inbox is None
        or inbox.state != "processing"
        or inbox.execution_token != execution_token
    ):
        raise PostgreSQLTransactionError(
            "gateway execution lease changed before run completion"
        )
    target = inbox.run_target
    if (
        target is None
        or target.scope.run_id != claim.run_id
        or target.scope.key != scope.key
        or claim.scope_key != scope.key
    ):
        raise PostgreSQLTransactionError(
            "gateway execution does not own the claimed run scope"
        )


def _require_sync_engine(
    engine: Engine,
    gateway: PostgreSQLGatewayStore,
    runs: PostgreSQLRunStore,
) -> None:
    if gateway.engine is not engine or runs.engine is not engine:
        raise ValueError(
            "atomic ownership transfers require stores from the same engine"
        )


def _require_async_engine(
    engine: AsyncEngine,
    gateway: AsyncPostgreSQLGatewayStore,
    runs: AsyncPostgreSQLRunStore,
) -> None:
    if gateway.async_engine is not engine or runs.async_engine is not engine:
        raise ValueError(
            "atomic ownership transfers require stores from the same async engine"
        )


__all__ = [
    "PostgreSQLTransactionError",
    "async_complete_run_and_enqueue",
    "async_ingest_and_submit_run",
    "complete_run_and_enqueue",
    "ingest_and_submit_run",
]
