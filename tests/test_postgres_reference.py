"""Real PostgreSQL contracts for the optional hosted persistence reference."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from threading import Barrier, Event, Lock, local
from typing import Any
from uuid import uuid4

import pytest
import chulk.runs._store_clock as run_store_clock
import chulk.runs._store_effects as run_store_effects_module
import chulk.runs._store_parent_child as run_store_parent_child_module


pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")
pytest.importorskip("alembic")

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import chulk.postgres._compat as postgres_compat_module
from chulk.approvals import ApprovalDecision, ApprovalSubmission
from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryTarget,
    GatewayBackpressureError,
    GatewayRunTarget,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
    TrustLevel,
)
from chulk.hosting import ExecutionScope
from chulk.postgres import (
    AsyncPostgreSQLApprovalStore,
    AsyncPostgreSQLGatewayStore,
    AsyncPostgreSQLRunStore,
    AsyncPostgreSQLScheduleStore,
    PostgreSQLApprovalStore,
    PostgreSQLGatewayStore,
    PostgreSQLRunStore,
    PostgreSQLScheduleStore,
    PostgreSQLTransactionError,
    async_ingest_and_submit_run,
    complete_run_and_enqueue,
    create_async_postgres_engine,
    create_postgres_engine,
    ingest_and_submit_run,
    upgrade_postgres,
)
from chulk.runs import (
    EffectConflictError,
    InvalidRunTransitionError,
    ParentCompletionStatus,
    ParentRunPolicy,
    ReconciliationDecision,
    RunConflictError,
    RunLeaseError,
    RunNotFoundError,
    RunStore,
    RunSubmission,
    SQLiteRunStore,
    StepDefinition,
)
from chulk.scheduling import (
    AutomationConflictError,
    AutomationDeliveryState,
    AutomationNotFoundError,
)
from chulk.testing import (
    assert_async_durable_execution_contract,
    assert_async_gateway_store_contract,
    assert_async_parent_child_run_contract,
    assert_durable_execution_contract,
    assert_gateway_store_contract,
    assert_parent_child_run_contract,
)
from chulk.usage import BudgetScope, RunBudget


@dataclass
class PostgreSQLTestDatabase:
    url: str
    connect_args: dict[str, str]
    engine: Any


def test_postgres_run_store_preserves_the_shared_store_contract() -> None:
    operations = {
        name
        for name, value in vars(RunStore).items()
        if not name.startswith("_") and callable(value)
    }

    assert issubclass(PostgreSQLRunStore, SQLiteRunStore)
    assert {
        name
        for name in operations
        if not callable(getattr(PostgreSQLRunStore, name, None))
    } == set()


@pytest.fixture
def postgres_database() -> Iterator[PostgreSQLTestDatabase]:
    url = os.environ.get("CHULK_POSTGRES_TEST_URL")
    if not url:
        pytest.skip("CHULK_POSTGRES_TEST_URL is required for PostgreSQL tests")
    schema = f"chulk_test_{uuid4().hex}"
    admin = create_postgres_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    connect_args = {"options": f"-csearch_path={schema}"}
    engine = create_postgres_engine(url, connect_args=connect_args)
    try:
        upgrade_postgres(engine)
        yield PostgreSQLTestDatabase(url, connect_args, engine)
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _scope(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace",
    run_id: str | None = None,
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        actor_id="actor",
        agent_id="agent",
        agent_version="1.0.0",
        run_id=run_id or uuid4().hex,
        conversation_id="conversation",
    )


def _target(scope: ExecutionScope) -> GatewayRunTarget:
    return GatewayRunTarget(
        scope=scope,
        definition_id="agent",
        definition_version="1.0.0",
        definition_digest="sha256:definition",
    )


def _submission(
    *,
    idempotency_key: str = "submission",
    input_digest: str = "sha256:input",
    definition_digest: str = "sha256:definition",
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=idempotency_key,
        input_digest=input_digest,
        definition_digest=definition_digest,
        steps=(StepDefinition(id="agent", name="Agent turn"),),
    )


def _submit_parent_with_child(
    store: PostgreSQLRunStore,
    parent_scope: ExecutionScope,
) -> ExecutionScope:
    child_budget = RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=1,
        max_tool_calls=1,
        max_tokens=100,
    )
    store.submit_parent(
        parent_scope,
        _submission(
            idempotency_key=f"{parent_scope.run_id}-parent-key",
        ),
        policy=ParentRunPolicy(
            required_children=1,
            max_children=1,
            budget=child_budget,
        ),
    )
    child_scope = parent_scope.child(
        run_id=f"{parent_scope.run_id}-child",
        agent_version="published-1",
    )
    store.submit_child(
        parent_scope,
        child_scope,
        RunSubmission(
            idempotency_key=f"{parent_scope.run_id}-child-key",
            input_digest=f"sha256:{parent_scope.run_id}-child-input",
            definition_digest=f"sha256:{parent_scope.run_id}-child-definition",
            steps=(StepDefinition(id="agent", name="Agent turn"),),
            budget=child_budget.to_dict(),
        ),
        definition_revision=child_scope.agent_version,
    )
    return child_scope


def _enqueue_parent_completion(
    store: PostgreSQLRunStore,
    parent_scope: ExecutionScope,
) -> None:
    child_scope = _submit_parent_with_child(store, parent_scope)
    child_claim = store.claim(
        child_scope,
        worker_id="child-worker",
        run_id=child_scope.run_id,
    )
    assert child_claim is not None
    store.start_step(child_scope, child_claim, "agent")
    store.complete_step(child_scope, child_claim, "agent")
    store.complete(child_scope, child_claim, result={"ok": True})
    store.aggregate_children(
        parent_scope,
        actor="host",
        idempotency_key=f"{parent_scope.run_id}-aggregate",
    )


def _inbound(key: str = "event") -> InboundEnvelope:
    return InboundEnvelope(
        event_id=key,
        idempotency_key=key,
        identity=ChannelIdentity("contract", "primary", "actor"),
        destination_id="destination",
        parts=(TextPart("contract input"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
    )


def _outbound(inbox_id: str) -> OutboundEnvelope:
    return OutboundEnvelope(
        envelope_id=f"reply-{inbox_id}",
        profile_id="contract",
        conversation_id="conversation",
        target=DeliveryTarget("contract", "primary", "destination"),
        text="done",
    )


def test_clean_and_repeated_upgrade(postgres_database: PostgreSQLTestDatabase) -> None:
    upgrade_postgres(postgres_database.engine)
    with postgres_database.engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        table_count = connection.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            )
        ).scalar_one()
    assert revision == "0003"
    assert table_count == 25


def test_upgrade_from_0001_preserves_idempotency_rows(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    schema = f"chulk_upgrade_{uuid4().hex}"
    admin = create_postgres_engine(postgres_database.url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_postgres_engine(
        postgres_database.url,
        connect_args={"options": f"-csearch_path={schema}"},
    )
    try:
        upgrade_postgres(engine, "0001")
        runs = PostgreSQLRunStore(engine)
        scope = _scope()
        submission = _submission(idempotency_key="existing-run-key")
        created = runs.submit(scope, submission)

        upgrade_postgres(engine)

        replayed = runs.submit(scope, submission)
        assert replayed.id == created.id
        with engine.connect() as connection:
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
        assert revision == "0003"
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_long_idempotency_keys_preserve_hosted_store_contracts(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    long_key = "".join(chr(0x10000 + index) for index in range(950))
    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    submission = _submission(idempotency_key=f"run:{long_key}")
    created_run = runs.submit(scope, submission)
    assert runs.submit(scope, submission).id == created_run.id
    event = runs.record_event(
        scope,
        scope.run_id,
        name="run.long_key",
        actor="host",
        payload={"kind": "idempotency-regression"},
        idempotency_key=f"event:{long_key}",
    )
    replayed_event = runs.record_event(
        scope,
        scope.run_id,
        name="run.long_key",
        actor="host",
        payload={"kind": "idempotency-regression"},
        idempotency_key=f"event:{long_key}",
    )
    assert replayed_event.id == event.id

    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    envelope = _inbound(f"gateway:{long_key}")
    ingested = gateway.ingest(envelope, profile_id="long-key-profile")
    replayed_ingest = gateway.ingest(envelope, profile_id="long-key-profile")
    assert replayed_ingest.record.id == ingested.record.id

    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="long-key-profile",
    )
    schedule_key = f"schedule:{long_key}"
    job = schedules.create(
        adapter="contract",
        destination_id="destination",
        prompt="long idempotency key",
        next_run_at=datetime.now(timezone.utc),
        idempotency_key=schedule_key,
    )
    assert (
        schedules.create(
            adapter="contract",
            destination_id="destination",
            prompt="long idempotency key",
            next_run_at=job.next_run_at,
            idempotency_key=schedule_key,
        ).id
        == job.id
    )
    control_key = f"control:{long_key}"
    requested = schedules.run_now(
        job.id,
        expected_revision=0,
        idempotency_key=control_key,
    )
    assert (
        schedules.run_now(
            job.id,
            expected_revision=0,
            idempotency_key=control_key,
        ).revision
        == requested.revision
    )

    approvals = PostgreSQLApprovalStore(postgres_database.engine)
    approval = approvals.create(
        scope,
        ApprovalSubmission(
            step_id="agent",
            tool_name="write_ticket",
            tool_version="1.0.0",
            schema_version="1",
            arguments_digest="sha256:long-key-arguments",
            policy_version="policy-1",
            preview={"ticket_id": "42"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        ),
    )
    decision_key = f"decision:{long_key}"
    decided = approvals.decide(
        scope,
        approval.id,
        ApprovalDecision.DENY,
        decided_by="operator",
        reason="not authorized",
        idempotency_key=decision_key,
    )
    assert (
        approvals.decide(
            scope,
            approval.id,
            ApprovalDecision.DENY,
            decided_by="operator",
            reason="not authorized",
            idempotency_key=decision_key,
        ).revision
        == decided.revision
    )


def test_postgres_rejects_partial_schedule_lease_tuple(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="lease-constraint",
    )
    job = schedules.create(
        adapter="contract",
        destination_id="destination",
        prompt="lease tuple",
        next_run_at=datetime.now(timezone.utc),
    )
    with pytest.raises(IntegrityError):
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE automation_jobs SET claim_token = 'partial' "
                    "WHERE id = :id"
                ),
                {"id": job.id},
            )
    persisted = schedules.get(job.id)
    assert persisted.claim_token is None
    assert persisted.lease_until is None
    assert persisted.active_run_id is None


def test_engine_factories_reject_non_psycopg_urls() -> None:
    with pytest.raises(ValueError, match="PostgreSQL URL"):
        create_postgres_engine("sqlite://")
    with pytest.raises(ValueError, match="psycopg 3"):
        create_postgres_engine("postgresql+psycopg2://localhost/chulk")


def test_sync_public_contracts_and_scope_isolation(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    approvals = PostgreSQLApprovalStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()

    assert assert_durable_execution_contract(runs, approvals, scope=scope).passed
    assert assert_parent_child_run_contract(
        runs,
        scope=_scope(run_id="parent-child-contract"),
    ).passed
    assert assert_gateway_store_contract(
        gateway,
        target=_target(_scope(run_id="gateway-contract-run")),
    ).passed

    with pytest.raises(RunNotFoundError):
        runs.get(
            _scope(
                tenant_id="tenant-b",
                workspace_id=scope.workspace_id,
                run_id=scope.run_id,
            ),
            scope.run_id,
        )


@pytest.mark.asyncio
async def test_native_async_public_contracts(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    engine = create_async_postgres_engine(
        postgres_database.url,
        connect_args=postgres_database.connect_args,
    )
    try:
        assert (
            await assert_async_durable_execution_contract(
                AsyncPostgreSQLRunStore(engine),
                AsyncPostgreSQLApprovalStore(engine),
                scope=_scope(),
            )
        ).passed
        assert (
            await assert_async_parent_child_run_contract(
                AsyncPostgreSQLRunStore(engine),
                scope=_scope(run_id="async-parent-child-contract"),
            )
        ).passed
        assert (
            await assert_async_gateway_store_contract(
                AsyncPostgreSQLGatewayStore(engine),
                target=_target(_scope(run_id="async-gateway-contract-run")),
            )
        ).passed
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_async_stores_retry_concurrent_idempotency_collisions(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    engine = create_async_postgres_engine(
        postgres_database.url,
        connect_args=postgres_database.connect_args,
    )
    try:
        run_stores = tuple(AsyncPostgreSQLRunStore(engine) for _index in range(2))
        scope = _scope()
        submission = _submission(idempotency_key="async-concurrent-run")
        submitted = await asyncio.gather(
            *(store.submit(scope, submission) for store in run_stores)
        )
        assert [run.id for run in submitted] == [scope.run_id, scope.run_id]

        gateway_stores = tuple(
            AsyncPostgreSQLGatewayStore(engine) for _index in range(2)
        )
        envelope = _inbound("async-concurrent-inbox")
        target = _target(_scope())
        ingested = await asyncio.gather(
            *(
                store.ingest(
                    envelope,
                    profile_id="async-concurrent",
                    conversation_key="async-concurrent",
                    run_target=target,
                )
                for store in gateway_stores
            )
        )
        assert ingested[0].record.id == ingested[1].record.id

        ignored_envelope = _inbound("async-concurrent-ignore")
        ignored = await asyncio.gather(
            *(
                store.ignore(
                    ignored_envelope,
                    profile_id="async-concurrent",
                    reason="challenge consumed",
                )
                for store in gateway_stores
            )
        )
        assert ignored[0].id == ignored[1].id
        assert ignored[0].state == ignored[1].state == "ignored"

        schedule_stores = tuple(
            AsyncPostgreSQLScheduleStore(
                engine,
                profile_id="async-concurrent",
            )
            for _index in range(2)
        )
        scheduled = await asyncio.gather(
            *(
                store.create(
                    adapter="contract",
                    destination_id="destination",
                    prompt="idempotent",
                    next_run_at=datetime.now(timezone.utc),
                    idempotency_key="async-concurrent-schedule",
                )
                for store in schedule_stores
            )
        )
        assert scheduled[0].id == scheduled[1].id

        jobs = await asyncio.gather(
            *(
                store.create(
                    adapter="contract",
                    destination_id=f"destination-{index}",
                    prompt=f"job-{index}",
                    next_run_at=datetime.now(timezone.utc),
                )
                for index, store in enumerate(schedule_stores)
            )
        )

        async def update_job(index: int) -> Any | None:
            try:
                return await schedule_stores[index].update(
                    jobs[index].id,
                    expected_revision=0,
                    idempotency_key=(
                        "async-shared-control-key"
                        if index == 0
                        else " async-shared-control-key "
                    ),
                    prompt=f"updated-{index}",
                )
            except AutomationConflictError:
                return None

        updated = await asyncio.gather(*(update_job(index) for index in range(2)))
        assert sum(result is not None for result in updated) == 1
    finally:
        await engine.dispose()


def test_parent_child_fanout_is_serialized_across_postgres_workers(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    stores = tuple(
        PostgreSQLRunStore(postgres_database.engine) for _index in range(2)
    )
    parent_scope = _scope(run_id="concurrent-parent-child")
    stores[0].submit_parent(
        parent_scope,
        _submission(idempotency_key="concurrent-parent"),
        policy=ParentRunPolicy(
            required_children=1,
            max_children=1,
            budget=RunBudget(
                scope=BudgetScope.CHILD_TASK,
                max_model_calls=1,
                max_tool_calls=1,
                max_tokens=100,
            ),
        ),
    )
    child_scope = parent_scope.child(
        run_id="concurrent-child",
        agent_version="published-1",
    )
    child_budget = RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=1,
        max_tool_calls=1,
        max_tokens=100,
    )
    child_submission = RunSubmission(
        idempotency_key="concurrent-child-key",
        input_digest="sha256:concurrent-child-input",
        definition_digest="sha256:concurrent-child-definition",
        steps=(StepDefinition(id="agent", name="Agent turn"),),
        budget=child_budget.to_dict(),
    )
    barrier = Barrier(2)

    def submit(store: PostgreSQLRunStore) -> str:
        barrier.wait()
        return store.submit_child(
            parent_scope,
            child_scope,
            child_submission,
            definition_revision="published-1",
        ).run.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        children = tuple(pool.map(submit, stores))

    assert children == (child_scope.run_id, child_scope.run_id)
    assert len(stores[0].children(parent_scope, parent_scope.run_id)) == 1

    with pytest.raises(InvalidRunTransitionError, match="fan-out"):
        stores[1].submit_child(
            parent_scope,
            parent_scope.child(
                run_id="competing-child",
                agent_version="published-2",
            ),
            RunSubmission(
                idempotency_key="competing-child-key",
                input_digest="sha256:competing-child-input",
                definition_digest="sha256:competing-child-definition",
                steps=(StepDefinition(id="agent", name="Agent turn"),),
                budget=child_budget.to_dict(),
            ),
            definition_revision="published-2",
        )


@pytest.mark.parametrize(
    "pause_for_approval",
    (False, True),
    ids=("queued", "waiting-for-approval"),
)
def test_expired_postgres_child_fails_before_claim(
    postgres_database: PostgreSQLTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
    pause_for_approval: bool,
) -> None:
    store = PostgreSQLRunStore(postgres_database.engine)
    observed = [datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(run_store_clock, "utc_now", lambda: observed[0])
    child_budget = RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=1,
        max_tool_calls=1,
        max_tokens=100,
        deadline=observed[0] + timedelta(minutes=1),
    )
    parent_scope = _scope(run_id="postgres-deadline-parent")
    store.submit_parent(
        parent_scope,
        _submission(idempotency_key="postgres-deadline-parent-key"),
        policy=ParentRunPolicy(
            required_children=1,
            max_children=1,
            budget=child_budget,
        ),
    )
    child_scope = parent_scope.child(
        run_id="postgres-deadline-child",
        agent_version="published-1",
    )
    store.submit_child(
        parent_scope,
        child_scope,
        RunSubmission(
            idempotency_key="postgres-deadline-child-key",
            input_digest="sha256:postgres-deadline-child-input",
            definition_digest="sha256:postgres-deadline-child-definition",
            steps=(StepDefinition(id="agent", name="Agent turn"),),
            budget=child_budget.to_dict(),
        ),
        definition_revision=child_scope.agent_version,
    )
    if pause_for_approval:
        claim = store.claim(
            child_scope,
            worker_id="approval-worker",
            run_id=child_scope.run_id,
        )
        assert claim is not None
        store.start_step(child_scope, claim, "agent")
        effect = store.begin_effect(
            child_scope,
            claim,
            "agent",
            logical_key="postgres-deadline-effect",
            tool_name="write",
            tool_version="1",
            schema_version="1",
            arguments_digest="sha256:postgres-deadline",
        )
        store.mark_effect_started(child_scope, claim, effect.id)
        with pytest.raises(
            InvalidRunTransitionError,
            match="reconciled before approval pause",
        ):
            store.pause_for_approval(
                child_scope,
                claim,
                "agent",
                approval_id="postgres-unsafe-deadline-approval",
                payload={"reason": "operator review"},
            )
        store.fail_effect(
            child_scope,
            claim,
            effect.id,
            reason="effect stopped before approval pause",
        )
        paused = store.pause_for_approval(
            child_scope,
            claim,
            "agent",
            approval_id="postgres-deadline-approval",
            payload={"reason": "operator review"},
        )
        assert paused.status.value == "waiting_for_approval"

    observed[0] += timedelta(minutes=2)
    assert store.claim(
        child_scope,
        worker_id="late-worker",
        run_id=child_scope.run_id,
    ) is None
    expired = store.get(child_scope, child_scope.run_id)
    assert expired.status.value == "failed"
    assert expired.error == "child run budget deadline expired before claim"


def test_parent_completion_transition_rejects_a_stale_postgres_cas(
    postgres_database: PostgreSQLTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = tuple(
        PostgreSQLRunStore(postgres_database.engine) for _index in range(2)
    )
    parent_scope = _scope(run_id="completion-cas-parent")
    _enqueue_parent_completion(stores[0], parent_scope)
    completion_claim = stores[0].claim_parent_completion(
        parent_scope,
        worker_id="delivery-worker",
        parent_run_id=parent_scope.run_id,
    )
    assert completion_claim is not None

    initial_reads = Barrier(2)
    failure_committed = Event()
    transition_kind = local()
    original_completion_row = run_store_parent_child_module._completion_row

    def coordinated_completion_row(conn: Any, completion_id: str) -> Any:
        row = original_completion_row(conn, completion_id)
        if str(row["status"]) == ParentCompletionStatus.CLAIMED.value:
            initial_reads.wait()
            if transition_kind.value == "complete":
                assert failure_committed.wait(timeout=5)
        return row

    monkeypatch.setattr(
        run_store_parent_child_module,
        "_completion_row",
        coordinated_completion_row,
    )

    def fail_delivery() -> ParentCompletionStatus:
        transition_kind.value = "fail"
        completion = stores[0].fail_parent_completion(
            parent_scope,
            completion_claim,
            reason="delivery did not start",
        )
        failure_committed.set()
        return completion.status

    def complete_delivery() -> str:
        transition_kind.value = "complete"
        try:
            stores[1].complete_parent_completion(
                parent_scope,
                completion_claim,
            )
        except RunLeaseError:
            return "stale"
        return "delivered"

    with ThreadPoolExecutor(max_workers=2) as executor:
        failed = executor.submit(fail_delivery)
        completed = executor.submit(complete_delivery)
        assert failed.result(timeout=10) is ParentCompletionStatus.PENDING
        assert completed.result(timeout=10) == "stale"

    events = stores[0].events(parent_scope, parent_scope.run_id)
    assert [event.name for event in events].count(
        "parent.completion_failed"
    ) == 1
    assert all(event.name != "parent.completion_delivered" for event in events)

    monkeypatch.setattr(
        run_store_parent_child_module,
        "_completion_row",
        original_completion_row,
    )
    retry_claim = stores[0].claim_parent_completion(
        parent_scope,
        worker_id="retry-delivery-worker",
        parent_run_id=parent_scope.run_id,
    )
    assert retry_claim is not None
    delivered = stores[0].complete_parent_completion(parent_scope, retry_claim)
    assert delivered.status is ParentCompletionStatus.DELIVERED


def test_parent_completion_recovery_locks_parent_before_outbox(
    postgres_database: PostgreSQLTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgreSQLRunStore(postgres_database.engine)
    parent_scope = _scope(run_id="completion-lock-order-parent")
    observed = [datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(run_store_clock, "utc_now", lambda: observed[0])
    _enqueue_parent_completion(store, parent_scope)
    completion_claim = store.claim_parent_completion(
        parent_scope,
        worker_id="delivery-worker",
        lease_seconds=1,
        parent_run_id=parent_scope.run_id,
    )
    assert completion_claim is not None
    observed[0] += timedelta(seconds=2)

    parent_lock_attempted = Event()
    original_lock_run = postgres_compat_module.PostgreSQLConnection._lock_run

    def signal_parent_lock(
        connection: postgres_compat_module.PostgreSQLConnection,
        run_id: str,
    ) -> None:
        if run_id == parent_scope.run_id:
            parent_lock_attempted.set()
        original_lock_run(connection, run_id)

    monkeypatch.setattr(
        postgres_compat_module.PostgreSQLConnection,
        "_lock_run",
        signal_parent_lock,
    )

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM durable_runs WHERE id = :id FOR UPDATE"),
                {"id": parent_scope.run_id},
            )
            recovery = executor.submit(
                store.claim_parent_completion,
                parent_scope,
                worker_id="recovery-worker",
                parent_run_id=parent_scope.run_id,
            )
            assert parent_lock_attempted.wait(timeout=5)
            locked_completion_id = connection.execute(
                text(
                    "SELECT id FROM durable_parent_completion_outbox "
                    "WHERE parent_run_id = :parent_run_id FOR UPDATE NOWAIT"
                ),
                {"parent_run_id": parent_scope.run_id},
            ).scalar_one()
            assert locked_completion_id == completion_claim.completion.id

        assert recovery.result(timeout=10) is None
    finally:
        executor.shutdown(wait=True)

    completion = store.parent_completion(parent_scope, parent_scope.run_id)
    assert completion is not None
    assert completion.status is ParentCompletionStatus.UNKNOWN


def test_get_child_locks_parent_before_child(
    postgres_database: PostgreSQLTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgreSQLRunStore(postgres_database.engine)
    parent_scope = _scope(run_id="get-child-lock-order-parent")
    child_scope = _submit_parent_with_child(store, parent_scope)

    parent_lock_attempted = Event()
    original_lock_run = postgres_compat_module.PostgreSQLConnection._lock_run

    def signal_parent_lock(
        connection: postgres_compat_module.PostgreSQLConnection,
        run_id: str,
    ) -> None:
        if run_id == parent_scope.run_id:
            parent_lock_attempted.set()
        original_lock_run(connection, run_id)

    monkeypatch.setattr(
        postgres_compat_module.PostgreSQLConnection,
        "_lock_run",
        signal_parent_lock,
    )

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM durable_runs WHERE id = :id FOR UPDATE"),
                {"id": parent_scope.run_id},
            )
            child_read = executor.submit(
                store.get_child,
                parent_scope,
                child_scope.run_id,
            )
            assert parent_lock_attempted.wait(timeout=5)
            locked_child_id = connection.execute(
                text(
                    "SELECT id FROM durable_runs WHERE id = :id "
                    "FOR UPDATE NOWAIT"
                ),
                {"id": child_scope.run_id},
            ).scalar_one()
            assert locked_child_id == child_scope.run_id

        child = child_read.result(timeout=10)
    finally:
        executor.shutdown(wait=True)

    assert child.run.id == child_scope.run_id


def test_concurrent_workers_claim_distinct_runs_and_schedules(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    scopes = (_scope(), _scope())
    for index, scope in enumerate(scopes):
        runs.submit(
            scope,
            _submission(idempotency_key=f"run-{index}"),
        )
    barrier = Barrier(2)

    def claim_run(scope: ExecutionScope) -> str | None:
        barrier.wait()
        claim = runs.claim(scope, worker_id=uuid4().hex)
        return claim.run_id if claim else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed_runs = tuple(executor.map(claim_run, scopes))
    assert set(claimed_runs) == {scope.run_id for scope in scopes}

    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="schedule-profile",
    )
    observed = datetime(2026, 7, 29, 20, tzinfo=timezone.utc)
    jobs = tuple(
        schedules.create(
            adapter="contract",
            destination_id=str(index),
            prompt=f"job {index}",
            next_run_at=observed,
        )
        for index in range(2)
    )
    barrier = Barrier(2)

    def claim_schedule(_index: int) -> str | None:
        barrier.wait()
        claims = schedules.claim_due(
            adapter="contract",
            now=observed,
            limit=1,
            worker_id=uuid4().hex,
        )
        return claims[0].id if claims else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed_jobs = tuple(executor.map(claim_schedule, range(2)))
    assert set(claimed_jobs) == {job.id for job in jobs}


def test_schedule_leases_recovery_delivery_and_profile_isolation(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="schedule-profile",
    )
    observed = datetime(2026, 7, 29, 20, tzinfo=timezone.utc)
    job = schedules.create(
        adapter="contract",
        destination_id="destination",
        prompt="recover me",
        next_run_at=observed,
    )
    stale = schedules.claim_due(
        now=observed,
        lease_seconds=10,
        worker_id="stale-worker",
    )[0]
    assert stale.claim_token is not None
    assert schedules.renew_lease(
        job.id,
        stale.claim_token,
        now=observed + timedelta(seconds=5),
        lease_seconds=10,
    )
    assert not schedules.renew_lease(
        job.id,
        "not-owner",
        now=observed + timedelta(seconds=6),
        lease_seconds=10,
    )
    assert schedules.recover_expired(now=observed + timedelta(seconds=11)) == ()

    recovered = schedules.recover_expired(
        now=observed + timedelta(seconds=16),
    )
    assert len(recovered) == 1
    assert recovered[0].status.value == "unknown"
    assert not schedules.complete(
        job.id,
        stale.claim_token,
        finished_at=observed + timedelta(seconds=16),
    )

    retry = schedules.claim_due(
        now=observed + timedelta(seconds=16),
        worker_id="current-worker",
    )[0]
    assert retry.claim_token is not None
    assert schedules.complete(
        job.id,
        retry.claim_token,
        finished_at=observed + timedelta(seconds=17),
        delivery_state=AutomationDeliveryState.PENDING,
    )
    run = schedules.runs(job.id)[0]
    delivered = schedules.mark_delivery(
        run.id,
        AutomationDeliveryState.DELIVERED,
    )
    assert delivered.delivery_state is AutomationDeliveryState.DELIVERED
    assert len(schedules.delivery_history(run.id)) == 2
    assert schedules.get(job.id).status.value == "completed"

    isolated = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="other-profile",
    )
    with pytest.raises(AutomationNotFoundError):
        isolated.get(job.id)


@pytest.mark.asyncio
async def test_native_async_schedule_lifecycle(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    engine = create_async_postgres_engine(
        postgres_database.url,
        connect_args=postgres_database.connect_args,
    )
    try:
        schedules = AsyncPostgreSQLScheduleStore(
            engine,
            profile_id="async-schedule",
        )
        observed = datetime(2026, 7, 29, 21, tzinfo=timezone.utc)
        job = await schedules.create(
            adapter="contract",
            destination_id="destination",
            prompt="async",
            next_run_at=observed,
        )
        claim = (await schedules.claim_due(now=observed))[0]
        assert claim.claim_token is not None
        assert await schedules.renew_lease(
            job.id,
            claim.claim_token,
            now=observed + timedelta(seconds=1),
        )
        assert await schedules.complete(
            job.id,
            claim.claim_token,
            finished_at=observed + timedelta(seconds=2),
        )
        assert (await schedules.get(job.id)).status.value == "completed"
        with pytest.raises(AttributeError):
            schedules.misspelled_method
    finally:
        await engine.dispose()


def test_concurrent_duplicate_submission_and_gateway_claims(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    submission = _submission(idempotency_key="concurrent-run")
    barrier = Barrier(2)

    def submit(_index: int) -> str:
        barrier.wait()
        return runs.submit(scope, submission).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        submitted = tuple(executor.map(submit, range(2)))
    assert submitted == (scope.run_id, scope.run_id)

    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    for index in range(2):
        gateway.ingest(
            _inbound(f"gateway-{index}"),
            profile_id="contract",
            conversation_key=f"conversation-{index}",
            run_target=_target(_scope()),
        )
    barrier = Barrier(2)

    def claim_gateway(_index: int) -> str | None:
        barrier.wait()
        claim = gateway.claim_execution(global_limit=2, profile_limit=2)
        return claim.record.id if claim else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = tuple(executor.map(claim_gateway, range(2)))
    assert None not in claimed
    assert len(set(claimed)) == 2


def test_idempotent_run_submission_locks_existing_aggregate(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    submission = _submission(idempotency_key="locked-idempotent-run")
    runs.submit(scope, submission)
    started = Event()

    def replay() -> Any:
        started.set()
        return runs.submit(scope, submission)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM durable_runs WHERE id = :id FOR UPDATE"),
                {"id": scope.run_id},
            )
            future = executor.submit(replay)
            assert started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.25)
            connection.execute(
                text(
                    """
                    UPDATE durable_runs
                    SET status = 'cancelled', revision = revision + 1
                    WHERE id = :id
                    """
                ),
                {"id": scope.run_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE durable_run_steps
                    SET status = 'cancelled', revision = revision + 1
                    WHERE run_id = :id
                    """
                ),
                {"id": scope.run_id},
            )

        replayed = future.result(timeout=2)

    assert replayed.status.value == "cancelled"
    assert replayed.steps[0].status.value == "cancelled"


def test_gateway_concurrency_caps_are_atomic(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    gateways = (
        PostgreSQLGatewayStore(postgres_database.engine),
        PostgreSQLGatewayStore(postgres_database.engine),
    )

    def race_claims(
        *,
        prefix: str,
        profiles: tuple[str, str],
        global_limit: int,
        profile_limit: int,
        selected_profile: str | None = None,
    ) -> tuple[Any | None, Any | None]:
        for index, profile_id in enumerate(profiles):
            gateways[0].ingest(
                _inbound(f"{prefix}-{index}"),
                profile_id=profile_id,
                conversation_key=f"{prefix}-conversation-{index}",
                run_target=_target(_scope()),
            )
        barrier = Barrier(2)

        def claim(gateway: PostgreSQLGatewayStore) -> Any | None:
            barrier.wait()
            return gateway.claim_execution(
                global_limit=global_limit,
                profile_limit=profile_limit,
                profile_id=selected_profile,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            return tuple(executor.map(claim, gateways))

    global_claims = race_claims(
        prefix="global-cap",
        profiles=("profile-a", "profile-b"),
        global_limit=1,
        profile_limit=1,
    )
    assert sum(claim is not None for claim in global_claims) == 1
    global_winner = next(claim for claim in global_claims if claim is not None)
    assert gateways[0].quarantine_execution(
        global_winner.record.id,
        global_winner.execution_token,
        error="test cleanup",
    )

    profile_claims = race_claims(
        prefix="profile-cap",
        profiles=("shared-profile", "shared-profile"),
        global_limit=2,
        profile_limit=1,
        selected_profile="shared-profile",
    )
    assert sum(claim is not None for claim in profile_claims) == 1


def test_concurrent_gateway_pending_admission_respects_limit(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    gateways = tuple(
        PostgreSQLGatewayStore(postgres_database.engine) for _index in range(2)
    )
    barrier = Barrier(2)

    def ingest(
        indexed_store: tuple[int, PostgreSQLGatewayStore],
    ) -> bool:
        index, store = indexed_store
        barrier.wait()
        try:
            store.ingest(
                _inbound(f"bounded-pending-{index}"),
                profile_id="bounded",
                conversation_key=f"bounded-conversation-{index}",
                max_pending=1,
                run_target=_target(_scope()),
            )
        except GatewayBackpressureError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        admitted = tuple(executor.map(ingest, enumerate(gateways)))
    assert sum(admitted) == 1
    assert gateways[0].pending_count() == 1


def test_concurrent_bounded_gateway_replay_is_idempotent(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    admission_barrier = Barrier(2)

    class CoordinatedGatewayStore(PostgreSQLGatewayStore):
        def _serialize_pending_admission(self, conn: Any) -> None:
            admission_barrier.wait()
            super()._serialize_pending_admission(conn)

    gateways = tuple(
        CoordinatedGatewayStore(postgres_database.engine) for _index in range(2)
    )
    envelope = _inbound("bounded-idempotent-replay")
    target = _target(_scope())

    def ingest(store: PostgreSQLGatewayStore) -> tuple[str, bool]:
        result = store.ingest(
            envelope,
            profile_id="bounded",
            conversation_key="bounded-idempotent-conversation",
            max_pending=1,
            run_target=target,
        )
        return result.record.id, result.created

    with ThreadPoolExecutor(max_workers=2) as executor:
        ingested = tuple(executor.map(ingest, gateways))

    assert len({record_id for record_id, _created in ingested}) == 1
    assert sorted(created for _record_id, created in ingested) == [False, True]
    assert gateways[0].pending_count() == 1


def test_ignored_ingestion_is_not_claimable_before_commit(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    accepted = Event()
    allow_ignore = Event()

    class PausedIgnoreStore(PostgreSQLGatewayStore):
        def _ingest_in_transaction(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._ingest_in_transaction(*args, **kwargs)
            if result.created:
                accepted.set()
                assert allow_ignore.wait(timeout=2)
            return result

    ignore_store = PausedIgnoreStore(postgres_database.engine)
    claim_store = PostgreSQLGatewayStore(postgres_database.engine)
    envelope = _inbound("atomic-ignore")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ignore_store.ignore,
            envelope,
            profile_id="contract",
            reason="pairing challenge consumed",
        )
        assert accepted.wait(timeout=2)
        assert claim_store.claim_execution(
            global_limit=1,
            profile_limit=1,
        ) is None
        allow_ignore.set()
        ignored = future.result(timeout=2)

    assert ignored.state == "ignored"
    assert claim_store.claim_execution(
        global_limit=1,
        profile_limit=1,
    ) is None


def test_conversation_cancellation_revalidates_completed_inboxes(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    cancellation_selected = Event()
    completion_finished = Event()

    class PausedCancellationStore(PostgreSQLGatewayStore):
        def _serialize_inbox_mutation(self, conn: Any, inbox_id: str) -> None:
            cancellation_selected.set()
            assert completion_finished.wait(timeout=2)
            super()._serialize_inbox_mutation(conn, inbox_id)

    completion_store = PostgreSQLGatewayStore(postgres_database.engine)
    cancellation_store = PausedCancellationStore(postgres_database.engine)
    conversation_key = "cancel-complete-conversation"
    active = completion_store.ingest(
        _inbound("cancel-complete-active"),
        profile_id="contract",
        conversation_key=conversation_key,
        run_target=_target(_scope()),
    ).record
    stop = completion_store.ingest(
        _inbound("cancel-complete-stop"),
        profile_id="contract",
        conversation_key=conversation_key,
        run_target=_target(_scope()),
    ).record
    execution = completion_store.claim_execution(
        global_limit=1,
        profile_limit=1,
    )
    assert execution is not None
    assert execution.record.id == active.id

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            cancellation_store.request_conversation_cancellation,
            profile_id="contract",
            conversation_key=conversation_key,
            exclude_inbox_id=stop.id,
        )
        assert cancellation_selected.wait(timeout=2)
        try:
            assert completion_store.complete_execution(
                active.id,
                execution.execution_token,
                (_outbound(active.id),),
            )
        finally:
            completion_finished.set()
        assert future.result(timeout=2) == ()

    persisted = completion_store.get_inbox(active.id)
    assert persisted is not None
    assert persisted.state == "executed"
    assert not persisted.cancellation_requested


def test_concurrent_approval_creation_reuses_pending_request(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    runs.submit(scope, _submission(idempotency_key="approval-run"))
    stores = tuple(
        PostgreSQLApprovalStore(postgres_database.engine) for _index in range(2)
    )
    submission = ApprovalSubmission(
        step_id="agent",
        tool_name="write_ticket",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:approval-arguments",
        policy_version="policy-1",
        preview={"ticket_id": "42"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    barrier = Barrier(2)

    def create(store: PostgreSQLApprovalStore) -> str:
        barrier.wait()
        return store.create(scope, submission).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        approval_ids = tuple(executor.map(create, stores))
    assert approval_ids[0] == approval_ids[1]
    assert len(stores[0].list(scope, run_id=scope.run_id)) == 1


def test_approval_expiration_revalidates_after_concurrent_decision(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    selected = Event()
    proceed = Event()

    class PausingApprovalStore(PostgreSQLApprovalStore):
        def _serialize_request_mutation(
            self,
            conn: Any,
            approval_id: str,
        ) -> None:
            selected.set()
            assert proceed.wait(timeout=2)
            super()._serialize_request_mutation(conn, approval_id)

    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    runs.submit(scope, _submission(idempotency_key="approval-expiration-run"))
    approvals = PostgreSQLApprovalStore(postgres_database.engine)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    approval = approvals.create(
        scope,
        ApprovalSubmission(
            step_id="agent",
            tool_name="write_ticket",
            tool_version="1.0.0",
            schema_version="1",
            arguments_digest="sha256:expiration-arguments",
            policy_version="policy-1",
            preview={"ticket_id": "42"},
            expires_at=expires_at,
        ),
    )
    expirer = PausingApprovalStore(postgres_database.engine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            expirer.expire,
            now=expires_at + timedelta(seconds=1),
        )
        assert selected.wait(timeout=2)
        decided = approvals.decide(
            scope,
            approval.id,
            ApprovalDecision.DENY,
            decided_by="operator",
            reason="not authorized",
            idempotency_key="deny-before-expire",
        )
        proceed.set()
        assert future.result(timeout=2) == ()
    assert approvals.get(scope, approval.id).revision == decided.revision


def test_concurrent_schedule_controls_preserve_revisions(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    stores = tuple(
        PostgreSQLScheduleStore(
            postgres_database.engine,
            profile_id="revision-profile",
        )
        for _index in range(2)
    )
    observed = datetime(2026, 7, 29, 23, tzinfo=timezone.utc)
    job = stores[0].create(
        adapter="contract",
        destination_id="destination",
        prompt="original",
        next_run_at=observed,
    )

    def race(action: Any) -> tuple[bool, bool]:
        barrier = Barrier(2)

        def invoke(indexed_store: tuple[int, PostgreSQLScheduleStore]) -> bool:
            index, store = indexed_store
            barrier.wait()
            try:
                action(index, store)
            except AutomationConflictError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            return tuple(executor.map(invoke, enumerate(stores)))

    updated = race(
        lambda index, store: store.update(
            job.id,
            expected_revision=0,
            idempotency_key=f"update-{index}",
            prompt=f"winner-{index}",
        )
    )
    assert sum(updated) == 1
    assert stores[0].get(job.id).revision == 1

    requested = race(
        lambda index, store: store.run_now(
            job.id,
            expected_revision=1,
            idempotency_key=f"run-now-{index}",
            now=observed,
        )
    )
    assert sum(requested) == 1
    assert stores[0].get(job.id).revision == 2

    paused = race(
        lambda index, store: store.pause(
            job.id,
            expected_revision=2,
            idempotency_key=f"pause-{index}",
        )
    )
    assert sum(paused) == 1
    persisted = stores[0].get(job.id)
    assert persisted.revision == 3
    assert persisted.status.value == "paused"
    actions = [event.action for event in stores[0].events(job.id)]
    assert actions.count("updated") == 1
    assert actions.count("run_now_requested") == 1
    assert actions.count("pause") == 1


def test_schedule_control_keys_are_serialized_across_jobs(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    barrier = Barrier(2)

    class CoordinatedScheduleStore(PostgreSQLScheduleStore):
        def _serialize_control_action(
            self,
            conn: Any,
            idempotency_key: str,
        ) -> None:
            barrier.wait()
            super()._serialize_control_action(conn, idempotency_key)

    stores = tuple(
        CoordinatedScheduleStore(
            postgres_database.engine,
            profile_id="shared-control-key-profile",
        )
        for _index in range(2)
    )
    observed = datetime(2026, 7, 30, tzinfo=timezone.utc)
    jobs = tuple(
        stores[0].create(
            adapter="contract",
            destination_id=f"destination-{index}",
            prompt=f"job-{index}",
            next_run_at=observed,
        )
        for index in range(2)
    )

    def update_job(index: int) -> bool:
        try:
            stores[index].update(
                jobs[index].id,
                expected_revision=0,
                idempotency_key=(
                    "shared-control-key" if index == 0 else " shared-control-key "
                ),
                prompt=f"updated-{index}",
            )
        except AutomationConflictError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        updated = tuple(executor.map(update_job, range(2)))

    assert sum(updated) == 1
    assert sum(stores[0].get(job.id).revision for job in jobs) == 1


def test_schedule_claim_skips_locked_due_job(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="skip-locked-claims",
    )
    observed = datetime(2026, 7, 29, 23, 30, tzinfo=timezone.utc)
    jobs = tuple(
        schedules.create(
            adapter="contract",
            destination_id=str(index),
            prompt=f"job {index}",
            next_run_at=observed,
        )
        for index in range(5)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text(
                    "SELECT id FROM automation_jobs "
                    "WHERE id = ANY(:ids) FOR UPDATE"
                ),
                {"ids": [job.id for job in jobs[:4]]},
            )
            future = executor.submit(
                schedules.claim_due,
                now=observed,
                limit=1,
                worker_id="parallel-worker",
            )
            claims = future.result(timeout=2)
    assert [claim.id for claim in claims] == [jobs[4].id]


def test_recovery_workers_skip_locked_leases(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    observed = datetime(2026, 7, 29, 23, 45, tzinfo=timezone.utc)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    inbox = gateway.ingest(
        _inbound("locked-expired-inbox"),
        profile_id="locked-recovery",
        run_target=_target(_scope()),
    ).record
    execution = gateway.claim_execution(
        global_limit=1,
        profile_limit=1,
        now=observed,
        lease_seconds=10,
    )
    assert execution is not None
    assert execution.record.id == inbox.id
    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM gateway_inbox WHERE id = :id FOR UPDATE"),
                {"id": inbox.id},
            )
            future = executor.submit(
                gateway.recover_expired_executions,
                now=observed + timedelta(seconds=11),
            )
            assert future.result(timeout=2) == ()
    assert gateway.renew_execution(
        inbox.id,
        execution.execution_token,
        now=observed + timedelta(seconds=5),
        lease_seconds=20,
    )
    assert gateway.recover_expired_executions(
        now=observed + timedelta(seconds=11),
    ) == ()

    schedules = PostgreSQLScheduleStore(
        postgres_database.engine,
        profile_id="locked-recovery",
    )
    job = schedules.create(
        adapter="contract",
        destination_id="destination",
        prompt="locked recovery",
        next_run_at=observed,
    )
    claimed = schedules.claim_due(
        now=observed,
        lease_seconds=10,
        worker_id="lease-owner",
    )[0]
    assert claimed.claim_token is not None
    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM automation_jobs WHERE id = :id FOR UPDATE"),
                {"id": job.id},
            )
            future = executor.submit(
                schedules.recover_expired,
                now=observed + timedelta(seconds=11),
            )
            assert future.result(timeout=2) == ()
    assert schedules.renew_lease(
        job.id,
        claimed.claim_token,
        now=observed + timedelta(seconds=5),
        lease_seconds=20,
    )
    assert schedules.recover_expired(
        now=observed + timedelta(seconds=11),
    ) == ()

    runs = PostgreSQLRunStore(postgres_database.engine)
    scope = _scope()
    runs.submit(scope, _submission(idempotency_key="locked-run-recovery"))
    run_claim = runs.claim(scope, worker_id="run-lease-owner")
    assert run_claim is not None
    run_expired_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with postgres_database.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE durable_runs SET lease_until = :lease_until "
                "WHERE id = :id"
            ),
            {
                "id": scope.run_id,
                "lease_until": run_expired_at.isoformat(),
            },
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with postgres_database.engine.begin() as connection:
            connection.execute(
                text("SELECT id FROM durable_runs WHERE id = :id FOR UPDATE"),
                {"id": scope.run_id},
            )
            future = executor.submit(
                runs.reconcile_expired,
                now=run_expired_at + timedelta(seconds=1),
            )
            assert future.result(timeout=2) == ()
    recovered_runs = runs.reconcile_expired(
        now=run_expired_at + timedelta(seconds=1),
    )
    assert [run.id for run in recovered_runs] == [scope.run_id]


def test_concurrent_run_events_have_unique_ordered_sequences(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    stores = tuple(
        PostgreSQLRunStore(postgres_database.engine)
        for _index in range(8)
    )
    scope = _scope()
    stores[0].submit(
        scope,
        _submission(idempotency_key="concurrent-event-run"),
    )
    barrier = Barrier(len(stores))

    def append_event(
        indexed_store: tuple[int, PostgreSQLRunStore],
    ) -> Any:
        index, store = indexed_store
        barrier.wait()
        return store.record_event(
            scope,
            scope.run_id,
            name=f"host.event.{index}",
            actor=f"worker-{index}",
            payload={"index": index},
        )

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        appended = tuple(executor.map(append_event, enumerate(stores)))
    assert sorted(event.sequence for event in appended) == list(range(2, 10))
    persisted = stores[0].events(scope, scope.run_id)
    assert [event.sequence for event in persisted] == list(range(1, 10))


def test_concurrent_effect_reconciliation_has_one_winner(
    postgres_database: PostgreSQLTestDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = tuple(
        PostgreSQLRunStore(postgres_database.engine) for _index in range(2)
    )
    scope = _scope()
    stores[0].submit(
        scope,
        _submission(idempotency_key="concurrent-effect-reconciliation"),
    )
    claim = stores[0].claim(scope, worker_id="effect-worker")
    assert claim is not None
    stores[0].start_step(scope, claim, "agent")
    effect = stores[0].begin_effect(
        scope,
        claim,
        "agent",
        logical_key="external:effect",
        tool_name="external_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:arguments",
    )
    stores[0].mark_effect_started(scope, claim, effect.id)
    stores[0].mark_effect_unknown(
        scope,
        claim,
        effect.id,
        reason="outcome is unknown",
    )

    initial_lock_barrier = Barrier(2)
    initial_call_count = 0
    initial_call_lock = Lock()
    original_run_row = run_store_effects_module._run_row

    def coordinated_run_row(conn: Any, run_id: str) -> Any:
        nonlocal initial_call_count
        with initial_call_lock:
            coordinate = initial_call_count < 2
            initial_call_count += 1
        if coordinate:
            initial_lock_barrier.wait()
        return original_run_row(conn, run_id)

    monkeypatch.setattr(
        run_store_effects_module,
        "_run_row",
        coordinated_run_row,
    )
    decisions = (
        ReconciliationDecision.RETRY,
        ReconciliationDecision.FAILED,
    )

    def reconcile(
        indexed_store: tuple[int, PostgreSQLRunStore],
    ) -> ReconciliationDecision | None:
        index, store = indexed_store
        try:
            return store.reconcile_effect(
                scope,
                effect.id,
                decision=decisions[index],
                actor=f"operator-{index}",
                reason=f"decision-{index}",
            ).decision
        except EffectConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(reconcile, enumerate(stores)))

    assert sum(outcome is not None for outcome in outcomes) == 1
    persisted_effect = stores[0].effects(scope, scope.run_id)[0]
    persisted_run = stores[0].get(scope, scope.run_id)
    if persisted_effect.status.value == "intended":
        assert ReconciliationDecision.RETRY in outcomes
        assert persisted_run.status.value == "queued"
        assert persisted_run.steps[0].status.value == "queued"
    else:
        assert persisted_effect.status.value == "failed"
        assert ReconciliationDecision.FAILED in outcomes
        assert persisted_run.status.value == "failed"
        assert persisted_run.steps[0].status.value == "failed"
    assert [
        event.name
        for event in stores[0].events(scope, scope.run_id)
    ].count("effect.reconciled") == 1


def test_run_cancellation_and_completion_are_serialized(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    stores = tuple(
        PostgreSQLRunStore(postgres_database.engine) for _index in range(2)
    )
    scope = _scope()
    stores[0].submit(
        scope,
        _submission(idempotency_key="cancel-complete-race"),
    )
    claim = stores[0].claim(scope, worker_id="race-worker")
    assert claim is not None
    stores[0].start_step(scope, claim, "agent")
    stores[0].complete_step(scope, claim, "agent", result={"ok": True})
    barrier = Barrier(2)

    def complete() -> str:
        barrier.wait()
        try:
            stores[0].complete(scope, claim, result={"ok": True})
        except InvalidRunTransitionError:
            return "rejected"
        return "completed"

    def cancel() -> str:
        barrier.wait()
        try:
            stores[1].request_cancellation(
                scope,
                scope.run_id,
                actor="operator",
                reason="stop",
            )
        except InvalidRunTransitionError:
            return "rejected"
        return "cancelled"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = (executor.submit(complete), executor.submit(cancel))
        results = tuple(outcome.result() for outcome in outcomes)
    assert results.count("rejected") == 1
    persisted = stores[0].get(scope, scope.run_id)
    assert (persisted.status.value, persisted.cancellation_requested) in {
        ("completed", False),
        ("running", True),
    }


def test_concurrent_adapter_lease_has_one_owner(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    gateways = (
        PostgreSQLGatewayStore(postgres_database.engine),
        PostgreSQLGatewayStore(postgres_database.engine),
    )
    observed = datetime(2026, 7, 29, 22, tzinfo=timezone.utc)
    winners: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(20):
            barrier = Barrier(2)
            account_id = f"lease-{index}"

            def acquire(gateway: PostgreSQLGatewayStore) -> str | None:
                barrier.wait()
                try:
                    return gateway.start_adapter(
                        "contract",
                        account_id,
                        now=observed,
                        lease_seconds=10,
                    ).instance_token
                except RuntimeError:
                    return None

            tokens = tuple(executor.map(acquire, gateways))
            assert sum(token is not None for token in tokens) == 1
            persisted = gateways[0].adapter_status("contract", account_id)
            assert persisted is not None
            assert persisted.instance_token in tokens
            assert persisted.instance_token is not None
            winners[account_id] = persisted.instance_token

    replacement = gateways[1].start_adapter(
        "contract",
        "lease-0",
        now=observed + timedelta(seconds=11),
        lease_seconds=10,
    )
    assert replacement.instance_token is not None
    assert replacement.instance_token != winners["lease-0"]


def test_concurrent_atomic_ingest_and_submission_is_idempotent(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()
    target = _target(scope)
    submission = _submission(idempotency_key="atomic-concurrent-run")
    envelope = _inbound("atomic-concurrent-event")
    barrier = Barrier(2)

    def transfer(_index: int) -> tuple[str, str]:
        barrier.wait()
        ingested, run = ingest_and_submit_run(
            postgres_database.engine,
            gateway,
            runs,
            envelope,
            profile_id="contract",
            target=target,
            submission=submission,
        )
        return ingested.record.id, run.id

    with ThreadPoolExecutor(max_workers=2) as executor:
        transferred = tuple(executor.map(transfer, range(2)))
    assert len({inbox_id for inbox_id, _run_id in transferred}) == 1
    assert {run_id for _inbox_id, run_id in transferred} == {scope.run_id}


def test_gateway_run_ownership_transfers_are_atomic(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()
    target = _target(scope)
    submission = _submission()

    ingested, run = ingest_and_submit_run(
        postgres_database.engine,
        gateway,
        runs,
        _inbound(),
        profile_id="contract",
        target=target,
        submission=submission,
    )
    assert ingested.created and run.id == scope.run_id
    execution = gateway.claim_execution(global_limit=1, profile_limit=1)
    claim = runs.claim(scope, worker_id="worker")
    assert execution is not None and claim is not None
    runs.start_step(scope, claim, "agent")
    runs.complete_step(scope, claim, "agent", result={"ok": True})

    with pytest.raises(PostgreSQLTransactionError):
        complete_run_and_enqueue(
            postgres_database.engine,
            gateway,
            runs,
            scope,
            claim,
            inbox_id=ingested.record.id,
            execution_token="stale",
            result={"ok": True},
            responses=(_outbound(ingested.record.id),),
        )
    assert runs.get(scope, scope.run_id).status.value == "running"

    completed = complete_run_and_enqueue(
        postgres_database.engine,
        gateway,
        runs,
        scope,
        claim,
        inbox_id=ingested.record.id,
        execution_token=execution.execution_token,
        result={"ok": True},
        responses=(_outbound(ingested.record.id),),
    )
    assert completed.status.value == "completed"
    assert len(gateway.list_outbox(ingested.record.id)) == 1


def test_completion_rejects_an_inbox_owned_by_another_run(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scopes = (_scope(), _scope())
    ingested = []
    for index, scope in enumerate(scopes):
        accepted, _run = ingest_and_submit_run(
            postgres_database.engine,
            gateway,
            runs,
            _inbound(f"ownership-{index}"),
            profile_id="contract",
            target=_target(scope),
            submission=_submission(idempotency_key=f"ownership-run-{index}"),
            conversation_key=f"ownership-conversation-{index}",
        )
        ingested.append(accepted)
    executions = tuple(
        gateway.claim_execution(global_limit=2, profile_limit=2)
        for _index in range(2)
    )
    assert all(execution is not None for execution in executions)
    execution_by_inbox = {
        execution.record.id: execution
        for execution in executions
        if execution is not None
    }
    claims = []
    for scope in scopes:
        claim = runs.claim(scope, worker_id=f"worker-{scope.run_id}")
        assert claim is not None
        runs.start_step(scope, claim, "agent")
        runs.complete_step(scope, claim, "agent", result={"ok": True})
        claims.append(claim)
    unrelated_inbox = ingested[1].record.id
    unrelated_execution = execution_by_inbox[unrelated_inbox]
    with pytest.raises(
        PostgreSQLTransactionError,
        match="does not own the claimed run scope",
    ):
        complete_run_and_enqueue(
            postgres_database.engine,
            gateway,
            runs,
            scopes[0],
            claims[0],
            inbox_id=unrelated_inbox,
            execution_token=unrelated_execution.execution_token,
            result={"ok": True},
            responses=(_outbound(unrelated_inbox),),
        )
    assert runs.get(scopes[0], scopes[0].run_id).status.value == "running"
    persisted_inbox = gateway.get_inbox(unrelated_inbox)
    assert persisted_inbox is not None
    assert persisted_inbox.state == "processing"


def test_completion_rejects_a_different_run_definition(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()
    ingested = gateway.ingest(
        _inbound("completion-definition-mismatch"),
        profile_id="contract",
        run_target=_target(scope),
    )
    runs.submit(
        scope,
        _submission(
            idempotency_key="completion-definition-mismatch",
            definition_digest="sha256:different-definition",
        ),
    )
    execution = gateway.claim_execution(global_limit=1, profile_limit=1)
    claim = runs.claim(scope, worker_id="worker")
    assert execution is not None and claim is not None
    runs.start_step(scope, claim, "agent")
    runs.complete_step(scope, claim, "agent", result={"ok": True})

    with pytest.raises(
        PostgreSQLTransactionError,
        match="definition digest",
    ):
        complete_run_and_enqueue(
            postgres_database.engine,
            gateway,
            runs,
            scope,
            claim,
            inbox_id=ingested.record.id,
            execution_token=execution.execution_token,
            result={"ok": True},
            responses=(_outbound(ingested.record.id),),
        )

    assert runs.get(scope, scope.run_id).status.value == "running"
    persisted_inbox = gateway.get_inbox(ingested.record.id)
    assert persisted_inbox is not None
    assert persisted_inbox.state == "processing"
    assert gateway.list_outbox(ingested.record.id) == ()


def test_ingest_rolls_back_when_run_submission_conflicts(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()
    target = _target(scope)
    runs.submit(scope, _submission(input_digest="sha256:first"))

    envelope = _inbound("conflicting-event")
    with pytest.raises(RunConflictError):
        ingest_and_submit_run(
            postgres_database.engine,
            gateway,
            runs,
            envelope,
            profile_id="contract",
            target=target,
            submission=_submission(input_digest="sha256:second"),
        )
    assert (
        gateway.find_inbox(
            adapter="contract",
            account_id="primary",
            idempotency_key=envelope.idempotency_key,
        )
        is None
    )


def test_ingest_rejects_definition_digest_mismatch_before_writes(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    runs = PostgreSQLRunStore(postgres_database.engine)
    gateway = PostgreSQLGatewayStore(postgres_database.engine)
    scope = _scope()
    envelope = _inbound("definition-mismatch")
    with pytest.raises(
        PostgreSQLTransactionError,
        match="definition digest",
    ):
        ingest_and_submit_run(
            postgres_database.engine,
            gateway,
            runs,
            envelope,
            profile_id="contract",
            target=_target(scope),
            submission=_submission(
                idempotency_key="definition-mismatch",
                definition_digest="sha256:different-definition",
            ),
        )
    assert (
        gateway.find_inbox(
            adapter="contract",
            account_id="primary",
            idempotency_key=envelope.idempotency_key,
        )
        is None
    )
    with pytest.raises(RunNotFoundError):
        runs.get(scope, scope.run_id)


@pytest.mark.asyncio
async def test_async_ingest_and_submit_is_atomic(
    postgres_database: PostgreSQLTestDatabase,
) -> None:
    engine = create_async_postgres_engine(
        postgres_database.url,
        connect_args=postgres_database.connect_args,
    )
    try:
        gateway = AsyncPostgreSQLGatewayStore(engine)
        runs = AsyncPostgreSQLRunStore(engine)
        scope = _scope()
        results = await asyncio.gather(
            *(
                async_ingest_and_submit_run(
                    engine,
                    gateway,
                    runs,
                    _inbound("async-event"),
                    profile_id="contract",
                    target=_target(scope),
                    submission=_submission(idempotency_key="async-run"),
                )
                for _index in range(2)
            )
        )
        assert len({ingested.record.id for ingested, _record in results}) == 1
        assert {record.id for _ingested, record in results} == {scope.run_id}
    finally:
        await engine.dispose()
