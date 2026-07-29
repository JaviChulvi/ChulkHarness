"""Real PostgreSQL contracts for the optional hosted persistence reference."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from threading import Barrier, Event
from typing import Any
from uuid import uuid4

import pytest


pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")
pytest.importorskip("alembic")

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
    RunSubmission,
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
    assert_durable_execution_contract,
    assert_gateway_store_contract,
)


@dataclass
class PostgreSQLTestDatabase:
    url: str
    connect_args: dict[str, str]
    engine: Any


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
    assert revision == "0001"
    assert table_count == 21


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
    finally:
        await engine.dispose()


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
