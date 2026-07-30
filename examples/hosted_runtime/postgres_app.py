"""Compose sync and native-async hosted stores against PostgreSQL."""

from __future__ import annotations

import asyncio
import os

from chulk import ExecutionScope, RunSubmission, StepDefinition
from chulk.postgres import (
    AsyncPostgreSQLRunStore,
    PostgreSQLApprovalStore,
    PostgreSQLGatewayStore,
    PostgreSQLRunStore,
    PostgreSQLScheduleStore,
    create_async_postgres_engine,
    create_postgres_engine,
    upgrade_postgres,
)


def scope() -> ExecutionScope:
    return ExecutionScope(
        tenant_id="example-tenant",
        workspace_id="support",
        actor_id="operator",
        agent_id="support-agent",
        agent_version="1.0.0",
        run_id="postgres-example-run",
        conversation_id="postgres-example-conversation",
    )


async def main() -> None:
    url = os.environ["CHULK_POSTGRES_URL"]
    engine = create_postgres_engine(url)
    upgrade_postgres(engine)

    runs = PostgreSQLRunStore(engine)
    approvals = PostgreSQLApprovalStore(engine)
    gateway = PostgreSQLGatewayStore(engine)
    schedules = PostgreSQLScheduleStore(engine, profile_id="example-tenant")
    _ = (approvals, gateway, schedules)

    submitted = runs.submit(
        scope(),
        RunSubmission(
            idempotency_key="example:postgres-run",
            input_digest="sha256:example-input",
            definition_digest="sha256:example-definition",
            steps=(StepDefinition(id="agent", name="Agent turn"),),
        ),
    )
    print(f"sync run: {submitted.id} ({submitted.status.value})")
    engine.dispose()

    async_engine = create_async_postgres_engine(url)
    try:
        loaded = await AsyncPostgreSQLRunStore(async_engine).get(
            scope(),
            submitted.id,
        )
        print(f"async read: {loaded.id} ({loaded.status.value})")
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
