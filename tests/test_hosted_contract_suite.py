"""The published hosted contract gate runs without credentials or infrastructure."""

from __future__ import annotations

import asyncio

import pytest

from chulk import ExecutionScope
from chulk.gateway import GatewayRunTarget, SQLiteGatewayLedger
from chulk.hosting.reference import InMemoryServiceHub
from chulk.testing import (
    assert_async_durable_execution_contract,
    assert_async_gateway_store_contract,
    assert_async_hosted_services_contract,
    assert_durable_execution_contract,
    assert_gateway_store_contract,
    assert_hosted_services_contract,
)


def _scope(
    tenant_id: str,
    *,
    run_id: str = "contract-run",
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id="workspace",
        actor_id="actor",
        agent_id="contract-agent",
        agent_version="1.0.0",
        run_id=run_id,
        conversation_id="contract-conversation",
    )


def _target() -> GatewayRunTarget:
    return GatewayRunTarget(
        scope=_scope("tenant-a"),
        definition_id="contract-agent",
        definition_version="1.0.0",
        definition_digest="sha256:contract-definition",
    )


def test_sync_hosted_service_and_gateway_contracts(tmp_path) -> None:
    hub = InMemoryServiceHub()
    service_report = assert_hosted_services_contract(
        hub.services(),
        first_scope=_scope("tenant-a"),
        second_scope=_scope("tenant-b"),
    )
    gateway_report = assert_gateway_store_contract(
        SQLiteGatewayLedger(tmp_path / "control.sqlite"),
        target=_target(),
    )
    durable_scope = _scope("tenant-a", run_id="durable-contract-run")
    durable = hub.services().resolve(durable_scope)
    durable_report = assert_durable_execution_contract(
        durable.runs,
        durable.approvals,
        scope=durable_scope,
    )

    assert service_report.passed
    assert gateway_report.passed
    assert durable_report.passed
    assert "delivery_reconciliation" in gateway_report.checks


@pytest.mark.asyncio
async def test_async_hosted_service_and_gateway_contracts(tmp_path) -> None:
    hub = InMemoryServiceHub()
    service_report = await assert_async_hosted_services_contract(
        hub.async_services(),
        first_scope=_scope("tenant-a"),
        second_scope=_scope("tenant-b"),
    )
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")

    class AsyncStore:
        async def ingest(self, *args, **kwargs):
            return await asyncio.to_thread(ledger.ingest, *args, **kwargs)

        async def claim_execution(self, **kwargs):
            return await asyncio.to_thread(ledger.claim_execution, **kwargs)

        async def complete_execution(self, *args, **kwargs):
            return await asyncio.to_thread(
                ledger.complete_execution,
                *args,
                **kwargs,
            )

        async def claim_delivery(self, **kwargs):
            return await asyncio.to_thread(ledger.claim_delivery, **kwargs)

        async def record_delivery(self, *args, **kwargs):
            return await asyncio.to_thread(
                ledger.record_delivery,
                *args,
                **kwargs,
            )

        async def reconcile_delivery(self, *args, **kwargs):
            return await asyncio.to_thread(
                ledger.reconcile_delivery,
                *args,
                **kwargs,
            )

    gateway_report = await assert_async_gateway_store_contract(
        AsyncStore(),
        target=_target(),
    )
    durable_scope = _scope("tenant-a", run_id="async-durable-contract-run")
    durable = await hub.async_services().resolve_async(durable_scope)
    durable_report = await assert_async_durable_execution_contract(
        durable.runs,
        durable.approvals,
        scope=durable_scope,
    )

    assert service_report.passed
    assert gateway_report.passed
    assert durable_report.passed
