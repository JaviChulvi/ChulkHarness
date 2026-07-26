"""Run the reusable hosted-service and resilience contracts offline."""

from pathlib import Path
from tempfile import TemporaryDirectory

from chulk import ExecutionScope
from chulk.gateway import GatewayRunTarget, SQLiteGatewayLedger
from chulk.hosting.reference import InMemoryServiceHub
from chulk.testing import (
    assert_durable_execution_contract,
    assert_gateway_store_contract,
    assert_hosted_services_contract,
)


def scope(tenant_id: str, run_id: str) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id="contract-workspace",
        actor_id="contract-operator",
        agent_id="contract-agent",
        agent_version="1.0.0",
        run_id=run_id,
        conversation_id=f"conversation-{run_id}",
    )


hub = InMemoryServiceHub()
service_report = assert_hosted_services_contract(
    hub.services(),
    first_scope=scope("tenant-a", "service-run"),
    second_scope=scope("tenant-b", "service-run"),
)

durable_scope = scope("tenant-a", "durable-run")
durable_services = hub.services().resolve(durable_scope)
durable_report = assert_durable_execution_contract(
    durable_services.runs,
    durable_services.approvals,
    scope=durable_scope,
)

with TemporaryDirectory(prefix="chulk-gateway-contract-") as temporary:
    target_scope = scope("tenant-a", "gateway-run")
    gateway_report = assert_gateway_store_contract(
        SQLiteGatewayLedger(Path(temporary) / "control.sqlite"),
        target=GatewayRunTarget(
            scope=target_scope,
            definition_id=target_scope.agent_id,
            definition_version=target_scope.agent_version,
            definition_digest="sha256:contract-definition",
        ),
    )

print("services:", ", ".join(service_report.checks))
print("durability:", ", ".join(durable_report.checks))
print("gateway:", ", ".join(gateway_report.checks))
