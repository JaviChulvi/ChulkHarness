"""Run credential-free sync and async hosted-runtime embedding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from chulk import (
    AgentConfig,
    AgentEvent,
    AsyncDurableHostedExecutor,
    AsyncHostedRuntime,
    DurableHostedExecutor,
    ExecutionScope,
    HostedRuntime,
    RunSubmission,
    StepDefinition,
    Tool,
    ToolContext,
    ToolEffect,
    ToolPolicy,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.testing import ScriptedLLMClient


READ_POLICY = ToolPolicy(
    version="1.0.0",
    required_grants=frozenset({"catalog:read"}),
    effect=ToolEffect.READ,
)


@Tool(policy=READ_POLICY)
def catalog_status(item_id: str, context: ToolContext) -> str:
    """Read one item from the example application's catalog."""
    scope = context.scope
    if not isinstance(scope, ExecutionScope):
        raise RuntimeError("hosted tool received no execution scope")
    return f"{scope.tenant_id}:{item_id}:available"


def execution_scope(run_id: str) -> ExecutionScope:
    return ExecutionScope(
        tenant_id="example-tenant",
        workspace_id="example-workspace",
        actor_id="default",
        agent_id="catalog-assistant",
        agent_version="1.0.0",
        run_id=run_id,
        grants=frozenset({"catalog:read"}),
    )


def script() -> ScriptedLLMClient:
    return ScriptedLLMClient(
        [
            {
                "type": "tool_call",
                "tool_name": "catalog_status",
                "arguments": {"item_id": "SKU-42"},
            },
            {
                "type": "final_answer",
                "content": "SKU-42 is available.",
            },
        ]
    )


def submission(trigger: str) -> RunSubmission:
    return RunSubmission(
        idempotency_key=trigger,
        input_digest=f"sha256:{trigger}:input",
        definition_digest="sha256:catalog-assistant:1.0.0",
        steps=(StepDefinition(id="agent", name="Run agent turn"),),
    )


def run_sync(root: Path, hub: InMemoryServiceHub) -> str:
    events: list[AgentEvent] = []
    with HostedRuntime(
        config=AgentConfig(project_root=root),
        llm=script(),
        tools=[catalog_status],
        skills=[],
        services=hub.services(),
        execution_scope=execution_scope("sync-run"),
        on_event=events.append,
    ) as runtime:
        outcome = DurableHostedExecutor(
            runtime,
            runtime.runtime.run_store,
        ).execute(
            "Check SKU-42.",
            submission("sync-trigger"),
            worker_id="sync-worker",
            step_id="agent",
        )
        assert outcome.result is not None
        assert all(
            event.extensions.get("execution_scope_key")
            == runtime.runtime.execution_scope.key
            for event in events
        )
        return outcome.result.content


async def run_async(root: Path, hub: InMemoryServiceHub) -> str:
    runtime = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=root),
        llm=script(),
        tools=[catalog_status],
        skills=[],
        services=hub.async_services(),
        execution_scope=execution_scope("async-run"),
    )
    async with runtime:
        outcome = await AsyncDurableHostedExecutor(
            runtime,
            runtime.runtime.run_store,
        ).execute(
            "Check SKU-42.",
            submission("async-trigger"),
            worker_id="async-worker",
            step_id="agent",
        )
        assert outcome.result is not None
        return outcome.result.content


async def main() -> None:
    hub = InMemoryServiceHub()
    with TemporaryDirectory(prefix="chulk-hosted-") as temporary:
        root = Path(temporary)
        print("sync:", run_sync(root, hub))
        print("async:", await run_async(root, hub))
        assert list(root.iterdir()) == []
        print("local runtime files: 0")


if __name__ == "__main__":
    asyncio.run(main())
