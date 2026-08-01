"""Policy-owned async tool batching remains safe and deterministic."""

from __future__ import annotations

import asyncio

import pytest

from chulk.core.state import TurnState
from chulk.core.tool_execution import ToolExecutor
from chulk.tools import ToolRegistry
from chulk.tools.permissions import ToolPermissionPolicy
from chulk.tools.policy import ToolConcurrency, ToolEffect, ToolPolicy
from chulk.tools.registry import Tool


def _executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(
        registry=registry,
        permission_policy=ToolPermissionPolicy(),
        permission_callback=None,
        trace=lambda _name, _payload=None: None,
        get_context=lambda _turn: None,
    )


@pytest.mark.asyncio
async def test_parallel_safe_read_batch_runs_concurrently_in_input_order() -> None:
    active = 0
    maximum = 0

    async def read(arguments):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return arguments["value"]

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    registry = ToolRegistry()
    for name in ("first", "second"):
        registry.register(
            Tool(
                name=name,
                description="Independent read.",
                args_schema=schema,
                callable=read,
                policy=ToolPolicy(
                    effect=ToolEffect.READ,
                    concurrency=ToolConcurrency.PARALLEL_SAFE,
                ),
            )
        )

    results = await _executor(registry).execute_batch_async(
        [
            ("first", {"value": "one"}),
            ("second", {"value": "two"}),
        ],
        TurnState("batch"),
    )

    assert maximum == 2
    assert [result.observation for result in results] == ["one", "two"]


@pytest.mark.asyncio
async def test_one_mutating_tool_makes_the_complete_batch_serial() -> None:
    active = 0
    maximum = 0
    order: list[str] = []

    async def execute(arguments):
        nonlocal active, maximum
        name = arguments["value"]
        order.append(f"start:{name}")
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        order.append(f"end:{name}")
        return name

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read",
            description="Read.",
            args_schema=schema,
            callable=execute,
            policy=ToolPolicy(
                effect=ToolEffect.READ,
                concurrency=ToolConcurrency.PARALLEL_SAFE,
            ),
        )
    )
    registry.register(
        Tool(
            name="write",
            description="Write.",
            args_schema=schema,
            callable=execute,
            policy=ToolPolicy(effect=ToolEffect.EXTERNAL_WRITE),
        )
    )

    results = await _executor(registry).execute_batch_async(
        [
            ("read", {"value": "one"}),
            ("write", {"value": "two"}),
        ],
        TurnState("batch"),
    )

    assert maximum == 1
    assert order == ["start:one", "end:one", "start:two", "end:two"]
    assert [result.observation for result in results] == ["one", "two"]
