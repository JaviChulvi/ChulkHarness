"""Policy-owned async tool batching remains safe and deterministic."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chulk.core.state import TurnState
from chulk.core.tool_execution import ToolExecutor
from chulk.tools import ToolRegistry
from chulk.tools.permissions import ToolPermissionPolicy
from chulk.tools.policy import ToolConcurrency, ToolEffect, ToolPolicy
from chulk.tools.registry import Tool


def _executor(
    registry: ToolRegistry,
    *,
    async_usage_accounting: object | None = None,
) -> ToolExecutor:
    return ToolExecutor(
        registry=registry,
        permission_policy=ToolPermissionPolicy(),
        permission_callback=None,
        trace=lambda _name, _payload=None: None,
        get_context=lambda _turn: None,
        async_usage_accounting=async_usage_accounting,
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
async def test_parallel_batch_uses_distinct_stable_usage_indexes() -> None:
    class Usage:
        def __init__(self) -> None:
            self.active: set[tuple[str, int, int]] = set()
            self.reserved: list[int] = []
            self.committed: list[int] = []

        async def reserve_tool_call(self, **kwargs):
            key = (
                kwargs["turn_id"],
                kwargs["tool_call_index"],
                kwargs["attempt"],
            )
            if key in self.active:
                raise AssertionError(f"duplicate reservation: {key}")
            self.active.add(key)
            self.reserved.append(kwargs["tool_call_index"])
            return SimpleNamespace(
                id=f"reservation-{kwargs['tool_call_index']}",
                budget=SimpleNamespace(scope=SimpleNamespace(value="turn")),
                reserved_tool_calls=1,
            )

        async def commit_tool_call(self, **kwargs):
            key = (
                kwargs["turn_id"],
                kwargs["tool_call_index"],
                kwargs["attempt"],
            )
            self.active.remove(key)
            self.committed.append(kwargs["tool_call_index"])
            return ()

        async def release_tool_call(self, **kwargs):
            key = (
                kwargs["turn_id"],
                kwargs["tool_call_index"],
                kwargs["attempt"],
            )
            self.active.discard(key)
            return None

    async def read(arguments):
        await asyncio.sleep(0.01)
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
    usage = Usage()
    turn = TurnState("batch")
    turn.tool_call_count = 7

    results = await _executor(
        registry,
        async_usage_accounting=usage,
    ).execute_batch_async(
        [
            ("first", {"value": "one"}),
            ("second", {"value": "two"}),
        ],
        turn,
    )

    assert [result.observation for result in results] == ["one", "two"]
    assert usage.reserved == [7, 8]
    assert sorted(usage.committed) == [7, 8]
    assert not usage.active


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
