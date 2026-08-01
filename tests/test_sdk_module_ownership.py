"""Identity guards for classes owned by focused SDK modules."""

import pytest

import chulk
import chulk.api as public_api
from chulk._sdk import facade
from chulk._sdk.agent import Agent
from chulk._sdk.async_agent import AsyncAgent
from chulk._sdk.handles import AgentHandle, AsyncAgentHandle
from chulk._sdk.hosted import AsyncHostedRuntime, HostedRuntime


@pytest.mark.parametrize(
    ("name", "owner"),
    [
        ("Agent", Agent),
        ("AsyncAgent", AsyncAgent),
        ("AgentHandle", AgentHandle),
        ("AsyncAgentHandle", AsyncAgentHandle),
        ("HostedRuntime", HostedRuntime),
        ("AsyncHostedRuntime", AsyncHostedRuntime),
    ],
)
def test_sdk_exports_share_the_owner_class_identity(name: str, owner: type) -> None:
    assert getattr(chulk, name) is owner
    assert getattr(public_api, name) is owner
    assert getattr(facade, name) is owner
