"""Identity guards for classes owned by focused SDK modules."""

import inspect

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


def test_async_hosted_create_exposes_closed_typed_signature() -> None:
    parameters = inspect.signature(AsyncHostedRuntime.create).parameters

    assert list(parameters) == [
        "services",
        "execution_scope",
        "config",
        "preset",
        "llm",
        "tools",
        "skills",
        "system_prompt",
        "conversation_id",
        "conversation_metadata",
        "runtime_metadata",
        "permission_callback",
        "plan_step_verifier",
        "async_plan_step_verifier",
        "on_event",
        "mcp",
        "redaction_callback",
        "redaction_fail_closed",
        "final_answer_streaming",
        "output_policy",
        "async_output_policy",
        "output_policy_failure_mode",
        "capabilities",
        "memory_mode",
        "deps",
        "shell_execution_policy",
        "require_shell_containment",
        "run_budget",
        "usage_dimensions",
        "goal_execution",
        "async_transcript_resolver",
        "transcript_timeout_seconds",
        "async_tool_catalog_resolver",
        "tool_catalog_timeout_seconds",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert all(
        parameter.annotation is not inspect.Parameter.empty
        for parameter in parameters.values()
    )
