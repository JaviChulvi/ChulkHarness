"""Tests for the stable public SDK exception contract."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from chulk import (
    Agent,
    AgentConfig,
    ChulkError,
    ConfigurationError,
    ErrorDetails,
    MemoryError,
    PermissionDeniedError,
    ProviderError,
    SafetyError,
    ToolExecutionError,
    TraceError,
)
from chulk.llm import LLMClient, LLMError
from chulk.mcp import MCPConfigError
from chulk.tools import Tool, ToolRegistry
from chulk.tools.permissions import TerminalPermissionDenied
from chulk.tools.schema import ToolValidationError, ToolValidationIssue
from chulk.tracing import Trace, TraceFormatError


class FakeLLMClient(LLMClient):
    provider = "test-provider"
    model = "test-model"

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps({"type": "final_answer", "content": "done"})


def _agent(tmp_path) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient(),
        tools=[],
        skills=[],
    )


def _raise(exc: Exception):
    raise exc


def test_hierarchy_is_available_from_stable_top_level():
    categories = {
        ConfigurationError: "configuration",
        ProviderError: "provider",
        ToolExecutionError: "tool_execution",
        PermissionDeniedError: "permission_denied",
        SafetyError: "safety",
        TraceError: "trace",
        MemoryError: "memory",
    }

    for error_type, category in categories.items():
        error = error_type("safe")
        assert isinstance(error, ChulkError)
        assert error.category == category
        assert error.to_dict()["category"] == category


def test_configuration_failure_maps_field_and_preserves_cause(tmp_path):
    with pytest.raises(ConfigurationError) as caught:
        Agent(
            config=AgentConfig(project_root=tmp_path, max_skills_per_turn=0),
            llm=FakeLLMClient(),
        )

    assert isinstance(caught.value, ChulkError)
    assert caught.value.details.invalid_field == "CHULK_MAX_SKILLS_PER_TURN"
    assert isinstance(caught.value.__cause__, ValueError)


def test_facade_maps_provider_tool_permission_safety_and_memory_failures(tmp_path):
    cases = [
        (LLMError("provider unavailable", retryable=True), ProviderError),
        (
            ToolValidationError(
                "lookup",
                [ToolValidationIssue(path="$.query", message="is required", expected="string")],
                {"type": "object"},
            ),
            ToolExecutionError,
        ),
        (TerminalPermissionDenied("write_file", "policy denied", policy_name="read-only"), PermissionDeniedError),
        (ValueError("Path is outside the project root"), SafetyError),
        (sqlite3.OperationalError("database is locked"), MemoryError),
    ]

    for index, (internal, public_type) in enumerate(cases):
        facade = _agent(tmp_path / str(index))
        facade._handle.run = lambda *args, _error=internal, **kwargs: _raise(_error)  # type: ignore[method-assign]
        with pytest.raises(public_type) as caught:
            facade.run("hello")
        assert isinstance(caught.value, ChulkError)
        assert caught.value.__cause__ is internal
        assert caught.value.details.conversation_id == facade.conversation_id
        assert caught.value.details.trace_path == str(facade.trace_path)

    provider = cases[0][0]
    assert provider.retryable is True  # type: ignore[attr-defined]


def test_provider_and_tool_details_are_structured(tmp_path):
    facade = _agent(tmp_path)
    provider_failure = LLMError("unavailable", retryable=True)
    facade._handle.run = lambda *args, **kwargs: _raise(provider_failure)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as provider_caught:
        facade.run("hello")

    assert provider_caught.value.details.provider == "test-provider"
    assert provider_caught.value.details.model == "test-model"
    assert provider_caught.value.details.retryable is True

    validation_failure = ToolValidationError(
        "lookup",
        [ToolValidationIssue(path="$.token", message="invalid", actual="token=super-secret")],
        {"type": "object"},
    )
    facade._handle.run = lambda *args, **kwargs: _raise(validation_failure)  # type: ignore[method-assign]
    with pytest.raises(ToolExecutionError) as tool_caught:
        facade.run("hello")

    details = tool_caught.value.details.to_dict()
    assert details["tool"] == "lookup"
    assert details["validation_errors"][0]["path"] == "$.token"
    assert "super-secret" not in json.dumps(details)


def test_public_error_redacts_messages_details_and_serialization():
    secrets = (
        "sk-abcdefghijklmnopqrstuvwxyz",
        "token=super-secret-token",
        "password: hunter2",
        "Authorization: Bearer abc.def.ghi",
    )
    error = ProviderError(
        "request failed with token=super-secret-token and sk-abcdefghijklmnopqrstuvwxyz",
        details=ErrorDetails(
            provider="openai",
            extensions={
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
                "nested": {"password": "hunter2", "message": "Authorization: Bearer abc.def.ghi"},
            },
        ),
    )
    serialized = json.dumps(error.to_dict(), sort_keys=True)

    for secret in secrets:
        assert secret not in str(error)
        assert secret not in serialized
    assert "[redacted]" in serialized


def test_direct_mcp_and_trace_errors_keep_legacy_bases_and_public_categories(tmp_path):
    mcp_error = MCPConfigError("bad MCP configuration")
    assert isinstance(mcp_error, ConfigurationError)
    assert isinstance(mcp_error, ValueError)

    trace_path = tmp_path / "broken.jsonl"
    trace_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(TraceFormatError) as caught:
        Trace.from_jsonl(trace_path)

    assert isinstance(caught.value, TraceError)
    assert isinstance(caught.value, ValueError)
    assert caught.value.details.trace_path == str(trace_path.resolve())
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.asyncio
async def test_async_cancellation_is_not_wrapped(tmp_path):
    from chulk import AsyncAgent

    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient(),
        tools=[],
        skills=[],
    )

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    facade._handle.run = cancel  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await facade.run("hello")


def test_recoverable_tool_failure_remains_a_result():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="explode",
            description="Fail safely",
            args_schema={"type": "object", "properties": {}, "additionalProperties": False},
            callable=lambda arguments: _raise(RuntimeError("token=secret-value")),
        )
    )

    result = registry.run("explode", {})

    assert result.success is False
    assert result.failure_kind == "environment_failure"
    assert "secret-value" not in result.observation
    assert "[redacted]" in result.observation
