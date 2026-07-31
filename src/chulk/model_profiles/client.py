"""Request-scoped provider client that refreshes referenced connection values."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

from chulk.llm import (
    LLMActionResult,
    LLMClient,
    LLMModelCapabilities,
    LLMResponse,
    LLMStreamChunk,
)
from chulk.llm.lifecycle import aclose_resources, close_resources
from chulk.llm.tools import PlanningToolAvailability


@dataclass(frozen=True, slots=True)
class RequestClientLease:
    """One provider client plus values that must never enter error output."""

    client: LLMClient
    sensitive_values: tuple[str, ...] = ()


class RefreshingLLMClient(LLMClient):
    """Create and close a bound provider client around every model request."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        model_profile_id: str,
        credential_ref: str | None,
        model_capabilities: LLMModelCapabilities,
        client_factory: Callable[[], RequestClientLease],
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_profile_id = model_profile_id
        self.credential_ref = credential_ref
        self.model_capabilities = model_capabilities
        self._client_factory = client_factory

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        return self.complete_response(
            messages,
            max_output_tokens=max_output_tokens,
        ).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        with self._lease() as client:
            return client.complete_response(
                messages,
                max_output_tokens=max_output_tokens,
            )

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        async with self._alease() as client:
            return await client.acomplete_response(
                messages,
                max_output_tokens=max_output_tokens,
            )

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        with self._lease() as client:
            yield from client.stream_complete(
                messages,
                max_output_tokens=max_output_tokens,
            )

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMActionResult:
        with self._lease() as client:
            return client.complete_action(
                messages,
                max_repair_attempts=max_repair_attempts,
                max_output_tokens=max_output_tokens,
                action_schema=action_schema,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )

    async def acomplete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMActionResult:
        async with self._alease() as client:
            return await client.acomplete_action(
                messages,
                max_repair_attempts=max_repair_attempts,
                max_output_tokens=max_output_tokens,
                action_schema=action_schema,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )

    @contextmanager
    def _lease(self) -> Iterator[LLMClient]:
        lease = self._client_factory()
        try:
            yield lease.client
        except Exception as exc:
            _redact_exception(exc, lease.sensitive_values)
            raise
        finally:
            close_resources((lease.client,))

    @asynccontextmanager
    async def _alease(self) -> AsyncIterator[LLMClient]:
        lease = self._client_factory()
        try:
            yield lease.client
        except Exception as exc:
            _redact_exception(exc, lease.sensitive_values)
            raise
        finally:
            await aclose_resources((lease.client,))


def _redact_exception(exc: Exception, values: tuple[str, ...]) -> None:
    clean_values = tuple(value for value in values if value)
    if not clean_values:
        return
    redacted_args = []
    for argument in exc.args:
        if not isinstance(argument, str):
            redacted_args.append(argument)
            continue
        redacted = argument
        for value in clean_values:
            redacted = redacted.replace(value, "[REDACTED]")
        redacted_args.append(redacted)
    exc.args = tuple(redacted_args)


__all__ = ["RefreshingLLMClient", "RequestClientLease"]
