"""Shared LLM client interfaces and errors."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import inspect
import json
from typing import TYPE_CHECKING, Any, Literal

from chulk.core.actions import ActionParseError, AgentAction, parse_model_response
from chulk.core.prompts import JSON_REPAIR_PROMPT
from chulk.llm.pricing import estimate_cost
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage, aggregate_cost, aggregate_usage, estimate_usage

if TYPE_CHECKING:
    from chulk.llm.capabilities import LLMModelCapabilities
    from chulk.llm.tools import PlanningToolAvailability


LLMErrorCode = Literal[
    "action_shape_error",
    "authentication_error",
    "configuration_error",
    "connection_error",
    "fallback_exhausted",
    "invalid_request",
    "invalid_response",
    "model_not_found",
    "permission_denied",
    "rate_limit",
    "server_error",
    "timeout",
    "unknown",
    "unsupported_feature",
]


@dataclass(frozen=True)
class LLMErrorClassification:
    """Provider-neutral retry and fallback semantics for one failure."""

    code: LLMErrorCode
    retryable: bool
    fallback_eligible: bool


class LLMError(RuntimeError):
    """Base error for model provider failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        code: LLMErrorCode = "unknown",
        retryable: bool = False,
        fallback_eligible: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.code = code
        self.retryable = retryable
        self.fallback_eligible = fallback_eligible

    @property
    def error_code(self) -> LLMErrorCode:
        """Compatibility-friendly explicit name for the provider error code."""
        return self.code

    def add_context(self, *, provider: str | None = None, model: str | None = None) -> "LLMError":
        """Fill missing provider identity without discarding existing metadata."""
        if self.provider is None:
            self.provider = provider
        if self.model is None:
            self.model = model
        return self


class LLMConfigurationError(LLMError):
    """Raised when the LLM client cannot be configured."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        code: LLMErrorCode = "configuration_error",
        retryable: bool = False,
        fallback_eligible: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            code=code,
            retryable=retryable,
            fallback_eligible=fallback_eligible,
        )


class LLMActionError(LLMError):
    """Raised when an LLM cannot produce a valid agent action."""

    def __init__(
        self,
        message: str,
        *,
        repair_attempts: int = 0,
        errors: list[str] | None = None,
        raw_response: str | None = None,
        usage: LLMUsage | None = None,
        cost: LLMCost | None = None,
        provider: str | None = None,
        model: str | None = None,
        code: LLMErrorCode = "action_shape_error",
        retryable: bool = False,
        fallback_eligible: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            code=code,
            retryable=retryable,
            fallback_eligible=fallback_eligible,
        )
        self.repair_attempts = repair_attempts
        self.errors = errors or []
        self.raw_response = raw_response
        self.usage = usage
        self.cost = cost if cost is not None else estimate_cost(None, None, usage)


@dataclass(frozen=True)
class LLMActionResult:
    """Validated agent action returned by an LLM provider."""

    action: AgentAction
    raw_response: str
    repair_attempts: int = 0
    errors: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    cost: LLMCost | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStreamChunk:
    """Provider-agnostic streamed text chunk."""

    type: Literal["text_delta", "completed"]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: LLMUsage | None = None
    cost: LLMCost | None = None


class LLMClient:
    """Small provider-agnostic LLM client interface."""

    model_capabilities: LLMModelCapabilities | None = None

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a normal text response."""
        raise NotImplementedError

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Return text without blocking the event loop."""
        return (await self.acomplete_response(messages, max_output_tokens=max_output_tokens)).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text plus normalized usage metadata."""
        kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
        content = call_with_supported_kwargs(self.complete, messages, **kwargs)
        return self._response_with_estimated_usage(messages, content)

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text and usage through a native async or sync compatibility path.

        Provider clients override this method when their SDK exposes a native
        async transport. Sync-only injected clients run in asyncio's bounded
        default executor so they do not block the caller's event loop.
        """
        kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
        return await asyncio.to_thread(
            call_with_supported_kwargs,
            self.complete_response,
            messages,
            **kwargs,
        )

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield a normal text response as chunks.

        Providers without native streaming use a one-shot compatibility stream.
        """
        response = self.complete_response(messages, max_output_tokens=max_output_tokens)
        text = response.content
        if text:
            yield LLMStreamChunk(type="text_delta", text=text)
        yield LLMStreamChunk(type="completed", usage=response.usage, cost=response.cost)

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a structured JSON response."""
        return _parse_json_object(self.complete(messages), client=self)

    async def acomplete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a structured JSON response without blocking the event loop."""
        return _parse_json_object(await self.acomplete(messages), client=self)

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMActionResult:
        """Return a validated agent action using provider-native structure when available."""
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be greater than zero")

        action_messages = list(messages)
        errors: list[str] = []
        usage_records: list[LLMUsage | None] = []
        cost_records: list[LLMCost | None] = []
        for attempt in range(max_repair_attempts + 1):
            response_kwargs: dict[str, Any] = {
                "tools": tools,
                "planning_tools": planning_tools,
                "hosted_mcp_servers": hosted_mcp_servers,
                "mcp_approval_callback": mcp_approval_callback,
            }
            if max_output_tokens is not None:
                response_kwargs["max_output_tokens"] = max_output_tokens
            response = call_with_supported_kwargs(
                self._complete_action_response_once,
                action_messages,
                **response_kwargs,
            )
            try:
                return _parse_action_response(
                    response,
                    attempt=attempt,
                    errors=errors,
                    usage_records=usage_records,
                    cost_records=cost_records,
                )
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt >= max_repair_attempts:
                    raise LLMActionError(
                        f"Model response was not valid action JSON: {exc}",
                        repair_attempts=attempt,
                        errors=errors,
                        raw_response=response.content,
                        usage=aggregate_usage(usage_records),
                        cost=aggregate_cost(cost_records),
                        provider=_provider_name(self),
                        model=_model_name(self),
                    ) from exc
                action_messages = [
                    *action_messages,
                    {
                        "role": "user",
                        "content": _format_json_repair_prompt(response.content, str(exc)),
                    },
                ]

        raise LLMActionError(
            "Model response was not valid action JSON",
            provider=_provider_name(self),
            model=_model_name(self),
        )

    async def acomplete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMActionResult:
        """Return a validated action through the client's async transport."""
        if (
            type(self).complete_action is not LLMClient.complete_action
            and type(self).acomplete_action is LLMClient.acomplete_action
        ):
            compatibility_kwargs: dict[str, Any] = {
                "max_repair_attempts": max_repair_attempts,
                "tools": tools,
                "planning_tools": planning_tools,
                "hosted_mcp_servers": hosted_mcp_servers,
                "mcp_approval_callback": mcp_approval_callback,
            }
            if max_output_tokens is not None:
                compatibility_kwargs["max_output_tokens"] = max_output_tokens
            return await asyncio.to_thread(
                call_with_supported_kwargs,
                self.complete_action,
                messages,
                **compatibility_kwargs,
            )
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be greater than zero")

        action_messages = list(messages)
        errors: list[str] = []
        usage_records: list[LLMUsage | None] = []
        cost_records: list[LLMCost | None] = []
        for attempt in range(max_repair_attempts + 1):
            response_kwargs: dict[str, Any] = {
                "tools": tools,
                "planning_tools": planning_tools,
                "hosted_mcp_servers": hosted_mcp_servers,
                "mcp_approval_callback": mcp_approval_callback,
            }
            if max_output_tokens is not None:
                response_kwargs["max_output_tokens"] = max_output_tokens
            response = await call_async_with_supported_kwargs(
                self._acomplete_action_response_once,
                action_messages,
                **response_kwargs,
            )
            try:
                return _parse_action_response(
                    response,
                    attempt=attempt,
                    errors=errors,
                    usage_records=usage_records,
                    cost_records=cost_records,
                )
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt >= max_repair_attempts:
                    raise LLMActionError(
                        f"Model response was not valid action JSON: {exc}",
                        repair_attempts=attempt,
                        errors=errors,
                        raw_response=response.content,
                        usage=aggregate_usage(usage_records),
                        cost=aggregate_cost(cost_records),
                        provider=_provider_name(self),
                        model=_model_name(self),
                    ) from exc
                action_messages = [
                    *action_messages,
                    {
                        "role": "user",
                        "content": _format_json_repair_prompt(response.content, str(exc)),
                    },
                ]

        raise LLMActionError(
            "Model response was not valid action JSON",
            provider=_provider_name(self),
            model=_model_name(self),
        )

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response attempt."""
        return self.complete(messages)

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        """Return one raw action response attempt plus metadata."""
        kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
        content = call_with_supported_kwargs(self._complete_action_once, messages, **kwargs)
        return self._response_with_estimated_usage(messages, content)

    async def _acomplete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        """Compatibility hook for sync-only custom clients."""
        kwargs: dict[str, Any] = {
            "tools": tools,
            "planning_tools": planning_tools,
            "hosted_mcp_servers": hosted_mcp_servers,
            "mcp_approval_callback": mcp_approval_callback,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return await asyncio.to_thread(
            call_with_supported_kwargs,
            self._complete_action_response_once,
            messages,
            **kwargs,
        )

    def _response_with_estimated_usage(self, messages: list[dict[str, str]], content: str) -> LLMResponse:
        provider = _provider_name(self)
        model = _model_name(self)
        usage = estimate_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost(provider, model, usage),
            provider=provider,
            model=model,
        )


def _parse_json_object(raw_response: str, *, client: LLMClient) -> dict[str, Any]:
    """Parse a provider response as one JSON object."""
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LLMError(
            "Model response was not valid JSON",
            provider=_provider_name(client),
            model=_model_name(client),
            code="action_shape_error",
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMError(
            "Model JSON response must be an object",
            provider=_provider_name(client),
            model=_model_name(client),
            code="action_shape_error",
        )
    return parsed


def _parse_action_response(
    response: LLMResponse,
    *,
    attempt: int,
    errors: list[str],
    usage_records: list[LLMUsage | None],
    cost_records: list[LLMCost | None],
) -> LLMActionResult:
    """Parse one sync or async action response and aggregate its accounting."""
    usage_records.append(response.usage)
    cost_records.append(response.cost)
    return LLMActionResult(
        action=parse_model_response(response.content),
        raw_response=response.content,
        repair_attempts=attempt,
        errors=errors,
        usage=aggregate_usage(usage_records),
        cost=aggregate_cost(cost_records),
        metadata=response.metadata,
    )


def _provider_name(client: object) -> str | None:
    value = getattr(client, "provider", None) or getattr(client, "name", None)
    return str(value) if value is not None else None


def _model_name(client: object) -> str | None:
    value = getattr(client, "model", None)
    return str(value) if value is not None else None


def provider_error_from_exception(
    exc: Exception,
    *,
    message: str,
    provider: str,
    model: str | None,
    action_transport: bool = False,
) -> LLMError:
    """Normalize an SDK exception while preserving an existing Chulk error."""
    if isinstance(exc, LLMError):
        return exc.add_context(provider=provider, model=model)
    classification = classify_provider_exception(exc, action_transport=action_transport)
    return LLMError(
        f"{message}: {exc}",
        provider=provider,
        model=model,
        code=classification.code,
        retryable=classification.retryable,
        fallback_eligible=classification.fallback_eligible,
    )


def classify_provider_exception(exc: Exception, *, action_transport: bool = False) -> LLMErrorClassification:
    """Classify OpenAI-style SDK failures without requiring the SDK at import time."""
    class_names = {item.__name__ for item in type(exc).__mro__}
    status_code = _exception_status_code(exc)
    provider_code = _exception_provider_code(exc)
    message = str(exc).lower()

    if "AuthenticationError" in class_names or status_code == 401 or provider_code in {
        "authentication_error",
        "invalid_api_key",
    }:
        return LLMErrorClassification("authentication_error", retryable=False, fallback_eligible=False)
    if "PermissionDeniedError" in class_names or status_code == 403:
        return LLMErrorClassification("permission_denied", retryable=False, fallback_eligible=False)
    if "RateLimitError" in class_names or status_code == 429:
        return LLMErrorClassification("rate_limit", retryable=True, fallback_eligible=True)
    if _has_timeout_class(class_names) or isinstance(exc, TimeoutError) or status_code == 408:
        return LLMErrorClassification("timeout", retryable=True, fallback_eligible=True)
    if _has_connection_class(class_names) or isinstance(exc, ConnectionError):
        return LLMErrorClassification("connection_error", retryable=True, fallback_eligible=True)
    if "InternalServerError" in class_names or (status_code is not None and status_code >= 500):
        return LLMErrorClassification("server_error", retryable=True, fallback_eligible=True)
    if "NotFoundError" in class_names or _is_model_error(provider_code, message, status_code):
        return LLMErrorClassification("model_not_found", retryable=False, fallback_eligible=False)
    if action_transport and _is_unsupported_action_transport(provider_code, message):
        return LLMErrorClassification("unsupported_feature", retryable=False, fallback_eligible=True)
    if (
        "BadRequestError" in class_names
        or "UnprocessableEntityError" in class_names
        or status_code in {400, 404, 409, 422}
    ):
        return LLMErrorClassification("invalid_request", retryable=False, fallback_eligible=False)
    return LLMErrorClassification("unknown", retryable=False, fallback_eligible=False)


def is_action_transport_fallback_error(exc: Exception) -> bool:
    """Return whether native tool calling may safely retry via Chulk action JSON."""
    return isinstance(exc, LLMError) and exc.code in {"action_shape_error", "unsupported_feature"}


def _exception_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _exception_provider_code(exc: Exception) -> str | None:
    value = getattr(exc, "code", None)
    if isinstance(value, str) and value:
        return value.lower()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            value = error.get("code")
            if isinstance(value, str) and value:
                return value.lower()
    return None


def _is_model_error(provider_code: str | None, message: str, status_code: int | None) -> bool:
    if provider_code in {"model_not_found", "invalid_model", "unknown_model"}:
        return True
    if status_code == 404:
        return True
    model_markers = ("model not found", "model_not_found", "unknown model", "does not exist")
    return any(marker in message for marker in model_markers) and status_code in {None, 400, 404, 422}


def _is_unsupported_action_transport(provider_code: str | None, message: str) -> bool:
    if provider_code in {"unsupported_feature", "unsupported_parameter"}:
        return True
    unsupported_markers = (
        "does not support tool",
        "doesn't support tool",
        "native tools unsupported",
        "tool calling is not supported",
        "tool calls are not supported",
        "tools unsupported",
        "unsupported parameter: tools",
        "unsupported_parameter: tools",
    )
    return any(marker in message for marker in unsupported_markers)


def _has_timeout_class(class_names: set[str]) -> bool:
    """Recognize common SDK and HTTP-client timeout families by class name."""
    known_names = {
        "APITimeoutError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }
    return bool(class_names & known_names) or any(name.endswith("TimeoutError") for name in class_names)


def _has_connection_class(class_names: set[str]) -> bool:
    """Recognize common SDK and HTTP-client connection families by class name."""
    known_names = {
        "APIConnectionError",
        "ConnectError",
        "NetworkError",
        "ProxyError",
        "RemoteProtocolError",
    }
    return bool(class_names & known_names) or any(name.endswith("ConnectionError") for name in class_names)


def _format_json_repair_prompt(raw_response: str, error: str) -> str:
    return "\n".join(
        [
            JSON_REPAIR_PROMPT,
            f"Parse error: {error}",
            "Previous invalid response:",
            raw_response[:2000],
        ]
    )


def call_with_supported_kwargs(call: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a compatibility hook once, omitting only unsupported keyword arguments.

    Older injected clients may implement the original, smaller Chulk method
    signatures. Inspecting the callable before invocation preserves that
    compatibility without catching an internal ``TypeError`` and accidentally
    issuing the same provider request twice.
    """
    supported_kwargs = _supported_kwargs(call, kwargs)
    return call(*args, **supported_kwargs)


async def call_async_with_supported_kwargs(
    call: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Await a compatibility hook once, omitting unsupported keywords."""
    result = call(*args, **_supported_kwargs(call, kwargs))
    if not inspect.isawaitable(result):
        raise TypeError(f"Async compatibility hook returned a non-awaitable: {call!r}")
    return await result


def _supported_kwargs(call: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return dict(kwargs)
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {name: value for name, value in kwargs.items() if name in accepted}
