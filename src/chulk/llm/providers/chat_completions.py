"""Shared transport for OpenAI-compatible Chat Completions providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from chulk.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMStreamChunk,
    is_action_transport_fallback_error,
    provider_error_from_exception,
)
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.lifecycle import aclose_resources, close_resources
from chulk.llm.pricing import estimate_cost
from chulk.llm.tools import (
    PlanningToolAvailability,
    action_payload_json,
    chat_completion_tools,
    native_final_answer_payload,
    native_tool_action_payload,
    parse_native_arguments,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse, LLMUsage


MessageNormalizer = Callable[[list[dict[str, str]]], list[dict[str, str]]]
UsageNormalizer = Callable[[object], LLMUsage | None]


@dataclass(frozen=True)
class ChatCompletionsTransportProfile:
    """Provider hooks used by the shared Chat Completions transport."""

    provider: str
    display_name: str
    capabilities: LLMCapabilities
    normalize_messages: MessageNormalizer
    normalize_usage: UsageNormalizer
    json_response_format: dict[str, Any] | None = None
    missing_api_key_message: str | None = None
    default_api_key: str | None = None
    max_output_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    models_supporting_required_tool_choice: frozenset[str] | None = None


class OpenAICompatibleChatCompletionsClient(LLMClient):
    """Explicit provider-neutral implementation of Chat Completions plumbing."""

    def __init__(
        self,
        *,
        profile: ChatCompletionsTransportProfile,
        model: str,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None,
        async_client: Any | None = None,
        owns_client: bool | None = None,
        owns_async_client: bool | None = None,
    ) -> None:
        self.profile = profile
        self.provider = profile.provider
        self.capabilities = profile.capabilities
        self.model = model
        self.base_url = _validate_base_url(base_url)
        self._async_client = async_client
        self._owns_client = (client is None) if owns_client is None else owns_client
        self._owns_async_client = (
            (client is None and async_client is None)
            if owns_async_client is None
            else owns_async_client
        )
        self._closed = False

        if client is not None:
            self._client = client
            return

        resolved_api_key = api_key or profile.default_api_key
        if not resolved_api_key and profile.missing_api_key_message:
            raise LLMConfigurationError(
                profile.missing_api_key_message,
                provider=self.provider,
                model=self.model,
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The openai package is required. Install it with: pip install -e '.[openai]'",
                provider=self.provider,
                model=self.model,
            ) from exc

        try:
            from openai import AsyncOpenAI
        except ImportError:
            AsyncOpenAI = None  # type: ignore[misc, assignment]

        self._client = OpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        if async_client is not None:
            self._async_client = async_client
        elif AsyncOpenAI is not None:
            try:
                self._async_client = AsyncOpenAI(
                    api_key=resolved_api_key,
                    base_url=self.base_url,
                    timeout=timeout_seconds,
                    max_retries=max_retries,
                )
            except BaseException:
                if self._owns_client:
                    close_resources((self._client,))
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_resources(self._owned_transports())

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await aclose_resources(self._owned_transports())

    def _owned_transports(self) -> tuple[object, ...]:
        resources: list[object] = []
        if self._owns_client:
            resources.append(self._client)
        if self._owns_async_client and self._async_client is not None:
            resources.append(self._async_client)
        return tuple(resources)

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response through the configured Chat Completions endpoint."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a text response plus normalized provider usage."""
        request = self._request(messages, max_output_tokens=max_output_tokens)
        response = self._create(request, operation="request")
        content = self._message_content(response, operation="response")
        return self._response_from_provider(messages, content, getattr(response, "usage", None))

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text through the native async Chat Completions transport."""
        if self._async_client is None:
            return await super().acomplete_response(messages, max_output_tokens=max_output_tokens)
        request = self._request(messages, max_output_tokens=max_output_tokens)
        response = await self._acreate(request, operation="request")
        content = self._message_content(response, operation="response")
        return self._response_from_provider(messages, content, getattr(response, "usage", None))

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield normalized chunks from a Chat Completions stream."""
        request = self._request(messages, max_output_tokens=max_output_tokens, stream=True)
        try:
            stream = self._client.chat.completions.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"{self.profile.display_name} streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        text_parts: list[str] = []
        usage_payload: object = None
        finish_reason: object = None
        try:
            for chunk in stream:
                chunk_usage = _value(chunk, "usage")
                if chunk_usage is not None:
                    usage_payload = chunk_usage
                choice = _first_choice(chunk)
                if choice is None:
                    continue
                finish_reason = _value(choice, "finish_reason") or finish_reason
                delta = _value(_value(choice, "delta"), "content")
                if isinstance(delta, str) and delta:
                    text_parts.append(delta)
                    yield LLMStreamChunk(
                        type="text_delta",
                        text=delta,
                        metadata={"transport": "chat_completions"},
                    )
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"{self.profile.display_name} streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        if not text_parts:
            raise LLMError(
                f"{self.profile.display_name} streaming response did not include message content",
                provider=self.provider,
                model=self.model,
                code="invalid_response",
                retryable=True,
                fallback_eligible=True,
            )

        content = "".join(text_parts)
        response = self._response_from_provider(messages, content, usage_payload)
        yield LLMStreamChunk(
            type="completed",
            metadata={
                "transport": "chat_completions",
                "finish_reason": finish_reason,
            },
            usage=response.usage,
            cost=response.cost,
        )

    async def astream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield chunks through the provider's native async client."""
        if self._async_client is None:
            async for chunk in super().astream_complete(
                messages, max_output_tokens=max_output_tokens
            ):
                yield chunk
            return
        request = self._request(messages, max_output_tokens=max_output_tokens, stream=True)
        try:
            stream = await self._async_client.chat.completions.create(**request)
            text_parts: list[str] = []
            usage_payload: object = None
            finish_reason: object = None
            async for chunk in stream:
                chunk_usage = _value(chunk, "usage")
                if chunk_usage is not None:
                    usage_payload = chunk_usage
                choice = _first_choice(chunk)
                if choice is None:
                    continue
                finish_reason = _value(choice, "finish_reason") or finish_reason
                delta = _value(_value(choice, "delta"), "content")
                if isinstance(delta, str) and delta:
                    text_parts.append(delta)
                    yield LLMStreamChunk(
                        type="text_delta", text=delta,
                        metadata={"transport": "chat_completions"},
                    )
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"{self.profile.display_name} streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc
        if not text_parts:
            raise LLMError(
                f"{self.profile.display_name} streaming response did not include message content",
                provider=self.provider, model=self.model, code="invalid_response",
                retryable=True, fallback_eligible=True,
            )
        response = self._response_from_provider(messages, "".join(text_parts), usage_payload)
        yield LLMStreamChunk(
            type="completed",
            metadata={"transport": "chat_completions", "finish_reason": finish_reason},
            usage=response.usage,
            cost=response.cost,
        )

    def _complete_action_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        return self._complete_action_response_once(messages, max_output_tokens=max_output_tokens).content

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        self._reject_hosted_mcp_arguments(
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        if tools is not None or bool(planning_tools and planning_tools.enabled):
            try:
                return self._complete_native_action_response_once(
                    messages,
                    tools=tools or [],
                    planning_tools=planning_tools,
                    max_output_tokens=max_output_tokens,
                )
            except LLMError as exc:
                if not is_action_transport_fallback_error(exc):
                    raise
                fallback = self._complete_json_action_response_once(
                    with_json_action_prompt(
                        messages,
                        tools=tools,
                        planning_tools=planning_tools,
                    ),
                    max_output_tokens=max_output_tokens,
                )
                fallback.metadata.update(
                    {
                        "action_transport": "chulk_json_fallback",
                        "native_tool_call_error": str(exc),
                    }
                )
                return fallback
        return self._complete_json_action_response_once(messages, max_output_tokens=max_output_tokens)

    async def _acomplete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        self._reject_hosted_mcp_arguments(
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        if self._async_client is None:
            return await super()._acomplete_action_response_once(
                messages,
                max_output_tokens=max_output_tokens,
                action_schema=action_schema,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )
        if tools is not None or bool(planning_tools and planning_tools.enabled):
            try:
                return await self._acomplete_native_action_response_once(
                    messages,
                    tools=tools or [],
                    planning_tools=planning_tools,
                    max_output_tokens=max_output_tokens,
                )
            except LLMError as exc:
                if not is_action_transport_fallback_error(exc):
                    raise
                fallback = await self._acomplete_json_action_response_once(
                    with_json_action_prompt(
                        messages,
                        tools=tools,
                        planning_tools=planning_tools,
                    ),
                    max_output_tokens=max_output_tokens,
                )
                fallback.metadata.update(
                    {
                        "action_transport": "chulk_json_fallback",
                        "native_tool_call_error": str(exc),
                    }
                )
                return fallback
        return await self._acomplete_json_action_response_once(
            messages,
            max_output_tokens=max_output_tokens,
        )

    def _reject_hosted_mcp_arguments(
        self,
        *,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None,
    ) -> None:
        unsupported: list[str] = []
        if hosted_mcp_servers:
            unsupported.append("hosted_mcp_servers")
        if mcp_approval_callback is not None:
            unsupported.append("mcp_approval_callback")
        if unsupported:
            raise LLMError(
                f"{self.profile.display_name} Chat Completions does not support "
                + " or ".join(unsupported),
                provider=self.provider,
                model=self.model,
                code="unsupported_feature",
            )

    def _complete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        if self.profile.json_response_format is not None:
            request["response_format"] = dict(self.profile.json_response_format)
        response = self._create(request, operation="structured action request")
        content = self._message_content(response, operation="structured action response")
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update({"action_transport": "chulk_json"})
        return result

    async def _acomplete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        if self.profile.json_response_format is not None:
            request["response_format"] = dict(self.profile.json_response_format)
        response = await self._acreate(request, operation="structured action request")
        content = self._message_content(response, operation="structured action response")
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update({"action_transport": "chulk_json"})
        return result

    def _complete_native_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[object],
        planning_tools: PlanningToolAvailability | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        native_tools = chat_completion_tools(
            tools,
            planning_tools=planning_tools,
        )
        if native_tools:
            request.update(
                {
                    "tools": native_tools,
                    "tool_choice": self._native_tool_choice(planning_tools),
                }
            )
        response = self._create(request, operation="native tool action request", action_transport=True)
        message = _response_message(
            response,
            display_name=self.profile.display_name,
            provider=self.provider,
            model=self.model,
            action_transport=True,
        )
        content, raw_tool_call = _normalize_native_action_message(
            message,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": raw_tool_call,
            }
        )
        return result

    async def _acomplete_native_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[object],
        planning_tools: PlanningToolAvailability | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        native_tools = chat_completion_tools(
            tools,
            planning_tools=planning_tools,
        )
        if native_tools:
            request.update(
                {
                    "tools": native_tools,
                    "tool_choice": self._native_tool_choice(planning_tools),
                }
            )
        response = await self._acreate(
            request,
            operation="native tool action request",
            action_transport=True,
        )
        message = _response_message(
            response,
            display_name=self.profile.display_name,
            provider=self.provider,
            model=self.model,
            action_transport=True,
        )
        content, raw_tool_call = _normalize_native_action_message(
            message,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": raw_tool_call,
            }
        )
        return result

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.profile.normalize_messages(messages),
            "stream": stream,
        }
        output_limit = _validate_max_output_tokens(max_output_tokens)
        if output_limit is not None:
            request[self.profile.max_output_tokens_field] = output_limit
        return request

    def _native_tool_choice(
        self,
        planning_tools: PlanningToolAvailability | None,
    ) -> str:
        if planning_tools is None or not planning_tools.enabled:
            return "auto"
        supported_models = self.profile.models_supporting_required_tool_choice
        if supported_models is None or self.model.lower() in supported_models:
            return "required"
        return "auto"

    def _create(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        action_transport: bool = False,
    ) -> object:
        try:
            return self._client.chat.completions.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"{self.profile.display_name} {operation} failed",
                provider=self.provider,
                model=self.model,
                action_transport=action_transport,
            )
            if error is exc:
                raise
            raise error from exc

    async def _acreate(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        action_transport: bool = False,
    ) -> object:
        async_client = self._async_client
        if async_client is None:
            raise RuntimeError("Async Chat Completions client is not configured")
        try:
            return await async_client.chat.completions.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"{self.profile.display_name} {operation} failed",
                provider=self.provider,
                model=self.model,
                action_transport=action_transport,
            )
            if error is exc:
                raise
            raise error from exc

    def _message_content(self, response: object, *, operation: str) -> str:
        message = _response_message(
            response,
            display_name=self.profile.display_name,
            provider=self.provider,
            model=self.model,
        )
        content = _value(message, "content")
        if isinstance(content, str) and content:
            return content
        raise LLMError(
            f"{self.profile.display_name} {operation} content was empty",
            provider=self.provider,
            model=self.model,
            code="invalid_response",
            retryable=True,
            fallback_eligible=True,
        )

    def _response_from_provider(
        self,
        messages: list[dict[str, str]],
        content: str,
        usage_payload: object,
    ) -> LLMResponse:
        usage = self.profile.normalize_usage(usage_payload)
        if usage is None:
            return self._response_with_estimated_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost(self.profile.provider, self.model, usage),
            provider=self.profile.provider,
            model=self.model,
        )


def _validate_base_url(value: str) -> str:
    if not value.strip():
        raise ValueError("base_url must be non-empty")
    return value.strip()


def _validate_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("max_output_tokens must be greater than zero")
    return value


def _response_message(
    response: object,
    *,
    display_name: str,
    provider: str,
    model: str,
    action_transport: bool = False,
) -> object:
    choice = _first_choice(response)
    message = _value(choice, "message") if choice is not None else None
    if message is None:
        raise LLMError(
            f"{display_name} response did not include message content",
            provider=provider,
            model=model,
            code="action_shape_error" if action_transport else "invalid_response",
            retryable=not action_transport,
            fallback_eligible=not action_transport,
        )
    return message


def _first_choice(response: object) -> object | None:
    choices = _value(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        return choices[0]
    return None


def _normalize_native_action_message(
    message: object,
    *,
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any] | None]:
    tool_calls = _value(message, "tool_calls")
    if isinstance(tool_calls, (list, tuple)) and tool_calls:
        if len(tool_calls) != 1:
            raise LLMError(
                "Native action response included multiple tool calls; Chulk accepts one action per turn",
                provider=provider,
                model=model,
                code="action_shape_error",
            )
        tool_call = tool_calls[0]
        function = _value(tool_call, "function")
        name = _value(function, "name")
        if not isinstance(name, str) or not name:
            raise LLMError(
                "Native tool call did not include a function name",
                provider=provider,
                model=model,
                code="action_shape_error",
            )
        try:
            arguments = parse_native_arguments(_value(function, "arguments"))
        except ValueError as exc:
            raise LLMError(
                str(exc),
                provider=provider,
                model=model,
                code="action_shape_error",
            ) from exc
        payload = native_tool_action_payload(name, arguments)
        raw_tool_call = public_value(tool_call)
        return action_payload_json(payload), raw_tool_call if isinstance(raw_tool_call, dict) else None

    content = _value(message, "content")
    if isinstance(content, str) and content.strip():
        return action_payload_json(native_final_answer_payload(content.strip())), None
    raise LLMError(
        "Native action response did not include a tool call or content",
        provider=provider,
        model=model,
        code="action_shape_error",
    )


def _value(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)
