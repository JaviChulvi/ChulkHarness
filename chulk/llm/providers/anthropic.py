"""Anthropic Messages API provider client."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from chulk.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMStreamChunk,
    is_action_transport_fallback_error,
    provider_error_from_exception,
)
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import split_instructions
from chulk.llm.pricing import estimate_cost
from chulk.llm.tools import (
    action_payload_json,
    native_final_answer_payload,
    native_tool_action_payload,
    parse_native_arguments,
    provider_action_tools,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse, LLMUsage


DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 4096

ANTHROPIC_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    supports_hosted_mcp_tools=False,
    api_style="messages",
)


class AnthropicMessagesClient(LLMClient):
    """LLM client backed by Anthropic's native Messages API."""

    capabilities = ANTHROPIC_CAPABILITIES
    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_output_tokens: int = DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
        client: Any | None = None,
    ) -> None:
        self.model = model.strip()
        if not self.model:
            raise LLMConfigurationError(
                "CHULK_MODEL is required for the Anthropic LLM client",
                provider=self.provider,
            )
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)

        if client is not None:
            self._client = client
            return

        resolved_api_key = api_key.strip() if api_key else ""
        if not resolved_api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is required for the Anthropic LLM client",
                provider=self.provider,
                model=self.model,
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "The anthropic package is required. Install it with: pip install -e '.[anthropic]'",
                provider=self.provider,
                model=self.model,
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = Anthropic(**client_kwargs)

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response using Anthropic's Messages API."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text plus normalized Anthropic usage metadata."""
        response = self._create(
            self._request(messages, max_output_tokens=max_output_tokens),
            operation="request",
        )
        content = _response_text(
            response,
            provider=self.provider,
            model=self.model,
            action_transport=False,
        )
        return self._response_from_provider(messages, content, _value(response, "usage"))

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield normalized text chunks from an Anthropic Messages stream."""
        request = self._request(messages, max_output_tokens=max_output_tokens)
        request["stream"] = True
        try:
            stream = self._client.messages.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="Anthropic streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        text_parts: list[str] = []
        usage_parts: dict[str, int] = {}
        stop_reason: object = None
        try:
            for event in stream:
                event_type = _value(event, "type")
                if event_type == "message_start":
                    _merge_stream_usage(usage_parts, _value(_value(event, "message"), "usage"))
                    continue
                if event_type == "content_block_delta":
                    delta = _value(event, "delta")
                    if _value(delta, "type") != "text_delta":
                        continue
                    text = _value(delta, "text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                        yield LLMStreamChunk(
                            type="text_delta",
                            text=text,
                            metadata={"event_type": event_type},
                        )
                    continue
                if event_type == "message_delta":
                    _merge_stream_usage(usage_parts, _value(event, "usage"))
                    stop_reason = _value(_value(event, "delta"), "stop_reason") or stop_reason
                    continue
                if event_type == "error":
                    raise LLMError(
                        f"Anthropic streaming request failed: {_event_error_message(event)}",
                        provider=self.provider,
                        model=self.model,
                        code="server_error",
                        retryable=True,
                        fallback_eligible=True,
                    )
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="Anthropic streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        if not text_parts:
            raise self._invalid_response_error("Anthropic streaming response did not include text content")

        content = "".join(text_parts)
        response = self._response_from_provider(messages, content, usage_parts or None)
        yield LLMStreamChunk(
            type="completed",
            metadata={"event_type": "message_stop", "stop_reason": stop_reason},
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
        tools: list[object] | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        del mcp_approval_callback
        if hosted_mcp_servers:
            raise LLMError(
                "Anthropic Messages does not support Chulk hosted MCP tools",
                provider=self.provider,
                model=self.model,
                code="unsupported_feature",
            )
        if tools is not None:
            try:
                return self._complete_native_action_response_once(
                    messages,
                    tools=tools,
                    max_output_tokens=max_output_tokens,
                )
            except LLMError as exc:
                if not is_action_transport_fallback_error(exc):
                    raise
                fallback = self._complete_json_action_response_once(
                    with_json_action_prompt(messages),
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

    def _complete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        response = self._create(
            self._request(messages, max_output_tokens=max_output_tokens),
            operation="structured action request",
        )
        content = _response_text(
            response,
            provider=self.provider,
            model=self.model,
            action_transport=True,
        )
        result = self._response_from_provider(messages, content, _value(response, "usage"))
        result.metadata.update({"action_transport": "chulk_json"})
        return result

    def _complete_native_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[object],
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        request.update(
            {
                "tools": _anthropic_tools(tools),
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }
        )
        response = self._create(
            request,
            operation="native tool action request",
            action_transport=True,
        )
        content, raw_tool_call = _normalize_native_action_response(
            response,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(messages, content, _value(response, "usage"))
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
    ) -> dict[str, Any]:
        system, conversation = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_conversation(conversation),
            "max_tokens": (
                self.max_output_tokens
                if max_output_tokens is None
                else _validate_max_output_tokens(max_output_tokens)
            ),
        }
        if system:
            request["system"] = system
        return request

    def _create(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        action_transport: bool = False,
    ) -> object:
        try:
            return self._client.messages.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message=f"Anthropic {operation} failed",
                provider=self.provider,
                model=self.model,
                action_transport=action_transport,
            )
            if error is exc:
                raise
            raise error from exc

    def _response_from_provider(
        self,
        messages: list[dict[str, str]],
        content: str,
        usage_payload: object,
    ) -> LLMResponse:
        usage = normalize_anthropic_usage(usage_payload)
        if usage is None:
            return self._response_with_estimated_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost(self.provider, self.model, usage),
            provider=self.provider,
            model=self.model,
        )

    def _invalid_response_error(self, message: str) -> LLMError:
        return LLMError(
            message,
            provider=self.provider,
            model=self.model,
            code="invalid_response",
            retryable=True,
            fallback_eligible=True,
        )


def normalize_anthropic_usage(usage: object) -> LLMUsage | None:
    """Return normalized usage from an Anthropic Messages response."""
    if usage is None:
        return None
    uncached_input = _int_value(_value(usage, "input_tokens"))
    output_tokens = _int_value(_value(usage, "output_tokens"))
    cache_creation = _int_value(_value(usage, "cache_creation_input_tokens"))
    cache_read = _int_value(_value(usage, "cache_read_input_tokens"))
    input_tokens = uncached_input + cache_creation + cache_read
    if not any([input_tokens, output_tokens]):
        return None
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=cache_read,
        cache_hit_input_tokens=cache_read,
        cache_miss_input_tokens=uncached_input + cache_creation,
        estimated=False,
        cache_split_estimated=False,
        source="provider",
        raw=_public_dict(usage),
    )


def _anthropic_tools(tools: list[object]) -> list[dict[str, Any]]:
    return [
        {
            "name": declaration["name"],
            "description": declaration["description"],
            "input_schema": declaration["parameters"],
        }
        for declaration in provider_action_tools(tools)
    ]


def _normalize_native_action_response(
    response: object,
    *,
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any] | None]:
    content = _value(response, "content")
    if not isinstance(content, list):
        raise LLMError(
            "Anthropic native action response did not include content blocks",
            provider=provider,
            model=model,
            code="action_shape_error",
        )

    tool_blocks = [block for block in content if _value(block, "type") == "tool_use"]
    if len(tool_blocks) > 1:
        raise LLMError(
            "Anthropic native action response included multiple tool calls",
            provider=provider,
            model=model,
            code="action_shape_error",
        )
    if tool_blocks:
        block = tool_blocks[0]
        name = _value(block, "name")
        if not isinstance(name, str) or not name:
            raise LLMError(
                "Anthropic native tool call did not include a function name",
                provider=provider,
                model=model,
                code="action_shape_error",
            )
        try:
            arguments = parse_native_arguments(_value(block, "input"))
        except ValueError as exc:
            raise LLMError(
                str(exc),
                provider=provider,
                model=model,
                code="action_shape_error",
            ) from exc
        return action_payload_json(native_tool_action_payload(name, arguments)), public_value(block)

    text = _text_from_content(content)
    if text:
        return action_payload_json(native_final_answer_payload(text)), None
    raise LLMError(
        "Anthropic native action response did not include a tool call or text content",
        provider=provider,
        model=model,
        code="action_shape_error",
    )


def _response_text(
    response: object,
    *,
    provider: str,
    model: str,
    action_transport: bool,
) -> str:
    content = _value(response, "content")
    text = _text_from_content(content)
    if text:
        return text
    raise LLMError(
        "Anthropic response did not include text content",
        provider=provider,
        model=model,
        code="action_shape_error" if action_transport else "invalid_response",
        retryable=not action_transport,
        fallback_eligible=True,
    )


def _text_from_content(content: object) -> str:
    if not isinstance(content, list):
        return ""
    parts = [
        text
        for block in content
        if _value(block, "type") == "text"
        for text in [_value(block, "text")]
        if isinstance(text, str) and text
    ]
    return "".join(parts).strip()


def _normalize_conversation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            role = "user"
        content = message.get("content", "")
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = "\n\n".join([normalized[-1]["content"], content])
        else:
            normalized.append({"role": role, "content": content})
    if not normalized:
        normalized.append({"role": "user", "content": "Continue from the available context."})
    return normalized


def _merge_stream_usage(target: dict[str, int], usage: object) -> None:
    if usage is None:
        return
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = _int_value(_value(usage, key))
        if value or key not in target:
            target[key] = value


def _event_error_message(event: object) -> str:
    error = _value(event, "error")
    message = _value(error, "message")
    if isinstance(message, str) and message:
        return message
    return str(error or event)


def _validate_max_output_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_output_tokens must be greater than zero")
    return value


def _value(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _int_value(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _public_dict(value: object) -> dict[str, Any]:
    public = public_value(value)
    return public if isinstance(public, dict) else {}
