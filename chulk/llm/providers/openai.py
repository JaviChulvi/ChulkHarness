"""OpenAI provider client."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from chulk.core.actions import STRICT_AGENT_ACTION_JSON_SCHEMA
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
    PlanningToolAvailability,
    action_payload_json,
    native_final_answer_payload,
    native_tool_action_payload,
    openai_response_tools,
    parse_native_arguments,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse, normalize_openai_usage


OPENAI_CAPABILITIES = LLMCapabilities(
    supports_structured_output=True,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    supports_hosted_mcp_tools=True,
    api_style="responses",
)


class OpenAIResponsesClient(LLMClient):
    """LLM client backed by the OpenAI Responses API."""

    capabilities = OPENAI_CAPABILITIES
    provider = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)
        self._async_client = async_client

        if client is not None:
            self._client = client
            return

        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI LLM client",
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

        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        if async_client is not None:
            self._async_client = async_client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                self._async_client = None
            else:
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    timeout=timeout_seconds,
                    max_retries=max_retries,
                )

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response using OpenAI's Responses API."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a text response plus OpenAI usage metadata."""
        request = self._text_request(messages, max_output_tokens=max_output_tokens)
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return self._response_from_provider(messages, output_text, getattr(response, "usage", None))
        raise self._invalid_response_error("OpenAI response did not include output_text")

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text through the native async Responses API."""
        if self._async_client is None:
            return await super().acomplete_response(messages, max_output_tokens=max_output_tokens)
        request = self._text_request(messages, max_output_tokens=max_output_tokens)
        async_client = self._async_client
        if async_client is None:
            raise RuntimeError("Async OpenAI client is not configured")
        try:
            response = await async_client.responses.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return self._response_from_provider(messages, output_text, getattr(response, "usage", None))
        raise self._invalid_response_error("OpenAI response did not include output_text")

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield a text response using OpenAI's Responses API streaming events."""
        request = self._text_request(messages, max_output_tokens=max_output_tokens)
        request["stream"] = True
        try:
            stream = self._client.responses.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        saw_text = False
        completed = False
        usage = None
        try:
            for event in stream:
                event_type = _event_value(event, "type")
                if event_type == "response.output_text.delta":
                    delta = _event_value(event, "delta")
                    if isinstance(delta, str) and delta:
                        saw_text = True
                        yield LLMStreamChunk(type="text_delta", text=delta, metadata={"event_type": event_type})
                    continue
                if event_type == "response.completed":
                    completed = True
                    completed_response = _event_value(event, "response")
                    usage = normalize_openai_usage(_event_value(completed_response, "usage"))
                    continue
                if event_type == "error":
                    raise LLMError(
                        f"OpenAI streaming request failed: {_event_error_message(event)}",
                        provider=self.provider,
                        model=self.model,
                        code="server_error",
                        retryable=True,
                        fallback_eligible=True,
                    )
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI streaming request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        if not saw_text:
            raise self._invalid_response_error("OpenAI streaming response did not include output text")
        if completed:
            cost = estimate_cost("openai", self.model, usage) if usage is not None else None
            yield LLMStreamChunk(
                type="completed",
                metadata={"event_type": "response.completed"},
                usage=usage,
                cost=cost,
            )
        else:
            yield LLMStreamChunk(type="completed", metadata={"event_type": "stream.closed"})

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response using OpenAI Structured Outputs."""
        return self._complete_action_response_once(messages, max_output_tokens=max_output_tokens).content

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
        """Return one raw action response plus OpenAI usage metadata."""
        if (
            tools is not None
            or bool(planning_tools and planning_tools.enabled)
            or bool(hosted_mcp_servers)
        ):
            try:
                return self._complete_native_action_response_once(
                    messages,
                    tools=tools or [],
                    planning_tools=planning_tools,
                    max_output_tokens=max_output_tokens,
                    hosted_mcp_servers=hosted_mcp_servers,
                    mcp_approval_callback=mcp_approval_callback,
                )
            except LLMError as exc:
                if hosted_mcp_servers:
                    if _hosted_mcp_can_advance_fallback(exc):
                        raise _hosted_mcp_fallback_error(exc) from exc
                    raise
                if not is_action_transport_fallback_error(exc):
                    raise
                fallback = self._complete_json_action_response_once(
                    with_json_action_prompt(messages, tools=tools),
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
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        if self._async_client is None:
            return await super()._acomplete_action_response_once(
                messages,
                max_output_tokens=max_output_tokens,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )
        if (
            tools is not None
            or bool(planning_tools and planning_tools.enabled)
            or bool(hosted_mcp_servers)
        ):
            try:
                return await self._acomplete_native_action_response_once(
                    messages,
                    tools=tools or [],
                    planning_tools=planning_tools,
                    max_output_tokens=max_output_tokens,
                    hosted_mcp_servers=hosted_mcp_servers,
                    mcp_approval_callback=mcp_approval_callback,
                )
            except LLMError as exc:
                if hosted_mcp_servers:
                    if _hosted_mcp_can_advance_fallback(exc):
                        raise _hosted_mcp_fallback_error(exc) from exc
                    raise
                if not is_action_transport_fallback_error(exc):
                    raise
                fallback = await self._acomplete_json_action_response_once(
                    with_json_action_prompt(messages, tools=tools),
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

    def _complete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        instructions, response_input = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or None,
            "input": response_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent_action",
                    "strict": True,
                    "schema": STRICT_AGENT_ACTION_JSON_SCHEMA,
                }
            },
        }
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI structured action request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            result = self._response_from_provider(messages, output_text, getattr(response, "usage", None))
            result.metadata.update({"action_transport": "chulk_json"})
            return result
        raise self._invalid_response_error("OpenAI structured action response did not include output_text")

    async def _acomplete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        instructions, response_input = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or None,
            "input": response_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent_action",
                    "strict": True,
                    "schema": STRICT_AGENT_ACTION_JSON_SCHEMA,
                }
            },
        }
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        async_client = self._async_client
        if async_client is None:
            raise RuntimeError("Async OpenAI client is not configured")
        try:
            response = await async_client.responses.create(**request)
        except Exception as exc:
            error = provider_error_from_exception(
                exc,
                message="OpenAI structured action request failed",
                provider=self.provider,
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            result = self._response_from_provider(messages, output_text, getattr(response, "usage", None))
            result.metadata.update({"action_transport": "chulk_json"})
            return result
        raise self._invalid_response_error("OpenAI structured action response did not include output_text")

    def _complete_native_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[object],
        planning_tools: PlanningToolAvailability | None = None,
        max_output_tokens: int | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        instructions, response_input = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or None,
            "input": response_input,
        }
        native_tools = openai_response_tools(
            tools,
            hosted_mcp_servers=hosted_mcp_servers,
            planning_tools=planning_tools,
        )
        if native_tools:
            request.update({"tools": native_tools, "tool_choice": "auto"})
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        response, approval_metadata = self._create_native_action_response(
            request,
            mcp_approval_callback=mcp_approval_callback,
        )

        content, metadata = _normalize_openai_native_action_response(
            response,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": metadata.get("provider_tool_call"),
                "provider_mcp_output": metadata.get("provider_mcp_output", []),
                "provider_mcp_approval": approval_metadata,
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
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> LLMResponse:
        instructions, response_input = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or None,
            "input": response_input,
        }
        native_tools = openai_response_tools(
            tools,
            hosted_mcp_servers=hosted_mcp_servers,
            planning_tools=planning_tools,
        )
        if native_tools:
            request.update({"tools": native_tools, "tool_choice": "auto"})
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        response, approval_metadata = await self._acreate_native_action_response(
            request,
            mcp_approval_callback=mcp_approval_callback,
        )

        content, metadata = _normalize_openai_native_action_response(
            response,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": metadata.get("provider_tool_call"),
                "provider_mcp_output": metadata.get("provider_mcp_output", []),
                "provider_mcp_approval": approval_metadata,
            }
        )
        return result

    def _create_native_action_response(
        self,
        request: dict[str, Any],
        *,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None,
    ) -> tuple[object, list[dict[str, Any]]]:
        approval_metadata: list[dict[str, Any]] = []
        current_request = dict(request)
        for _ in range(4):
            try:
                response = self._client.responses.create(**current_request)
            except Exception as exc:
                error = provider_error_from_exception(
                    exc,
                    message="OpenAI native tool action request failed",
                    provider=self.provider,
                    model=self.model,
                    action_transport=True,
                )
                if error is exc:
                    raise
                raise error from exc

            approval_request = _find_mcp_approval_request(response)
            if approval_request is None:
                return response, approval_metadata
            if mcp_approval_callback is None:
                raise LLMError(
                    "OpenAI MCP approval request could not be handled without a permission callback",
                    provider=self.provider,
                    model=self.model,
                    code="configuration_error",
                )

            approval_payload = public_value(approval_request)
            approval_id = _mcp_approval_request_id(approval_payload)
            if not approval_id:
                raise LLMError(
                    "OpenAI MCP approval request did not include an approval id",
                    provider=self.provider,
                    model=self.model,
                    code="invalid_response",
                )
            approved = bool(mcp_approval_callback(approval_payload))
            approval_metadata.append(
                {
                    "approval_request_id": approval_id,
                    "server_label": approval_payload.get("server_label"),
                    "name": approval_payload.get("name"),
                    "approved": approved,
                }
            )
            current_request = {
                "model": self.model,
                "previous_response_id": _response_id(
                    response,
                    provider=self.provider,
                    model=self.model,
                ),
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval_id,
                        "approve": approved,
                    }
                ],
                "tools": request["tools"],
                "tool_choice": "auto",
            }
            if request.get("max_output_tokens") is not None:
                current_request["max_output_tokens"] = request["max_output_tokens"]

        raise LLMError(
            "OpenAI MCP approval loop exceeded the maximum continuation count",
            provider=self.provider,
            model=self.model,
            code="invalid_response",
        )

    async def _acreate_native_action_response(
        self,
        request: dict[str, Any],
        *,
        mcp_approval_callback: Callable[[dict[str, Any]], bool] | None,
    ) -> tuple[object, list[dict[str, Any]]]:
        async_client = self._async_client
        if async_client is None:
            raise RuntimeError("Async OpenAI client is not configured")
        approval_metadata: list[dict[str, Any]] = []
        current_request = dict(request)
        for _ in range(4):
            try:
                response = await async_client.responses.create(**current_request)
            except Exception as exc:
                error = provider_error_from_exception(
                    exc,
                    message="OpenAI native tool action request failed",
                    provider=self.provider,
                    model=self.model,
                    action_transport=True,
                )
                if error is exc:
                    raise
                raise error from exc

            approval_request = _find_mcp_approval_request(response)
            if approval_request is None:
                return response, approval_metadata
            if mcp_approval_callback is None:
                raise LLMError(
                    "OpenAI MCP approval request could not be handled without a permission callback",
                    provider=self.provider,
                    model=self.model,
                    code="configuration_error",
                )

            approval_payload = public_value(approval_request)
            approval_id = _mcp_approval_request_id(approval_payload)
            if not approval_id:
                raise LLMError(
                    "OpenAI MCP approval request did not include an approval id",
                    provider=self.provider,
                    model=self.model,
                    code="invalid_response",
                )
            approved = bool(mcp_approval_callback(approval_payload))
            approval_metadata.append(
                {
                    "approval_request_id": approval_id,
                    "server_label": approval_payload.get("server_label"),
                    "name": approval_payload.get("name"),
                    "approved": approved,
                }
            )
            current_request = {
                "model": self.model,
                "previous_response_id": _response_id(
                    response,
                    provider=self.provider,
                    model=self.model,
                ),
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval_id,
                        "approve": approved,
                    }
                ],
                "tools": request["tools"],
                "tool_choice": "auto",
            }
            if request.get("max_output_tokens") is not None:
                current_request["max_output_tokens"] = request["max_output_tokens"]

        raise LLMError(
            "OpenAI MCP approval loop exceeded the maximum continuation count",
            provider=self.provider,
            model=self.model,
            code="invalid_response",
        )

    def _text_request(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> dict[str, Any]:
        instructions, response_input = split_instructions(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or None,
            "input": response_input,
        }
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        return request

    def _response_from_provider(self, messages: list[dict[str, str]], content: str, usage_payload: object) -> LLMResponse:
        usage = normalize_openai_usage(usage_payload)
        if usage is None:
            return self._response_with_estimated_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost("openai", self.model, usage),
            provider="openai",
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


def _validate_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("max_output_tokens must be greater than zero")
    return value


def _hosted_mcp_fallback_error(exc: LLMError) -> LLMError:
    """Allow a fallback provider to recover when JSON retry would drop hosted MCP."""
    return LLMError(
        str(exc),
        provider=exc.provider,
        model=exc.model,
        code=exc.code,
        retryable=exc.retryable,
        fallback_eligible=True,
    )


def _hosted_mcp_can_advance_fallback(exc: LLMError) -> bool:
    return (
        exc.fallback_eligible
        or exc.code == "invalid_response"
        or is_action_transport_fallback_error(exc)
    )


def _request_max_output_tokens(model_limit: int | None, request_limit: int | None) -> int | None:
    if model_limit is None:
        return _validate_max_output_tokens(request_limit)
    if request_limit is None:
        return model_limit
    return min(model_limit, _validate_max_output_tokens(request_limit) or model_limit)


def _event_value(event: object, key: str) -> object:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _event_error_message(event: object) -> str:
    error = _event_value(event, "error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    message = _event_value(event, "message")
    if isinstance(message, str) and message:
        return message
    return str(error or event)


def _normalize_openai_native_action_response(
    response: object,
    *,
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {"provider_tool_call": None, "provider_mcp_output": []}
    output = _event_value(response, "output")
    if isinstance(output, list):
        for item in output:
            item_type = _event_value(item, "type")
            if isinstance(item_type, str) and item_type.startswith("mcp_"):
                metadata["provider_mcp_output"].append(public_value(item))
                continue
            if item_type != "function_call":
                continue
            name = _event_value(item, "name")
            if not isinstance(name, str) or not name:
                raise LLMError(
                    "OpenAI native tool call did not include a function name",
                    provider=provider,
                    model=model,
                    code="action_shape_error",
                )
            try:
                arguments = parse_native_arguments(_event_value(item, "arguments"))
            except ValueError as exc:
                raise LLMError(
                    str(exc),
                    provider=provider,
                    model=model,
                    code="action_shape_error",
                ) from exc
            payload = native_tool_action_payload(name, arguments)
            metadata["provider_tool_call"] = public_value(item)
            return action_payload_json(payload), metadata

    output_text = _event_value(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return action_payload_json(native_final_answer_payload(output_text.strip())), metadata
    raise LLMError(
        "OpenAI native action response did not include a function call or output_text",
        provider=provider,
        model=model,
        code="action_shape_error",
    )


def _find_mcp_approval_request(response: object) -> object | None:
    output = _event_value(response, "output")
    if not isinstance(output, list):
        return None
    for item in output:
        if _event_value(item, "type") == "mcp_approval_request":
            return item
    return None


def _mcp_approval_request_id(payload: dict[str, Any]) -> str | None:
    for key in ("approval_request_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _response_id(response: object, *, provider: str, model: str) -> str:
    response_id = _event_value(response, "id")
    if isinstance(response_id, str) and response_id:
        return response_id
    raise LLMError(
        "OpenAI response did not include an id for MCP approval continuation",
        provider=provider,
        model=model,
        code="invalid_response",
    )
