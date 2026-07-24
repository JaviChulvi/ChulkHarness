"""Native Google Gemini provider client."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

from chulk.core.actions import STRICT_AGENT_ACTION_JSON_SCHEMA
from chulk.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMErrorCode,
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
    native_final_answer_payload,
    native_tool_action_payload,
    parse_native_arguments,
    provider_action_tools,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse, LLMUsage


GEMINI_CAPABILITIES = LLMCapabilities(
    supports_structured_output=True,
    supports_json_mode=True,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="generate_content",
)


class GeminiGenerateContentClient(LLMClient):
    """LLM client backed by the native ``google-genai`` GenerateContent API."""

    capabilities = GEMINI_CAPABILITIES
    provider = "gemini"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        async_client: Any | None = None,
        owns_client: bool | None = None,
        owns_async_client: bool | None = None,
    ) -> None:
        if not model.strip():
            raise LLMConfigurationError(
                "CHULK_MODEL is required for the Gemini LLM client",
                provider=self.provider,
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.model = model.strip()
        self.base_url = base_url.strip() if base_url is not None else None
        self._async_client = async_client
        self._owns_client = (client is None) if owns_client is None else owns_client
        self._owns_async_client = (
            (client is None and async_client is None)
            if owns_async_client is None
            else owns_async_client
        )
        self._closed = False
        if base_url is not None and not self.base_url:
            raise ValueError("base_url must be non-empty when provided")

        if client is not None:
            self._client = client
            if async_client is None:
                self._async_client = getattr(client, "aio", None)
            return

        resolved_api_key = api_key.strip() if api_key else ""
        if not resolved_api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY or CHULK_GEMINI_API_KEY is required for Gemini",
                provider=self.provider,
                model=self.model,
            )

        try:
            from google import genai
        except ImportError as exc:
            raise LLMConfigurationError(
                "The google-genai package is required. Install it with: "
                "pip install -e '.[gemini]'",
                provider=self.provider,
                model=self.model,
            ) from exc

        http_options: dict[str, Any] = {
            "timeout": max(1, round(timeout_seconds * 1000)),
            "retry_options": {"attempts": max_retries + 1},
        }
        if self.base_url is not None:
            http_options["base_url"] = self.base_url
        self._client = genai.Client(
            api_key=resolved_api_key,
            http_options=cast(Any, http_options),
        )
        self._async_client = async_client or getattr(self._client, "aio", None)

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

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Return a text response using Gemini GenerateContent."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text plus normalized Gemini usage metadata."""
        request = self._request(messages, max_output_tokens=max_output_tokens)
        response = self._generate(request, operation="request")
        content = _response_text(response)
        if not content:
            raise self._invalid_response_error("Gemini response did not include text")
        return self._response_from_provider(messages, content, _value(response, "usage_metadata"))

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text through Gemini's native async GenerateContent API."""
        if self._async_client is None:
            return await super().acomplete_response(messages, max_output_tokens=max_output_tokens)
        request = self._request(messages, max_output_tokens=max_output_tokens)
        response = await self._agenerate(request, operation="request")
        content = _response_text(response)
        if not content:
            raise self._invalid_response_error("Gemini response did not include text")
        return self._response_from_provider(messages, content, _value(response, "usage_metadata"))

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield normalized text chunks from Gemini's native streaming API."""
        request = self._request(messages, max_output_tokens=max_output_tokens)
        try:
            stream = self._client.models.generate_content_stream(**request)
        except Exception as exc:
            error = _gemini_error_from_exception(
                exc,
                message="Gemini streaming request failed",
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        text_parts: list[str] = []
        usage_payload: object = None
        finish_reason: str | None = None
        try:
            for chunk in stream:
                usage = _value(chunk, "usage_metadata")
                if usage is not None:
                    usage_payload = usage
                current_finish_reason = _finish_reason(chunk)
                if current_finish_reason is not None:
                    finish_reason = current_finish_reason
                text = _response_text(chunk, strip=False)
                if text:
                    text_parts.append(text)
                    yield LLMStreamChunk(
                        type="text_delta",
                        text=text,
                        metadata={"transport": "generate_content"},
                    )
        except Exception as exc:
            error = _gemini_error_from_exception(
                exc,
                message="Gemini streaming request failed",
                model=self.model,
            )
            if error is exc:
                raise
            raise error from exc

        if not text_parts:
            raise self._invalid_response_error("Gemini streaming response did not include text")

        response = self._response_from_provider(messages, "".join(text_parts), usage_payload)
        yield LLMStreamChunk(
            type="completed",
            metadata={
                "transport": "generate_content",
                "finish_reason": finish_reason,
            },
            usage=response.usage,
            cost=response.cost,
        )

    def _complete_action_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        return self._complete_action_response_once(
            messages,
            max_output_tokens=max_output_tokens,
        ).content

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
        if hosted_mcp_servers:
            raise LLMError(
                "Gemini does not support Chulk hosted MCP tools",
                provider=self.provider,
                model=self.model,
                code="unsupported_feature",
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
                    action_schema=action_schema,
                )
                fallback.metadata.update(
                    {
                        "action_transport": "chulk_json_fallback",
                        "native_tool_call_error": str(exc),
                    }
                )
                return fallback
        return self._complete_json_action_response_once(
            messages,
            max_output_tokens=max_output_tokens,
            action_schema=action_schema,
        )

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
        if hosted_mcp_servers:
            raise LLMError(
                "Gemini does not support Chulk hosted MCP tools",
                provider=self.provider,
                model=self.model,
                code="unsupported_feature",
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
                    action_schema=action_schema,
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
            action_schema=action_schema,
        )

    def _complete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        request["config"].update(
            {
                "response_mime_type": "application/json",
                "response_json_schema": action_schema or STRICT_AGENT_ACTION_JSON_SCHEMA,
            }
        )
        response = self._generate(request, operation="structured action request")
        content = _response_text(response)
        if not content:
            raise LLMError(
                "Gemini structured action response did not include JSON text",
                provider=self.provider,
                model=self.model,
                code="action_shape_error",
                fallback_eligible=True,
            )
        result = self._response_from_provider(
            messages,
            content,
            _value(response, "usage_metadata"),
        )
        result.metadata.update({"action_transport": "chulk_json"})
        return result

    async def _acomplete_json_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        request = self._request(messages, max_output_tokens=max_output_tokens)
        request["config"].update(
            {
                "response_mime_type": "application/json",
                "response_json_schema": action_schema or STRICT_AGENT_ACTION_JSON_SCHEMA,
            }
        )
        response = await self._agenerate(request, operation="structured action request")
        content = _response_text(response)
        if not content:
            raise LLMError(
                "Gemini structured action response did not include JSON text",
                provider=self.provider,
                model=self.model,
                code="action_shape_error",
                fallback_eligible=True,
            )
        result = self._response_from_provider(
            messages,
            content,
            _value(response, "usage_metadata"),
        )
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
        request["config"].update(
            _native_tool_config(tools, planning_tools=planning_tools)
        )
        response = self._generate(
            request,
            operation="native tool action request",
            action_transport=True,
        )
        content, raw_tool_call = _normalize_native_action_response(
            response,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(
            messages,
            content,
            _value(response, "usage_metadata"),
        )
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
        request["config"].update(
            _native_tool_config(tools, planning_tools=planning_tools)
        )
        response = await self._agenerate(
            request,
            operation="native tool action request",
            action_transport=True,
        )
        content, raw_tool_call = _normalize_native_action_response(
            response,
            provider=self.provider,
            model=self.model,
        )
        result = self._response_from_provider(
            messages,
            content,
            _value(response, "usage_metadata"),
        )
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
        system_instruction, contents = _gemini_messages(messages)
        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        output_limit = _validate_max_output_tokens(max_output_tokens)
        if output_limit is not None:
            config["max_output_tokens"] = output_limit
        return {
            "model": self.model,
            "contents": contents,
            "config": config,
        }

    def _generate(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        action_transport: bool = False,
    ) -> object:
        try:
            return self._client.models.generate_content(**request)
        except Exception as exc:
            error = _gemini_error_from_exception(
                exc,
                message=f"Gemini {operation} failed",
                model=self.model,
                action_transport=action_transport,
            )
            if error is exc:
                raise
            raise error from exc

    async def _agenerate(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        action_transport: bool = False,
    ) -> object:
        async_client = self._async_client
        if async_client is None:
            raise RuntimeError("Async Gemini client is not configured")
        try:
            return await async_client.models.generate_content(**request)
        except Exception as exc:
            error = _gemini_error_from_exception(
                exc,
                message=f"Gemini {operation} failed",
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
        usage = normalize_gemini_usage(usage_payload)
        if usage is None:
            estimated = self._response_with_estimated_usage(messages, content)
            return LLMResponse(
                content=estimated.content,
                usage=estimated.usage,
                cost=estimated.cost,
                provider=self.provider,
                model=self.model,
            )
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


def normalize_gemini_usage(usage: object) -> LLMUsage | None:
    """Normalize Gemini ``usage_metadata`` into Chulk usage accounting."""
    if usage is None:
        return None
    prompt_tokens = _int_value(_value(usage, "prompt_token_count"))
    tool_use_tokens = _int_value(_value(usage, "tool_use_prompt_token_count"))
    input_tokens = prompt_tokens + tool_use_tokens
    candidate_tokens = _int_value(_value(usage, "candidates_token_count"))
    reasoning_tokens = _int_value(_value(usage, "thoughts_token_count"))
    output_tokens = candidate_tokens + reasoning_tokens
    total_tokens = _int_value(_value(usage, "total_token_count")) or input_tokens + output_tokens
    cached_tokens = _int_value(_value(usage, "cached_content_token_count"))
    if not any([input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens]):
        return None
    raw = public_value(usage)
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_tokens,
        cache_hit_input_tokens=cached_tokens,
        cache_miss_input_tokens=max(input_tokens - cached_tokens, 0),
        reasoning_tokens=reasoning_tokens,
        estimated=False,
        source="provider",
        raw=raw if isinstance(raw, dict) else {},
    )


def _gemini_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    instruction_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            if content:
                instruction_parts.append(content)
            continue
        if role == "assistant":
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            gemini_role = "user"
            content = f"{role or 'message'}: {content}"
        _append_content(contents, gemini_role, content)

    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Continue."}]})
    return "\n\n".join(instruction_parts), contents


def _append_content(contents: list[dict[str, Any]], role: str, text: str) -> None:
    if contents and contents[-1]["role"] == role:
        current_text = str(contents[-1]["parts"][0]["text"])
        contents[-1]["parts"][0]["text"] = "\n\n".join([current_text, text])
        return
    contents.append({"role": role, "parts": [{"text": text}]})


def _native_tool_config(
    tools: list[object],
    *,
    planning_tools: PlanningToolAvailability | None = None,
) -> dict[str, Any]:
    declarations = [
        {
            "name": declaration["name"],
            "description": declaration["description"],
            "parameters_json_schema": declaration["parameters"],
        }
        for declaration in provider_action_tools(
            tools,
            planning_tools=planning_tools,
        )
    ]
    if not declarations:
        return {}
    planning_required = planning_tools is not None and planning_tools.enabled
    return {
        "tools": [{"function_declarations": declarations}],
        "tool_config": {
            "function_calling_config": {
                "mode": "ANY" if planning_required else "AUTO"
            }
        },
        "automatic_function_calling": {"disable": True},
    }


def _normalize_native_action_response(
    response: object,
    *,
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any] | None]:
    function_calls = _function_calls(response)
    if function_calls:
        if len(function_calls) != 1:
            raise LLMError(
                "Gemini native action response included multiple function calls; "
                "Chulk accepts one action per turn",
                provider=provider,
                model=model,
                code="action_shape_error",
            )
        function_call = function_calls[0]
        name = _value(function_call, "name")
        if not isinstance(name, str) or not name:
            raise LLMError(
                "Gemini native function call did not include a function name",
                provider=provider,
                model=model,
                code="action_shape_error",
            )
        raw_arguments = _value(function_call, "args")
        if not isinstance(raw_arguments, (dict, str)):
            public_arguments = public_value(raw_arguments)
            raw_arguments = public_arguments if isinstance(public_arguments, dict) else raw_arguments
        try:
            arguments = parse_native_arguments(raw_arguments)
        except ValueError as exc:
            raise LLMError(
                str(exc),
                provider=provider,
                model=model,
                code="action_shape_error",
            ) from exc
        payload = native_tool_action_payload(name, arguments)
        raw_tool_call = public_value(function_call)
        return (
            action_payload_json(payload),
            raw_tool_call if isinstance(raw_tool_call, dict) else None,
        )

    text = _response_text(response)
    if text:
        return action_payload_json(native_final_answer_payload(text)), None
    raise LLMError(
        "Gemini native action response did not include a function call or text",
        provider=provider,
        model=model,
        code="action_shape_error",
    )


def _function_calls(response: object) -> list[object]:
    direct = _safe_value(response, "function_calls")
    if isinstance(direct, (list, tuple)):
        return list(direct)

    calls: list[object] = []
    candidates = _safe_value(response, "candidates")
    if not isinstance(candidates, (list, tuple)):
        return calls
    for candidate in candidates:
        content = _safe_value(candidate, "content")
        parts = _safe_value(content, "parts")
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            function_call = _safe_value(part, "function_call")
            if function_call is not None:
                calls.append(function_call)
    return calls


def _response_text(response: object, *, strip: bool = True) -> str:
    direct = _safe_value(response, "text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip() if strip else direct

    text_parts: list[str] = []
    candidates = _safe_value(response, "candidates")
    if not isinstance(candidates, (list, tuple)):
        return ""
    for candidate in candidates:
        content = _safe_value(candidate, "content")
        parts = _safe_value(content, "parts")
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            text = _safe_value(part, "text")
            if isinstance(text, str) and text:
                text_parts.append(text)
    joined = "".join(text_parts)
    return joined.strip() if strip else joined


def _finish_reason(response: object) -> str | None:
    candidates = _safe_value(response, "candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    finish_reason = _safe_value(candidates[0], "finish_reason")
    if finish_reason is None:
        return None
    value = getattr(finish_reason, "value", finish_reason)
    return str(value)


def _gemini_error_from_exception(
    exc: Exception,
    *,
    message: str,
    model: str,
    action_transport: bool = False,
) -> LLMError:
    status_code = _gemini_status_code(exc)
    error_text = str(exc).lower()
    if isinstance(exc, LLMError):
        return provider_error_from_exception(
            exc,
            message=message,
            provider="gemini",
            model=model,
            action_transport=action_transport,
        )

    if _has_any(error_text, "api key not valid", "invalid api key", "api_key_invalid"):
        classification: tuple[LLMErrorCode, bool, bool] = (
            "authentication_error",
            False,
            False,
        )
    elif action_transport and status_code in {None, 400, 409, 422} and _has_any(
        error_text,
        "function calling is not supported",
        "function calls are not supported",
        "tools are not supported",
        "unsupported function calling",
    ):
        classification = ("unsupported_feature", False, True)
    else:
        normalized = provider_error_from_exception(
            exc,
            message=message,
            provider="gemini",
            model=model,
            action_transport=action_transport,
        )
        if normalized.code != "unknown":
            return normalized

        if status_code == 401:
            classification = ("authentication_error", False, False)
        elif status_code == 403:
            classification = ("permission_denied", False, False)
        elif status_code == 404 or _has_any(
            error_text,
            "model not found",
            "unknown model",
            "model does not exist",
        ):
            classification = ("model_not_found", False, False)
        elif status_code == 429:
            classification = ("rate_limit", True, True)
        elif status_code == 408:
            classification = ("timeout", True, True)
        elif status_code is not None and status_code >= 500:
            classification = ("server_error", True, True)
        elif status_code in {400, 409, 422}:
            classification = ("invalid_request", False, False)
        else:
            return normalized

    code, retryable, fallback_eligible = classification
    return LLMError(
        f"{message}: {exc}",
        provider="gemini",
        model=model,
        code=code,
        retryable=retryable,
        fallback_eligible=fallback_eligible,
    )


def _gemini_status_code(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def _has_any(value: str, *markers: str) -> bool:
    return any(marker in value for marker in markers)


def _validate_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or value < 1:
        raise ValueError("max_output_tokens must be greater than zero")
    return value


def _safe_value(source: object, key: str) -> object:
    try:
        return _value(source, key)
    except Exception:
        return None


def _value(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _int_value(value: object) -> int:
    if value is None:
        return 0
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
