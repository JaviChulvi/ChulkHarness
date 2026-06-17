"""Local OpenAI-compatible provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.base import LLMClient, LLMConfigurationError, LLMError
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import local_chat_messages
from chulk.llm.tools import (
    action_payload_json,
    chat_completion_tools,
    native_final_answer_payload,
    native_tool_action_payload,
    parse_native_arguments,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse


DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LOCAL_API_KEY = "local"

LOCAL_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=False,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)


class LocalOpenAICompatibleClient(LLMClient):
    """LLM client backed by a local OpenAI-compatible chat-completions server."""

    capabilities = LOCAL_CAPABILITIES
    provider = "local"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_LOCAL_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = _validate_base_url(base_url)

        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The openai package is required. Install it with: pip install -e '.[openai]'"
            ) from exc

        self._client = OpenAI(
            api_key=api_key or DEFAULT_LOCAL_API_KEY,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response using a local OpenAI-compatible chat endpoint."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a text response with estimated token usage."""
        request = {
            "model": self.model,
            "messages": local_chat_messages(messages),
            "stream": False,
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"Local LLM request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("Local LLM response did not include message content") from exc

        if isinstance(content, str) and content:
            return self._response_with_estimated_usage(messages, content)
        raise LLMError("Local LLM response content was empty")

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response from the local model."""
        return self.complete(messages)

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
    ) -> LLMResponse:
        """Return one raw action response, using native local tool calls by default."""
        if tools is not None:
            try:
                return self._complete_native_action_response_once(messages, tools=tools)
            except LLMError as exc:
                fallback = self._complete_json_action_response_once(with_json_action_prompt(messages))
                fallback.metadata.update(
                    {
                        "action_transport": "chulk_json_fallback",
                        "native_tool_call_error": str(exc),
                    }
                )
                return fallback
        return self._complete_json_action_response_once(messages)

    def _complete_json_action_response_once(self, messages: list[dict[str, str]]) -> LLMResponse:
        content = self.complete(messages)
        response = self._response_with_estimated_usage(messages, content)
        response.metadata.update({"action_transport": "chulk_json"})
        return response

    def _complete_native_action_response_once(self, messages: list[dict[str, str]], *, tools: list[object]) -> LLMResponse:
        request = {
            "model": self.model,
            "messages": local_chat_messages(messages),
            "stream": False,
            "tools": chat_completion_tools(tools),
            "tool_choice": "auto",
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"Local native tool action request failed: {exc}") from exc

        message = _response_message(response)
        content, raw_tool_call = _normalize_chat_native_action_message(message)
        result = self._response_with_estimated_usage(messages, content)
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": raw_tool_call,
            }
        )
        return result


def _validate_base_url(value: str) -> str:
    if not value.strip():
        raise ValueError("base_url must be non-empty")
    return value.strip()


def _response_message(response: object) -> object:
    try:
        return response.choices[0].message
    except (AttributeError, IndexError) as exc:
        raise LLMError("Local LLM response did not include message content") from exc


def _normalize_chat_native_action_message(message: object) -> tuple[str, dict | None]:
    tool_calls = _value(message, "tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        tool_call = tool_calls[0]
        function = _value(tool_call, "function")
        name = _value(function, "name")
        if not isinstance(name, str) or not name:
            raise LLMError("Native tool call did not include a function name")
        arguments = parse_native_arguments(_value(function, "arguments"))
        payload = native_tool_action_payload(name, arguments)
        return action_payload_json(payload), public_value(tool_call)

    content = _value(message, "content")
    if isinstance(content, str) and content.strip():
        return action_payload_json(native_final_answer_payload(content.strip())), None
    raise LLMError("Native action response did not include a tool call or content")


def _value(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)
