"""DeepSeek provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.base import LLMClient, LLMConfigurationError, LLMError
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import chat_messages
from chulk.llm.pricing import estimate_cost
from chulk.llm.tools import (
    action_payload_json,
    chat_completion_tools,
    native_final_answer_payload,
    native_tool_action_payload,
    parse_native_arguments,
    public_value,
    with_json_action_prompt,
)
from chulk.llm.usage import LLMResponse, normalize_deepseek_usage


DEEPSEEK_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=True,
    supports_streaming=False,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)


class DeepSeekChatCompletionsClient(LLMClient):
    """LLM client backed by DeepSeek's OpenAI-compatible Chat Completions API."""

    capabilities = DEEPSEEK_CAPABILITIES
    provider = "deepseek"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url

        if client is not None:
            self._client = client
            return

        if not api_key:
            raise LLMConfigurationError("DEEPSEEK_API_KEY or CHULK_DEEPSEEK_API_KEY is required for DeepSeek")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The openai package is required. Install it with: pip install -e '.[openai]'"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response using DeepSeek chat completions."""
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a text response plus DeepSeek usage metadata."""
        request = {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": False,
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"DeepSeek request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek response did not include message content") from exc

        if isinstance(content, str) and content:
            return self._response_from_provider(messages, content, getattr(response, "usage", None))
        raise LLMError("DeepSeek response content was empty")

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response using DeepSeek JSON Output mode."""
        return self._complete_action_response_once(messages, max_output_tokens=max_output_tokens).content

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
    ) -> LLMResponse:
        """Return one raw action response plus DeepSeek usage metadata."""
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
        request = {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"DeepSeek structured action request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek structured action response did not include message content") from exc

        if isinstance(content, str) and content:
            result = self._response_from_provider(messages, content, getattr(response, "usage", None))
            result.metadata.update({"action_transport": "chulk_json"})
            return result
        raise LLMError("DeepSeek structured action response content was empty")

    def _complete_native_action_response_once(self, messages: list[dict[str, str]], *, tools: list[object]) -> LLMResponse:
        request = {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": False,
            "tools": chat_completion_tools(tools),
            "tool_choice": "auto",
        }
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"DeepSeek native tool action request failed: {exc}") from exc

        message = _response_message(response)
        content, raw_tool_call = _normalize_chat_native_action_message(message)
        result = self._response_from_provider(messages, content, getattr(response, "usage", None))
        result.metadata.update(
            {
                "action_transport": "provider_native",
                "provider_tool_call": raw_tool_call,
            }
        )
        return result

    def _response_from_provider(self, messages: list[dict[str, str]], content: str, usage_payload: object) -> LLMResponse:
        usage = normalize_deepseek_usage(usage_payload)
        if usage is None:
            return self._response_with_estimated_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost("deepseek", self.model, usage),
            provider="deepseek",
            model=self.model,
        )


def _response_message(response: object) -> object:
    try:
        return response.choices[0].message
    except (AttributeError, IndexError) as exc:
        raise LLMError("DeepSeek response did not include message content") from exc


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
