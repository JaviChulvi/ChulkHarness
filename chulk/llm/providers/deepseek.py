"""DeepSeek provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.base import LLMClient, LLMConfigurationError, LLMError
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import chat_messages


DEEPSEEK_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=True,
    supports_streaming=False,
    api_style="chat_completions",
)


class DeepSeekChatCompletionsClient(LLMClient):
    """LLM client backed by DeepSeek's OpenAI-compatible Chat Completions API."""

    capabilities = DEEPSEEK_CAPABILITIES

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)

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
        request = {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": False,
        }
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_tokens"] = output_limit
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"DeepSeek request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek response did not include message content") from exc

        if isinstance(content, str) and content:
            return content
        raise LLMError("DeepSeek response content was empty")

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response using DeepSeek JSON Output mode."""
        request = {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        output_limit = _request_max_output_tokens(self.max_output_tokens, max_output_tokens)
        if output_limit is not None:
            request["max_tokens"] = output_limit
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"DeepSeek structured action request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek structured action response did not include message content") from exc

        if isinstance(content, str) and content:
            return content
        raise LLMError("DeepSeek structured action response content was empty")


def _validate_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("max_output_tokens must be greater than zero")
    return value


def _request_max_output_tokens(model_limit: int | None, request_limit: int | None) -> int | None:
    if model_limit is None:
        return _validate_max_output_tokens(request_limit)
    if request_limit is None:
        return model_limit
    return min(model_limit, _validate_max_output_tokens(request_limit) or model_limit)
