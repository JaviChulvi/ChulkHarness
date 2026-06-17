"""Local OpenAI-compatible provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.base import LLMClient, LLMConfigurationError, LLMError
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import local_chat_messages
from chulk.llm.usage import LLMResponse


DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LOCAL_API_KEY = "local"

LOCAL_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=False,
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


def _validate_base_url(value: str) -> str:
    if not value.strip():
        raise ValueError("base_url must be non-empty")
    return value.strip()
