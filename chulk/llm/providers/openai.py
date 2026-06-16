"""OpenAI provider client."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from chulk.core.actions import STRICT_AGENT_ACTION_JSON_SCHEMA
from chulk.llm.base import LLMClient, LLMConfigurationError, LLMError, LLMStreamChunk
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import split_instructions


OPENAI_CAPABILITIES = LLMCapabilities(
    supports_structured_output=True,
    supports_json_mode=False,
    supports_streaming=True,
    api_style="responses",
)


class OpenAIResponsesClient(LLMClient):
    """LLM client backed by the OpenAI Responses API."""

    capabilities = OPENAI_CAPABILITIES

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)

        if client is not None:
            self._client = client
            return

        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for the OpenAI LLM client")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "The openai package is required. Install it with: pip install -e '.[openai]'"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a text response using OpenAI's Responses API."""
        request = self._text_request(messages, max_output_tokens=max_output_tokens)
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        raise LLMError("OpenAI response did not include output_text")

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
            raise LLMError(f"OpenAI streaming request failed: {exc}") from exc

        saw_text = False
        completed = False
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
                continue
            if event_type == "error":
                raise LLMError(f"OpenAI streaming request failed: {_event_error_message(event)}")

        if not saw_text:
            raise LLMError("OpenAI streaming response did not include output text")
        if completed:
            yield LLMStreamChunk(type="completed", metadata={"event_type": "response.completed"})
        else:
            yield LLMStreamChunk(type="completed", metadata={"event_type": "stream.closed"})

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response using OpenAI Structured Outputs."""
        instructions, response_input = split_instructions(messages)
        request = {
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
            raise LLMError(f"OpenAI structured action request failed: {exc}") from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        raise LLMError("OpenAI structured action response did not include output_text")

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
