"""Shared LLM client interfaces and errors."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from chulk.core.actions import ActionParseError, AgentAction, parse_model_response
from chulk.core.prompts import JSON_REPAIR_PROMPT


class LLMError(RuntimeError):
    """Base error for model provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM client cannot be configured."""


class LLMActionError(LLMError):
    """Raised when an LLM cannot produce a valid agent action."""

    def __init__(
        self,
        message: str,
        *,
        repair_attempts: int = 0,
        errors: list[str] | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.repair_attempts = repair_attempts
        self.errors = errors or []
        self.raw_response = raw_response


@dataclass(frozen=True)
class LLMActionResult:
    """Validated agent action returned by an LLM provider."""

    action: AgentAction
    raw_response: str
    repair_attempts: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMStreamChunk:
    """Provider-agnostic streamed text chunk."""

    type: Literal["text_delta", "completed"]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Small provider-agnostic LLM client interface."""

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a normal text response."""
        raise NotImplementedError

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield a normal text response as chunks.

        Providers without native streaming use a one-shot compatibility stream.
        """
        try:
            if max_output_tokens is None:
                text = self.complete(messages)
            else:
                text = self.complete(messages, max_output_tokens=max_output_tokens)
        except TypeError:
            text = self.complete(messages)
        if text:
            yield LLMStreamChunk(type="text_delta", text=text)
        yield LLMStreamChunk(type="completed")

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a structured JSON response."""
        raw_response = self.complete(messages)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMError("Model response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Model JSON response must be an object")
        return parsed

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
    ) -> LLMActionResult:
        """Return a validated agent action using provider-native structure when available."""
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be greater than zero")

        action_messages = list(messages)
        errors: list[str] = []
        for attempt in range(max_repair_attempts + 1):
            if max_output_tokens is None:
                raw_response = self._complete_action_once(action_messages)
            else:
                raw_response = self._complete_action_once(action_messages, max_output_tokens=max_output_tokens)
            try:
                return LLMActionResult(
                    action=parse_model_response(raw_response),
                    raw_response=raw_response,
                    repair_attempts=attempt,
                    errors=errors,
                )
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt >= max_repair_attempts:
                    raise LLMActionError(
                        f"Model response was not valid action JSON: {exc}",
                        repair_attempts=attempt,
                        errors=errors,
                        raw_response=raw_response,
                    ) from exc
                action_messages = [
                    *action_messages,
                    {
                        "role": "user",
                        "content": _format_json_repair_prompt(raw_response, str(exc)),
                    },
                ]

        raise LLMActionError("Model response was not valid action JSON")

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return one raw action response attempt."""
        return self.complete(messages)


def _format_json_repair_prompt(raw_response: str, error: str) -> str:
    return "\n".join(
        [
            JSON_REPAIR_PROMPT,
            f"Parse error: {error}",
            "Previous invalid response:",
            raw_response[:2000],
        ]
    )
