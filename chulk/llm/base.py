"""Shared LLM client interfaces and errors."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from typing import Any, Literal

from chulk.core.actions import ActionParseError, AgentAction, parse_model_response
from chulk.core.prompts import JSON_REPAIR_PROMPT
from chulk.llm.pricing import estimate_cost
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage, aggregate_cost, aggregate_usage, estimate_usage


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
        usage: LLMUsage | None = None,
        cost: LLMCost | None = None,
    ) -> None:
        super().__init__(message)
        self.repair_attempts = repair_attempts
        self.errors = errors or []
        self.raw_response = raw_response
        self.usage = usage
        self.cost = cost if cost is not None else estimate_cost(None, None, usage)


@dataclass(frozen=True)
class LLMActionResult:
    """Validated agent action returned by an LLM provider."""

    action: AgentAction
    raw_response: str
    repair_attempts: int = 0
    errors: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    cost: LLMCost | None = None


@dataclass(frozen=True)
class LLMStreamChunk:
    """Provider-agnostic streamed text chunk."""

    type: Literal["text_delta", "completed"]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: LLMUsage | None = None
    cost: LLMCost | None = None


class LLMClient:
    """Small provider-agnostic LLM client interface."""

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        """Return a normal text response."""
        raise NotImplementedError

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return text plus normalized usage metadata."""
        try:
            if max_output_tokens is None:
                content = self.complete(messages)
            else:
                content = self.complete(messages, max_output_tokens=max_output_tokens)
        except TypeError:
            content = self.complete(messages)
        return self._response_with_estimated_usage(messages, content)

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield a normal text response as chunks.

        Providers without native streaming use a one-shot compatibility stream.
        """
        response = self.complete_response(messages, max_output_tokens=max_output_tokens)
        text = response.content
        if text:
            yield LLMStreamChunk(type="text_delta", text=text)
        yield LLMStreamChunk(type="completed", usage=response.usage, cost=response.cost)

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
        usage_records: list[LLMUsage | None] = []
        cost_records: list[LLMCost | None] = []
        for attempt in range(max_repair_attempts + 1):
            if max_output_tokens is None:
                response = self._complete_action_response_once(action_messages)
            else:
                response = self._complete_action_response_once(action_messages, max_output_tokens=max_output_tokens)
            raw_response = response.content
            usage_records.append(response.usage)
            cost_records.append(response.cost)
            try:
                return LLMActionResult(
                    action=parse_model_response(raw_response),
                    raw_response=raw_response,
                    repair_attempts=attempt,
                    errors=errors,
                    usage=aggregate_usage(usage_records),
                    cost=aggregate_cost(cost_records),
                )
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt >= max_repair_attempts:
                    raise LLMActionError(
                        f"Model response was not valid action JSON: {exc}",
                        repair_attempts=attempt,
                        errors=errors,
                        raw_response=raw_response,
                        usage=aggregate_usage(usage_records),
                        cost=aggregate_cost(cost_records),
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

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Return one raw action response attempt plus metadata."""
        try:
            if max_output_tokens is None:
                content = self._complete_action_once(messages)
            else:
                content = self._complete_action_once(messages, max_output_tokens=max_output_tokens)
        except TypeError:
            content = self._complete_action_once(messages)
        return self._response_with_estimated_usage(messages, content)

    def _response_with_estimated_usage(self, messages: list[dict[str, str]], content: str) -> LLMResponse:
        provider = _provider_name(self)
        model = _model_name(self)
        usage = estimate_usage(messages, content)
        return LLMResponse(
            content=content,
            usage=usage,
            cost=estimate_cost(provider, model, usage),
            provider=provider,
            model=model,
        )


def _provider_name(client: object) -> str | None:
    value = getattr(client, "provider", None) or getattr(client, "name", None)
    return str(value) if value is not None else None


def _model_name(client: object) -> str | None:
    value = getattr(client, "model", None)
    return str(value) if value is not None else None


def _format_json_repair_prompt(raw_response: str, error: str) -> str:
    return "\n".join(
        [
            JSON_REPAIR_PROMPT,
            f"Parse error: {error}",
            "Previous invalid response:",
            raw_response[:2000],
        ]
    )
