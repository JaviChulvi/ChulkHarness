"""Provider wrapper for language model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from src.core.actions import STRICT_AGENT_ACTION_JSON_SCHEMA, ActionParseError, AgentAction, parse_model_response
from src.core.prompts import JSON_REPAIR_PROMPT


class LLMError(RuntimeError):
    """Base error for model provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM client cannot be configured."""


class LLMActionError(LLMError):
    """Raised when an LLM cannot produce a valid agent action."""

    def __init__(self, message: str, *, repair_attempts: int = 0, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.repair_attempts = repair_attempts
        self.errors = errors or []


@dataclass(frozen=True)
class LLMActionResult:
    """Validated agent action returned by an LLM provider."""

    action: AgentAction
    raw_response: str
    repair_attempts: int = 0
    errors: list[str] = field(default_factory=list)


class LLMClient:
    """Small provider-agnostic LLM client interface."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a normal text response."""
        raise NotImplementedError

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

    def complete_action(self, messages: list[dict[str, str]], *, max_repair_attempts: int = 2) -> LLMActionResult:
        """Return a validated agent action using provider-native structure when available."""
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")

        action_messages = list(messages)
        errors: list[str] = []
        for attempt in range(max_repair_attempts + 1):
            raw_response = self._complete_action_once(action_messages)
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
                    ) from exc
                action_messages = [
                    *action_messages,
                    {
                        "role": "user",
                        "content": _format_json_repair_prompt(raw_response, str(exc)),
                    },
                ]

        raise LLMActionError("Model response was not valid action JSON")

    def _complete_action_once(self, messages: list[dict[str, str]]) -> str:
        """Return one raw action response attempt."""
        return self.complete(messages)


class OpenAIResponsesClient(LLMClient):
    """LLM client backed by the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model

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

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a text response using OpenAI's Responses API."""
        instructions, response_input = _split_instructions(messages)
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions or None,
                input=response_input,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        raise LLMError("OpenAI response did not include output_text")

    def _complete_action_once(self, messages: list[dict[str, str]]) -> str:
        """Return one raw action response using OpenAI Structured Outputs."""
        instructions, response_input = _split_instructions(messages)
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions or None,
                input=response_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "agent_action",
                        "strict": True,
                        "schema": STRICT_AGENT_ACTION_JSON_SCHEMA,
                    }
                },
            )
        except Exception as exc:
            raise LLMError(f"OpenAI structured action request failed: {exc}") from exc

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        raise LLMError("OpenAI structured action response did not include output_text")


class DeepSeekChatCompletionsClient(LLMClient):
    """LLM client backed by DeepSeek's OpenAI-compatible Chat Completions API."""

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

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a text response using DeepSeek chat completions."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=_chat_messages(messages),
                stream=False,
            )
        except Exception as exc:
            raise LLMError(f"DeepSeek request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek response did not include message content") from exc

        if isinstance(content, str) and content:
            return content
        raise LLMError("DeepSeek response content was empty")

    def _complete_action_once(self, messages: list[dict[str, str]]) -> str:
        """Return one raw action response using DeepSeek JSON Output mode."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=_chat_messages(messages),
                stream=False,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMError(f"DeepSeek structured action request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError("DeepSeek structured action response did not include message content") from exc

        if isinstance(content, str) and content:
            return content
        raise LLMError("DeepSeek structured action response content was empty")


def _split_instructions(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Split system/developer instructions from conversational input."""
    instruction_parts: list[str] = []
    response_input: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            instruction_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
            content = f"{message.get('role', 'unknown')}: {content}"
        response_input.append({"role": role, "content": content})

    return "\n\n".join(part for part in instruction_parts if part), response_input


def _chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize messages for OpenAI-compatible chat-completions providers."""
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
            content = f"{message.get('role', 'unknown')}: {content}"
        chat_messages.append({"role": role, "content": content})
    return chat_messages


def _format_json_repair_prompt(raw_response: str, error: str) -> str:
    return "\n".join(
        [
            JSON_REPAIR_PROMPT,
            f"Parse error: {error}",
            "Previous invalid response:",
            raw_response[:2000],
        ]
    )


def create_llm_client(
    *,
    provider: str,
    model: str,
    openai_api_key: str | None,
    deepseek_api_key: str | None,
    deepseek_base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> LLMClient:
    """Create an LLM client for the selected provider."""
    normalized_provider = provider.lower()
    if normalized_provider == "openai":
        return OpenAIResponsesClient(
            model=model,
            api_key=openai_api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if normalized_provider == "deepseek":
        return DeepSeekChatCompletionsClient(
            model=model,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")
