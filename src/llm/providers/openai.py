"""OpenAI provider client."""

from __future__ import annotations

from typing import Any

from src.core.actions import STRICT_AGENT_ACTION_JSON_SCHEMA
from src.llm.base import LLMClient, LLMConfigurationError, LLMError
from src.llm.capabilities import LLMCapabilities
from src.llm.messages import split_instructions


OPENAI_CAPABILITIES = LLMCapabilities(
    supports_structured_output=True,
    supports_json_mode=False,
    supports_streaming=False,
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
        instructions, response_input = split_instructions(messages)
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
        instructions, response_input = split_instructions(messages)
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
