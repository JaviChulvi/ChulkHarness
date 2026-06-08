"""Provider wrapper for language model calls."""

from typing import Any


class LLMClient:
    """Small provider-agnostic LLM client interface."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return a normal text response."""
        raise NotImplementedError("LLM text completion is planned for Phase 1.")

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a structured JSON response."""
        raise NotImplementedError("Structured LLM responses are planned for Phase 2.")
