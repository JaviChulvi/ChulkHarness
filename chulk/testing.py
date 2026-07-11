"""Deterministic test utilities for SDK embeddings and examples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
import json
from threading import Lock
from types import MappingProxyType
from typing import Any

from chulk.core.actions import AgentAction
from chulk.llm.base import LLMClient, LLMError, LLMStreamChunk


ScriptedResponse = str | Mapping[str, Any] | AgentAction


@dataclass(frozen=True)
class _ScriptedCall:
    messages: tuple[Mapping[str, str], ...]
    max_output_tokens: int | None
    response: str


class ScriptedLLMClient(LLMClient):
    """Return an ordered script of model responses without network access.

    The client uses the normal :class:`LLMClient` action parser and usage
    estimator, so tests exercise the same provider-neutral contract as a live
    integration. Each response is consumed exactly once.
    """

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses: Iterable[ScriptedResponse], *, chunk_size: int = 16) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        self._responses = [_serialize_response(response) for response in responses]
        self._chunk_size = chunk_size
        self._calls: list[_ScriptedCall] = []
        self._lock = Lock()

    @property
    def call_log(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable snapshots of completed scripted calls."""
        with self._lock:
            return tuple(
                MappingProxyType(
                    {
                        "messages": tuple(MappingProxyType(dict(message)) for message in call.messages),
                        "max_output_tokens": call.max_output_tokens,
                        "response": call.response,
                    }
                )
                for call in self._calls
            )

    @property
    def remaining(self) -> int:
        """Return the number of unconsumed scripted responses."""
        with self._lock:
            return len(self._responses)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Consume and return the next scripted response."""
        with self._lock:
            if not self._responses:
                raise LLMError(
                    "ScriptedLLMClient response script is exhausted",
                    provider=self.provider,
                    model=self.model,
                    retryable=False,
                )
            response = self._responses.pop(0)
            self._calls.append(
                _ScriptedCall(
                    messages=tuple(dict(message) for message in messages),
                    max_output_tokens=max_output_tokens,
                    response=response,
                )
            )
        return response

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield the next scripted response in deterministic text chunks."""
        response = self.complete_response(messages, max_output_tokens=max_output_tokens)
        for offset in range(0, len(response.content), self._chunk_size):
            yield LLMStreamChunk(
                type="text_delta",
                text=response.content[offset : offset + self._chunk_size],
            )
        yield LLMStreamChunk(type="completed", usage=response.usage, cost=response.cost)


def _serialize_response(response: ScriptedResponse) -> str:
    if isinstance(response, str):
        return response
    if is_dataclass(response):
        response = asdict(response)
    if isinstance(response, Mapping):
        return json.dumps(dict(response), sort_keys=True)
    raise TypeError("Scripted responses must be strings, mappings, or AgentAction dataclasses")


__all__ = ["ScriptedLLMClient"]
