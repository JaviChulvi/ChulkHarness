"""Externally owned conversation transcripts and minimal execution journals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

from chulk.hosting.scope import ExecutionScope
from chulk.errors import ChulkError
from chulk.resources import HostResource


_TRANSCRIPT_ROLES = frozenset({"system", "user", "assistant", "tool", "observation"})
_SEMANTIC_CONTEXT_PREFIXES = {
    "tool": "[External tool context]\n",
    "observation": "[External observation context]\n",
}
MAX_TRANSCRIPT_MESSAGES = 500
MAX_TRANSCRIPT_BYTES = 1_000_000


class TranscriptResolutionError(ChulkError):
    """A host transcript could not be resolved before model work."""


class TranscriptConflictError(TranscriptResolutionError):
    """The externally owned transcript changed across a resume boundary."""


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    """One stable ordered message supplied by the authoritative host."""

    id: str
    role: str
    content: str
    ordinal: int
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("transcript message id cannot be empty")
        if self.role not in _TRANSCRIPT_ROLES:
            raise ValueError(f"unsupported transcript message role: {self.role}")
        if isinstance(self.ordinal, bool) or self.ordinal < 1:
            raise ValueError("transcript message ordinal must be positive")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(deepcopy(dict(self.metadata))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "ordinal": self.ordinal,
            "created_at": self.created_at,
            "metadata": deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class ExternalTranscriptSnapshot:
    """Bounded canonical transcript input owned and versioned by the host."""

    conversation_id: str
    messages: tuple[TranscriptMessage, ...] = ()
    summary: str | None = None
    summary_message_count: int = 0
    revision: str | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            raise ValueError("external transcript conversation_id cannot be empty")
        messages = tuple(self.messages)
        if len(messages) > MAX_TRANSCRIPT_MESSAGES:
            raise ValueError(
                f"external transcript exceeds {MAX_TRANSCRIPT_MESSAGES} messages"
            )
        if len({message.id for message in messages}) != len(messages):
            raise ValueError("external transcript contains duplicate message ids")
        ordinals = [message.ordinal for message in messages]
        if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
            raise ValueError(
                "external transcript message ordinals must be unique and ordered"
            )
        if (
            isinstance(self.summary_message_count, bool)
            or self.summary_message_count < 0
        ):
            raise ValueError("summary_message_count cannot be negative")
        if self.summary_message_count and not (self.summary or "").strip():
            raise ValueError("summary_message_count requires a transcript summary")
        payload = {
            "conversation_id": self.conversation_id,
            "messages": [message.to_dict() for message in messages],
            "summary": self.summary.strip() if self.summary else None,
            "summary_message_count": self.summary_message_count,
            "revision": self.revision,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(
                f"external transcript exceeds {MAX_TRANSCRIPT_BYTES} bytes"
            )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "summary", payload["summary"])
        object.__setattr__(self, "digest", sha256(encoded).hexdigest())

    def prompt_messages(self) -> list[dict[str, str]]:
        """Return semantic host messages for history selection."""
        messages: list[dict[str, str]] = []
        for message in self.messages:
            role = message.role
            content = message.content
            prefix = _SEMANTIC_CONTEXT_PREFIXES.get(role)
            if prefix is not None:
                if not content.strip():
                    raise TranscriptResolutionError(
                        f"external transcript {role} context cannot be empty"
                    )
                content = f"{prefix}{content}"
            messages.append({"role": role, "content": content})
        return messages

    def evidence(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "digest": self.digest,
            "revision": self.revision,
            "message_ids": [message.id for message in self.messages],
            "message_count": len(self.messages),
            "summary_message_count": self.summary_message_count,
        }


def project_external_transcript_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Project selected external tool context into portable user messages."""
    projected: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prefix = _SEMANTIC_CONTEXT_PREFIXES.get(role)
        if prefix is not None and content.startswith(prefix):
            role = "user"
        projected.append({"role": role, "content": content})
    return projected


@dataclass(frozen=True, slots=True)
class TranscriptRequest:
    """Immutable scope and turn identity supplied to a transcript resolver."""

    scope: ExecutionScope
    conversation_id: str
    turn_id: str

    def __post_init__(self) -> None:
        if self.scope.conversation_id != self.conversation_id:
            raise ValueError("transcript request does not match execution scope")
        if not self.turn_id.strip():
            raise ValueError("transcript request turn_id cannot be empty")


TranscriptResolver: TypeAlias = Callable[[TranscriptRequest], ExternalTranscriptSnapshot]
AsyncTranscriptResolver: TypeAlias = Callable[
    [TranscriptRequest], Awaitable[ExternalTranscriptSnapshot]
]


@dataclass(frozen=True, slots=True)
class TranscriptProjection:
    """Idempotent terminal assistant intent delivered to the host."""

    idempotency_key: str
    conversation_id: str
    turn_id: str
    input_transcript_digest: str
    status: str
    content: str
    resources: tuple[HostResource, ...] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(
            self,
            "extensions",
            MappingProxyType(deepcopy(dict(self.extensions))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "input_transcript_digest": self.input_transcript_digest,
            "status": self.status,
            "role": "assistant",
            "content": self.content,
            "resources": [resource.to_dict() for resource in self.resources],
            "extensions": deepcopy(dict(self.extensions)),
        }


@runtime_checkable
class ExecutionJournal(Protocol):
    """Minimal Chulk-owned recovery evidence, separate from business messages."""

    def bind_scope(self, conversation_id: str, scope: ExecutionScope) -> None: ...

    def load_scope(self, conversation_id: str) -> ExecutionScope | None: ...

    def load_turns(self, conversation_id: str) -> list[Any]: ...

    def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None: ...

    def append_event(
        self,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


@runtime_checkable
class AsyncExecutionJournal(Protocol):
    async def bind_scope(
        self,
        conversation_id: str,
        scope: ExecutionScope,
    ) -> None: ...

    async def load_scope(self, conversation_id: str) -> ExecutionScope | None: ...

    async def load_turns(self, conversation_id: str) -> list[Any]: ...

    async def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None: ...

    async def append_event(
        self,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


@runtime_checkable
class TranscriptProjectionSink(Protocol):
    def emit(self, projection: TranscriptProjection) -> None: ...


@runtime_checkable
class AsyncTranscriptProjectionSink(Protocol):
    async def emit(self, projection: TranscriptProjection) -> None: ...


@dataclass(frozen=True, slots=True)
class ExternalTranscriptSessionRuntimeServices:
    """Sync external-transcript journal and terminal projection boundary."""

    journal: ExecutionJournal
    projections: TranscriptProjectionSink


@dataclass(frozen=True, slots=True)
class AsyncExternalTranscriptSessionRuntimeServices:
    """Native-async external-transcript journal and projection boundary."""

    journal: AsyncExecutionJournal
    projections: AsyncTranscriptProjectionSink


__all__ = [
    "AsyncExecutionJournal",
    "AsyncExternalTranscriptSessionRuntimeServices",
    "AsyncTranscriptProjectionSink",
    "AsyncTranscriptResolver",
    "ExecutionJournal",
    "ExternalTranscriptSessionRuntimeServices",
    "ExternalTranscriptSnapshot",
    "MAX_TRANSCRIPT_BYTES",
    "MAX_TRANSCRIPT_MESSAGES",
    "TranscriptConflictError",
    "TranscriptMessage",
    "TranscriptProjection",
    "TranscriptProjectionSink",
    "TranscriptResolutionError",
    "TranscriptRequest",
    "TranscriptResolver",
]
