"""Channel-neutral gateway envelopes and delivery records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeAlias
from uuid import uuid4

from chulk.redaction import redact_data
from chulk.results import freeze_mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChannelScope(str, Enum):
    DIRECT = "direct"
    GROUP = "group"
    WEBHOOK = "webhook"
    LOCAL_OPERATOR = "local_operator"


class AuthenticationState(str, Enum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    UNKNOWN = "unknown"


class TrustLevel(str, Enum):
    OWNER = "owner"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class DeliveryState(str, Enum):
    ACCEPTED = "accepted"
    DELIVERED = "delivered"
    RETRYABLE = "retryable"
    FAILED = "failed"
    UNKNOWN = "unknown"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True, slots=True)
class ChannelIdentity:
    """One adapter/account/principal identity tuple."""

    adapter: str
    account_id: str
    principal_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", _required(self.adapter, "adapter"))
        object.__setattr__(self, "account_id", _required(self.account_id, "account_id"))
        object.__setattr__(self, "principal_id", _required(self.principal_id, "principal_id"))


@dataclass(frozen=True, slots=True)
class MediaReference:
    """Opaque bounded media metadata; bytes live in a dedicated content store."""

    content_ref: str
    content_type: str
    size_bytes: int
    file_name: str | None = None
    sha256: str | None = None
    external_content: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_ref", _required(self.content_ref, "content_ref"))
        object.__setattr__(self, "content_type", _required(self.content_type, "content_type"))
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str
    external_content: bool = True
    kind: str = field(default="text", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required(self.text, "text"))


@dataclass(frozen=True, slots=True)
class MediaPart:
    media: MediaReference
    caption: str | None = None
    kind: str = field(default="media", init=False)


@dataclass(frozen=True, slots=True)
class ReplyPart:
    event_id: str
    excerpt: str | None = None
    external_content: bool = True
    kind: str = field(default="reply", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))


@dataclass(frozen=True, slots=True)
class ReactionPart:
    reaction: str
    target_event_id: str
    external_content: bool = True
    kind: str = field(default="reaction", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reaction", _required(self.reaction, "reaction"))
        object.__setattr__(
            self,
            "target_event_id",
            _required(self.target_event_id, "target_event_id"),
        )


InboundPart: TypeAlias = TextPart | MediaPart | ReplyPart | ReactionPart


@dataclass(frozen=True, slots=True)
class InboundEnvelope:
    """Normalized authenticated input received from any adapter."""

    event_id: str
    idempotency_key: str
    identity: ChannelIdentity
    destination_id: str
    parts: tuple[InboundPart, ...]
    scope: ChannelScope
    authentication: AuthenticationState
    trust: TrustLevel
    thread_id: str | None = None
    received_at: str = field(default_factory=_utc_now)
    external_content: bool = True
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "idempotency_key",
            _required(self.idempotency_key, "idempotency_key"),
        )
        object.__setattr__(
            self,
            "destination_id",
            _required(self.destination_id, "destination_id"),
        )
        if not self.parts:
            raise ValueError("inbound envelope must contain at least one part")
        object.__setattr__(self, "scope", ChannelScope(self.scope))
        object.__setattr__(
            self,
            "authentication",
            AuthenticationState(self.authentication),
        )
        object.__setattr__(self, "trust", TrustLevel(self.trust))
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(
            self,
            "extensions",
            freeze_mapping(redact_data(dict(self.extensions))),
        )


@dataclass(frozen=True, slots=True)
class DeliveryTarget:
    """Adapter-neutral destination for one outbound delivery."""

    adapter: str
    account_id: str
    destination_id: str
    thread_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", _required(self.adapter, "adapter"))
        object.__setattr__(self, "account_id", _required(self.account_id, "account_id"))
        object.__setattr__(
            self,
            "destination_id",
            _required(self.destination_id, "destination_id"),
        )


@dataclass(frozen=True, slots=True)
class OutboundEnvelope:
    """One retryable delivery unit produced after agent execution."""

    profile_id: str
    conversation_id: str
    target: DeliveryTarget
    text: str | None = None
    attachments: tuple[MediaReference, ...] = ()
    reply_to_event_id: str | None = None
    envelope_id: str = field(default_factory=lambda: str(uuid4()))
    checkpoint: str | None = None
    sequence: int = 0
    final: bool = True
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope_id", _required(self.envelope_id, "envelope_id"))
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile_id"))
        object.__setattr__(
            self,
            "conversation_id",
            _required(self.conversation_id, "conversation_id"),
        )
        if self.text is None and not self.attachments:
            raise ValueError("outbound envelope requires text or an attachment")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(self.final, bool):
            raise ValueError("final must be a boolean")
        object.__setattr__(self, "attachments", tuple(self.attachments))
        object.__setattr__(
            self,
            "extensions",
            freeze_mapping(redact_data(dict(self.extensions))),
        )


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Adapter delivery outcome and durable retry checkpoint."""

    envelope_id: str
    state: DeliveryState
    attempt: int
    adapter_message_id: str | None = None
    checkpoint: str | None = None
    retry_after_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    recorded_at: str = field(default_factory=_utc_now)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope_id", _required(self.envelope_id, "envelope_id"))
        object.__setattr__(self, "state", DeliveryState(self.state))
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if self.retry_after_seconds is not None:
            if (
                isinstance(self.retry_after_seconds, bool)
                or not isinstance(self.retry_after_seconds, (int, float))
                or self.retry_after_seconds < 0
            ):
                raise ValueError("retry_after_seconds must be a non-negative number")
        if self.state is DeliveryState.RETRYABLE and self.error_message is None:
            raise ValueError("retryable delivery receipts require an error_message")
        object.__setattr__(
            self,
            "extensions",
            freeze_mapping(redact_data(dict(self.extensions))),
        )


def _required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} cannot contain NUL characters")
    return normalized


__all__ = [
    "AuthenticationState",
    "ChannelIdentity",
    "ChannelScope",
    "DeliveryReceipt",
    "DeliveryState",
    "DeliveryTarget",
    "InboundEnvelope",
    "InboundPart",
    "MediaPart",
    "MediaReference",
    "OutboundEnvelope",
    "ReactionPart",
    "ReplyPart",
    "TextPart",
    "TrustLevel",
]
