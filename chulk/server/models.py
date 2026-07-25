"""Versioned request and error models for local control clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


API_SCHEMA_VERSION = 1
MessageMode = Literal["run", "plan"]
PermissionAnswer = Literal["allow", "deny"]


def _required_text(value: object, field: str, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    if len(normalized) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    if "\x00" in normalized:
        raise ValueError(f"{field} cannot contain NUL characters")
    return normalized


def _optional_text(value: object, field: str, *, max_chars: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, field, max_chars=max_chars)


def _object(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("request body must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ConversationCreateRequest:
    conversation_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: object) -> ConversationCreateRequest:
        body = _object(value)
        metadata = body.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            conversation_id=_optional_text(
                body.get("conversation_id"),
                "conversation_id",
                max_chars=128,
            ),
            metadata=dict(metadata) if metadata is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ConversationMessageRequest:
    message: str
    mode: MessageMode = "run"
    idempotency_key: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> ConversationMessageRequest:
        body = _object(value)
        mode = body.get("mode", "run")
        if mode not in {"run", "plan"}:
            raise ValueError("mode must be 'run' or 'plan'")
        return cls(
            message=_required_text(body.get("message"), "message", max_chars=100_000),
            mode=mode,
            idempotency_key=_optional_text(
                body.get("idempotency_key"),
                "idempotency_key",
                max_chars=256,
            ),
        )


@dataclass(frozen=True, slots=True)
class PermissionDecisionRequest:
    decision: PermissionAnswer
    idempotency_key: str
    reason: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> PermissionDecisionRequest:
        body = _object(value)
        decision = body.get("decision")
        if decision not in {"allow", "deny"}:
            raise ValueError("decision must be 'allow' or 'deny'")
        return cls(
            decision=decision,
            idempotency_key=_required_text(
                body.get("idempotency_key"),
                "idempotency_key",
                max_chars=256,
            ),
            reason=_optional_text(body.get("reason"), "reason", max_chars=2_000),
        )


@dataclass(frozen=True, slots=True)
class ApiError:
    code: str
    message: str
    status: int
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error["details"] = dict(self.details)
        return {
            "schema_version": API_SCHEMA_VERSION,
            "error": error,
        }


__all__ = [
    "API_SCHEMA_VERSION",
    "ApiError",
    "ConversationCreateRequest",
    "ConversationMessageRequest",
    "MessageMode",
    "PermissionAnswer",
    "PermissionDecisionRequest",
]
