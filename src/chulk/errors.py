"""Stable public exception hierarchy for the Chulk SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from chulk.redaction import redact_data, redact_text


@dataclass(frozen=True)
class ErrorDetails:
    """Safe, immutable context attached to a public SDK error."""

    provider: str | None = None
    model: str | None = None
    tool: str | None = None
    invalid_field: str | None = None
    validation_errors: tuple[Mapping[str, Any], ...] = ()
    retryable: bool | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    trace_path: str | None = None
    failure_kind: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "model",
            "tool",
            "invalid_field",
            "conversation_id",
            "turn_id",
            "trace_path",
            "failure_kind",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, redact_text(str(value)))
        safe_issues = tuple(MappingProxyType(redact_data(dict(issue))) for issue in self.validation_errors)
        object.__setattr__(self, "validation_errors", safe_issues)
        object.__setattr__(self, "extensions", MappingProxyType(redact_data(dict(self.extensions))))

    def to_dict(self) -> dict[str, Any]:
        """Return applicable details as redacted plain data."""
        values: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "tool": self.tool,
            "invalid_field": self.invalid_field,
            "validation_errors": [dict(issue) for issue in self.validation_errors] or None,
            "retryable": self.retryable,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "trace_path": self.trace_path,
            "failure_kind": self.failure_kind,
            "extensions": dict(self.extensions) or None,
        }
        return redact_data({key: value for key, value in values.items() if value is not None})


class ChulkError(Exception):
    """Base class for failures that escape a stable SDK boundary."""

    category: ClassVar[str] = "chulk"

    def __init__(self, message: str, *, details: ErrorDetails | None = None) -> None:
        self.message = redact_text(str(message))
        self.details = details or ErrorDetails()
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted plain-data representation for logs and adapters."""
        return {
            "type": type(self).__name__,
            "category": self.category,
            "message": self.message,
            "details": self.details.to_dict(),
        }


class ConfigurationError(ChulkError):
    """Invalid or incomplete SDK configuration."""

    category = "configuration"


class HostedServiceDisabledError(ConfigurationError):
    """A hosted capability attempted to use an explicitly disabled service."""

    category = "hosted_service_disabled"


class ProviderError(ChulkError):
    """A model provider request or response failure."""

    category = "provider"


class ToolExecutionError(ChulkError):
    """A tool failure that escapes the recoverable agent loop."""

    category = "tool_execution"


class PermissionDeniedError(ChulkError):
    """A terminal permission denial at an SDK boundary."""

    category = "permission_denied"


class SafetyError(ChulkError):
    """A request rejected by a Chulk safety boundary."""

    category = "safety"


class TraceError(ChulkError):
    """A trace could not be read or written safely."""

    category = "trace"


class MemoryError(ChulkError):
    """A durable memory operation failed."""

    category = "memory"


__all__ = [
    "ChulkError",
    "ConfigurationError",
    "ErrorDetails",
    "HostedServiceDisabledError",
    "MemoryError",
    "PermissionDeniedError",
    "ProviderError",
    "SafetyError",
    "ToolExecutionError",
    "TraceError",
]
