"""Translate internal failures at public SDK boundaries."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from chulk.errors import (
    ChulkError,
    ConfigurationError,
    ErrorDetails,
    MemoryError,
    PermissionDeniedError,
    ProviderError,
    SafetyError,
    ToolExecutionError,
    TraceError,
)
from chulk.core.signals import DurableApprovalPaused
from chulk.llm.base import LLMConfigurationError, LLMError
from chulk.mcp.config import MCPConfigError
from chulk.memory.security import MemorySecretError
from chulk.plugins import (
    PluginLoadError,
    PluginLockError,
    PluginRegistrationError,
    PluginVerificationError,
)
from chulk.tools.permissions import TerminalPermissionDenied
from chulk.tools.schema import ToolValidationError


def map_public_error(
    exc: Exception,
    *,
    runtime: object | None = None,
    config: object | None = None,
    operation: str | None = None,
) -> ChulkError:
    """Return the documented public error corresponding to an internal failure."""
    if isinstance(exc, DurableApprovalPaused):
        return exc  # type: ignore[return-value]
    if isinstance(exc, ChulkError):
        return exc

    details = _details(exc, runtime=runtime, config=config, operation=operation)
    if isinstance(
        exc,
        (
            LLMConfigurationError,
            MCPConfigError,
            PluginLoadError,
            PluginLockError,
            PluginRegistrationError,
            PluginVerificationError,
        ),
    ):
        return ConfigurationError(str(exc), details=details)
    if isinstance(exc, LLMError):
        return ProviderError(str(exc), details=details)
    if isinstance(exc, MemorySecretError):
        return SafetyError(str(exc), details=details)
    if isinstance(exc, TerminalPermissionDenied):
        return SafetyError(str(exc), details=details)
    if isinstance(exc, PermissionError):
        return PermissionDeniedError(str(exc), details=details)
    if isinstance(exc, ToolValidationError) or getattr(exc, "tool_name", None):
        return ToolExecutionError(str(exc), details=details)
    if isinstance(exc, sqlite3.Error):
        return MemoryError("The memory store operation failed.", details=details)
    if _looks_like_trace_error(exc, operation):
        return TraceError(str(exc), details=details)
    if _looks_like_safety_error(exc):
        return SafetyError(str(exc), details=details)
    if isinstance(exc, (ValueError, TypeError)) or _is_closed_error(exc):
        return ConfigurationError(str(exc), details=details)
    return ChulkError(str(exc), details=details)


def _details(
    exc: Exception,
    *,
    runtime: object | None,
    config: object | None,
    operation: str | None,
) -> ErrorDetails:
    state = getattr(runtime, "state", None)
    llm = getattr(runtime, "llm_client", None)
    provider = getattr(exc, "provider", None) or getattr(llm, "provider", None)
    model = getattr(exc, "model", None) or getattr(llm, "model", None)
    if provider is None:
        provider = getattr(config, "llm_provider", None) or getattr(config, "provider", None)
    if model is None:
        model = getattr(config, "model", None)
    trace_logger = getattr(runtime, "trace_logger", None)
    trace_path = getattr(exc, "trace_path", None) or getattr(trace_logger, "path", None)
    validation_errors = tuple(
        issue.to_dict() if hasattr(issue, "to_dict") else {"message": str(issue)}
        for issue in getattr(exc, "issues", ())
    )
    invalid_field = getattr(exc, "field", None) or _invalid_field(str(exc))
    retryable = getattr(exc, "retryable", None)
    extensions: dict[str, Any] = {}
    if operation:
        extensions["operation"] = operation
    memory_operation = getattr(exc, "memory_operation", None)
    if memory_operation:
        extensions["memory_operation"] = memory_operation
    policy_name = getattr(exc, "policy_name", None)
    if policy_name:
        extensions["policy_name"] = policy_name
    if isinstance(exc, LLMError):
        extensions["error_code"] = exc.code
        extensions["fallback_eligible"] = exc.fallback_eligible
    return ErrorDetails(
        provider=str(provider) if provider is not None else None,
        model=str(model) if model is not None else None,
        tool=getattr(exc, "tool_name", None),
        invalid_field=invalid_field,
        validation_errors=validation_errors,
        retryable=retryable,
        conversation_id=getattr(state, "conversation_id", None),
        turn_id=getattr(state, "current_turn_id", None),
        trace_path=str(trace_path) if trace_path is not None else None,
        failure_kind=getattr(exc, "failure_kind", None),
        extensions=extensions,
    )


def _invalid_field(message: str) -> str | None:
    match = re.match(r"([A-Z][A-Z0-9_]+)\s+must\b", message)
    return match.group(1) if match else None


def _looks_like_trace_error(exc: Exception, operation: str | None) -> bool:
    return operation in {"trace", "trace_read", "trace_write"} or type(exc).__name__ == "TraceFormatError"


def _looks_like_safety_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ("outside the project root", "unsafe path", "destructive", "refusing"))


def _is_closed_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc).strip().lower() == "agent is closed"


__all__ = ["map_public_error"]
