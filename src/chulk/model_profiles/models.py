"""Immutable named-model, diagnostic, selection, and health models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
import re
from typing import Any, Literal

from chulk.profiles import CredentialRef, normalize_profile_id
from chulk.results import freeze_mapping


DEFAULT_MODEL_PROFILE_ID = "default"
EndpointSource = Literal["config", "env", "host"]
_REFERENCE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")


@dataclass(frozen=True, slots=True)
class EndpointRef:
    """Reference to a host-owned endpoint configuration, not a persisted URL."""

    name: str
    source: EndpointSource = "config"

    def __post_init__(self) -> None:
        name = self.name.strip()
        if _REFERENCE_NAME.fullmatch(name) is None:
            raise ValueError("endpoint reference contains unsupported characters")
        if self.source not in {"config", "env", "host"}:
            raise ValueError("endpoint references must use config:, env:, or host:")
        object.__setattr__(self, "name", name)

    @property
    def uri(self) -> str:
        return f"{self.source}:{self.name}"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source}

    @classmethod
    def parse(cls, value: str) -> EndpointRef:
        source, separator, name = value.strip().partition(":")
        if not separator or not name or source not in {"config", "env", "host"}:
            raise ValueError("endpoint references must use config:, env:, or host:")
        return cls(name=name, source=source)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModelCapabilityRequirements:
    """Provider transport capabilities required by one named profile."""

    structured_output: bool = False
    json_mode: bool = False
    streaming: bool = False
    native_tool_calling: bool = False
    hosted_mcp_tools: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "structured_output": self.structured_output,
            "json_mode": self.json_mode,
            "streaming": self.streaming,
            "native_tool_calling": self.native_tool_calling,
            "hosted_mcp_tools": self.hosted_mcp_tools,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> ModelCapabilityRequirements:
        data = value or {}
        return cls(
            structured_output=bool(data.get("structured_output")),
            json_mode=bool(data.get("json_mode")),
            streaming=bool(data.get("streaming")),
            native_tool_calling=bool(data.get("native_tool_calling")),
            hosted_mcp_tools=bool(data.get("hosted_mcp_tools")),
        )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One named provider/model configuration without credential values."""

    id: str
    provider: str
    model: str
    credential_ref: CredentialRef | None = None
    endpoint_ref: EndpointRef | None = None
    fallback_profile_ids: tuple[str, ...] = ()
    required_capabilities: ModelCapabilityRequirements = field(
        default_factory=ModelCapabilityRequirements
    )
    context_window_tokens: int | None = None
    response_reserve_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_per_turn: Decimal | None = None
    implicit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_profile_id(self.id))
        provider = self.provider.strip().lower()
        model = self.model.strip()
        if not provider:
            raise ValueError("model profile provider cannot be empty")
        if not model:
            raise ValueError("model profile model cannot be empty")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        fallbacks = tuple(
            dict.fromkeys(
                normalize_profile_id(value) for value in self.fallback_profile_ids
            )
        )
        if self.id in fallbacks:
            raise ValueError("a model profile cannot directly fall back to itself")
        object.__setattr__(self, "fallback_profile_ids", fallbacks)
        for field_name in (
            "context_window_tokens",
            "response_reserve_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            self.context_window_tokens is not None
            and self.response_reserve_tokens is not None
            and self.response_reserve_tokens >= self.context_window_tokens
        ):
            raise ValueError(
                "response_reserve_tokens must be smaller than context_window_tokens"
            )
        if self.max_cost_per_turn is not None:
            if (
                not isinstance(self.max_cost_per_turn, Decimal)
                or not self.max_cost_per_turn.is_finite()
                or self.max_cost_per_turn <= 0
            ):
                raise ValueError("max_cost_per_turn must be a positive finite Decimal")
        if (
            self.implicit
            and self.id != DEFAULT_MODEL_PROFILE_ID
            and not self.id.startswith("default-fallback-")
        ):
            raise ValueError("implicit model profile ids are reserved")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "credential_ref": self.credential_ref.uri
            if self.credential_ref is not None
            else None,
            "endpoint_ref": self.endpoint_ref.uri
            if self.endpoint_ref is not None
            else None,
            "fallback_profile_ids": list(self.fallback_profile_ids),
            "required_capabilities": self.required_capabilities.to_dict(),
            "context_window_tokens": self.context_window_tokens,
            "response_reserve_tokens": self.response_reserve_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_cost_per_turn": (
                str(self.max_cost_per_turn)
                if self.max_cost_per_turn is not None
                else None
            ),
            "implicit": self.implicit,
        }


class DiagnosticCategory(StrEnum):
    READY = "ready"
    MISSING_CREDENTIAL = "missing_credential"
    INVALID_MODEL = "invalid_model"
    UNAVAILABLE_ENDPOINT = "unavailable_endpoint"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    COOLDOWN = "cooldown"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelDiagnostic:
    """Sanitized static or explicitly probed diagnostic result."""

    profile_id: str
    category: DiagnosticCategory
    ok: bool
    message: str
    provider: str
    model: str
    probed: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", DiagnosticCategory(self.category))
        object.__setattr__(self, "details", freeze_mapping(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "category": self.category.value,
            "ok": self.ok,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "probed": self.probed,
            "details": dict(self.details),
        }


class ProviderHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    health_key: str
    model_profile_id: str
    provider: str
    credential_ref: str | None
    endpoint_ref: str | None
    status: ProviderHealthStatus
    consecutive_failures: int
    cooldown_until: datetime | None = None
    last_error_category: DiagnosticCategory | None = None
    last_error_at: datetime | None = None
    last_success_at: datetime | None = None
    successful_requests: int = 0
    failed_requests: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProviderHealthStatus(self.status))
        if self.last_error_category is not None:
            object.__setattr__(
                self,
                "last_error_category",
                DiagnosticCategory(self.last_error_category),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_key": self.health_key,
            "model_profile_id": self.model_profile_id,
            "provider": self.provider,
            "credential_ref": self.credential_ref,
            "endpoint_ref": self.endpoint_ref,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": (
                self.cooldown_until.isoformat()
                if self.cooldown_until is not None
                else None
            ),
            "last_error_category": (
                self.last_error_category.value
                if self.last_error_category is not None
                else None
            ),
            "last_error_at": (
                self.last_error_at.isoformat()
                if self.last_error_at is not None
                else None
            ),
            "last_success_at": (
                self.last_success_at.isoformat()
                if self.last_success_at is not None
                else None
            ),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }


@dataclass(frozen=True, slots=True)
class ModelSelectionSkip:
    profile_id: str
    reason: str
    category: DiagnosticCategory

    def to_dict(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "reason": self.reason,
            "category": self.category.value,
        }


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    requested_profile_id: str
    selected_profile_id: str
    fallback_path: tuple[str, ...]
    reason: str
    skipped: tuple[ModelSelectionSkip, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_profile_id": self.requested_profile_id,
            "selected_profile_id": self.selected_profile_id,
            "fallback_path": list(self.fallback_path),
            "reason": self.reason,
            "skipped": [item.to_dict() for item in self.skipped],
        }


__all__ = [
    "DEFAULT_MODEL_PROFILE_ID",
    "DiagnosticCategory",
    "EndpointRef",
    "EndpointSource",
    "ModelCapabilityRequirements",
    "ModelDiagnostic",
    "ModelProfile",
    "ModelSelectionResult",
    "ModelSelectionSkip",
    "ProviderHealth",
    "ProviderHealthStatus",
]
