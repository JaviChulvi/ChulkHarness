"""Versioned hosted tool identity, policy, and host hook contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
from typing import Any
import types

from packaging.version import InvalidVersion, Version

from chulk.hosting.scope import ExecutionScope
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.schema import schema_digest


_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class ToolRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolEffect(StrEnum):
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"
    UNKNOWN = "unknown"


class ToolConcurrency(StrEnum):
    SERIAL = "serial"
    PARALLEL_SAFE = "parallel_safe"


class ToolApprovalMode(StrEnum):
    NEVER = "never"
    POLICY = "policy"
    ALWAYS = "always"


class ToolIdempotencyStrategy(StrEnum):
    NONE = "none"
    CALLER_KEY = "caller_key"
    PROVIDER_KEY = "provider_key"
    RECONCILIATION = "reconciliation"


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    """Stable semantic identity for code and schemas exposed by one tool."""

    name: str
    version: str = "1.0.0"
    input_schema_version: str = "1.0.0"
    output_schema_version: str = "1.0.0"
    input_schema_digest: str = ""
    output_schema_digest: str = ""
    implementation_digest: str = ""

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("tool identity name cannot be empty")
        object.__setattr__(self, "name", clean_name)
        for field_name in ("version", "input_schema_version", "output_schema_version"):
            value = str(getattr(self, field_name)).strip()
            if _SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
                raise ValueError(
                    f"tool identity {field_name} must be a semantic version"
                )
            try:
                Version(value)
            except InvalidVersion as exc:
                raise ValueError(
                    f"tool identity {field_name} must be a semantic version"
                ) from exc
            object.__setattr__(self, field_name, value)
        for field_name in (
            "input_schema_digest",
            "output_schema_digest",
            "implementation_digest",
        ):
            digest = str(getattr(self, field_name)).strip().lower()
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"tool identity {field_name} must be a SHA-256 digest")
            object.__setattr__(self, field_name, digest)

    @classmethod
    def from_schemas(
        cls,
        name: str,
        *,
        version: str = "1.0.0",
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        input_schema_version: str = "1.0.0",
        output_schema_version: str = "1.0.0",
    ) -> "ToolIdentity":
        return cls(
            name=name,
            version=version,
            input_schema_version=input_schema_version,
            output_schema_version=output_schema_version,
            input_schema_digest=schema_digest(input_schema or {}),
            output_schema_digest=schema_digest(output_schema or {}),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return schema_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Validated authorization and side-effect contract for a tool."""

    version: str = "1.0.0"
    required_grants: frozenset[str] = frozenset()
    risk: ToolRisk = ToolRisk.LOW
    effect: ToolEffect = ToolEffect.READ
    concurrency: ToolConcurrency = ToolConcurrency.SERIAL
    approval: ToolApprovalMode = ToolApprovalMode.POLICY
    idempotency: ToolIdempotencyStrategy = ToolIdempotencyStrategy.NONE
    supports_dry_run: bool = False
    supports_compensation: bool = False
    input_classification: DataClassification = DataClassification.INTERNAL
    output_classification: DataClassification = DataClassification.INTERNAL

    def __post_init__(self) -> None:
        if _SEMANTIC_VERSION_PATTERN.fullmatch(self.version) is None:
            raise ValueError("tool policy version must be a semantic version")
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError("tool policy version must be a semantic version") from exc
        object.__setattr__(
            self,
            "required_grants",
            frozenset(_clean_grant(value) for value in self.required_grants),
        )
        object.__setattr__(self, "risk", ToolRisk(self.risk))
        object.__setattr__(self, "effect", ToolEffect(self.effect))
        object.__setattr__(self, "concurrency", ToolConcurrency(self.concurrency))
        object.__setattr__(self, "approval", ToolApprovalMode(self.approval))
        object.__setattr__(
            self,
            "idempotency",
            ToolIdempotencyStrategy(self.idempotency),
        )
        object.__setattr__(
            self,
            "input_classification",
            DataClassification(self.input_classification),
        )
        object.__setattr__(
            self,
            "output_classification",
            DataClassification(self.output_classification),
        )
        if (
            self.concurrency is ToolConcurrency.PARALLEL_SAFE
            and self.effect is not ToolEffect.READ
        ):
            raise ValueError("only read-only tools may be parallel-safe")
        if self.supports_compensation and self.effect is ToolEffect.READ:
            raise ValueError("read-only tools cannot declare compensation")
        if (
            self.effect in {ToolEffect.EXTERNAL_WRITE, ToolEffect.DESTRUCTIVE}
            and self.approval is ToolApprovalMode.NEVER
        ):
            raise ValueError("external or destructive effects cannot disable approval")

    @classmethod
    def safe_default(
        cls,
        permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ,
        *,
        requires_confirmation: bool = False,
        idempotent: bool = False,
    ) -> "ToolPolicy":
        level = ToolPermissionLevel(permission_level)
        if level is ToolPermissionLevel.READ:
            effect = ToolEffect.READ
            risk = ToolRisk.LOW
        elif level is ToolPermissionLevel.DESTRUCTIVE:
            effect = ToolEffect.DESTRUCTIVE
            risk = ToolRisk.CRITICAL
        elif level in {ToolPermissionLevel.NETWORK, ToolPermissionLevel.EXTERNAL_SERVICE}:
            effect = ToolEffect.EXTERNAL_WRITE
            risk = ToolRisk.HIGH
        else:
            effect = ToolEffect.LOCAL_WRITE
            risk = ToolRisk.MEDIUM
        return cls(
            risk=risk,
            effect=effect,
            approval=(
                ToolApprovalMode.ALWAYS
                if requires_confirmation
                else ToolApprovalMode.POLICY
            ),
            idempotency=(
                ToolIdempotencyStrategy.CALLER_KEY
                if idempotent
                else ToolIdempotencyStrategy.NONE
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_grants"] = sorted(self.required_grants)
        for key in (
            "risk",
            "effect",
            "concurrency",
            "approval",
            "idempotency",
            "input_classification",
            "output_classification",
        ):
            payload[key] = getattr(self, key).value
        return payload

    @property
    def digest(self) -> str:
        return schema_digest(self.to_dict())

    @property
    def retry_safe(self) -> bool:
        return self.effect is ToolEffect.READ or self.idempotency is not ToolIdempotencyStrategy.NONE


@dataclass(frozen=True, slots=True)
class ToolAuthorization:
    allowed: bool
    reason: str = ""


ToolHookResult = Any | Awaitable[Any]
ToolAuthorizer = Callable[
    [ExecutionScope, ToolIdentity, ToolPolicy, Mapping[str, Any]],
    ToolAuthorization | bool | Awaitable[ToolAuthorization | bool],
]
CredentialResolver = Callable[
    [ExecutionScope, ToolIdentity, ToolPolicy, Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]
ToolPolicyHook = Callable[
    [ExecutionScope, ToolIdentity, ToolPolicy, Mapping[str, Any]],
    ToolHookResult,
]


@dataclass(frozen=True, slots=True)
class ToolPolicyHooks:
    """Host-owned hooks around policy evaluation and external effects."""

    authorize: ToolAuthorizer | None = None
    resolve_credentials: CredentialResolver | None = None
    preview: ToolPolicyHook | None = None
    derive_effect_key: ToolPolicyHook | None = None
    reconcile: ToolPolicyHook | None = None
    compensate: ToolPolicyHook | None = None
    redact: ToolPolicyHook | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def validate_tool_contract(
    *,
    name: str,
    args_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any] | None,
    identity: ToolIdentity,
    policy: ToolPolicy,
) -> None:
    if identity.name != name:
        raise ValueError("tool identity name must match the registered tool name")
    if identity.input_schema_digest != schema_digest(args_schema):
        raise ValueError("tool input schema does not match its identity digest")
    if identity.output_schema_digest != schema_digest(output_schema or {}):
        raise ValueError("tool output schema does not match its identity digest")
    if (
        policy.effect is not ToolEffect.READ
        and policy.concurrency is ToolConcurrency.PARALLEL_SAFE
    ):
        raise ValueError("mutating tools must execute serially")


def callable_digest(value: Callable[..., Any]) -> str:
    """Return a stable digest of Python callable code and nested closures."""
    payload = _callable_payload(value, seen=set())
    return schema_digest(payload)


def _clean_grant(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("tool policy grants must be non-empty strings")
    return value.strip()


def _callable_payload(value: object, *, seen: set[int]) -> dict[str, Any]:
    identity = id(value)
    if identity in seen:
        return {"recursive": True}
    seen.add(identity)
    if isinstance(value, types.MethodType):
        return _callable_payload(value.__func__, seen=seen)
    if isinstance(value, types.FunctionType):
        code = value.__code__
        closures: list[dict[str, Any]] = []
        for cell in value.__closure__ or ():
            try:
                item = cell.cell_contents
            except ValueError:
                closures.append({"empty": True})
                continue
            if isinstance(item, (types.FunctionType, types.MethodType)):
                closures.append(_callable_payload(item, seen=seen))
            else:
                closures.append(
                    {
                        "type": f"{type(item).__module__}.{type(item).__qualname__}",
                        "value": _stable_constant(item),
                    }
                )
        return {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code": code.co_code.hex(),
            "constants": [_stable_constant(item) for item in code.co_consts],
            "names": list(code.co_names),
            "closures": closures,
        }
    call = getattr(value, "__call__", None)
    if call is not None and call is not value:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "call": _callable_payload(call, seen=seen),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _stable_constant(value: object) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return [_stable_constant(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(
            (_stable_constant(item) for item in value),
            key=repr,
        )
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
