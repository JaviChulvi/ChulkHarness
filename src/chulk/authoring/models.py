"""Portable, immutable agent-definition and workflow contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import PurePath
import re
from types import MappingProxyType
from typing import Any

from packaging.version import InvalidVersion, Version


AGENT_DEFINITION_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_DIGEST_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SECRET_FIELD_MARKERS = frozenset(
    {"secret", "password", "token", "credential", "api_key", "private_key"}
)
_PATH_FIELD_MARKERS = frozenset(
    {"path", "project_root", "runtime_dir", "store_path", "traces_dir"}
)
_EXECUTABLE_FIELD_MARKERS = frozenset(
    {"callable", "import", "module", "python", "script", "shell"}
)


class DefinitionStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


class WorkflowEffect(StrEnum):
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"
    UNKNOWN = "unknown"


class WorkflowApproval(StrEnum):
    NEVER = "never"
    POLICY = "policy"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class VersionedReference:
    """Exact portable reference to one immutable published artifact."""

    name: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _reference(self.name, "reference name"))
        object.__setattr__(self, "version", _semantic_version(self.version))
        object.__setattr__(self, "digest", _digest(self.digest))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VersionedReference:
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            digest=str(value["digest"]),
        )


@dataclass(frozen=True, slots=True)
class ToolReference(VersionedReference):
    """Exact tool, schemas, implementation, and policy identity."""

    input_schema_version: str = "1.0.0"
    input_schema_digest: str = "sha256:" + ("0" * 64)
    output_schema_version: str = "1.0.0"
    output_schema_digest: str = "sha256:" + ("0" * 64)
    implementation_digest: str = "sha256:" + ("0" * 64)
    policy_version: str = "1.0.0"
    policy_digest: str = "sha256:" + ("0" * 64)

    def __post_init__(self) -> None:
        VersionedReference.__post_init__(self)
        for field_name in (
            "input_schema_version",
            "output_schema_version",
            "policy_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _semantic_version(str(getattr(self, field_name))),
            )
        for field_name in (
            "input_schema_digest",
            "output_schema_digest",
            "implementation_digest",
            "policy_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(str(getattr(self, field_name))),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            **VersionedReference.to_dict(self),
            "input_schema_version": self.input_schema_version,
            "input_schema_digest": self.input_schema_digest,
            "output_schema_version": self.output_schema_version,
            "output_schema_digest": self.output_schema_digest,
            "implementation_digest": self.implementation_digest,
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolReference:
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            digest=str(value["digest"]),
            input_schema_version=str(value["input_schema_version"]),
            input_schema_digest=str(value["input_schema_digest"]),
            output_schema_version=str(value["output_schema_version"]),
            output_schema_digest=str(value["output_schema_digest"]),
            implementation_digest=str(value["implementation_digest"]),
            policy_version=str(value["policy_version"]),
            policy_digest=str(value["policy_digest"]),
        )


@dataclass(frozen=True, slots=True)
class TriggerDefinition:
    """One declarative trigger with no executable imports or credentials."""

    kind: str
    version: str = "1.0.0"
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _reference(self.kind, "trigger kind"))
        object.__setattr__(self, "version", _semantic_version(self.version))
        clean = _portable_mapping(self.config, field_path="trigger.config")
        object.__setattr__(self, "config", MappingProxyType(clean))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "config": _plain(self.config),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TriggerDefinition:
        config = value.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("trigger config must be an object")
        return cls(
            kind=str(value["kind"]),
            version=str(value.get("version", "1.0.0")),
            config=config,
        )


@dataclass(frozen=True, slots=True)
class BudgetDefinition:
    """Portable upper bounds enforced by a runtime host."""

    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: str | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        for field_name in (
            "max_model_requests",
            "max_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be positive")
        currency = self.currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("budget currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)
        if self.max_cost is not None:
            try:
                cost = Decimal(self.max_cost)
            except InvalidOperation as exc:
                raise ValueError("max_cost must be numeric") from exc
            if not cost.is_finite() or cost <= 0:
                raise ValueError("max_cost must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_model_requests": self.max_model_requests,
            "max_tool_calls": self.max_tool_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_cost": self.max_cost,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BudgetDefinition:
        return cls(
            max_model_requests=_optional_int(value.get("max_model_requests")),
            max_tool_calls=_optional_int(value.get("max_tool_calls")),
            max_input_tokens=_optional_int(value.get("max_input_tokens")),
            max_output_tokens=_optional_int(value.get("max_output_tokens")),
            max_cost=_optional_string(value.get("max_cost")),
            currency=str(value.get("currency", "USD")),
        )


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One tool-bound node in a deterministic declarative workflow graph."""

    id: str
    tool: ToolReference
    depends_on: tuple[str, ...] = ()
    effect: WorkflowEffect = WorkflowEffect.READ
    approval: WorkflowApproval = WorkflowApproval.POLICY
    retry_limit: int = 0
    timeout_seconds: float | None = None
    failure_step: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "workflow step id"))
        dependencies = tuple(
            dict.fromkeys(
                _identifier(value, "workflow dependency") for value in self.depends_on
            )
        )
        if self.id in dependencies:
            raise ValueError("workflow step cannot depend on itself")
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "effect", WorkflowEffect(self.effect))
        object.__setattr__(self, "approval", WorkflowApproval(self.approval))
        if self.retry_limit < 0 or self.retry_limit > 10:
            raise ValueError("workflow retry_limit must be between 0 and 10")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("workflow timeout_seconds must be positive")
        if self.failure_step is not None:
            object.__setattr__(
                self,
                "failure_step",
                _identifier(self.failure_step, "workflow failure step"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool.to_dict(),
            "depends_on": list(self.depends_on),
            "effect": self.effect.value,
            "approval": self.approval.value,
            "retry_limit": self.retry_limit,
            "timeout_seconds": self.timeout_seconds,
            "failure_step": self.failure_step,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkflowStep:
        tool = value.get("tool")
        if not isinstance(tool, Mapping):
            raise ValueError("workflow step tool must be an object")
        return cls(
            id=str(value["id"]),
            tool=ToolReference.from_dict(tool),
            depends_on=tuple(str(item) for item in value.get("depends_on", ())),
            effect=WorkflowEffect(str(value.get("effect", "read"))),
            approval=WorkflowApproval(str(value.get("approval", "policy"))),
            retry_limit=int(value.get("retry_limit", 0)),
            timeout_seconds=(
                float(value["timeout_seconds"])
                if value.get("timeout_seconds") is not None
                else None
            ),
            failure_step=_optional_string(value.get("failure_step")),
        )


@dataclass(frozen=True, slots=True)
class WorkflowGraph:
    """Validated acyclic workflow with explicit effects and failure paths."""

    steps: tuple[WorkflowStep, ...] = ()

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        by_id = {step.id: step for step in steps}
        if len(by_id) != len(steps):
            raise ValueError("workflow step ids must be unique")
        for step in steps:
            missing = set(step.depends_on) - set(by_id)
            if missing:
                raise ValueError(
                    f"workflow step {step.id!r} has unknown dependencies: "
                    + ", ".join(sorted(missing))
                )
            if step.failure_step is not None and step.failure_step not in by_id:
                raise ValueError(
                    f"workflow step {step.id!r} has an unknown failure step"
                )
        _assert_acyclic(by_id)
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkflowGraph:
        raw_steps = value.get("steps", ())
        if not isinstance(raw_steps, Sequence) or isinstance(
            raw_steps, (str, bytes, bytearray)
        ):
            raise ValueError("workflow steps must be an array")
        return cls(
            steps=tuple(
                WorkflowStep.from_dict(_mapping(item, "workflow step"))
                for item in raw_steps
            )
        )


@dataclass(frozen=True, slots=True)
class DefinitionProvenance:
    """Bounded, portable authorship and generation evidence."""

    author: str
    generator: str | None = None
    source_request_digest: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "author", _text(self.author, "author", 256))
        if self.generator is not None:
            object.__setattr__(
                self,
                "generator",
                _text(self.generator, "generator", 256),
            )
        if self.source_request_digest is not None:
            object.__setattr__(
                self,
                "source_request_digest",
                _digest(self.source_request_digest),
            )
        object.__setattr__(self, "created_at", _timestamp(self.created_at))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "author": self.author,
            "generator": self.generator,
            "source_request_digest": self.source_request_digest,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DefinitionProvenance:
        return cls(
            author=str(value["author"]),
            generator=_optional_string(value.get("generator")),
            source_request_digest=_optional_string(
                value.get("source_request_digest")
            ),
            created_at=str(value["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Portable behavior definition with deterministic canonical identity."""

    agent_id: str
    version: str
    prompt: VersionedReference
    model_profile: VersionedReference
    approval_policy: VersionedReference
    tools: tuple[ToolReference, ...] = ()
    skills: tuple[VersionedReference, ...] = ()
    triggers: tuple[TriggerDefinition, ...] = ()
    workflow: WorkflowGraph = field(default_factory=WorkflowGraph)
    budget: BudgetDefinition = field(default_factory=BudgetDefinition)
    locale: str = "en"
    provenance: DefinitionProvenance = field(
        default_factory=lambda: DefinitionProvenance(author="unknown")
    )
    schema_version: int = AGENT_DEFINITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_DEFINITION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported agent definition schema version: {self.schema_version}"
            )
        object.__setattr__(
            self,
            "agent_id",
            _identifier(self.agent_id, "agent id"),
        )
        object.__setattr__(self, "version", _semantic_version(self.version))
        tools = tuple(sorted(self.tools, key=lambda item: item.name))
        skills = tuple(sorted(self.skills, key=lambda item: item.name))
        if len({item.name for item in tools}) != len(tools):
            raise ValueError("agent definition tool names must be unique")
        if len({item.name for item in skills}) != len(skills):
            raise ValueError("agent definition skill names must be unique")
        workflow_tools = {step.tool.name: step.tool for step in self.workflow.steps}
        declared_tools = {tool.name: tool for tool in tools}
        for name, reference in workflow_tools.items():
            if declared_tools.get(name) != reference:
                raise ValueError(
                    f"workflow tool {name!r} is not pinned by the definition"
                )
        locale = self.locale.strip()
        if _LOCALE_PATTERN.fullmatch(locale) is None:
            raise ValueError("locale must be a BCP-47 language tag")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "triggers", tuple(self.triggers))
        object.__setattr__(self, "locale", locale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "version": self.version,
            "prompt": self.prompt.to_dict(),
            "model_profile": self.model_profile.to_dict(),
            "approval_policy": self.approval_policy.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
            "skills": [skill.to_dict() for skill in self.skills],
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "workflow": self.workflow.to_dict(),
            "budget": self.budget.to_dict(),
            "locale": self.locale,
            "provenance": self.provenance.to_dict(),
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentDefinition:
        _reject_unknown_fields(
            value,
            {
                "schema_version",
                "agent_id",
                "version",
                "prompt",
                "model_profile",
                "approval_policy",
                "tools",
                "skills",
                "triggers",
                "workflow",
                "budget",
                "locale",
                "provenance",
            },
            "agent definition",
        )
        prompt = _mapping(value.get("prompt"), "prompt")
        model_profile = _mapping(value.get("model_profile"), "model_profile")
        approval_policy = _mapping(
            value.get("approval_policy"), "approval_policy"
        )
        workflow = _mapping(value.get("workflow", {}), "workflow")
        budget = _mapping(value.get("budget", {}), "budget")
        provenance = _mapping(value.get("provenance"), "provenance")
        return cls(
            schema_version=int(
                value.get("schema_version", AGENT_DEFINITION_SCHEMA_VERSION)
            ),
            agent_id=str(value["agent_id"]),
            version=str(value["version"]),
            prompt=VersionedReference.from_dict(prompt),
            model_profile=VersionedReference.from_dict(model_profile),
            approval_policy=VersionedReference.from_dict(approval_policy),
            tools=tuple(
                ToolReference.from_dict(_mapping(item, "tool"))
                for item in _sequence(value.get("tools", ()), "tools")
            ),
            skills=tuple(
                VersionedReference.from_dict(_mapping(item, "skill"))
                for item in _sequence(value.get("skills", ()), "skills")
            ),
            triggers=tuple(
                TriggerDefinition.from_dict(_mapping(item, "trigger"))
                for item in _sequence(value.get("triggers", ()), "triggers")
            ),
            workflow=WorkflowGraph.from_dict(workflow),
            budget=BudgetDefinition.from_dict(budget),
            locale=str(value.get("locale", "en")),
            provenance=DefinitionProvenance.from_dict(provenance),
        )

    @classmethod
    def from_json(cls, value: str) -> AgentDefinition:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("agent definition JSON must contain an object")
        return cls.from_dict(decoded)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return a stable cross-process and cross-platform JSON encoding."""
    portable = _portable_mapping(value, field_path="root")
    return json.dumps(
        portable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def request_digest(value: Mapping[str, Any]) -> str:
    """Hash a structured source request without retaining its raw text."""
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _portable_mapping(
    value: Mapping[str, Any],
    *,
    field_path: str,
) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{field_path} keys must be strings")
        key = raw_key
        lowered = key.lower()
        if lowered in _SECRET_FIELD_MARKERS or any(
            lowered.endswith(f"_{marker}") for marker in _SECRET_FIELD_MARKERS
        ):
            raise ValueError(f"{field_path}.{key} cannot contain secret material")
        if lowered in _PATH_FIELD_MARKERS or lowered.endswith("_path"):
            raise ValueError(f"{field_path}.{key} cannot contain local paths")
        if lowered in _EXECUTABLE_FIELD_MARKERS:
            raise ValueError(
                f"{field_path}.{key} cannot contain executable imports"
            )
        clean[key] = _portable_value(raw_value, field_path=f"{field_path}.{key}")
    return clean


def _portable_value(value: Any, *, field_path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            if "\x00" in value:
                raise ValueError(f"{field_path} cannot contain NUL characters")
            if value.startswith(("file://", "/", "\\\\")):
                raise ValueError(f"{field_path} cannot contain local paths")
            if value.startswith(("./", "../", ".\\", "..\\")):
                raise ValueError(f"{field_path} cannot contain local paths")
            if re.match(r"^[A-Za-z]:[\\/]", value):
                raise ValueError(f"{field_path} cannot contain local paths")
            lowered = value.lower()
            if any(
                marker in lowered
                for marker in (
                    "api_key=",
                    "password=",
                    "private_key=",
                    "bearer ",
                )
            ):
                raise ValueError(f"{field_path} cannot contain secret material")
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{field_path} must be a finite number")
        return value
    if isinstance(value, PurePath):
        raise ValueError(f"{field_path} cannot contain local paths")
    if isinstance(value, Mapping):
        return _portable_mapping(value, field_path=field_path)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _portable_value(item, field_path=f"{field_path}[]")
            for item in value
        ]
    raise ValueError(f"{field_path} is not JSON-serializable")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _assert_acyclic(by_id: Mapping[str, WorkflowStep]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError("workflow dependencies must be acyclic")
        visiting.add(step_id)
        for dependency in by_id[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in sorted(by_id):
        visit(step_id)


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if _ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must start with a letter and contain only lowercase "
            "letters, digits, underscores, or hyphens"
        )
    return normalized


def _reference(value: str, field_name: str) -> str:
    normalized = value.strip()
    if _REFERENCE_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _semantic_version(value: str) -> str:
    normalized = value.strip()
    try:
        parsed = Version(normalized)
    except InvalidVersion as exc:
        raise ValueError("version must be semantic") from exc
    if parsed.epoch or len(parsed.release) != 3:
        raise ValueError("version must be semantic")
    return str(parsed)


def _digest(value: str) -> str:
    normalized = value.strip().lower()
    if _DIGEST_PATTERN.fullmatch(normalized) is None:
        raise ValueError("digest must be SHA-256")
    return (
        normalized if normalized.startswith("sha256:") else f"sha256:{normalized}"
    )


def _text(value: str, field_name: str, maximum: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum} characters")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} cannot contain NUL characters")
    return normalized


def _timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must use ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{field_name} must be an array")
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: " + ", ".join(sorted(unknown))
        )


__all__ = [
    "AGENT_DEFINITION_SCHEMA_VERSION",
    "AgentDefinition",
    "BudgetDefinition",
    "DefinitionProvenance",
    "DefinitionStatus",
    "ToolReference",
    "TriggerDefinition",
    "VersionedReference",
    "WorkflowApproval",
    "WorkflowEffect",
    "WorkflowGraph",
    "WorkflowStep",
    "canonical_json",
    "request_digest",
]
