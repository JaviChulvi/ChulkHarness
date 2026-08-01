"""Host-owned catalogs and publication state for portable definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from chulk.authoring.models import (
    AgentDefinition,
    DefinitionStatus,
    ToolReference,
    VersionedReference,
    canonical_json,
)
from chulk.hosting import ExecutionScope
from chulk.tools.policy import schema_digest
from chulk.tools.registry import Tool, ToolRegistry


class PublicationError(RuntimeError):
    """A portable artifact cannot make the requested lifecycle transition."""


class ArtifactAvailability(StrEnum):
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    findings: tuple[ValidationFinding, ...] = ()
    validator: str = "chulk"
    artifact_digest: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "findings": [finding.to_dict() for finding in self.findings],
            "validator": self.validator,
            "artifact_digest": self.artifact_digest,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    passed: bool
    cases: tuple[EvaluationCaseResult, ...]
    evaluator: str
    artifact_digest: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
            "evaluator": self.evaluator,
            "artifact_digest": self.artifact_digest,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class AgentDefinitionRecord:
    """One immutable definition revision plus host-owned lifecycle state."""

    definition: AgentDefinition
    status: DefinitionStatus = DefinitionStatus.DRAFT
    validation_report: ValidationReport | None = None
    evaluation_report: EvaluationReport | None = None
    reviewer: str | None = None
    published_at: str | None = None
    deprecated_at: str | None = None
    revoked_at: str | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None

    @property
    def identity(self) -> dict[str, str]:
        return {
            "agent_id": self.definition.agent_id,
            "version": self.definition.version,
            "digest": self.definition.digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "digest": self.definition.digest,
            "status": self.status.value,
            "validation_report": (
                self.validation_report.to_dict()
                if self.validation_report is not None
                else None
            ),
            "evaluation_report": (
                self.evaluation_report.to_dict()
                if self.evaluation_report is not None
                else None
            ),
            "reviewer": self.reviewer,
            "published_at": self.published_at,
            "deprecated_at": self.deprecated_at,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
            "revocation_reason": self.revocation_reason,
        }


@runtime_checkable
class AgentDefinitionStore(Protocol):
    """Sync host storage for immutable portable definition revisions."""

    def save_draft(
        self,
        scope: ExecutionScope,
        definition: AgentDefinition,
    ) -> AgentDefinitionRecord: ...

    def get(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...

    def publish(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        validation_report: ValidationReport,
        evaluation_report: EvaluationReport,
        reviewer: str,
    ) -> AgentDefinitionRecord: ...

    def deprecate(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...

    def revoke(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> AgentDefinitionRecord: ...

    def resolve_for_run(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...


@runtime_checkable
class AsyncAgentDefinitionStore(Protocol):
    """Async host storage contract matching :class:`AgentDefinitionStore`."""

    async def save_draft(
        self,
        scope: ExecutionScope,
        definition: AgentDefinition,
    ) -> AgentDefinitionRecord: ...

    async def get(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...

    async def publish(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        validation_report: ValidationReport,
        evaluation_report: EvaluationReport,
        reviewer: str,
    ) -> AgentDefinitionRecord: ...

    async def deprecate(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...

    async def revoke(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> AgentDefinitionRecord: ...

    async def resolve_for_run(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord: ...


class InMemoryAgentDefinitionStore:
    """Thread-safe tenant/workspace-scoped reference definition store."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str], AgentDefinitionRecord] = {}
        self._lock = RLock()

    def save_draft(
        self,
        scope: ExecutionScope,
        definition: AgentDefinition,
    ) -> AgentDefinitionRecord:
        _definition_matches_scope(scope, definition)
        key = _definition_key(scope, definition.agent_id, definition.version)
        with self._lock:
            current = self._records.get(key)
            if current is not None:
                if current.definition.digest != definition.digest:
                    raise PublicationError(
                        "agent definition version already identifies different behavior"
                    )
                return current
            record = AgentDefinitionRecord(definition=definition)
            self._records[key] = record
            return record

    def get(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        key = _definition_key(scope, agent_id, version)
        with self._lock:
            try:
                return self._records[key]
            except KeyError as exc:
                raise KeyError(
                    f"agent definition {agent_id!r}@{version!r} does not exist"
                ) from exc

    def publish(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        validation_report: ValidationReport,
        evaluation_report: EvaluationReport,
        reviewer: str,
    ) -> AgentDefinitionRecord:
        reviewer = _operator(reviewer, "reviewer")
        if not validation_report.valid:
            raise PublicationError("invalid agent definitions cannot be published")
        if not evaluation_report.passed:
            raise PublicationError(
                "agent definitions with failed evaluations cannot be published"
            )
        with self._lock:
            current = self.get(scope, agent_id, version)
            _reports_match_digest(
                current.definition.digest,
                validation_report,
                evaluation_report,
            )
            if current.status is DefinitionStatus.REVOKED:
                raise PublicationError("revoked agent definitions cannot be published")
            if current.status is DefinitionStatus.PUBLISHED:
                return current
            published = replace(
                current,
                status=DefinitionStatus.PUBLISHED,
                validation_report=validation_report,
                evaluation_report=evaluation_report,
                reviewer=reviewer,
                published_at=_now(),
                deprecated_at=None,
            )
            self._records[_definition_key(scope, agent_id, version)] = published
            return published

    def deprecate(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        with self._lock:
            current = self.get(scope, agent_id, version)
            if current.status is not DefinitionStatus.PUBLISHED:
                raise PublicationError("only published definitions may be deprecated")
            deprecated = replace(
                current,
                status=DefinitionStatus.DEPRECATED,
                deprecated_at=_now(),
            )
            self._records[_definition_key(scope, agent_id, version)] = deprecated
            return deprecated

    def revoke(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> AgentDefinitionRecord:
        revoked_by = _operator(revoked_by, "revoked_by")
        reason = _operator(reason, "reason", maximum=2_000)
        with self._lock:
            current = self.get(scope, agent_id, version)
            if current.status is DefinitionStatus.DRAFT:
                raise PublicationError("draft definitions should be deleted, not revoked")
            if current.status is DefinitionStatus.REVOKED:
                return current
            revoked = replace(
                current,
                status=DefinitionStatus.REVOKED,
                revoked_at=_now(),
                revoked_by=revoked_by,
                revocation_reason=reason,
            )
            self._records[_definition_key(scope, agent_id, version)] = revoked
            return revoked

    def resolve_for_run(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        record = self.get(scope, agent_id, version)
        _definition_matches_scope(scope, record.definition)
        if record.status is DefinitionStatus.DRAFT:
            raise PublicationError("draft agent definitions cannot execute")
        if record.status is DefinitionStatus.REVOKED:
            raise PublicationError("revoked agent definitions cannot start new runs")
        return record

    def list(
        self,
        scope: ExecutionScope,
        *,
        agent_id: str | None = None,
    ) -> tuple[AgentDefinitionRecord, ...]:
        prefix = (scope.tenant_id, scope.workspace_id)
        with self._lock:
            records = [
                record
                for key, record in self._records.items()
                if key[:2] == prefix
                and (agent_id is None or record.definition.agent_id == agent_id)
            ]
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.definition.agent_id,
                    record.definition.version,
                ),
            )
        )


class AsyncInMemoryAgentDefinitionStore:
    """Native async contract adapter for the in-memory reference store."""

    def __init__(self, store: InMemoryAgentDefinitionStore | None = None) -> None:
        self.sync_store = store or InMemoryAgentDefinitionStore()

    async def save_draft(
        self,
        scope: ExecutionScope,
        definition: AgentDefinition,
    ) -> AgentDefinitionRecord:
        return self.sync_store.save_draft(scope, definition)

    async def get(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        return self.sync_store.get(scope, agent_id, version)

    async def publish(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        validation_report: ValidationReport,
        evaluation_report: EvaluationReport,
        reviewer: str,
    ) -> AgentDefinitionRecord:
        return self.sync_store.publish(
            scope,
            agent_id,
            version,
            validation_report=validation_report,
            evaluation_report=evaluation_report,
            reviewer=reviewer,
        )

    async def deprecate(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        return self.sync_store.deprecate(scope, agent_id, version)

    async def revoke(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> AgentDefinitionRecord:
        return self.sync_store.revoke(
            scope,
            agent_id,
            version,
            revoked_by=revoked_by,
            reason=reason,
        )

    async def resolve_for_run(
        self,
        scope: ExecutionScope,
        agent_id: str,
        version: str,
    ) -> AgentDefinitionRecord:
        return self.sync_store.resolve_for_run(scope, agent_id, version)


@dataclass(frozen=True, slots=True)
class PublishedTool:
    tool: Tool
    reference: ToolReference
    availability: ArtifactAvailability = ArtifactAvailability.PUBLISHED


class ToolCatalog:
    """Explicit trusted tool selection; hidden tools are never discoverable."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[tuple[str, str], PublishedTool] = {}
        self._active: dict[str, ToolReference] = {}
        for tool in tools:
            self.publish(tool)

    def publish(self, tool: Tool) -> PublishedTool:
        registry = ToolRegistry()
        registry.register(tool)
        resolved = registry.get(tool.name)
        identity = resolved.resolved_identity()
        policy = resolved.resolved_policy()
        reference = ToolReference(
            name=identity.name,
            version=identity.version,
            digest="sha256:" + identity.digest,
            input_schema_version=identity.input_schema_version,
            input_schema_digest="sha256:" + identity.input_schema_digest,
            output_schema_version=identity.output_schema_version,
            output_schema_digest="sha256:" + identity.output_schema_digest,
            implementation_digest="sha256:" + identity.implementation_digest,
            policy_version=policy.version,
            policy_digest="sha256:" + schema_digest(policy.to_dict()),
        )
        key = (reference.name, reference.version)
        current = self._tools.get(key)
        if current is not None and current.reference != reference:
            raise PublicationError(
                f"tool {resolved.name!r}@{reference.version!r} is already "
                "published with another identity"
            )
        published = PublishedTool(tool=resolved, reference=reference)
        self._tools[key] = published
        self._active[resolved.name] = reference
        return published

    def reference(
        self,
        name: str,
        version: str | None = None,
    ) -> ToolReference:
        return self._published(name, version).reference

    def resolve(self, reference: ToolReference) -> Tool:
        published = self._published(reference.name, reference.version)
        if published.reference != reference:
            raise PublicationError(
                f"tool {reference.name!r} does not match its published identity"
            )
        return published.tool

    def deprecate(
        self,
        name: str,
        version: str | None = None,
    ) -> PublishedTool:
        current = self._published(name, version)
        deprecated = replace(
            current,
            availability=ArtifactAvailability.DEPRECATED,
        )
        self._tools[(current.reference.name, current.reference.version)] = deprecated
        return deprecated

    def revoke(
        self,
        name: str,
        version: str | None = None,
    ) -> PublishedTool:
        current = self._published(name, version)
        revoked = replace(current, availability=ArtifactAvailability.REVOKED)
        key = (current.reference.name, current.reference.version)
        self._tools[key] = revoked
        if self._active.get(name) == current.reference:
            self._active.pop(name, None)
        return revoked

    def activate(self, reference: ToolReference) -> PublishedTool:
        """Select one exact non-revoked revision for new compilations."""
        published = self._published(reference.name, reference.version)
        if published.reference != reference:
            raise PublicationError("tool reference digest does not match")
        self._active[reference.name] = reference
        return published

    def selected(self, names: Sequence[str]) -> tuple[PublishedTool, ...]:
        """Resolve only caller-selected names without catalog-wide discovery."""
        return tuple(self._published(name) for name in names)

    def _published(
        self,
        name: str,
        version: str | None = None,
    ) -> PublishedTool:
        if version is None:
            try:
                reference = self._active[name]
            except KeyError as exc:
                raise KeyError(
                    f"tool {name!r} is not in the published catalog "
                    "with an active revision"
                ) from exc
            version = reference.version
        try:
            published = self._tools[(name, version)]
        except KeyError as exc:
            raise KeyError(
                f"tool {name!r}@{version!r} is not in the published catalog"
            ) from exc
        if published.availability is ArtifactAvailability.REVOKED:
            raise PublicationError(f"tool {name!r}@{version!r} has been revoked")
        return published


@dataclass(frozen=True, slots=True)
class PublishedPrompt:
    reference: VersionedReference
    content: str
    availability: ArtifactAvailability = ArtifactAvailability.PUBLISHED


class PromptCatalog:
    """Host-owned immutable prompt text catalog."""

    def __init__(self) -> None:
        self._prompts: dict[tuple[str, str], PublishedPrompt] = {}

    def publish(self, *, name: str, version: str, content: str) -> PublishedPrompt:
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("prompt content cannot be empty")
        if "\x00" in clean_content:
            raise ValueError("prompt content cannot contain NUL characters")
        digest = "sha256:" + schema_digest({"content": clean_content})
        reference = VersionedReference(name=name, version=version, digest=digest)
        key = (reference.name, reference.version)
        current = self._prompts.get(key)
        if current is not None and current.reference != reference:
            raise PublicationError(
                "prompt version already identifies different content"
            )
        published = PublishedPrompt(reference=reference, content=clean_content)
        self._prompts[key] = published
        return published

    def resolve(self, reference: VersionedReference) -> str:
        try:
            published = self._prompts[(reference.name, reference.version)]
        except KeyError as exc:
            raise KeyError(
                f"prompt {reference.name!r}@{reference.version!r} is not published"
            ) from exc
        if published.reference != reference:
            raise PublicationError("prompt reference digest does not match")
        if published.availability is ArtifactAvailability.REVOKED:
            raise PublicationError("revoked prompts cannot start new runs")
        return published.content


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Immutable non-secret JSON artifact such as a model or approval policy."""

    reference: VersionedReference
    payload: Mapping[str, Any]
    availability: ArtifactAvailability = ArtifactAvailability.PUBLISHED


class ArtifactCatalog:
    """Host-owned immutable catalog for portable policy documents."""

    def __init__(self) -> None:
        self._artifacts: dict[tuple[str, str], PublishedArtifact] = {}

    def publish(
        self,
        *,
        name: str,
        version: str,
        payload: Mapping[str, Any],
    ) -> PublishedArtifact:
        clean_payload = json.loads(canonical_json(payload))
        if not isinstance(clean_payload, dict):
            raise ValueError("artifact payload must be an object")
        digest = "sha256:" + schema_digest(clean_payload)
        reference = VersionedReference(name=name, version=version, digest=digest)
        key = (reference.name, reference.version)
        current = self._artifacts.get(key)
        if current is not None and current.reference != reference:
            raise PublicationError(
                "artifact version already identifies different content"
            )
        artifact = PublishedArtifact(
            reference=reference,
            payload=_freeze_json_mapping(clean_payload),
        )
        self._artifacts[key] = artifact
        return artifact

    def resolve(self, reference: VersionedReference) -> Mapping[str, Any]:
        try:
            artifact = self._artifacts[(reference.name, reference.version)]
        except KeyError as exc:
            raise KeyError(
                f"artifact {reference.name!r}@{reference.version!r} "
                "is not published"
            ) from exc
        if artifact.reference != reference:
            raise PublicationError("artifact reference digest does not match")
        if artifact.availability is ArtifactAvailability.REVOKED:
            raise PublicationError("revoked artifacts cannot start new runs")
        return dict(artifact.payload)


def validate_definition_catalogs(
    definition: AgentDefinition,
    *,
    tools: ToolCatalog,
    prompts: PromptCatalog,
    model_profiles: ArtifactCatalog | None = None,
    approval_policies: ArtifactCatalog | None = None,
    skill_references: Mapping[str, VersionedReference] | None = None,
) -> ValidationReport:
    """Validate every pinned definition dependency without executing it."""
    if skill_references is None:
        skill_references = {}
    findings: list[ValidationFinding] = []
    try:
        prompts.resolve(definition.prompt)
    except (KeyError, PublicationError) as exc:
        findings.append(
            ValidationFinding(
                code="prompt_unavailable",
                field="prompt",
                message=str(exc),
            )
        )
    for field_name, reference, catalog in (
        ("model_profile", definition.model_profile, model_profiles),
        ("approval_policy", definition.approval_policy, approval_policies),
    ):
        if catalog is None:
            continue
        try:
            catalog.resolve(reference)
        except (KeyError, PublicationError) as exc:
            findings.append(
                ValidationFinding(
                    code=f"{field_name}_unavailable",
                    field=field_name,
                    message=str(exc),
                )
            )
    for reference in definition.tools:
        try:
            tools.resolve(reference)
        except (KeyError, PublicationError) as exc:
            findings.append(
                ValidationFinding(
                    code="tool_unavailable",
                    field=f"tools.{reference.name}",
                    message=str(exc),
                )
            )
    for reference in definition.skills:
        current = skill_references.get(reference.name)
        if current != reference:
            findings.append(
                ValidationFinding(
                    code="skill_unavailable",
                    field=f"skills.{reference.name}",
                    message="skill is not published with the pinned identity",
                )
            )
    return ValidationReport(
        valid=not findings,
        findings=tuple(findings),
        artifact_digest=definition.digest,
    )


def _reports_match_digest(
    expected_digest: str,
    validation_report: ValidationReport,
    evaluation_report: EvaluationReport,
) -> None:
    if validation_report.artifact_digest != expected_digest:
        raise PublicationError(
            "validation report does not match the agent definition digest"
        )
    if evaluation_report.artifact_digest != expected_digest:
        raise PublicationError(
            "evaluation report does not match the agent definition digest"
        )


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_json_value(item)
            for key, item in value.items()
        }
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _definition_matches_scope(
    scope: ExecutionScope,
    definition: AgentDefinition,
) -> None:
    if scope.agent_id != definition.agent_id:
        raise PublicationError("execution scope agent does not match definition")
    if scope.agent_version != definition.version:
        raise PublicationError(
            "execution scope agent version does not match definition"
        )


def _definition_key(
    scope: ExecutionScope,
    agent_id: str,
    version: str,
) -> tuple[str, str, str, str]:
    return (scope.tenant_id, scope.workspace_id, agent_id, version)


def _operator(value: str, field_name: str, *, maximum: int = 256) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} cannot be empty")
    if len(clean) > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum} characters")
    return clean


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AgentDefinitionRecord",
    "AgentDefinitionStore",
    "ArtifactCatalog",
    "ArtifactAvailability",
    "AsyncAgentDefinitionStore",
    "AsyncInMemoryAgentDefinitionStore",
    "EvaluationCaseResult",
    "EvaluationReport",
    "InMemoryAgentDefinitionStore",
    "PromptCatalog",
    "PublicationError",
    "PublishedArtifact",
    "PublishedPrompt",
    "PublishedTool",
    "ToolCatalog",
    "ValidationFinding",
    "ValidationReport",
    "validate_definition_catalogs",
]
