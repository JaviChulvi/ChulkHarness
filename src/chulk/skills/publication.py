"""Portable host-owned skill validation, publication, and revocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from chulk.authoring.catalog import (
    EvaluationCaseResult,
    EvaluationReport,
    PublicationError,
    ToolCatalog,
    ValidationFinding,
    ValidationReport,
)
from chulk.authoring.models import (
    DefinitionProvenance,
    ToolReference,
    VersionedReference,
)
from chulk.hosting import ExecutionScope
from chulk.skills.lifecycle_models import SkillLifecycleStatus
from chulk.tools.policy import schema_digest


@dataclass(frozen=True, slots=True)
class PortableSkill:
    """Declarative procedural instructions with only pinned dependencies."""

    name: str
    version: str
    description: str
    instructions: str
    required_tools: tuple[ToolReference, ...] = ()
    includes: tuple[VersionedReference, ...] = ()
    provenance: DefinitionProvenance = field(
        default_factory=lambda: DefinitionProvenance(author="unknown")
    )

    def __post_init__(self) -> None:
        reference = VersionedReference(
            name=self.name,
            version=self.version,
            digest="sha256:" + ("0" * 64),
        )
        object.__setattr__(self, "name", reference.name)
        object.__setattr__(self, "version", reference.version)
        description = _text(self.description, "description", maximum=1_000)
        instructions = _text(
            self.instructions,
            "instructions",
            maximum=100_000,
        )
        _validate_instruction_text(instructions)
        tools = tuple(sorted(self.required_tools, key=lambda item: item.name))
        includes = tuple(sorted(self.includes, key=lambda item: item.name))
        if len({item.name for item in tools}) != len(tools):
            raise ValueError("portable skill tool references must be unique")
        if len({item.name for item in includes}) != len(includes):
            raise ValueError("portable skill includes must be unique")
        if self.name in {item.name for item in includes}:
            raise ValueError("portable skill cannot include itself")
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "required_tools", tools)
        object.__setattr__(self, "includes", includes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "instructions": self.instructions,
            "required_tools": [
                reference.to_dict() for reference in self.required_tools
            ],
            "includes": [reference.to_dict() for reference in self.includes],
            "provenance": self.provenance.to_dict(),
        }

    @property
    def digest(self) -> str:
        return "sha256:" + schema_digest(self.to_dict())

    @property
    def reference(self) -> VersionedReference:
        return VersionedReference(
            name=self.name,
            version=self.version,
            digest=self.digest,
        )


@dataclass(frozen=True, slots=True)
class SkillPublicationRecord:
    skill: PortableSkill
    status: SkillLifecycleStatus = SkillLifecycleStatus.DRAFT
    validation_report: ValidationReport | None = None
    evaluation_report: EvaluationReport | None = None
    reviewer: str | None = None
    published_at: str | None = None
    deprecated_at: str | None = None
    revoked_at: str | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None
    affected_definitions: tuple[str, ...] = ()

    @property
    def reference(self) -> VersionedReference:
        return self.skill.reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "digest": self.skill.digest,
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
            "affected_definitions": list(self.affected_definitions),
        }


@dataclass(frozen=True, slots=True)
class SkillActivationRecord:
    """Immutable provenance for publication and rollback pointer changes."""

    reference: VersionedReference
    activated_by: str
    reason: str
    previous_reference: VersionedReference | None = None
    activated_at: str = field(default_factory=lambda: _now())


@runtime_checkable
class SkillPublicationStore(Protocol):
    """Host-backed sync store for portable governed skill revisions."""

    def save_draft(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> SkillPublicationRecord: ...

    def get(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord: ...

    def replace(
        self,
        scope: ExecutionScope,
        record: SkillPublicationRecord,
    ) -> SkillPublicationRecord: ...

    def list(
        self,
        scope: ExecutionScope,
    ) -> tuple[SkillPublicationRecord, ...]: ...

    def activate(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        activated_by: str,
        reason: str,
    ) -> SkillPublicationRecord: ...

    def active(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> SkillPublicationRecord: ...

    def activation_history(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> tuple[SkillActivationRecord, ...]: ...


@runtime_checkable
class AsyncSkillPublicationStore(Protocol):
    """Native async store contract matching :class:`SkillPublicationStore`."""

    async def save_draft(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> SkillPublicationRecord: ...

    async def get(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord: ...

    async def replace(
        self,
        scope: ExecutionScope,
        record: SkillPublicationRecord,
    ) -> SkillPublicationRecord: ...

    async def list(
        self,
        scope: ExecutionScope,
    ) -> tuple[SkillPublicationRecord, ...]: ...

    async def activate(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        activated_by: str,
        reason: str,
    ) -> SkillPublicationRecord: ...

    async def active(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> SkillPublicationRecord: ...

    async def activation_history(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> tuple[SkillActivationRecord, ...]: ...


class InMemorySkillPublicationStore:
    """Thread-safe tenant/workspace-scoped publication reference store."""

    def __init__(self) -> None:
        self._records: dict[
            tuple[str, str, str, str], SkillPublicationRecord
        ] = {}
        self._active: dict[tuple[str, str, str], VersionedReference] = {}
        self._activation_history: dict[
            tuple[str, str, str], list[SkillActivationRecord]
        ] = {}
        self._lock = RLock()

    def save_draft(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> SkillPublicationRecord:
        key = _key(scope, skill.name, skill.version)
        with self._lock:
            current = self._records.get(key)
            if current is not None:
                if current.skill.digest != skill.digest:
                    raise PublicationError(
                        "skill version already identifies different content"
                    )
                return current
            record = SkillPublicationRecord(skill=skill)
            self._records[key] = record
            return record

    def get(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord:
        with self._lock:
            try:
                return self._records[_key(scope, name, version)]
            except KeyError as exc:
                raise KeyError(f"skill {name!r}@{version!r} does not exist") from exc

    def replace(
        self,
        scope: ExecutionScope,
        record: SkillPublicationRecord,
    ) -> SkillPublicationRecord:
        key = _key(scope, record.skill.name, record.skill.version)
        with self._lock:
            current = self._records.get(key)
            if current is None:
                raise KeyError(
                    f"skill {record.skill.name!r}@{record.skill.version!r} "
                    "does not exist"
                )
            if current.skill.digest != record.skill.digest:
                raise PublicationError("published skill revisions are immutable")
            self._records[key] = record
            return record

    def list(
        self,
        scope: ExecutionScope,
    ) -> tuple[SkillPublicationRecord, ...]:
        prefix = (scope.tenant_id, scope.workspace_id)
        with self._lock:
            records = [
                record
                for key, record in self._records.items()
                if key[:2] == prefix
            ]
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.skill.name,
                    record.skill.version,
                ),
            )
        )

    def activate(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        activated_by: str,
        reason: str,
    ) -> SkillPublicationRecord:
        activated_by = _text(activated_by, "activated_by", maximum=256)
        reason = _text(reason, "reason", maximum=2_000)
        with self._lock:
            record = self.get(scope, reference.name, reference.version)
            if record.reference != reference:
                raise PublicationError("skill reference digest does not match")
            if record.status not in {
                SkillLifecycleStatus.PUBLISHED,
                SkillLifecycleStatus.DEPRECATED,
            }:
                raise PublicationError(
                    "only published skill revisions may become active"
                )
            active_key = _active_key(scope, reference.name)
            previous = self._active.get(active_key)
            self._active[active_key] = reference
            self._activation_history.setdefault(active_key, []).append(
                SkillActivationRecord(
                    reference=reference,
                    activated_by=activated_by,
                    reason=reason,
                    previous_reference=previous,
                )
            )
            return record

    def active(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> SkillPublicationRecord:
        with self._lock:
            try:
                reference = self._active[_active_key(scope, name)]
            except KeyError as exc:
                raise KeyError(f"skill {name!r} has no active revision") from exc
            record = self.get(scope, reference.name, reference.version)
            if record.status is SkillLifecycleStatus.REVOKED:
                raise PublicationError("revoked skills cannot execute")
            return record

    def activation_history(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> tuple[SkillActivationRecord, ...]:
        with self._lock:
            return tuple(
                self._activation_history.get(_active_key(scope, name), ())
            )


class AsyncInMemorySkillPublicationStore:
    """Native async adapter used by hosted contract tests."""

    def __init__(self, store: InMemorySkillPublicationStore | None = None) -> None:
        self.sync_store = store or InMemorySkillPublicationStore()

    async def save_draft(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> SkillPublicationRecord:
        return self.sync_store.save_draft(scope, skill)

    async def get(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord:
        return self.sync_store.get(scope, name, version)

    async def replace(
        self,
        scope: ExecutionScope,
        record: SkillPublicationRecord,
    ) -> SkillPublicationRecord:
        return self.sync_store.replace(scope, record)

    async def list(
        self,
        scope: ExecutionScope,
    ) -> tuple[SkillPublicationRecord, ...]:
        return self.sync_store.list(scope)

    async def activate(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        activated_by: str,
        reason: str,
    ) -> SkillPublicationRecord:
        return self.sync_store.activate(
            scope,
            reference,
            activated_by=activated_by,
            reason=reason,
        )

    async def active(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> SkillPublicationRecord:
        return self.sync_store.active(scope, name)

    async def activation_history(
        self,
        scope: ExecutionScope,
        name: str,
    ) -> tuple[SkillActivationRecord, ...]:
        return self.sync_store.activation_history(scope, name)


class SkillPublicationManager:
    """Validate and transition portable skill revisions through host review."""

    def __init__(
        self,
        store: SkillPublicationStore,
        *,
        tools: ToolCatalog,
        affected_definitions: Callable[
            [ExecutionScope, VersionedReference], tuple[str, ...]
        ]
        | None = None,
    ) -> None:
        self.store = store
        self.tools = tools
        self.affected_definitions = affected_definitions

    def submit(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
        *,
        evaluator: str = "chulk-skill-evaluator",
    ) -> SkillPublicationRecord:
        draft = self.store.save_draft(scope, skill)
        if draft.status is not SkillLifecycleStatus.DRAFT:
            return draft
        validating = self.store.replace(
            scope,
            replace(draft, status=SkillLifecycleStatus.VALIDATING),
        )
        validation = self.validate(scope, skill)
        if not validation.valid:
            return self.store.replace(
                scope,
                replace(validating, validation_report=validation),
            )
        evaluating = self.store.replace(
            scope,
            replace(
                validating,
                status=SkillLifecycleStatus.EVALUATING,
                validation_report=validation,
            ),
        )
        evaluation = self.evaluate(scope, skill, evaluator=evaluator)
        return self.store.replace(
            scope,
            replace(
                evaluating,
                status=(
                    SkillLifecycleStatus.AWAITING_REVIEW
                    if evaluation.passed
                    else SkillLifecycleStatus.EVALUATING
                ),
                evaluation_report=evaluation,
            ),
        )

    def validate(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> ValidationReport:
        findings: list[ValidationFinding] = []
        findings.extend(
            _skill_validation_findings(
                skill,
                tools=self.tools,
                records=self.store.list(scope),
            )
        )
        return ValidationReport(
            valid=not findings,
            findings=tuple(findings),
            validator="chulk-skill-validator",
            artifact_digest=skill.digest,
            created_at=skill.provenance.created_at,
        )

    def evaluate(
        self,
        _scope: ExecutionScope,
        skill: PortableSkill,
        *,
        evaluator: str,
    ) -> EvaluationReport:
        return _skill_evaluation(skill, evaluator=evaluator)

    def publish(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
        *,
        reviewer: str,
    ) -> SkillPublicationRecord:
        current = self.store.get(scope, name, version)
        if current.status is not SkillLifecycleStatus.AWAITING_REVIEW:
            raise PublicationError("skill must pass evaluation before publication")
        if current.validation_report is None or not current.validation_report.valid:
            raise PublicationError("skill validation did not pass")
        if current.evaluation_report is None or not current.evaluation_report.passed:
            raise PublicationError("skill evaluation did not pass")
        _skill_reports_match(current)
        published = self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.PUBLISHED,
                reviewer=_text(reviewer, "reviewer", maximum=256),
                published_at=_now(),
            ),
        )
        self.store.activate(
            scope,
            published.reference,
            activated_by=published.reviewer or "host",
            reason="publication",
        )
        return published

    def deprecate(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord:
        current = self.store.get(scope, name, version)
        if current.status is not SkillLifecycleStatus.PUBLISHED:
            raise PublicationError("only published skills may be deprecated")
        return self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.DEPRECATED,
                deprecated_at=_now(),
            ),
        )

    def revoke(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> SkillPublicationRecord:
        current = self.store.get(scope, name, version)
        if current.status not in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }:
            raise PublicationError("only published skills may be revoked")
        affected = (
            self.affected_definitions(scope, current.reference)
            if self.affected_definitions is not None
            else ()
        )
        return self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.REVOKED,
                revoked_at=_now(),
                revoked_by=_text(revoked_by, "revoked_by", maximum=256),
                revocation_reason=_text(reason, "reason", maximum=2_000),
                affected_definitions=tuple(sorted(set(affected))),
            ),
        )

    def rollback(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        approved_by: str,
    ) -> SkillPublicationRecord:
        return self.store.activate(
            scope,
            reference,
            activated_by=_text(approved_by, "approved_by", maximum=256),
            reason="rollback",
        )

    def resolve_for_run(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
    ) -> PortableSkill:
        record = self.store.get(scope, reference.name, reference.version)
        if record.reference != reference:
            raise PublicationError("skill reference digest does not match")
        if record.status not in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }:
            raise PublicationError("unpublished or revoked skills cannot execute")
        return record.skill


class AsyncSkillPublicationManager:
    """Async lifecycle owner using native async store calls."""

    def __init__(
        self,
        store: AsyncSkillPublicationStore,
        *,
        tools: ToolCatalog,
        affected_definitions: Callable[
            [ExecutionScope, VersionedReference], tuple[str, ...]
        ]
        | None = None,
    ) -> None:
        self.store = store
        self.tools = tools
        self.affected_definitions = affected_definitions

    async def submit(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
        *,
        evaluator: str = "chulk-skill-evaluator",
    ) -> SkillPublicationRecord:
        draft = await self.store.save_draft(scope, skill)
        if draft.status is not SkillLifecycleStatus.DRAFT:
            return draft
        validating = await self.store.replace(
            scope,
            replace(draft, status=SkillLifecycleStatus.VALIDATING),
        )
        validation = await self.validate(scope, skill)
        if not validation.valid:
            return await self.store.replace(
                scope,
                replace(validating, validation_report=validation),
            )
        evaluation = await self.evaluate(
            scope,
            skill,
            evaluator=evaluator,
        )
        return await self.store.replace(
            scope,
            replace(
                validating,
                status=SkillLifecycleStatus.AWAITING_REVIEW,
                validation_report=validation,
                evaluation_report=evaluation,
            ),
        )

    async def publish(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
        *,
        reviewer: str,
    ) -> SkillPublicationRecord:
        current = await self.store.get(scope, name, version)
        if current.status is not SkillLifecycleStatus.AWAITING_REVIEW:
            raise PublicationError("skill must pass evaluation before publication")
        if current.validation_report is None or not current.validation_report.valid:
            raise PublicationError("skill validation did not pass")
        if current.evaluation_report is None or not current.evaluation_report.passed:
            raise PublicationError("skill evaluation did not pass")
        _skill_reports_match(current)
        published = await self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.PUBLISHED,
                reviewer=_text(reviewer, "reviewer", maximum=256),
                published_at=_now(),
            ),
        )
        await self.store.activate(
            scope,
            published.reference,
            activated_by=published.reviewer or "host",
            reason="publication",
        )
        return published

    async def validate(
        self,
        scope: ExecutionScope,
        skill: PortableSkill,
    ) -> ValidationReport:
        findings = _skill_validation_findings(
            skill,
            tools=self.tools,
            records=await self.store.list(scope),
        )
        return ValidationReport(
            valid=not findings,
            findings=findings,
            validator="chulk-skill-validator",
            artifact_digest=skill.digest,
            created_at=skill.provenance.created_at,
        )

    async def evaluate(
        self,
        _scope: ExecutionScope,
        skill: PortableSkill,
        *,
        evaluator: str,
    ) -> EvaluationReport:
        return _skill_evaluation(skill, evaluator=evaluator)

    async def deprecate(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
    ) -> SkillPublicationRecord:
        current = await self.store.get(scope, name, version)
        if current.status is not SkillLifecycleStatus.PUBLISHED:
            raise PublicationError("only published skills may be deprecated")
        return await self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.DEPRECATED,
                deprecated_at=_now(),
            ),
        )

    async def revoke(
        self,
        scope: ExecutionScope,
        name: str,
        version: str,
        *,
        revoked_by: str,
        reason: str,
    ) -> SkillPublicationRecord:
        current = await self.store.get(scope, name, version)
        if current.status not in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }:
            raise PublicationError("only published skills may be revoked")
        affected = (
            self.affected_definitions(scope, current.reference)
            if self.affected_definitions is not None
            else ()
        )
        return await self.store.replace(
            scope,
            replace(
                current,
                status=SkillLifecycleStatus.REVOKED,
                revoked_at=_now(),
                revoked_by=_text(revoked_by, "revoked_by", maximum=256),
                revocation_reason=_text(reason, "reason", maximum=2_000),
                affected_definitions=tuple(sorted(set(affected))),
            ),
        )

    async def rollback(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
        *,
        approved_by: str,
    ) -> SkillPublicationRecord:
        return await self.store.activate(
            scope,
            reference,
            activated_by=_text(approved_by, "approved_by", maximum=256),
            reason="rollback",
        )

    async def resolve_for_run(
        self,
        scope: ExecutionScope,
        reference: VersionedReference,
    ) -> PortableSkill:
        record = await self.store.get(scope, reference.name, reference.version)
        if record.reference != reference:
            raise PublicationError("skill reference digest does not match")
        if record.status not in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }:
            raise PublicationError("unpublished or revoked skills cannot execute")
        return record.skill


def _skill_validation_findings(
    candidate: PortableSkill,
    *,
    tools: ToolCatalog,
    records: tuple[SkillPublicationRecord, ...],
) -> tuple[ValidationFinding, ...]:
    findings: list[ValidationFinding] = []
    for reference in candidate.required_tools:
        try:
            tools.resolve(reference)
        except (KeyError, PublicationError) as exc:
            findings.append(
                ValidationFinding(
                    code="tool_unavailable",
                    field=f"required_tools.{reference.name}",
                    message=str(exc),
                )
            )
    published = {
        record.reference: record
        for record in records
        if record.status
        in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }
    }
    for include in candidate.includes:
        if include not in published:
            findings.append(
                ValidationFinding(
                    code="include_unavailable",
                    field=f"includes.{include.name}",
                    message="include must pin a published version and digest",
                )
            )
    if findings:
        return tuple(findings)
    try:
        _assert_no_include_cycle(candidate, records)
    except PublicationError as exc:
        findings.append(
            ValidationFinding(
                code="include_cycle",
                field="includes",
                message=str(exc),
            )
        )
    return tuple(findings)


def _assert_no_include_cycle(
    candidate: PortableSkill,
    records: tuple[SkillPublicationRecord, ...],
) -> None:
    graph: dict[VersionedReference, tuple[VersionedReference, ...]] = {
        record.reference: record.skill.includes
        for record in records
    }
    graph[candidate.reference] = candidate.includes
    visiting: set[VersionedReference] = set()
    visited: set[VersionedReference] = set()

    def visit(reference: VersionedReference) -> None:
        if reference in visited:
            return
        if reference in visiting:
            raise PublicationError("skill include graph contains a cycle")
        visiting.add(reference)
        for included in graph.get(reference, ()):
            visit(included)
        visiting.remove(reference)
        visited.add(reference)

    for reference in sorted(
        graph,
        key=lambda item: (item.name, item.version, item.digest),
    ):
        visit(reference)


def _skill_evaluation(
    skill: PortableSkill,
    *,
    evaluator: str,
) -> EvaluationReport:
    cases = (
        EvaluationCaseResult(
            name="instructions_present",
            passed=bool(skill.instructions),
            detail="procedural instructions are present",
        ),
        EvaluationCaseResult(
            name="dependencies_pinned",
            passed=all(reference.digest for reference in skill.includes),
            detail="all includes carry immutable versions and digests",
        ),
        EvaluationCaseResult(
            name="declarative_only",
            passed=True,
            detail="portable skill model exposes no executable resource fields",
        ),
    )
    return EvaluationReport(
        passed=all(case.passed for case in cases),
        cases=cases,
        evaluator=_text(evaluator, "evaluator", maximum=256),
        artifact_digest=skill.digest,
        created_at=skill.provenance.created_at,
    )


def _skill_reports_match(record: SkillPublicationRecord) -> None:
    validation = record.validation_report
    evaluation = record.evaluation_report
    if validation is None or validation.artifact_digest != record.skill.digest:
        raise PublicationError("validation report does not match the skill digest")
    if evaluation is None or evaluation.artifact_digest != record.skill.digest:
        raise PublicationError("evaluation report does not match the skill digest")


def skill_reference_map(
    records: tuple[SkillPublicationRecord, ...],
) -> Mapping[str, VersionedReference]:
    return {
        record.skill.name: record.reference
        for record in records
        if record.status
        in {
            SkillLifecycleStatus.PUBLISHED,
            SkillLifecycleStatus.DEPRECATED,
        }
    }


def _key(
    scope: ExecutionScope,
    name: str,
    version: str,
) -> tuple[str, str, str, str]:
    return (scope.tenant_id, scope.workspace_id, name, version)


def _active_key(
    scope: ExecutionScope,
    name: str,
) -> tuple[str, str, str]:
    return (scope.tenant_id, scope.workspace_id, name)


def _text(value: str, field_name: str, *, maximum: int) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} cannot be empty")
    if len(clean) > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum} characters")
    if "\x00" in clean:
        raise ValueError(f"{field_name} cannot contain NUL characters")
    return clean


def _validate_instruction_text(value: str) -> None:
    lowered = value.lower()
    if any(marker in lowered for marker in ("pip install ", "npm install ")):
        raise ValueError("portable skill instructions cannot install packages")
    if any(
        marker in lowered
        for marker in (
            "subprocess.",
            "os.system(",
            "importlib.",
            "```python",
            "```shell",
            "```bash",
            "mcp.json",
        )
    ):
        raise ValueError(
            "portable skill instructions cannot execute code, import modules, "
            "or configure MCP"
        )
    if any(
        marker in lowered
        for marker in (
            "api_key=",
            "password=",
            "private_key=",
            "bearer ",
        )
    ):
        raise ValueError(
            "portable skill instructions cannot contain credential values"
        )
    if value.startswith(("/", "\\\\", "./", "../", ".\\", "..\\")) or any(
        marker in value
        for marker in (
            " file://",
            " C:\\",
            " /Users/",
            " /home/",
            " ./",
            " ../",
        )
    ):
        raise ValueError("portable skill instructions cannot contain local paths")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AsyncInMemorySkillPublicationStore",
    "AsyncSkillPublicationManager",
    "AsyncSkillPublicationStore",
    "InMemorySkillPublicationStore",
    "PortableSkill",
    "SkillPublicationManager",
    "SkillActivationRecord",
    "SkillPublicationRecord",
    "SkillPublicationStore",
    "skill_reference_map",
]
