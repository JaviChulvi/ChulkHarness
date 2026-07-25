"""Host-governed delegation policy and parent-side result validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import uuid4

from chulk.capabilities import Capabilities, FileAccess, MemoryMode
from chulk.children.models import (
    ChildTask,
    ChildTaskLineage,
    ChildTaskResult,
    ChildTaskSpec,
    ChildTaskStatus,
)
from chulk.children.store import ChildTaskStore
from chulk.execution import WorkspaceMode
from chulk.goals import GoalRevisionConflictError, GoalService
from chulk.tools.schema import (
    ToolValidationError,
    validate_tool_output,
    validate_tool_output_schema,
)


_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "conversation_history",
        "full_transcript",
        "history",
        "messages",
        "transcript",
    }
)


@dataclass(frozen=True, slots=True)
class ChildAuthority:
    """Maximum authority a parent is permitted to delegate."""

    profile_id: str
    capabilities: Capabilities
    tool_names: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()
    mcp_server_labels: tuple[str, ...] = ()
    model_profile_ids: tuple[str, ...] = ()
    backend_names: tuple[str, ...] = ("host",)
    workspace_modes: tuple[WorkspaceMode, ...] = (WorkspaceMode.HOST,)
    max_depth: int = 1

    def __post_init__(self) -> None:
        clean_profile = self.profile_id.strip()
        if not clean_profile:
            raise ValueError("delegation authority profile cannot be empty")
        object.__setattr__(self, "profile_id", clean_profile)
        for field_name in (
            "tool_names",
            "skill_names",
            "mcp_server_labels",
            "model_profile_ids",
            "backend_names",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique(getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "workspace_modes",
            tuple(dict.fromkeys(WorkspaceMode(value) for value in self.workspace_modes)),
        )
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth < 1
        ):
            raise ValueError("delegation authority max_depth must be positive")


@dataclass(frozen=True, slots=True)
class DelegationPolicy:
    """Host ceilings independent of model-authored task specifications."""

    max_depth: int = 1
    max_active_tasks: int = 32
    max_parallel_workers: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "max_depth",
            "max_active_tasks",
            "max_parallel_workers",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """Explicit child package; no parent transcript is accepted."""

    spec: ChildTaskSpec
    task_id: str | None = None
    parent_task_id: str | None = None
    dependency_ids: tuple[str, ...] = ()
    goal_id: str | None = None
    goal_step_id: str | None = None
    parent_conversation_id: str | None = None
    parent_turn_id: str | None = None
    parent_trace_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChildResultValidation:
    """Parent-side decision about a non-authoritative child result."""

    valid: bool
    issues: tuple[str, ...] = ()
    evidenced_criterion_ids: tuple[str, ...] = ()


class ChildResultRejectedError(ValueError):
    """Raised when a child completion claim fails parent-side validation."""

    def __init__(self, validation: ChildResultValidation) -> None:
        self.validation = validation
        super().__init__("; ".join(validation.issues))


class ParentCompletionValidator:
    """Validate evidence and machine-readable output without completing a goal."""

    def validate(
        self,
        task: ChildTask,
        result: ChildTaskResult,
    ) -> ChildResultValidation:
        issues: list[str] = []
        schema = task.spec.to_dict()["result_schema"]
        try:
            validate_tool_output(
                "child_task_result",
                result.to_dict()["structured_output"],
                schema,
            )
        except ToolValidationError as exc:
            issues.extend(issue.to_prompt_line() for issue in exc.issues)
        evidenced = tuple(
            dict.fromkeys(
                criterion_id
                for item in result.evidence
                for criterion_id in item.criterion_ids
            )
        )
        missing_claims = tuple(
            claim for claim in result.completion_claims if claim not in evidenced
        )
        if missing_claims:
            issues.append(
                "completion claims lack evidence: " + ", ".join(missing_claims)
            )
        if result.trace_id is None:
            issues.append("child result must link its separate trace")
        if result.changed_files and not task.spec.mutable:
            issues.append("read-only child result cannot report changed files")
        if result.changed_files and result.change_set_id is None:
            issues.append("changed files require an isolated change_set_id")
        return ChildResultValidation(
            valid=not issues,
            issues=tuple(issues),
            evidenced_criterion_ids=evidenced,
        )

    def require_valid(
        self,
        task: ChildTask,
        result: ChildTaskResult,
    ) -> ChildResultValidation:
        validation = self.validate(task, result)
        if not validation.valid:
            raise ChildResultRejectedError(validation)
        return validation


class DelegationService:
    """Create narrower child work and coordinate cancellation with the host."""

    def __init__(
        self,
        store: ChildTaskStore,
        *,
        policy: DelegationPolicy | None = None,
        goal_service: GoalService | None = None,
        cancellation_propagator: Callable[[ChildTask], None] | None = None,
        event_callback: Callable[[str, ChildTask], None] | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or DelegationPolicy()
        self.goal_service = goal_service
        self.cancellation_propagator = cancellation_propagator
        self.event_callback = event_callback

    def delegate(
        self,
        request: DelegationRequest,
        *,
        authority: ChildAuthority,
        actor: str = "parent",
        idempotency_key: str | None = None,
    ) -> ChildTask:
        if authority.profile_id != self.store.profile_id:
            raise ValueError("delegation authority belongs to another profile")
        _reject_hidden_transcript(request.spec.context)
        validate_tool_output_schema(
            "child_task_result",
            request.spec.to_dict()["result_schema"],
        )
        self._validate_authority(request.spec, authority)
        lineage = self._lineage(request.parent_task_id)
        if lineage.depth > min(self.policy.max_depth, authority.max_depth):
            raise ValueError("child task exceeds the host delegation depth")
        if request.spec.max_depth > min(self.policy.max_depth, authority.max_depth):
            raise ValueError("child task delegation allowance exceeds host authority")
        self._validate_parent(request.spec, lineage)
        clean_key = _optional(idempotency_key)
        task_id = _optional(request.task_id) or _task_id(
            self.store.profile_id,
            clean_key,
        )
        existing = False
        try:
            self.store.get(task_id)
        except LookupError:
            pass
        else:
            existing = True
        active = sum(1 for task in self.store.list(limit=1000) if not task.terminal)
        if not existing and active >= self.policy.max_active_tasks:
            raise ValueError("profile child-task limit is already active")
        task = ChildTask(
            id=task_id,
            profile_id=self.store.profile_id,
            spec=request.spec,
            lineage=lineage,
            dependency_ids=request.dependency_ids,
            goal_id=_optional(request.goal_id),
            goal_step_id=_optional(request.goal_step_id),
            parent_conversation_id=_optional(request.parent_conversation_id),
            parent_turn_id=_optional(request.parent_turn_id),
            parent_trace_id=_optional(request.parent_trace_id),
        )
        created = self.store.create(
            task,
            actor=actor,
            idempotency_key=clean_key,
        )
        if created.goal_id is not None:
            self._link_goal(created, actor=actor)
        if self.event_callback is not None:
            self.event_callback("child.created", created)
        return created

    def cancel(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor: str,
        reason: str = "Cancellation requested by parent.",
    ) -> tuple[ChildTask, ...]:
        changed = self.store.request_cancel(
            task_id,
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
        )
        for task in changed:
            if self.cancellation_propagator is not None:
                self.cancellation_propagator(task)
            if self.event_callback is not None:
                self.event_callback(
                    "child.cancelled"
                    if task.status is ChildTaskStatus.CANCELLED
                    else "child.cancellation_requested",
                    task,
                )
        return changed

    def _lineage(self, parent_task_id: str | None) -> ChildTaskLineage:
        clean_parent = _optional(parent_task_id)
        if clean_parent is None:
            return ChildTaskLineage()
        parent = self.store.get(clean_parent)
        return ChildTaskLineage(
            parent_task_id=parent.id,
            root_task_id=parent.lineage.root_task_id or parent.id,
            depth=parent.lineage.depth + 1,
        )

    def _validate_parent(
        self,
        spec: ChildTaskSpec,
        lineage: ChildTaskLineage,
    ) -> None:
        if lineage.parent_task_id is None:
            return
        parent = self.store.get(lineage.parent_task_id)
        parent_authority = ChildAuthority(
            profile_id=parent.profile_id,
            capabilities=parent.spec.capabilities,
            tool_names=parent.spec.tool_names,
            skill_names=parent.spec.skill_names,
            mcp_server_labels=parent.spec.mcp_server_labels,
            model_profile_ids=(
                (parent.spec.model_profile_id,)
                if parent.spec.model_profile_id is not None
                else ()
            ),
            backend_names=(parent.spec.backend_name,),
            workspace_modes=(parent.spec.workspace_mode,),
            max_depth=parent.spec.max_depth,
        )
        self._validate_authority(spec, parent_authority)

    @staticmethod
    def _validate_authority(
        spec: ChildTaskSpec,
        authority: ChildAuthority,
    ) -> None:
        if not _capabilities_are_subset(spec.capabilities, authority.capabilities):
            raise ValueError("child capabilities exceed parent authority")
        _require_subset("tools", spec.tool_names, authority.tool_names)
        _require_subset("skills", spec.skill_names, authority.skill_names)
        _require_subset(
            "MCP servers",
            spec.mcp_server_labels,
            authority.mcp_server_labels,
        )
        if (
            spec.model_profile_id is not None
            and spec.model_profile_id not in authority.model_profile_ids
        ):
            raise ValueError("child model profile exceeds parent authority")
        if spec.backend_name not in authority.backend_names:
            raise ValueError("child execution backend exceeds parent authority")
        if spec.workspace_mode not in authority.workspace_modes:
            raise ValueError("child workspace mode exceeds parent authority")

    def _link_goal(self, task: ChildTask, *, actor: str) -> None:
        if self.goal_service is None:
            raise ValueError("goal-linked child task requires a goal service")
        for _ in range(3):
            goal = self.goal_service.store.get(task.goal_id or "")
            if goal.profile_id != task.profile_id:
                raise ValueError("child task goal belongs to another profile")
            try:
                self.goal_service.link_resource(
                    goal.id,
                    expected_revision=goal.revision,
                    kind="child_task",
                    resource_id=task.id,
                    actor=actor,
                )
                return
            except GoalRevisionConflictError:
                continue
        raise RuntimeError("goal changed repeatedly while linking child task")


def _capabilities_are_subset(
    child: Capabilities,
    parent: Capabilities,
) -> bool:
    file_rank = {
        FileAccess.OFF: 0,
        FileAccess.READ: 1,
        FileAccess.WRITE: 2,
    }
    memory_rank = {
        MemoryMode.OFF: 0,
        MemoryMode.READ_ONLY: 1,
        MemoryMode.MANUAL: 2,
        MemoryMode.AUTOMATIC: 3,
    }
    return (
        file_rank[FileAccess(child.files)] <= file_rank[FileAccess(parent.files)]
        and memory_rank[MemoryMode(child.memory)]
        <= memory_rank[MemoryMode(parent.memory)]
        and (not child.shell or parent.shell)
        and (not child.network or parent.network)
        and (not child.external_services or parent.external_services)
        and (not child.utilities or parent.utilities)
    )


def _require_subset(
    label: str,
    requested: tuple[str, ...],
    allowed: tuple[str, ...],
) -> None:
    extra = sorted(set(requested) - set(allowed))
    if extra:
        raise ValueError(f"child {label} exceed parent authority: {', '.join(extra)}")


def _reject_hidden_transcript(context: Mapping[str, Any]) -> None:
    blocked = sorted(_FORBIDDEN_CONTEXT_KEYS.intersection(context))
    if blocked:
        raise ValueError(
            "child context must be explicit and cannot contain a parent transcript: "
            + ", ".join(blocked)
        )


def _task_id(profile_id: str, idempotency_key: str | None) -> str:
    if idempotency_key is None:
        return uuid4().hex
    digest = sha256(f"{profile_id}:{idempotency_key}".encode()).hexdigest()
    return f"child-{digest[:24]}"


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    return clean or None


__all__ = [
    "ChildAuthority",
    "ChildResultRejectedError",
    "ChildResultValidation",
    "DelegationPolicy",
    "DelegationRequest",
    "DelegationService",
    "ParentCompletionValidator",
]
