"""Tests for bounded delegation, supervision, and parent validation."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from threading import Event
import time

import pytest

from chulk.capabilities import Capabilities, FileAccess, MemoryMode
from chulk.children import (
    ChildAuthority,
    ChildEvidenceRef,
    ChildResultRejectedError,
    ChildTask,
    ChildTaskLeaseConflictError,
    ChildTaskLineage,
    ChildTaskResult,
    ChildTaskRole,
    ChildTaskSpec,
    ChildTaskStatus,
    ChildTaskStore,
    DelegationPolicy,
    DelegationRequest,
    DelegationService,
    ParentCompletionValidator,
    RuntimeChildAgentFactory,
    TaskSupervisor,
)
from chulk.execution import WorkspaceMode
from chulk.goals import GoalService, GoalStep, GoalStore
from chulk.usage import (
    BudgetExceededError,
    BudgetScope,
    RunBudget,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path) -> ChildTaskStore:
    return ChildTaskStore(
        tmp_path / "control.sqlite",
        clock=lambda: NOW,
    )


def _authority(*, max_depth: int = 2) -> ChildAuthority:
    return ChildAuthority(
        profile_id="default",
        capabilities=Capabilities(
            files=FileAccess.WRITE,
            shell=True,
            memory=MemoryMode.READ_ONLY,
            network=True,
            external_services=True,
        ),
        tool_names=("read_file", "write_file", "shell"),
        skill_names=("review", "testing"),
        mcp_server_labels=("github",),
        model_profile_ids=("default-model",),
        backend_names=("host", "temporary"),
        workspace_modes=(WorkspaceMode.HOST, WorkspaceMode.TEMPORARY),
        max_depth=max_depth,
    )


def _service(tmp_path, *, cancel=None) -> DelegationService:
    return DelegationService(
        _store(tmp_path),
        policy=DelegationPolicy(
            max_depth=2,
            max_active_tasks=16,
            max_parallel_workers=2,
        ),
        cancellation_propagator=cancel,
    )


def _spec(
    instruction: str = "Inspect the selected files.",
    *,
    context=None,
    role: ChildTaskRole = ChildTaskRole.LEAF,
    max_depth: int = 1,
    max_parallelism: int = 1,
    result_schema=None,
) -> ChildTaskSpec:
    return ChildTaskSpec(
        instruction=instruction,
        context=context or {},
        capabilities=Capabilities(memory=MemoryMode.READ_ONLY),
        tool_names=("read_file",),
        skill_names=("review",),
        mcp_server_labels=(),
        model_profile_id="default-model",
        role=role,
        max_depth=max_depth,
        max_parallelism=max_parallelism,
        budget=RunBudget(scope=BudgetScope.CHILD_TASK, max_model_calls=2),
        result_schema=result_schema or {"type": "object"},
    )


def _result(
    *,
    output=None,
    claims: tuple[str, ...] = (),
    evidence: tuple[ChildEvidenceRef, ...] = (),
) -> ChildTaskResult:
    return ChildTaskResult(
        summary="Child execution finished.",
        structured_output=output or {"status": "ok"},
        completion_claims=claims,
        evidence=evidence,
        trace_id="child-trace",
        usage={"model_calls": 1},
    )


class _Runner:
    def __init__(
        self,
        result: ChildTaskResult | None = None,
        *,
        error: Exception | None = None,
        started: Event | None = None,
        released: Event | None = None,
    ) -> None:
        self.result = result or _result()
        self.error = error
        self.started = started
        self.released = released
        self.cancelled = False
        self.closed = False
        self.boundary_count = 0

    def run(self, *, boundary):
        boundary()
        self.boundary_count += 1
        if self.started is not None:
            self.started.set()
        if self.released is not None:
            self.released.wait(timeout=2)
        if self.error is not None:
            raise self.error
        boundary()
        self.boundary_count += 1
        return self.result

    def cancel(self) -> None:
        self.cancelled = True
        if self.released is not None:
            self.released.set()

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, runners=None) -> None:
        self.runners = list(runners or [])
        self.created: list[tuple[ChildTask, object]] = []

    def create(self, task, claim):
        self.created.append((task, claim))
        return self.runners.pop(0) if self.runners else _Runner()


def test_delegation_rejects_escalation_transcripts_and_invalid_schema(tmp_path) -> None:
    service = _service(tmp_path)
    authority = _authority()

    with pytest.raises(ValueError, match="transcript"):
        service.delegate(
            DelegationRequest(
                _spec(context={"messages": [{"role": "user", "content": "all"}]})
            ),
            authority=authority,
        )
    read_only_authority = ChildAuthority(
        profile_id="default",
        capabilities=Capabilities.read_only(),
        tool_names=("read_file",),
        skill_names=("review",),
        model_profile_ids=("default-model",),
    )
    with pytest.raises(ValueError, match="capabilities exceed"):
        service.delegate(
            DelegationRequest(
                ChildTaskSpec(
                    instruction="Use unavailable network access.",
                    capabilities=Capabilities(network=True),
                    tool_names=("read_file",),
                    skill_names=("review",),
                    model_profile_id="default-model",
                )
            ),
            authority=read_only_authority,
        )
    with pytest.raises(ValueError, match="memory mutation authority"):
        ChildTaskSpec(
            instruction="Use unavailable memory mutation.",
            capabilities=Capabilities(memory=MemoryMode.MANUAL),
        )
    with pytest.raises(ValueError, match="child tools"):
        service.delegate(
            DelegationRequest(
                ChildTaskSpec(
                    instruction="Use a tool outside the parent set.",
                    tool_names=("delete_everything",),
                )
            ),
            authority=authority,
        )
    with pytest.raises(ValueError, match="Invalid output schema"):
        service.delegate(
            DelegationRequest(_spec(result_schema={"type": "mystery"})),
            authority=authority,
        )


def test_delegation_builds_fresh_lineage_and_deterministic_idempotency(tmp_path) -> None:
    observed: list[str] = []
    service = _service(tmp_path)
    service.event_callback = lambda kind, _task: observed.append(kind)
    authority = _authority()
    parent = service.delegate(
        DelegationRequest(
            _spec(
                "Coordinate two bounded reviews.",
                role=ChildTaskRole.ORCHESTRATOR,
                max_depth=2,
                max_parallelism=2,
            )
        ),
        authority=authority,
        idempotency_key="coordinate",
    )
    replayed = service.delegate(
        DelegationRequest(
            _spec(
                "Coordinate two bounded reviews.",
                role=ChildTaskRole.ORCHESTRATOR,
                max_depth=2,
                max_parallelism=2,
            )
        ),
        authority=authority,
        idempotency_key="coordinate",
    )
    nested = service.delegate(
        DelegationRequest(
            _spec("Review only the storage package.", max_depth=2),
            parent_task_id=parent.id,
            parent_trace_id="parent-trace",
        ),
        authority=authority,
    )

    assert replayed.id == parent.id
    assert observed.count("child.created") == 2
    assert nested.lineage.parent_task_id == parent.id
    assert nested.lineage.root_task_id == parent.id
    assert nested.lineage.depth == 2
    assert nested.parent_trace_id == "parent-trace"


def test_goal_linking_and_shared_budget_resolution_are_automatic(tmp_path) -> None:
    db_path = tmp_path / "control.sqlite"
    goal_service = GoalService(GoalStore(db_path))
    goal = goal_service.create(
        title="Coordinate child work",
        acceptance_criteria=("The child provides evidence.",),
        steps=(
            GoalStep(
                id="delegate",
                title="Delegate",
                description="Run bounded child work.",
                acceptance_criterion_ids=("criterion-1",),
            ),
        ),
        budget=RunBudget(max_model_calls=4),
    )
    service = DelegationService(
        ChildTaskStore(db_path),
        policy=DelegationPolicy(max_depth=2),
        goal_service=goal_service,
    )

    task = service.delegate(
        DelegationRequest(
            _spec(),
            goal_id=goal.id,
            goal_step_id="delegate",
        ),
        authority=_authority(),
    )

    linked = goal_service.store.get(goal.id)
    assert linked.child_task_ids == (task.id,)
    assert service.shared_budgets(task) == (linked.budget,)


def test_store_enforces_parent_parallelism_at_claim_time(tmp_path) -> None:
    service = _service(tmp_path)
    parent = service.delegate(
        DelegationRequest(
            _spec(
                role=ChildTaskRole.ORCHESTRATOR,
                max_depth=2,
                max_parallelism=1,
            )
        ),
        authority=_authority(),
    )
    first = service.delegate(
        DelegationRequest(_spec(max_depth=2), parent_task_id=parent.id),
        authority=_authority(),
    )
    second = service.delegate(
        DelegationRequest(_spec(max_depth=2), parent_task_id=parent.id),
        authority=_authority(),
    )
    service.store.claim(
        first.id,
        expected_revision=first.revision,
        worker_id="worker-a",
    )
    with pytest.raises(ChildTaskLeaseConflictError, match="parallelism"):
        service.store.claim(
            second.id,
            expected_revision=second.revision,
            worker_id="worker-b",
        )


def test_parent_validator_requires_schema_evidence_trace_and_change_set() -> None:
    validator = ParentCompletionValidator()
    task = ChildTask(
        id="validate",
        profile_id="default",
        spec=_spec(
            result_schema={
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
            }
        ),
        lineage=ChildTaskLineage(),
        created_at=NOW,
        updated_at=NOW,
    )
    invalid = ChildTaskResult(
        summary="Unsupported claim.",
        structured_output={"count": "one"},
        completion_claims=("criterion-1",),
    )

    with pytest.raises(ChildResultRejectedError) as exc_info:
        validator.require_valid(task, invalid)

    assert "completion claims lack evidence" in str(exc_info.value)
    assert "separate trace" in str(exc_info.value)


def test_supervisor_executes_sync_parallel_and_budget_failure(tmp_path) -> None:
    service = _service(tmp_path)
    first = service.delegate(
        DelegationRequest(_spec("First review.")),
        authority=_authority(),
    )
    service.delegate(
        DelegationRequest(_spec("Second review.")),
        authority=_authority(),
    )
    budget_error = BudgetExceededError(
        scope=BudgetScope.CHILD_TASK,
        dimension="model_calls",
        limit="1",
        committed="1",
        reserved="0",
        requested="1",
    )
    factory = _Factory([_Runner(), _Runner(error=budget_error)])
    supervisor = TaskSupervisor(
        service.store,
        factory,
        worker_id="supervisor",
        policy=service.policy,
        lease_seconds=10,
        heartbeat_seconds=0.01,
    )

    completed = supervisor.run_task(first.id)
    remaining = supervisor.run_batch(max_tasks=2)

    assert completed.status is ChildTaskStatus.COMPLETED
    assert [task.status for task in remaining] == [
        ChildTaskStatus.BUDGET_EXHAUSTED
    ]


def test_detached_cancellation_stops_live_runner_and_finishes_cancelled(tmp_path) -> None:
    started = Event()
    released = Event()
    runner = _Runner(started=started, released=released)
    factory = _Factory([runner])
    service = _service(tmp_path)
    task = service.delegate(
        DelegationRequest(_spec("Wait for cancellation.")),
        authority=_authority(),
    )
    supervisor = TaskSupervisor(
        service.store,
        factory,
        worker_id="detached",
        policy=service.policy,
        lease_seconds=10,
        heartbeat_seconds=0.01,
    )
    service.cancellation_propagator = supervisor.cancel_active
    handle = supervisor.start_detached(workers=1, poll_seconds=0.01)
    assert started.wait(timeout=1)

    service.cancel(
        task.id,
        expected_revision=service.store.get(task.id).revision,
        actor="operator",
    )
    deadline = time.monotonic() + 2
    while (
        service.store.get(task.id).status is not ChildTaskStatus.CANCELLED
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    handle.stop(timeout=1)

    assert runner.cancelled
    assert runner.closed
    assert service.store.get(task.id).status is ChildTaskStatus.CANCELLED


def test_completion_delivery_passes_idempotency_key_to_consumer(tmp_path) -> None:
    service = _service(tmp_path)
    task = service.delegate(
        DelegationRequest(_spec()),
        authority=_authority(),
    )
    supervisor = TaskSupervisor(
        service.store,
        _Factory(),
        worker_id="worker",
        lease_seconds=10,
    )
    supervisor.run_task(task.id)
    observed: list[tuple[str, str]] = []

    delivery = supervisor.deliver_once(
        lambda item, completed: observed.append(
            (item.idempotency_key, completed.id)
        )
    )

    assert delivery is not None
    assert observed == [(delivery.idempotency_key, task.id)]
    assert supervisor.deliver_once(lambda _item, _task: None) is None


def test_runtime_factory_forwards_only_selected_child_scope(tmp_path) -> None:
    profile = SimpleNamespace(
        id="default",
        model_profile_id="default-model",
        execution_backend_id="host",
    )

    class ProfileFactory:
        def __init__(self) -> None:
            self.calls = []

        def resolve(self, profile_id):
            assert profile_id == "default"
            return SimpleNamespace(profile=profile)

        def create_agent(self, profile_id, **kwargs):
            self.calls.append((profile_id, kwargs))
            return SimpleNamespace()

    profile_factory = ProfileFactory()
    factory = RuntimeChildAgentFactory(
        profile_factory,  # type: ignore[arg-type]
        tool_catalog={"read_file": object(), "shell": object()},
        additional_budget_resolver=lambda _task: (
            RunBudget(scope=BudgetScope.GOAL, max_model_calls=4),
        ),
    )
    service = DelegationService(
        ChildTaskStore(tmp_path / "runtime.sqlite", clock=lambda: NOW),
        policy=DelegationPolicy(max_depth=2),
    )
    task = service.delegate(
        DelegationRequest(_spec(context={"files": ["src/chulk/runtime.py"]})),
        authority=_authority(),
    )
    claim = service.store.claim(
        task.id,
        expected_revision=task.revision,
        worker_id="worker",
    )

    factory.create(task, claim)

    _, kwargs = profile_factory.calls[0]
    assert len(kwargs["tool_specs"]) == 1
    assert kwargs["allowed_skill_names"] == ("review",)
    assert kwargs["conversation_id"] is None
    assert kwargs["conversation_metadata"]["child_task_id"] == task.id
    assert kwargs["usage_dimensions"].child_task_id == task.id
    assert kwargs["additional_run_budgets"][0].scope is BudgetScope.GOAL
