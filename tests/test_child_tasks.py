"""Outside-in tests for the durable child-task graph."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from chulk.children import (
    ChildDeliveryConflictError,
    ChildDeliveryStatus,
    ChildTask,
    ChildTaskConflictError,
    ChildTaskLeaseConflictError,
    ChildTaskLineage,
    ChildTaskResult,
    ChildTaskRevisionConflictError,
    ChildTaskRole,
    ChildTaskSpec,
    ChildTaskStatus,
    ChildTaskStore,
)
from chulk.usage import BudgetScope, RunBudget


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path, *, profile_id: str = "default") -> ChildTaskStore:
    return ChildTaskStore(
        tmp_path / "control.sqlite",
        profile_id=profile_id,
        clock=lambda: NOW,
    )


def _task(
    task_id: str,
    *,
    profile_id: str = "default",
    instruction: str = "Inspect the requested subsystem.",
    dependency_ids: tuple[str, ...] = (),
    lineage: ChildTaskLineage | None = None,
    role: ChildTaskRole = ChildTaskRole.LEAF,
    max_depth: int = 1,
    budget: RunBudget | None = None,
) -> ChildTask:
    return ChildTask(
        id=task_id,
        profile_id=profile_id,
        spec=ChildTaskSpec(
            instruction=instruction,
            role=role,
            max_depth=max_depth,
            budget=budget or RunBudget(scope=BudgetScope.CHILD_TASK),
        ),
        lineage=lineage or ChildTaskLineage(),
        dependency_ids=dependency_ids,
        created_at=NOW,
        updated_at=NOW,
    )


def _result(summary: str = "The child finished.") -> ChildTaskResult:
    return ChildTaskResult(
        summary=summary,
        structured_output={"status": "done"},
        completion_claims=("criterion-1",),
        trace_id="trace-child",
    )


def test_creation_is_profile_scoped_revisioned_and_idempotent(tmp_path) -> None:
    store = _store(tmp_path, profile_id="work")
    requested = _task("child-a", profile_id="work")

    created = store.create(requested, idempotency_key="create:child-a")
    replayed = store.create(requested, idempotency_key="create:child-a")

    assert created.status is ChildTaskStatus.READY
    assert replayed == created
    assert [event.kind for event in store.events(created.id)] == ["child.created"]
    with pytest.raises(ChildTaskConflictError):
        store.create(
            replace(
                requested,
                spec=replace(requested.spec, instruction="Different work."),
            ),
            idempotency_key="create:child-a",
        )
    with pytest.raises(LookupError):
        _store(tmp_path, profile_id="personal").get(created.id)


def test_lineage_enforces_orchestrator_role_root_and_depth(tmp_path) -> None:
    store = _store(tmp_path)
    leaf = store.create(_task("leaf"))
    with pytest.raises(ValueError, match="leaf child task"):
        store.create(
            _task(
                "leaf-child",
                lineage=ChildTaskLineage(
                    parent_task_id=leaf.id,
                    root_task_id=leaf.id,
                    depth=2,
                ),
                max_depth=2,
            )
        )

    parent = store.create(
        _task(
            "orchestrator",
            role=ChildTaskRole.ORCHESTRATOR,
            max_depth=2,
        )
    )
    nested = store.create(
        _task(
            "nested",
            lineage=ChildTaskLineage(
                parent_task_id=parent.id,
                root_task_id=parent.id,
                depth=2,
            ),
            max_depth=2,
        )
    )
    assert nested.lineage.depth == 2

    with pytest.raises(ValueError, match="root lineage"):
        store.create(
            _task(
                "wrong-root",
                lineage=ChildTaskLineage(
                    parent_task_id=parent.id,
                    root_task_id="another-root",
                    depth=2,
                ),
                max_depth=2,
            )
        )


def test_dependencies_become_ready_or_require_explicit_retry(tmp_path) -> None:
    store = _store(tmp_path)
    dependency = store.create(_task("dependency"))
    dependent = store.create(
        _task("dependent", dependency_ids=(dependency.id,))
    )
    assert dependent.status is ChildTaskStatus.PENDING

    claim = store.claim(
        dependency.id,
        expected_revision=dependency.revision,
        worker_id="worker-a",
    )
    dependency = store.fail(claim, "The dependency failed.")
    assert store.get(dependent.id).status is ChildTaskStatus.BLOCKED

    dependency = store.retry(
        dependency.id,
        expected_revision=dependency.revision,
        actor="operator",
    )
    claim = store.claim(
        dependency.id,
        expected_revision=dependency.revision,
        worker_id="worker-a",
    )
    store.complete(claim, _result())
    assert store.get(dependent.id).status is ChildTaskStatus.BLOCKED

    dependent = store.retry(
        dependent.id,
        expected_revision=store.get(dependent.id).revision,
        actor="operator",
    )
    assert dependent.status is ChildTaskStatus.READY


def test_attempt_claim_heartbeat_and_terminal_fencing(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(_task("leased"))
    claim = store.claim(
        task.id,
        expected_revision=task.revision,
        worker_id="worker-a",
        lease_seconds=10,
    )

    with pytest.raises(
        (ChildTaskRevisionConflictError, ChildTaskLeaseConflictError)
    ):
        store.claim(
            task.id,
            expected_revision=task.revision,
            worker_id="worker-b",
        )
    claim = store.heartbeat(claim, lease_seconds=20, now=NOW + timedelta(seconds=1))
    completed = store.complete(claim, _result(), now=NOW + timedelta(seconds=2))
    assert completed.status is ChildTaskStatus.COMPLETED
    assert store.attempts(task.id)[0]["status"] == "completed"
    with pytest.raises(ChildTaskLeaseConflictError):
        store.heartbeat(claim, now=NOW + timedelta(seconds=3))


def test_expired_inflight_work_becomes_unknown_and_is_never_replayed(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(_task("uncertain"))
    store.claim(
        task.id,
        expected_revision=task.revision,
        worker_id="worker-a",
        lease_seconds=1,
    )

    recovered = store.recover_expired(now=NOW + timedelta(seconds=2))

    assert [item.status for item in recovered] == [ChildTaskStatus.UNKNOWN]
    assert store.claim_next(worker_id="worker-b") is None
    retried = store.retry(
        task.id,
        expected_revision=recovered[0].revision,
        actor="operator",
    )
    second_claim = store.claim(
        task.id,
        expected_revision=retried.revision,
        worker_id="worker-b",
    )
    assert second_claim.attempt_number == 2


def test_cancellation_propagates_and_wins_over_active_completion(tmp_path) -> None:
    store = _store(tmp_path)
    parent = store.create(
        _task(
            "parent",
            role=ChildTaskRole.ORCHESTRATOR,
            max_depth=2,
        )
    )
    child = store.create(
        _task(
            "child",
            lineage=ChildTaskLineage(
                parent_task_id=parent.id,
                root_task_id=parent.id,
                depth=2,
            ),
            max_depth=2,
        )
    )
    claim = store.claim(
        parent.id,
        expected_revision=parent.revision,
        worker_id="worker-a",
    )

    cancelled = store.request_cancel(
        parent.id,
        expected_revision=store.get(parent.id).revision,
        actor="operator",
    )

    assert {item.id for item in cancelled} == {parent.id, child.id}
    assert store.get(child.id).status is ChildTaskStatus.CANCELLED
    parent = store.complete(claim, _result())
    assert parent.status is ChildTaskStatus.CANCELLED
    assert parent.result is not None


def test_completion_outbox_recovers_delivery_without_duplicate_ack(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(_task("delivery"))
    claim = store.claim(
        task.id,
        expected_revision=task.revision,
        worker_id="worker",
    )
    store.complete(claim, _result())

    first = store.claim_delivery(
        worker_id="delivery-a",
        lease_seconds=1,
        now=NOW,
    )
    assert first is not None
    second = store.claim_delivery(
        worker_id="delivery-b",
        now=NOW + timedelta(seconds=2),
    )
    assert second is not None
    assert second.id == first.id
    assert second.attempts == 2
    delivered = store.finish_delivery(
        second,
        delivered=True,
        now=NOW + timedelta(seconds=3),
    )
    assert delivered.status is ChildDeliveryStatus.DELIVERED
    with pytest.raises(ChildDeliveryConflictError):
        store.finish_delivery(
            first,
            delivered=True,
            now=NOW + timedelta(seconds=3),
        )


def test_deadline_and_secret_redaction_are_enforced_before_execution(tmp_path) -> None:
    store = _store(tmp_path)
    task = store.create(
        _task(
            "deadline",
            instruction="Use OPENAI_API_KEY=sk-fake-secret-value to inspect.",
            budget=RunBudget(scope=BudgetScope.CHILD_TASK, deadline=NOW),
        )
    )
    assert "sk-fake-secret-value" not in store.get(task.id).spec.instruction

    with pytest.raises(ChildTaskLeaseConflictError, match="deadline"):
        store.claim(
            task.id,
            expected_revision=task.revision,
            worker_id="worker",
        )

    expired = store.get(task.id)
    assert expired.status is ChildTaskStatus.BUDGET_EXHAUSTED
    assert store.list_deliveries()[0].task_id == task.id
