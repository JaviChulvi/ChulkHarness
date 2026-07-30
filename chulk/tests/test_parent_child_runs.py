"""Contract and adversarial tests for durable parent/child runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from chulk.hosting import ExecutionScope
from chulk.runs import (
    AsyncSQLiteRunStore,
    InvalidRunTransitionError,
    ParentCompletionStatus,
    ParentRunPolicy,
    RunLeaseError,
    RunNotFoundError,
    RunStatus,
    RunSubmission,
    SQLiteRunStore,
    StepDefinition,
)
from chulk.testing import (
    assert_async_parent_child_run_contract,
    assert_parent_child_run_contract,
)
from chulk.usage import BudgetScope, RunBudget


def _scope(run_id: str = "parent-run") -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        actor_id="actor-a",
        agent_id="parent-agent",
        agent_version="published-4",
        run_id=run_id,
        grants=frozenset({"files:read", "network"}),
    )


def _submission(
    key: str,
    *,
    budget: RunBudget | None = None,
    step_id: str = "agent",
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=key,
        input_digest=f"sha256:{key}:input",
        definition_digest=f"sha256:{key}:definition",
        steps=(StepDefinition(id=step_id, name=step_id.title()),),
        budget=budget.to_dict() if budget is not None else {},
    )


def _budget(
    *,
    model_calls: int,
    tool_calls: int | None = None,
    tokens: int = 100,
) -> RunBudget:
    return RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=model_calls,
        max_tool_calls=tool_calls or model_calls,
        max_tokens=tokens,
    )


def _policy(
    *,
    required: int = 1,
    maximum: int = 2,
    model_calls: int = 4,
    tokens: int = 400,
) -> ParentRunPolicy:
    return ParentRunPolicy(
        required_children=required,
        max_children=maximum,
        budget=_budget(model_calls=model_calls, tokens=tokens),
    )


def _submit_parent(
    store: SQLiteRunStore,
    scope: ExecutionScope,
    *,
    policy: ParentRunPolicy | None = None,
) -> None:
    store.submit_parent(
        scope,
        _submission(f"{scope.run_id}-key", step_id="orchestrate"),
        policy=policy or _policy(),
    )


def _submit_child(
    store: SQLiteRunStore,
    parent: ExecutionScope,
    index: int,
    *,
    budget: RunBudget | None = None,
) -> ExecutionScope:
    scope = parent.child(
        run_id=f"{parent.run_id}-child-{index}",
        agent_id="child-agent",
        agent_version=f"published-{index}",
        grants=frozenset({"files:read"}),
    )
    store.submit_child(
        parent,
        scope,
        _submission(
            f"{parent.run_id}-child-key-{index}",
            budget=budget or _budget(model_calls=1),
        ),
        definition_revision=scope.agent_version,
    )
    return scope


def _complete_child(
    store: SQLiteRunStore,
    scope: ExecutionScope,
    *,
    worker: str,
) -> None:
    claim = store.claim(scope, worker_id=worker, run_id=scope.run_id)
    assert claim is not None
    store.start_step(scope, claim, "agent")
    store.complete_step(scope, claim, "agent", result={"worker": worker})
    store.complete(scope, claim, result={"worker": worker})


def test_sqlite_parent_child_contract(tmp_path: Path) -> None:
    report = assert_parent_child_run_contract(
        SQLiteRunStore(tmp_path / "runs.sqlite"),
        scope=_scope("sync-contract-parent"),
    )

    assert report.passed
    assert "exactly_once_parent_delivery" in report.checks


@pytest.mark.asyncio
async def test_async_sqlite_parent_child_contract(tmp_path: Path) -> None:
    report = await assert_async_parent_child_run_contract(
        AsyncSQLiteRunStore(tmp_path / "runs.sqlite"),
        scope=_scope("async-contract-parent"),
    )

    assert report.passed
    assert "async_exactly_once_parent_delivery" in report.checks


def test_child_creation_fails_closed_for_scope_budget_and_required_set(
    tmp_path: Path,
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    parent = _scope()
    _submit_parent(
        store,
        parent,
        policy=_policy(required=2, maximum=2, model_calls=2, tokens=200),
    )

    crossing_actor = replace(
        parent.child(
            run_id="crossing-actor",
            grants=frozenset({"files:read"}),
        ),
        actor_id="actor-b",
    )
    with pytest.raises(RunNotFoundError, match="authority boundary"):
        store.submit_child(
            parent,
            crossing_actor,
            _submission("crossing-actor", budget=_budget(model_calls=1)),
            definition_revision=crossing_actor.agent_version,
        )

    broader_grants = parent.child(
        run_id="broader-grants",
        grants=parent.grants,
    )
    broader_grants = replace(
        broader_grants,
        grants=frozenset({*parent.grants, "files:write"}),
    )
    with pytest.raises(RunNotFoundError, match="broadens"):
        store.submit_child(
            parent,
            broader_grants,
            _submission("broader-grants", budget=_budget(model_calls=1)),
            definition_revision=broader_grants.agent_version,
        )

    mismatched_revision = parent.child(
        run_id="mismatched-revision",
        agent_version="published-9",
        grants=frozenset({"files:read"}),
    )
    with pytest.raises(ValueError, match="must match its execution scope"):
        store.submit_child(
            parent,
            mismatched_revision,
            _submission("mismatched-revision", budget=_budget(model_calls=1)),
            definition_revision="published-8",
        )

    child = _submit_child(store, parent, 1)
    with pytest.raises(ValueError, match="allocation exceeds"):
        _submit_child(
            store,
            parent,
            2,
            budget=_budget(model_calls=2, tokens=150),
        )
    with pytest.raises(InvalidRunTransitionError, match="required child set"):
        store.aggregate_children(
            parent,
            actor="host",
            idempotency_key="too-early",
        )

    sibling = replace(child, run_id="unlinked-sibling")
    with pytest.raises(RunNotFoundError, match="sibling"):
        store.get(sibling, child.run_id)
    with pytest.raises(RunNotFoundError, match="sibling"):
        store.claim(child, worker_id="wrong-child", run_id="unlinked-sibling")

    with pytest.raises(
        InvalidRunTransitionError,
        match="request_parent_cancellation",
    ):
        store.request_cancellation(
            parent,
            parent.run_id,
            actor="operator",
            reason="generic parent cancellation must be rejected",
        )


def test_multiple_workers_cannot_exceed_parent_fanout(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    parent = _scope("concurrent-parent")
    _submit_parent(
        SQLiteRunStore(path),
        parent,
        policy=_policy(required=1, maximum=2, model_calls=2, tokens=200),
    )

    def submit(index: int) -> str | None:
        store = SQLiteRunStore(path)
        try:
            return _submit_child(store, parent, index).run_id
        except InvalidRunTransitionError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        submitted = tuple(pool.map(submit, range(1, 9)))

    successful = tuple(item for item in submitted if item is not None)
    assert len(successful) == 2
    assert len(set(successful)) == 2
    assert len(SQLiteRunStore(path).children(parent, parent.run_id)) == 2


def test_unknown_and_partial_child_outcomes_block_or_fail_parent(
    tmp_path: Path,
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    parent = _scope("partial-parent")
    _submit_parent(store, parent)
    child = _submit_child(store, parent, 1)
    claim = store.claim(child, worker_id="worker-a", run_id=child.run_id)
    assert claim is not None
    store.start_step(child, claim, "agent")
    effect = store.begin_effect(
        child,
        claim,
        "agent",
        logical_key="external-write",
        tool_name="write",
        tool_version="1",
        schema_version="1",
        arguments_digest="sha256:arguments",
    )
    store.mark_effect_started(child, claim, effect.id)
    store.mark_effect_unknown(
        child,
        claim,
        effect.id,
        reason="transport disconnected",
    )

    with pytest.raises(InvalidRunTransitionError, match="nonterminal"):
        store.aggregate_children(
            parent,
            actor="host",
            idempotency_key="blocked-aggregate",
        )

    store.reconcile_effect(
        child,
        effect.id,
        decision="failed",
        actor="operator",
        reason="target confirms a failed write",
    )
    aggregate = store.aggregate_children(
        parent,
        actor="host",
        idempotency_key="failed-aggregate",
    )

    assert aggregate.run.status is RunStatus.FAILED
    assert aggregate.children[0].terminal_evidence is not None
    assert store.parent_completion(parent, parent.run_id) is not None


def test_parent_cancellation_propagates_and_stale_child_cannot_progress(
    tmp_path: Path,
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    parent = _scope("cancel-parent")
    _submit_parent(store, parent)
    child = _submit_child(store, parent, 1)
    claim = store.claim(child, worker_id="worker-a", run_id=child.run_id)
    assert claim is not None
    store.start_step(child, claim, "agent")

    cancelling = store.request_parent_cancellation(
        parent,
        actor="operator",
        reason="operator cancelled the parent",
    )
    assert cancelling.run.cancellation_requested
    assert cancelling.children[0].run.cancellation_requested

    cancelled_child = store.cancel(
        child,
        child.run_id,
        actor="worker-a",
        reason="parent cancellation observed",
        claim=claim,
    )
    assert cancelled_child.status is RunStatus.CANCELLED
    with pytest.raises(RunLeaseError):
        store.record_child_progress(
            child,
            claim,
            sequence=1,
            payload={"late": True},
            idempotency_key="late-progress",
        )

    aggregate = store.aggregate_children(
        parent,
        actor="operator",
        idempotency_key="cancel-aggregate",
    )
    assert aggregate.run.status is RunStatus.CANCELLED
    completion_claim = store.claim_parent_completion(
        parent,
        worker_id="delivery",
        parent_run_id=parent.run_id,
    )
    assert completion_claim is not None
    delivered = store.complete_parent_completion(parent, completion_claim)
    assert delivered.status is ParentCompletionStatus.DELIVERED


def test_parent_completion_claim_is_bound_to_the_exact_parent_scope(
    tmp_path: Path,
) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    first = _scope("first-parent")
    second = replace(
        _scope("second-parent"),
        actor_id="actor-b",
    )
    for parent in (first, second):
        _submit_parent(store, parent)
        child = _submit_child(store, parent, 1)
        _complete_child(store, child, worker=f"worker-{parent.run_id}")
        store.aggregate_children(
            parent,
            actor="host",
            idempotency_key=f"aggregate-{parent.run_id}",
        )

    with pytest.raises(RunNotFoundError, match="execution scope"):
        store.claim_parent_completion(
            first,
            worker_id="wrong-parent",
            parent_run_id=second.run_id,
        )

    first_claim = store.claim_parent_completion(
        first,
        worker_id="first-delivery",
    )
    second_claim = store.claim_parent_completion(
        second,
        worker_id="second-delivery",
    )

    assert first_claim is not None
    assert first_claim.completion.parent_run_id == first.run_id
    assert second_claim is not None
    assert second_claim.completion.parent_run_id == second.run_id
