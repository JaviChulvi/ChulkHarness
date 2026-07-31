"""Credential-free durable parent/child hosted orchestration."""

from __future__ import annotations

from chulk import (
    BudgetScope,
    ExecutionScope,
    InMemoryRunStore,
    ParentRunPolicy,
    RunBudget,
    RunSubmission,
    StepDefinition,
)


def main() -> None:
    runs = InMemoryRunStore()
    parent_scope = ExecutionScope(
        tenant_id="example",
        workspace_id="project",
        actor_id="user-42",
        agent_id="coordinator",
        agent_version="published-3",
        run_id="review-parent",
        grants=frozenset({"repository:read"}),
    )
    policy = ParentRunPolicy(
        required_children=2,
        max_children=2,
        budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=4,
            max_tool_calls=4,
            max_tokens=4_000,
        ),
    )
    runs.submit_parent(
        parent_scope,
        RunSubmission(
            idempotency_key="review-parent",
            input_digest="sha256:review-input",
            definition_digest="sha256:coordinator-v3",
            steps=(StepDefinition(id="aggregate", name="Aggregate reviews"),),
        ),
        policy=policy,
    )

    for index in (1, 2):
        child_scope = parent_scope.child(
            run_id=f"review-child-{index}",
            agent_id="reviewer",
            agent_version="published-7",
            grants=frozenset({"repository:read"}),
        )
        child_budget = RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=2,
            max_tool_calls=2,
            max_tokens=2_000,
        )
        child = runs.submit_child(
            parent_scope,
            child_scope,
            RunSubmission(
                idempotency_key=f"review-child-{index}",
                input_digest=f"sha256:review-child-{index}-input",
                definition_digest="sha256:reviewer-v7",
                steps=(StepDefinition(id="review", name="Review repository"),),
                budget=child_budget.to_dict(),
            ),
            definition_revision="published-7",
        )
        claim = runs.claim(
            child_scope,
            worker_id=f"worker-{index}",
            run_id=child.run.id,
        )
        assert claim is not None
        runs.start_step(child_scope, claim, "review")
        runs.record_child_progress(
            child_scope,
            claim,
            sequence=1,
            payload={"phase": "reviewed"},
            idempotency_key=f"review-progress-{index}",
        )
        runs.complete_step(child_scope, claim, "review")
        runs.complete(
            child_scope,
            claim,
            result={"summary": f"review {index} completed"},
        )

    parent = runs.aggregate_children(
        parent_scope,
        actor="host",
        idempotency_key="review-aggregation",
    )
    delivery = runs.claim_parent_completion(
        parent_scope,
        worker_id="application-outbox",
        parent_run_id=parent_scope.run_id,
    )
    assert delivery is not None
    runs.complete_parent_completion(parent_scope, delivery)

    print(parent.run.status.value)
    print([child.run.status.value for child in parent.children])
    print(delivery.completion.payload["status"])
    runs.close()


if __name__ == "__main__":
    main()
