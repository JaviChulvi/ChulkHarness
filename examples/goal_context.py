"""Credential-free claimed goal context and steering incorporation."""

from pathlib import Path
from tempfile import TemporaryDirectory

from chulk import Agent, AgentConfig, GoalModelRequest
from chulk.goals import GoalService, GoalStep, GoalStore
from chulk.testing import ScriptedLLMClient
from chulk.usage import RunBudget


def main() -> None:
    with TemporaryDirectory(prefix="chulk-goal-context-") as directory:
        root = Path(directory)
        service = GoalService(GoalStore(root / "goals.sqlite"))
        goal = service.create(
            title="Review", description="Inspect the change before publishing",
            constraints=("Keep the review read-only",),
            acceptance_criteria=("The host verifies the evidence",),
            steps=(GoalStep(id="review", title="Review", description="Inspect evidence",
                            acceptance_criterion_ids=("criterion-1",)),),
            budget=RunBudget(max_model_calls=10),
        )
        goal = service.approve(goal.id, expected_revision=goal.revision, approved_by="owner")
        goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
        goal = service.start_step(goal.id, "review", expected_revision=goal.revision, actor="owner")
        goal = service.steer(goal.id, expected_revision=goal.revision,
                             instruction="Report evidence before asserting success", created_by="owner")
        execution = service.claim_execution(goal.id, "review", expected_revision=goal.revision,
                                            runner_id="example")
        try:
            with Agent(config=AgentConfig(project_root=root, max_reflection_attempts=0),
                       llm=ScriptedLLMClient([{"type": "final_answer", "content": "Evidence is ready for host review"}]),
                       tools=[], skills=[], goal_execution=execution) as agent:
                result = agent.run_result("Inspect the current goal")
            receipt: GoalModelRequest = service.store.model_requests(goal.id)[0]
            assert receipt.response_ref is not None
            assert service.store.incorporated_steering_ids(goal.id) == frozenset({goal.steering[0].id})
            assert service.store.get(goal.id).evidence == ()
            print(result.content)
            print("Steering incorporated; fulfillment and step completion still require host evidence.")
        finally:
            execution.close()


if __name__ == "__main__":
    main()
