"""Credential-free, verified foreground goal execution."""
from pathlib import Path
from tempfile import TemporaryDirectory

from chulk import Agent, AgentConfig, GoalExecutionContext, GoalRunner, GoalService, GoalStep, GoalStore, PlanStepVerification, RunBudget
from chulk.testing import ScriptedLLMClient


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        store = GoalStore(root / "state.sqlite")
        service = GoalService(store)
        goal = service.create(title="Check the answer", acceptance_criteria=("Answer equals 42",),
            steps=(GoalStep(id="check", title="Check", description="Verify 42",
                            acceptance_criterion_ids=("criterion-1",)),),
            budget=RunBudget(max_model_calls=10))
        goal = service.approve(goal.id, expected_revision=goal.revision, approved_by="example-host")
        llm = ScriptedLLMClient([
            {"type": "plan_step_update", "step_update": {"step_id": "check", "status": "completed", "evidence": "42"}},
            {"type": "final_answer", "content": "42"},
        ])
        def factory(context: GoalExecutionContext, conversation_id: str | None) -> Agent:
            return Agent(config=AgentConfig(project_root=root, store_path=store.db_path, max_reflection_attempts=0),
                         llm=llm, tools=[], skills=[], goal_execution=context, conversation_id=conversation_id)
        runner = GoalRunner(store, agent_factory=factory,
                            verifier=lambda request: PlanStepVerification(request.asserted_evidence == "42", "Checked answer equals 42"))
        result = runner.run(goal.id)
        assert result.stop_reason.value == "completed"
        print(result.stop_reason.value, result.content)


if __name__ == "__main__":
    main()
