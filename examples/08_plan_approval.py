"""Create a plan, inspect its snapshot, and approve continuation."""

import bootstrap  # noqa: F401
from chulk import Agent, Tools

from common import live_config, print_run_result


def print_plan(plan_result) -> None:
    print("=== Plan ===")
    print(plan_result.content)
    if plan_result.plan is None:
        return
    print(plan_result.to_dict())
    print(f"summary: {plan_result.plan.summary}")
    print(f"status: {plan_result.plan.status}")
    for step in plan_result.plan.steps:
        print(f"- {step.get('id')}: {step.get('title')} [{step.get('status')}]")


def main() -> None:
    assistant = Agent(
        config=live_config("08-plan-approval", permission_profile="workspace-write"),
        tools=[Tools.calculator],
        skills=[],
    )
    plan_result = assistant.plan_result(
        "Plan how to calculate 19 * 23 with the calculator tool, then summarize the answer."
    )
    print_plan(plan_result)

    if plan_result.plan is None:
        print("The model did not create a plan, so there is nothing to approve.")
        return

    print("\n=== Approved Continuation ===")
    run_result = assistant.approve_result()
    print_run_result(run_result)


if __name__ == "__main__":
    main()
