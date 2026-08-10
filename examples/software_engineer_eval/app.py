"""Evaluate the public software-engineer preset without model credentials."""

from __future__ import annotations

from pathlib import Path

from chulk import Agent, AgentConfig
from chulk.evals import (
    EvalContext,
    EvalDataset,
    EvalReport,
    EvalRunner,
    EvalSafetyPolicy,
    EvalSuite,
    EvalTarget,
    ExactAnswerGrader,
    MetricThreshold,
    NoErrorGrader,
    StatusGrader,
    ToolCallGrader,
)
from chulk.presets import SoftwareEngineer


ROOT = Path(__file__).resolve().parent


def seed_buggy_project(context: EvalContext) -> dict[str, str]:
    """Create the disposable project that the coding agent must repair."""
    (context.workspace / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left - right\n",
        encoding="utf-8",
    )
    (context.workspace / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import add\n\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(2, 3), 5)\n",
        encoding="utf-8",
    )
    return {"fixture": "buggy-calculator"}


def create_software_engineer(context: EvalContext) -> Agent:
    return Agent(
        config=AgentConfig(project_root=context.workspace),
        preset=SoftwareEngineer(),
        llm=context.llm,
    )


suite = EvalSuite(
    name="software-engineer-preset",
    dataset=EvalDataset.from_jsonl(ROOT / "cases.jsonl"),
    targets=(EvalTarget("software-engineer", create_software_engineer),),
    graders=(
        ExactAnswerGrader(),
        StatusGrader(),
        NoErrorGrader(),
        ToolCallGrader(),
    ),
    required_graders=(
        "answer.exact",
        "run.status",
        "run.no_errors",
        "tools.calls",
    ),
    thresholds={"pass_rate": MetricThreshold(min=1.0)},
    fixtures={"buggy-project": seed_buggy_project},
    safety=EvalSafetyPolicy(allowed_tool_names=("apply_patch", "run_cmd")),
)


def main() -> None:
    report = EvalRunner().run(suite)
    assert isinstance(report, EvalReport)
    report.assert_thresholds()
    tool_names = [
        call.resolved_tool_name or call.tool_name
        for case in report.cases
        for trial in case.trials
        for turn in trial.turns
        for call in turn.result.tool_calls
    ]
    print(f"suite: {report.suite_name}")
    print(f"target: {report.cases[0].trials[0].target_name}")
    print(f"tools: {' -> '.join(tool_names)}")
    print(f"pass_rate: {report.metrics['pass_rate']:.0%}")


if __name__ == "__main__":
    main()
