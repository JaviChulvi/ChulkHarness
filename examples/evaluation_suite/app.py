"""Credential-free multi-turn evaluation of a public SDK agent."""

from __future__ import annotations

from pathlib import Path

from chulk import Agent, AgentConfig
from chulk.evals import (
    EvalDataset,
    EvalReport,
    EvalRunner,
    EvalSuite,
    EvalTarget,
    ExactAnswerGrader,
    MetricThreshold,
    StatusGrader,
)


ROOT = Path(__file__).resolve().parent


def create_agent(context):
    return Agent(
        config=AgentConfig(project_root=context.workspace),
        llm=context.llm,
        tools=[],
        skills=[],
    )


suite = EvalSuite(
    name="sdk-example",
    dataset=EvalDataset.from_jsonl(ROOT / "cases.jsonl"),
    targets=(EvalTarget("assistant", create_agent),),
    graders=(ExactAnswerGrader(), StatusGrader()),
    required_graders=("answer.exact", "run.status"),
    thresholds={"pass_rate": MetricThreshold(min=1.0)},
)


report = EvalRunner().run(suite)
assert isinstance(report, EvalReport)
report.assert_thresholds()
print(f"suite: {report.suite_name}")
print(f"cases: {int(report.metrics['case_count'])}")
print(f"pass_rate: {report.metrics['pass_rate']:.0%}")
