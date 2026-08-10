"""Complete deterministic, callable, and judge grader coverage."""

from __future__ import annotations

from decimal import Decimal

import pytest

from chulk.evals import (
    CallableGrader,
    ContainsGrader,
    CostBudgetGrader,
    EvalCase,
    EvalDataset,
    EvalReference,
    EvalRunner,
    EvalSuite,
    EvalTarget,
    EvalTurn,
    EvalTurnResult,
    EventSequenceGrader,
    ExactAnswerGrader,
    JSONSchemaGrader,
    LLMJudgeGrader,
    LatencyGrader,
    MemoryRetrievalGrader,
    NoErrorGrader,
    PlanGrader,
    RegexGrader,
    SkillSelectionGrader,
    StatusGrader,
    TokenBudgetGrader,
    ToolCallGrader,
    TrialResult,
)
from chulk.events import AgentEvent, SerializedEventPayload
from chulk.llm import LLMClient
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage
from chulk.results import (
    Cost,
    Observation,
    Plan,
    PlanStatus,
    RunResult,
    RunStatus,
    ToolCall,
    Usage,
)


def _case(reference: EvalReference | None = None) -> EvalCase:
    return EvalCase("case", (EvalTurn("question"),), reference)


def _trial(
    *,
    content: str = "answer",
    status: RunStatus = RunStatus.COMPLETED,
    errors: tuple[str, ...] = (),
    tool_calls: tuple[ToolCall, ...] = (),
    observations: tuple[Observation, ...] = (),
    skills: tuple[str, ...] = (),
    memories: tuple[str, ...] = (),
    events: tuple[AgentEvent, ...] = (),
    usage: Usage | None = Usage(total_tokens=7),
    cost: Cost | None = Cost(amount=Decimal("0.02"), pricing_known=True),
    plan: Plan | None = None,
    duration: float = 0.1,
) -> TrialResult:
    result = RunResult(
        content,
        status,
        None,
        "conversation",
        None,
        usage=usage,
        cost=cost,
        tool_calls=tool_calls,
        observations=observations,
        loaded_skill_names=skills,
        loaded_memory_ids=memories,
        errors=errors,
        plan=plan,
    )
    return TrialResult(
        "case",
        "target",
        1,
        (EvalTurnResult(0, result, events, duration),),
        duration,
    )


def test_answer_status_and_error_graders_cover_success_and_failure() -> None:
    reference = EvalReference(
        answer='{"ok": true}',
        contains=("ok", "true"),
        regex=r'"ok"\s*:\s*true',
        json_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        status="completed",
    )
    case = _case(reference)
    trial = _trial(content='{"ok": true}')

    assert ExactAnswerGrader().grade(case, trial).passed
    assert ContainsGrader().grade(case, trial).passed
    assert RegexGrader().grade(case, trial).passed
    assert JSONSchemaGrader().grade(case, trial).passed
    assert StatusGrader().grade(case, trial).passed
    assert NoErrorGrader().grade(case, trial).passed
    assert not ExactAnswerGrader().grade(case, _trial(content="wrong")).passed
    assert not NoErrorGrader().grade(case, _trial(errors=("failed",))).passed


def test_tool_event_skill_memory_and_plan_graders_match_public_evidence() -> None:
    reference = EvalReference(
        tool_sequence=("lookup",),
        tool_arguments={"lookup": {"id": 7}},
        tool_results={"lookup": {"found": True}},
        tool_failures=(),
        event_sequence=("run.started", "run.completed"),
        skill_names=("research",),
        memory_ids=("memory-1",),
    )
    events = tuple(
        AgentEvent(name, "conversation", SerializedEventPayload())
        for name in ("run.started", "model.response.completed", "run.completed")
    )
    trial = _trial(
        tool_calls=(ToolCall("lookup", {"id": 7}, 1, success=True),),
        observations=(Observation("lookup", '{"found": true}'),),
        skills=("research",),
        memories=("memory-1",),
        events=events,
        plan=Plan("done", PlanStatus.COMPLETED),
    )
    case = _case(reference)

    assert ToolCallGrader().grade(case, trial).passed
    assert EventSequenceGrader().grade(case, trial).passed
    assert SkillSelectionGrader().grade(case, trial).passed
    assert MemoryRetrievalGrader().grade(case, trial).passed
    assert PlanGrader().grade(case, trial).passed


def test_latency_token_and_cost_graders_fail_closed_on_unknown_accounting() -> None:
    case = _case()
    known = _trial(duration=0.1)
    unknown = _trial(usage=None, cost=None)

    assert LatencyGrader(0.2).grade(case, known).passed
    assert TokenBudgetGrader(10).grade(case, known).passed
    assert CostBudgetGrader(0.03).grade(case, known).passed
    token_grade = TokenBudgetGrader(10).grade(case, unknown)
    cost_grade = CostBudgetGrader(0.03).grade(case, unknown)
    assert not token_grade.passed and token_grade.details["known"] is False
    assert not cost_grade.passed and cost_grade.details["known"] is False


@pytest.mark.asyncio
async def test_callable_grader_normalizes_sync_and_async_values() -> None:
    case = _case()
    trial = _trial()
    sync = CallableGrader("sync", lambda _case, _trial: 0.75)

    async def async_grade(_case, _trial):
        return True

    async_grader = CallableGrader("async", async_grade)

    assert sync.grade(case, trial).score == 0.75
    assert (await async_grader.grade_async(case, trial)).passed
    with pytest.raises(TypeError, match="use AsyncEvalRunner"):
        async_grader.grade(case, trial)


class _JudgeClient(LLMClient):
    def __init__(self, content: str) -> None:
        self.content = content

    def complete_response(self, messages, *, max_output_tokens=None):
        del messages, max_output_tokens
        return LLMResponse(
            self.content,
            usage=LLMUsage(total_tokens=11),
            cost=LLMCost(Decimal("0.03"), pricing_known=True),
            provider="judge-provider",
            model="judge-model",
        )


class _FailingJudgeClient(LLMClient):
    def complete_response(self, messages, *, max_output_tokens=None):
        del messages, max_output_tokens
        raise RuntimeError("judge unavailable")


def test_llm_judge_is_strict_records_identity_and_supports_pairwise() -> None:
    case = _case(EvalReference(answer="reference"))
    trial = _trial()
    grader = LLMJudgeGrader(
        _JudgeClient('{"score": 0.9, "passed": true, "reason": "clear"}'),
        "Be clear.",
    )

    grade = grader.grade(case, trial)
    pairwise = grader.grade_pairwise(case, trial, _trial(content="baseline"))

    assert grade.passed and pairwise.passed
    assert grade.details["judge_provider"] == "judge-provider"
    assert grade.details["judge_model"] == "judge-model"
    malformed = LLMJudgeGrader(
        _JudgeClient('{"score": 0.9, "passed": true, "reason": "clear", "extra": 1}'),
        "Be clear.",
    ).grade(case, trial)
    assert malformed.error and "fields must be exactly" in malformed.reason


def test_judge_provider_failure_is_an_operational_error_and_is_accounted() -> None:
    class Agent:
        def run_result(self, _message, **_kwargs):
            return _trial().final_result

        def close(self):
            return None

    case = _case()
    successful = EvalSuite(
        "judge-accounting",
        EvalDataset((case,)),
        (EvalTarget("target", lambda _context: Agent()),),
        (LLMJudgeGrader(_JudgeClient('{"score": 1, "passed": true, "reason": "ok"}'), "Good"),),
        required_graders=("quality.judge",),
        max_total_cost=1.0,
    )
    report = EvalRunner().run(successful)
    assert report.metrics["agent_tokens"] == 7
    assert report.metrics["judge_tokens"] == 11
    assert report.metrics["total_tokens"] == 18
    assert report.metrics["judge_cost"] == pytest.approx(0.03)
    assert report.metrics["total_cost"] == pytest.approx(0.05)

    failing = EvalSuite(
        "judge-failure",
        EvalDataset((case,)),
        (EvalTarget("target", lambda _context: Agent()),),
        (LLMJudgeGrader(_FailingJudgeClient(), "Good"),),
        required_graders=("quality.judge",),
        max_total_cost=1.0,
    )
    failed_report = EvalRunner().run(failing)
    assert failed_report.operational_errors
    assert "judge unavailable" in failed_report.operational_errors[0]


def test_model_judges_require_and_incrementally_enforce_a_cost_cap() -> None:
    class CountingJudgeClient(_JudgeClient):
        calls = 0

        def complete_response(self, messages, *, max_output_tokens=None):
            self.calls += 1
            response = super().complete_response(
                messages,
                max_output_tokens=max_output_tokens,
            )
            return LLMResponse(
                response.content,
                usage=response.usage,
                cost=LLMCost(Decimal("0.60"), pricing_known=True),
                provider=response.provider,
                model=response.model,
            )

    judge = CountingJudgeClient(
        '{"score": 1, "passed": true, "reason": "ok"}'
    )
    cases = tuple(
        EvalCase(f"case-{index}", (EvalTurn("question"),))
        for index in range(3)
    )

    class Agent:
        def run_result(self, _message, **_kwargs):
            return _trial().final_result

        def close(self):
            return None

    base = {
        "name": "judge-budget",
        "dataset": EvalDataset(cases),
        "targets": (EvalTarget("target", lambda _context: Agent()),),
        "graders": (LLMJudgeGrader(judge, "Good"),),
    }

    with pytest.raises(ValueError, match="model judges require max_total_cost"):
        EvalSuite(**base)

    report = EvalRunner().run(EvalSuite(**base, max_total_cost=1.0))

    assert judge.calls == 2
    assert len(report.cases) == 2
    assert any("exceeded cap" in error for error in report.operational_errors)
