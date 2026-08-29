"""Complete deterministic, callable, and judge grader coverage."""

from __future__ import annotations

from decimal import Decimal
import json

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
    RubricDimension,
    RubricJudgeGrader,
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
            usage=LLMUsage(
                input_tokens=8,
                output_tokens=3,
                total_tokens=11,
                cache_hit_input_tokens=4,
                cache_miss_input_tokens=4,
            ),
            cost=LLMCost(Decimal("0.03"), pricing_known=True),
            provider="judge-provider",
            model="judge-model",
        )


class _FailingJudgeClient(LLMClient):
    def complete_response(self, messages, *, max_output_tokens=None):
        del messages, max_output_tokens
        raise RuntimeError("judge unavailable")


class _SequencedJudgeClient(_JudgeClient):
    def __init__(self, *contents: str) -> None:
        super().__init__("")
        self.contents = list(contents)
        self.messages = []

    def complete_response(self, messages, *, max_output_tokens=None):
        self.messages.append(messages)
        self.content = self.contents.pop(0)
        return super().complete_response(
            messages,
            max_output_tokens=max_output_tokens,
        )


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
    assert malformed.details["judge"] is True
    assert malformed.details["cost"]["amount"] == "0.03"


def test_rubric_judge_weights_dimensions_and_enforces_vetoes() -> None:
    dimensions = (
        RubricDimension(
            "Accuracy",
            {"excellent": "All facts are correct.", "fail": "Core facts are wrong."},
            weight=2,
        ),
        RubricDimension(
            "Completeness",
            {"excellent": "No required facts are missing.", "fail": "Key facts are missing."},
        ),
        RubricDimension(
            "Hallucination",
            {
                "pass": "Every claim is supported.",
                "fail": "Any unsupported claim is present.",
            },
            veto=True,
        ),
    )
    response = json.dumps(
        {
            "dimensions": [
                {"name": "Accuracy", "score": 0.9, "reason": "Accurate."},
                {"name": "Completeness", "score": 0.6, "reason": "One omission."},
                {"name": "Hallucination", "score": 1, "reason": "Supported."},
            ],
            "reason": "Strong overall.",
        }
    )
    grade = RubricJudgeGrader(
        _JudgeClient(response),
        dimensions,
    ).grade(_case(), _trial())

    assert grade.passed
    assert grade.score == pytest.approx(0.8)
    assert grade.details["weighted_score_before_veto"] == pytest.approx(0.8)
    assert grade.details["dimensions"][0]["name"] == "Accuracy"
    assert grade.details["veto_triggered"] is False

    vetoed_response = json.loads(response)
    vetoed_response["dimensions"][2] = {
        "name": "Hallucination",
        "score": 0,
        "reason": "Unsupported claim found.",
    }
    vetoed = RubricJudgeGrader(
        _JudgeClient(json.dumps(vetoed_response)),
        dimensions,
    ).grade(_case(), _trial())

    assert not vetoed.passed
    assert vetoed.score == 0.0
    assert vetoed.details["weighted_score_before_veto"] == pytest.approx(0.8)
    assert vetoed.details["failed_vetoes"] == ("Hallucination",)


@pytest.mark.asyncio
async def test_rubric_pairwise_judge_swaps_order_and_flags_position_bias() -> None:
    dimension = RubricDimension(
        "Quality",
        {"excellent": "Better answer.", "fail": "Worse answer."},
    )
    position_biased = _JudgeClient('{"winner": "A", "reason": "A is better."}')
    biased_grade = RubricJudgeGrader(
        position_biased,
        (dimension,),
    ).grade_pairwise(_case(), _trial(), _trial(content="baseline"))

    assert not biased_grade.passed
    assert biased_grade.score == 0.5
    assert biased_grade.details["position_consistent"] is False
    assert biased_grade.details["needs_review"] is True
    assert biased_grade.details["usage"]["total_tokens"] == 22
    assert biased_grade.details["cost"]["amount"] == "0.06"

    consistent = _SequencedJudgeClient(
        '{"winner": "A", "reason": "A is better."}',
        '{"winner": "B", "reason": "B is better."}',
    )
    consistent_grade = await RubricJudgeGrader(
        consistent,
        (dimension,),
    ).grade_pairwise_async(_case(), _trial(), _trial(content="baseline"))

    assert consistent_grade.passed
    assert consistent_grade.details["winner"] == "candidate"
    first_payload = json.loads(consistent.messages[0][1]["content"])
    second_payload = json.loads(consistent.messages[1][1]["content"])
    assert first_payload["answer_a"] == "answer"
    assert second_payload["answer_a"] == "baseline"


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
    assert report.metrics["cache_hit_input_tokens"] == 4
    assert report.metrics["cache_miss_input_tokens"] == 4
    assert report.metrics["cache_hit_ratio"] == 0.5

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
