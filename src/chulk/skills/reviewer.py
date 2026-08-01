"""Restricted model review that can only return learning proposal drafts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
import json
from typing import TYPE_CHECKING, Any, Protocol

from chulk.skills.lifecycle_models import LearningProposalKind
from chulk.skills.lifecycle_store import SQLiteSkillLifecycleStore
from chulk.skills.proposals import (
    AutomaticLearningBlocked,
    LearningProposalDraft,
    LearningProposalService,
)

if TYPE_CHECKING:
    from chulk.llm.usage import LLMCost, LLMUsage


class ReviewerLLM(Protocol):
    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Any: ...


class LearningReviewError(RuntimeError):
    """A reviewer response is invalid or unsafe to persist."""


class LearningReviewQuotaExceeded(LearningReviewError):
    """A daily learning-review quota has been exhausted."""


class LearningReviewTrigger(StrEnum):
    MANUAL = "manual"
    EXPLICIT_CORRECTION = "explicit_correction"
    SUCCESSFUL_TURN = "successful_turn"


@dataclass(frozen=True, slots=True)
class LearningReviewContext:
    """Bounded evidence supplied by the host, never gathered by the reviewer."""

    trigger: LearningReviewTrigger
    user_message: str
    assistant_response: str
    turn_id: str
    source_trace: str | None = None
    tool_call_count: int = 0
    host_confirmed_success: bool = False
    current_skill_manifests: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class LearningReviewPolicy:
    """Explicit trigger policy; successful-turn review is opt-in."""

    successful_turn_review_enabled: bool = False
    minimum_tool_calls: int = 3

    def should_review(self, context: LearningReviewContext) -> bool:
        if context.trigger in {
            LearningReviewTrigger.MANUAL,
            LearningReviewTrigger.EXPLICIT_CORRECTION,
        }:
            return True
        return (
            self.successful_turn_review_enabled
            and context.host_confirmed_success
            and context.tool_call_count >= self.minimum_tool_calls
        )


@dataclass(frozen=True, slots=True)
class LearningReviewQuota:
    max_proposals_per_day: int = 20
    max_proposals_per_review: int = 3
    max_tokens_per_day: int = 40_000
    max_output_tokens: int = 2_000
    max_cost_per_day: Decimal | None = None
    max_cost_per_review: Decimal | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        for field_name in (
            "max_proposals_per_day",
            "max_proposals_per_review",
            "max_tokens_per_day",
            "max_output_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in ("max_cost_per_day", "max_cost_per_review"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a finite non-negative Decimal"
                )
        if (
            self.max_cost_per_day is not None
            and self.max_cost_per_review is None
        ):
            raise ValueError(
                "max_cost_per_review is required for fail-closed cost quotas"
            )
        currency = self.currency.strip().upper()
        if not currency or len(currency) > 16:
            raise ValueError("currency must be a short non-empty code")
        object.__setattr__(self, "currency", currency)


@dataclass(frozen=True, slots=True)
class LearningReviewResult:
    drafts: tuple[LearningProposalDraft, ...]
    rationale: str
    usage: LLMUsage | None = None
    cost: LLMCost | None = None
    reviewer_model: str | None = None


@dataclass(frozen=True, slots=True)
class LearningReviewOutcome:
    skipped: bool
    rationale: str
    proposal_ids: tuple[str, ...] = ()
    review_run_id: str | None = None


class RestrictedLearningReviewer:
    """Call one model without exposing tools, stores, registries, or mutators."""

    def __init__(self, llm: ReviewerLLM) -> None:
        self._llm = llm

    @property
    def model(self) -> str | None:
        value = getattr(self._llm, "model", None)
        return str(value) if value is not None else None

    def review(
        self,
        context: LearningReviewContext,
        *,
        max_proposals: int,
        max_output_tokens: int,
    ) -> LearningReviewResult:
        messages = _review_messages(context, max_proposals=max_proposals)
        response = self._llm.complete_response(
            messages,
            max_output_tokens=max_output_tokens,
        )
        drafts, rationale = _parse_review_response(
            response.content,
            context=context,
            max_proposals=max_proposals,
        )
        return LearningReviewResult(
            drafts=drafts,
            rationale=rationale,
            usage=response.usage,
            cost=response.cost,
            reviewer_model=response.model
            or getattr(self._llm, "model", None),
        )


class LearningReviewCoordinator:
    """Enforce trigger and daily quotas around a restricted reviewer."""

    def __init__(
        self,
        *,
        reviewer: RestrictedLearningReviewer,
        proposal_service: LearningProposalService,
        lifecycle_store: SQLiteSkillLifecycleStore,
        policy: LearningReviewPolicy | None = None,
        quota: LearningReviewQuota | None = None,
        automatic_approval: bool = False,
        granted_capabilities: tuple[str, ...] = (),
    ) -> None:
        self.reviewer = reviewer
        self.proposal_service = proposal_service
        self.lifecycle_store = lifecycle_store
        self.policy = policy or LearningReviewPolicy()
        self.quota = quota or LearningReviewQuota()
        self.automatic_approval = automatic_approval
        self.granted_capabilities = tuple(granted_capabilities)

    def review(self, context: LearningReviewContext) -> LearningReviewOutcome:
        if not self.policy.should_review(context):
            return LearningReviewOutcome(
                skipped=True,
                rationale="learning review trigger is disabled",
            )

        period_start = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).isoformat()
        try:
            used = self.lifecycle_store.review_usage_since(
                period_start,
                currency=self.quota.currency,
            )
        except ValueError as exc:
            raise LearningReviewQuotaExceeded(str(exc)) from exc
        proposal_capacity = min(
            self.quota.max_proposals_per_review,
            self.quota.max_proposals_per_day - used.proposal_count,
        )
        if proposal_capacity < 1:
            raise LearningReviewQuotaExceeded(
                "daily learning proposal quota exceeded"
            )
        messages = _review_messages(context, max_proposals=proposal_capacity)
        reserved_tokens = (
            _estimate_message_tokens(messages)
            + self.quota.max_output_tokens
        )
        reserved_cost = self.quota.max_cost_per_review or Decimal(0)
        try:
            run_id = self.lifecycle_store.reserve_review_run(
                trigger=context.trigger.value,
                reviewer_model=self.reviewer.model,
                proposal_count=proposal_capacity,
                token_count=reserved_tokens,
                cost_amount=reserved_cost,
                occurred_at=period_start,
                max_proposals=self.quota.max_proposals_per_day,
                max_tokens=self.quota.max_tokens_per_day,
                max_cost=self.quota.max_cost_per_day,
                currency=self.quota.currency,
            )
        except ValueError as exc:
            raise LearningReviewQuotaExceeded(str(exc)) from exc

        result: LearningReviewResult | None = None
        try:
            result = self.reviewer.review(
                context,
                max_proposals=proposal_capacity,
                max_output_tokens=self.quota.max_output_tokens,
            )
            actual_tokens = (
                result.usage.total_tokens
                if result.usage is not None
                else reserved_tokens
            )
            actual_cost = _actual_cost(
                result.cost,
                reserved=reserved_cost,
                fail_closed=self.quota.max_cost_per_day is not None,
            )
            if (
                result.cost is not None
                and result.cost.amount is not None
                and result.cost.currency.upper() != self.quota.currency
            ):
                raise LearningReviewQuotaExceeded(
                    "learning reviewer cost currency does not match its quota"
                )
            if actual_tokens > reserved_tokens:
                raise LearningReviewQuotaExceeded(
                    "learning reviewer exceeded its reserved token budget"
                )
            if (
                self.quota.max_cost_per_day is not None
                and actual_cost > reserved_cost
            ):
                raise LearningReviewQuotaExceeded(
                    "learning reviewer exceeded its reserved cost budget"
                )
            records = self.proposal_service.create_many(
                result.drafts,
                reviewer_model=result.reviewer_model,
                cost=(
                    str(result.cost.amount)
                    if result.cost is not None
                    and result.cost.amount is not None
                    else None
                ),
                reviewer_metadata={
                    "review_run_id": run_id,
                    "reviewer_usage": (
                        result.usage.to_dict()
                        if result.usage is not None
                        else None
                    ),
                },
                review_run_id=run_id,
                review_token_count=actual_tokens,
                review_cost_amount=actual_cost,
            )
            proposal_ids = tuple(record.id for record in records)
            if self.automatic_approval:
                for record in records:
                    try:
                        self.proposal_service.approve(
                            record.id,
                            approved_by="automatic-learning-review",
                            automatic=True,
                            granted_capabilities=self.granted_capabilities,
                        )
                    except AutomaticLearningBlocked:
                        continue
            return LearningReviewOutcome(
                skipped=False,
                rationale=result.rationale,
                proposal_ids=proposal_ids,
                review_run_id=run_id,
            )
        except BaseException as exc:
            usage = result.usage if result is not None else None
            cost = result.cost if result is not None else None
            self.lifecycle_store.finalize_review_run(
                run_id,
                proposal_count=0,
                token_count=(
                    usage.total_tokens if usage is not None else reserved_tokens
                ),
                cost_amount=_actual_cost(
                    cost,
                    reserved=reserved_cost,
                    fail_closed=self.quota.max_cost_per_day is not None,
                ),
                failed=True,
                error=str(exc),
            )
            raise


def _review_messages(
    context: LearningReviewContext,
    *,
    max_proposals: int,
) -> list[dict[str, str]]:
    payload = {
        "trigger": context.trigger.value,
        "turn_id": _bounded(context.turn_id, 128),
        "source_trace": _optional_bounded(context.source_trace, 2_000),
        "user_message": _bounded(context.user_message, 20_000),
        "assistant_response": _bounded(context.assistant_response, 20_000),
        "tool_call_count": max(context.tool_call_count, 0),
        "host_confirmed_success": context.host_confirmed_success,
        "current_skill_manifests": list(context.current_skill_manifests)[:100],
    }
    system = f"""\
You are a restricted learning reviewer. You have no tools and no authority to
read files, use the network, call plugins, mutate memory, or mutate skills.
Review only the supplied evidence. Return one JSON object and nothing else.

Return either:
{{"decision":"no_action","rationale":"..."}}
or:
{{"decision":"propose","rationale":"...","proposals":[...]}}

Each proposal uses kind memory_create, memory_update, skill_create, skill_patch,
or skill_archive. Include rationale, confidence, and verification_steps.
Memory changes include content and memory_update includes target_name.
Skill changes include target_name, full SKILL.md content where applicable,
required_capabilities, metadata.scope (project or profile), and a unified diff
for patch/archive. Do not emit secrets. At most {max_proposals} proposals.
"""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True),
        },
    ]


def _parse_review_response(
    value: str,
    *,
    context: LearningReviewContext,
    max_proposals: int,
) -> tuple[tuple[LearningProposalDraft, ...], str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LearningReviewError("learning reviewer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LearningReviewError("learning reviewer response must be an object")
    decision = payload.get("decision")
    rationale = _required_string(payload.get("rationale"), "rationale", 4_000)
    if decision == "no_action":
        if payload.get("proposals") not in (None, []):
            raise LearningReviewError("no_action cannot include proposals")
        return (), rationale
    if decision != "propose":
        raise LearningReviewError("reviewer decision must be no_action or propose")
    raw_proposals = payload.get("proposals")
    if not isinstance(raw_proposals, list) or not raw_proposals:
        raise LearningReviewError("propose requires a non-empty proposals list")
    if len(raw_proposals) > max_proposals:
        raise LearningReviewError("reviewer exceeded the proposal count limit")
    drafts = tuple(
        _parse_draft(item, context=context) for item in raw_proposals
    )
    return drafts, rationale


def _parse_draft(
    value: object,
    *,
    context: LearningReviewContext,
) -> LearningProposalDraft:
    if not isinstance(value, dict):
        raise LearningReviewError("each proposal must be an object")
    raw_kind = value.get("kind")
    if not isinstance(raw_kind, str):
        raise LearningReviewError("proposal kind is invalid")
    try:
        kind = LearningProposalKind(raw_kind)
    except ValueError as exc:
        raise LearningReviewError("proposal kind is invalid") from exc
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise LearningReviewError("proposal metadata must be an object")
    required_capabilities = _string_tuple(
        value.get("required_capabilities", []),
        "required_capabilities",
        50,
    )
    verification_steps = _string_tuple(
        value.get("verification_steps", []),
        "verification_steps",
        50,
    )
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise LearningReviewError("proposal confidence must be between 0 and 1")
    return LearningProposalDraft(
        kind=kind,
        target_name=_optional_string(value.get("target_name"), 128),
        rationale=_required_string(
            value.get("rationale"),
            "proposal rationale",
            4_000,
        ),
        evidence_turn_ids=(context.turn_id,),
        source_trace=context.source_trace,
        content=_optional_string(value.get("content"), 500_000),
        diff=_optional_string(value.get("diff"), 200_000),
        required_capabilities=required_capabilities,
        confidence=float(confidence),
        verification_steps=verification_steps,
        metadata=dict(metadata),
    )


def _actual_cost(
    cost: LLMCost | None,
    *,
    reserved: Decimal,
    fail_closed: bool,
) -> Decimal:
    if cost is not None and cost.amount is not None:
        return cost.amount
    return reserved if fail_closed else Decimal(0)


def _estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return max(sum(len(item["content"]) for item in messages) // 4, 1)


def _required_string(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearningReviewError(f"{label} must be a non-empty string")
    clean = value.strip()
    if len(clean) > limit:
        raise LearningReviewError(f"{label} exceeds its size limit")
    return clean


def _optional_string(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return _required_string(value, "proposal field", limit)


def _string_tuple(value: object, label: str, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise LearningReviewError(f"{label} must be a bounded list")
    return tuple(
        _required_string(item, label, 1_000)
        for item in value
    )


def _bounded(value: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("review context text cannot be empty")
    return value[:limit]


def _optional_bounded(value: str | None, limit: int) -> str | None:
    return None if value is None else _bounded(value, limit)


__all__ = [
    "LearningReviewContext",
    "LearningReviewCoordinator",
    "LearningReviewError",
    "LearningReviewOutcome",
    "LearningReviewPolicy",
    "LearningReviewQuota",
    "LearningReviewQuotaExceeded",
    "LearningReviewResult",
    "LearningReviewTrigger",
    "RestrictedLearningReviewer",
]
