"""Deterministic test utilities for SDK embeddings and examples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import inspect
import json
from threading import Lock
from types import MappingProxyType
from typing import Any

from chulk.core.actions import AgentAction
from chulk.approvals import (
    ApprovalDecision,
    ApprovalStatus,
    ApprovalSubmission,
    ApprovalValidation,
    AsyncDurableApprovalService,
    DurableApprovalService,
)
from chulk.events import AgentEvent, RunLifecyclePayload
from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    GatewayRunTarget,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
    TrustLevel,
)
from chulk.hosting import AsyncRuntimeServices, ExecutionScope, RuntimeServices
from chulk.llm.base import LLMClient, LLMError, LLMStreamChunk
from chulk.runs import (
    EffectStatus,
    ParentCompletionStatus,
    ParentRunPolicy,
    ReconciliationDecision,
    RetryPolicy,
    RunStatus,
    RunSubmission,
    StepDefinition,
)
from chulk.usage import BudgetScope, RunBudget


ScriptedResponse = str | Mapping[str, Any] | AgentAction


@dataclass(frozen=True)
class _ScriptedCall:
    messages: tuple[Mapping[str, str], ...]
    max_output_tokens: int | None
    response: str


class ScriptedLLMClient(LLMClient):
    """Return an ordered script of model responses without network access.

    The client uses the normal :class:`LLMClient` action parser and usage
    estimator, so tests exercise the same provider-neutral contract as a live
    integration. Each response is consumed exactly once.
    """

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses: Iterable[ScriptedResponse], *, chunk_size: int = 16) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        self._responses = [_serialize_response(response) for response in responses]
        self._chunk_size = chunk_size
        self._calls: list[_ScriptedCall] = []
        self._lock = Lock()

    @property
    def call_log(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable snapshots of completed scripted calls."""
        with self._lock:
            return tuple(
                MappingProxyType(
                    {
                        "messages": tuple(MappingProxyType(dict(message)) for message in call.messages),
                        "max_output_tokens": call.max_output_tokens,
                        "response": call.response,
                    }
                )
                for call in self._calls
            )

    @property
    def remaining(self) -> int:
        """Return the number of unconsumed scripted responses."""
        with self._lock:
            return len(self._responses)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Consume and return the next scripted response."""
        with self._lock:
            if not self._responses:
                raise LLMError(
                    "ScriptedLLMClient response script is exhausted",
                    provider=self.provider,
                    model=self.model,
                    retryable=False,
                )
            response = self._responses.pop(0)
            self._calls.append(
                _ScriptedCall(
                    messages=tuple(dict(message) for message in messages),
                    max_output_tokens=max_output_tokens,
                    response=response,
                )
            )
        return response

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield the next scripted response in deterministic text chunks."""
        response = self.complete_response(messages, max_output_tokens=max_output_tokens)
        for offset in range(0, len(response.content), self._chunk_size):
            yield LLMStreamChunk(
                type="text_delta",
                text=response.content[offset : offset + self._chunk_size],
            )
        yield LLMStreamChunk(type="completed", usage=response.usage, cost=response.cost)


class HostedContractError(AssertionError):
    """Raised when an application-owned hosted service violates the SDK contract."""


@dataclass(frozen=True, slots=True)
class HostedContractReport:
    """Names of the portable hosted-runtime checks that completed."""

    checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks)


def assert_hosted_services_contract(
    services: RuntimeServices,
    *,
    first_scope: ExecutionScope,
    second_scope: ExecutionScope,
) -> HostedContractReport:
    """Exercise the minimum sync isolation and lifecycle service contract."""
    _contract_scopes(first_scope, second_scope)
    first = services.resolve(first_scope)
    second = services.resolve(second_scope)
    checks: list[str] = []
    conversation_id = first_scope.conversation_id or "contract-conversation"
    first.sessions.store.create_conversation(
        conversation_id,
        provider="contract",
        model="contract",
    )
    _must_reject(
        lambda: second.sessions.store.get_conversation(conversation_id),
        "sessions allowed a cross-scope read",
    )
    checks.append("session_isolation")

    artifact = first.artifacts.write("contract", "private")
    _must_reject(
        lambda: second.artifacts.read(artifact.artifact_id),
        "artifacts allowed a cross-scope read",
    )
    checks.append("artifact_isolation")

    first.runs.submit(
        first_scope,
        _contract_submission(),
        actor="contract",
    )
    _must_reject(
        lambda: second.runs.get(second_scope, first_scope.run_id),
        "durable runs allowed a cross-scope read",
    )
    checks.append("run_isolation")

    event = _contract_event(first_scope)
    first.events.emit(event)
    _must_reject(
        lambda: second.events.emit(event),
        "event sink accepted another scope",
    )
    checks.append("event_isolation")
    return HostedContractReport(tuple(checks))


async def assert_async_hosted_services_contract(
    services: AsyncRuntimeServices,
    *,
    first_scope: ExecutionScope,
    second_scope: ExecutionScope,
) -> HostedContractReport:
    """Exercise native async run, approval, and event service boundaries."""
    _contract_scopes(first_scope, second_scope)
    first = await services.resolve_async(first_scope)
    second = await services.resolve_async(second_scope)
    checks: list[str] = []
    await first.runs.submit(
        first_scope,
        _contract_submission(),
        actor="contract",
    )
    await _must_reject_async(
        lambda: second.runs.get(second_scope, first_scope.run_id),
        "async durable runs allowed a cross-scope read",
    )
    checks.append("async_run_isolation")

    emit = first.events.emit(_contract_event(first_scope))
    _require(inspect.isawaitable(emit), "async event sink did not return an awaitable")
    await emit
    await _must_reject_async(
        lambda: second.events.emit(_contract_event(first_scope)),
        "async event sink accepted another scope",
    )
    checks.append("async_event_isolation")

    first_approvals = await first.approvals.list(
        first_scope,
        run_id=first_scope.run_id,
    )
    second_approvals = await second.approvals.list(
        second_scope,
        run_id=second_scope.run_id,
    )
    _require(
        first_approvals == second_approvals == (),
        "async approval stores did not start isolated",
    )
    checks.append("async_approval_isolation")
    return HostedContractReport(tuple(checks))


def assert_durable_execution_contract(
    runs: Any,
    approvals: Any,
    *,
    scope: ExecutionScope,
) -> HostedContractReport:
    """Exercise duplicate, unknown-effect, reconciliation, and approval rules."""
    first = runs.submit(scope, _contract_submission(), actor="contract")
    duplicate = runs.submit(scope, _contract_submission(), actor="contract")
    _require(first.id == duplicate.id, "duplicate trigger created another run")
    claim = runs.claim(scope, worker_id="contract-worker-a")
    _require(claim is not None, "durable run could not be claimed")
    runs.start_step(scope, claim, "contract")
    effect = runs.begin_effect(
        scope,
        claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    repeated = runs.begin_effect(
        scope,
        claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    _require(effect.id == repeated.id, "logical effect was not idempotent")
    runs.mark_effect_started(scope, claim, effect.id)
    unknown = runs.mark_effect_unknown(
        scope,
        claim,
        effect.id,
        reason="contract transport disconnected after dispatch",
    )
    _require(unknown.status is EffectStatus.UNKNOWN, "effect did not become unknown")
    _require(
        runs.get(scope, scope.run_id).status is RunStatus.UNKNOWN,
        "unknown effect did not stop its run",
    )
    reconciled = runs.reconcile_effect(
        scope,
        effect.id,
        decision=ReconciliationDecision.RETRY,
        actor="contract-operator",
        reason="contract target confirms no write",
    )
    _require(
        reconciled.effect.status is EffectStatus.INTENDED,
        "reconciliation did not make the logical effect retryable",
    )
    retry_claim = runs.claim(scope, worker_id="contract-worker-b")
    _require(retry_claim is not None, "reconciled run could not be reclaimed")
    runs.start_step(scope, retry_claim, "contract")
    retry_effect = runs.begin_effect(
        scope,
        retry_claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    _require(retry_effect.id == effect.id, "reconciliation duplicated the effect")
    runs.mark_effect_started(scope, retry_claim, effect.id)
    runs.complete_effect(
        scope,
        retry_claim,
        effect.id,
        result_digest="sha256:contract-result",
    )
    runs.complete_step(
        scope,
        retry_claim,
        "contract",
        result={"status": "completed"},
    )

    approval_scope = scope.child(run_id=f"{scope.run_id}-approval")
    runs.submit(
        approval_scope,
        _contract_submission(idempotency_key="hosted-contract-approval"),
        actor="contract",
    )
    approval_claim = runs.claim(
        approval_scope,
        worker_id="contract-worker-a",
    )
    _require(approval_claim is not None, "approval run could not be claimed")
    runs.start_step(approval_scope, approval_claim, "contract")
    coordinator = DurableApprovalService(approvals, runs)
    paused = coordinator.request(
        approval_scope,
        approval_claim,
        _contract_approval(),
    )
    _require(
        paused.run.status is RunStatus.WAITING_FOR_APPROVAL,
        "approval did not release the worker",
    )
    DurableApprovalService(approvals, runs).decide(
        approval_scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="contract-operator",
        reason="contract approval",
        idempotency_key="hosted-contract-decision",
    )
    resumed = DurableApprovalService(approvals, runs).resume(
        approval_scope,
        paused.approval.id,
        _contract_approval_validation(approval_scope),
        actor="contract-worker-b",
    )
    _require(resumed.resumed, "approved run did not resume")
    _require(
        resumed.approval.status is ApprovalStatus.CONSUMED,
        "approval was not consumed exactly once",
    )
    sequences = [
        event.sequence
        for event in runs.events(scope, scope.run_id)
    ]
    _require(
        sequences == list(range(1, len(sequences) + 1)),
        "durable event sequence is not deterministic",
    )
    return HostedContractReport(
        (
            "duplicate_run",
            "idempotent_effect",
            "unknown_effect",
            "effect_reconciliation",
            "cross_process_approval",
            "event_ordering",
        )
    )


async def assert_async_durable_execution_contract(
    runs: Any,
    approvals: Any,
    *,
    scope: ExecutionScope,
) -> HostedContractReport:
    """Native async equivalent of :func:`assert_durable_execution_contract`."""
    first = await runs.submit(scope, _contract_submission(), actor="contract")
    duplicate = await runs.submit(scope, _contract_submission(), actor="contract")
    _require(first.id == duplicate.id, "async duplicate trigger created another run")
    claim = await runs.claim(scope, worker_id="contract-worker-a")
    _require(claim is not None, "async durable run could not be claimed")
    await runs.start_step(scope, claim, "contract")
    effect = await runs.begin_effect(
        scope,
        claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    repeated = await runs.begin_effect(
        scope,
        claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    _require(effect.id == repeated.id, "async logical effect was not idempotent")
    await runs.mark_effect_started(scope, claim, effect.id)
    unknown = await runs.mark_effect_unknown(
        scope,
        claim,
        effect.id,
        reason="contract transport disconnected after dispatch",
    )
    _require(unknown.status is EffectStatus.UNKNOWN, "async effect did not become unknown")
    await runs.reconcile_effect(
        scope,
        effect.id,
        decision=ReconciliationDecision.RETRY,
        actor="contract-operator",
        reason="contract target confirms no write",
    )
    retry_claim = await runs.claim(scope, worker_id="contract-worker-b")
    _require(retry_claim is not None, "async reconciled run could not be reclaimed")
    await runs.start_step(scope, retry_claim, "contract")
    retry_effect = await runs.begin_effect(
        scope,
        retry_claim,
        "contract",
        logical_key="contract:external-write",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
    )
    _require(retry_effect.id == effect.id, "async reconciliation duplicated the effect")
    await runs.mark_effect_started(scope, retry_claim, effect.id)
    await runs.complete_effect(
        scope,
        retry_claim,
        effect.id,
        result_digest="sha256:contract-result",
    )
    await runs.complete_step(
        scope,
        retry_claim,
        "contract",
        result={"status": "completed"},
    )

    approval_scope = scope.child(run_id=f"{scope.run_id}-approval")
    await runs.submit(
        approval_scope,
        _contract_submission(idempotency_key="hosted-contract-approval"),
        actor="contract",
    )
    approval_claim = await runs.claim(
        approval_scope,
        worker_id="contract-worker-a",
    )
    _require(approval_claim is not None, "async approval run could not be claimed")
    await runs.start_step(approval_scope, approval_claim, "contract")
    coordinator = AsyncDurableApprovalService(approvals, runs)
    paused = await coordinator.request(
        approval_scope,
        approval_claim,
        _contract_approval(),
    )
    await AsyncDurableApprovalService(approvals, runs).decide(
        approval_scope,
        paused.approval.id,
        ApprovalDecision.APPROVE,
        decided_by="contract-operator",
        reason="contract approval",
        idempotency_key="hosted-contract-decision",
    )
    resumed = await AsyncDurableApprovalService(approvals, runs).resume(
        approval_scope,
        paused.approval.id,
        _contract_approval_validation(approval_scope),
        actor="contract-worker-b",
    )
    _require(resumed.resumed, "async approved run did not resume")
    sequences = [
        event.sequence
        for event in await runs.events(scope, scope.run_id)
    ]
    _require(
        sequences == list(range(1, len(sequences) + 1)),
        "async durable event sequence is not deterministic",
    )
    return HostedContractReport(
        (
            "async_duplicate_run",
            "async_idempotent_effect",
            "async_effect_reconciliation",
            "async_cross_process_approval",
            "async_event_ordering",
        )
    )


def assert_parent_child_run_contract(
    runs: Any,
    *,
    scope: ExecutionScope,
) -> HostedContractReport:
    """Exercise bounded fan-out, isolation, aggregation, and delivery recovery."""
    policy = _contract_parent_policy(required_children=2, max_children=2)
    parent = runs.submit_parent(
        scope,
        _contract_parent_submission(scope.run_id),
        policy=policy,
        actor="contract",
    )
    _require(
        parent.run.status is RunStatus.WAITING_FOR_CHILDREN,
        "parent run entered the worker queue",
    )
    child_scopes = (
        scope.child(
            run_id=f"{scope.run_id}-child-1",
            agent_id="contract-child",
            agent_version="published-1",
            grants=frozenset(),
        ),
        scope.child(
            run_id=f"{scope.run_id}-child-2",
            agent_id="contract-child",
            agent_version="published-2",
            grants=frozenset(),
        ),
    )
    children = tuple(
        runs.submit_child(
            scope,
            child_scope,
            _contract_child_submission(
                f"{scope.run_id}-child-key-{index}",
                model_calls=2,
                tokens=200,
            ),
            definition_revision=f"published-{index}",
            actor="contract",
        )
        for index, child_scope in enumerate(child_scopes, start=1)
    )
    duplicate = runs.submit_child(
        scope,
        child_scopes[0],
        _contract_child_submission(
            f"{scope.run_id}-child-key-1",
            model_calls=2,
            tokens=200,
        ),
        definition_revision="published-1",
        actor="contract",
    )
    _require(
        duplicate.run.id == children[0].run.id,
        "duplicate child submission created another run",
    )
    _must_reject(
        lambda: runs.get_child(child_scopes[0], children[1].run.id),
        "child scope could inspect a sibling run",
    )
    parentless_child_scope = replace(
        child_scopes[0],
        actor_id="contract-other-actor",
        parent_run_id=None,
    )
    _must_reject(
        lambda: runs.get(parentless_child_scope, children[0].run.id),
        "parentless scope could inspect a linked child run",
    )
    ordinary_scope = replace(
        child_scopes[0],
        run_id=f"{scope.run_id}-ordinary",
        parent_run_id=None,
    )
    runs.submit(
        ordinary_scope,
        _contract_child_submission(
            f"{scope.run_id}-ordinary-key",
            model_calls=1,
            tokens=100,
        ),
        actor="contract",
    )
    ordinary_claim = runs.claim(
        ordinary_scope,
        worker_id="contract-ordinary-worker",
    )
    _require(
        ordinary_claim is not None
        and ordinary_claim.run_id == ordinary_scope.run_id,
        "parentless queue claim was blocked by a linked child",
    )
    runs.start_step(ordinary_scope, ordinary_claim, "contract")
    runs.complete_step(
        ordinary_scope,
        ordinary_claim,
        "contract",
        result={"status": "completed"},
    )
    runs.complete(
        ordinary_scope,
        ordinary_claim,
        result={"status": "completed"},
    )
    _must_reject(
        lambda: runs.submit_child(
            scope,
            scope.child(
                run_id=f"{scope.run_id}-child-3",
                agent_id="contract-child",
                agent_version="published-3",
                grants=frozenset(),
            ),
            _contract_child_submission(
                f"{scope.run_id}-child-key-3",
                model_calls=1,
                tokens=100,
            ),
            definition_revision="published-3",
            actor="contract",
        ),
        "parent accepted child work beyond its fan-out limit",
    )
    for index, (child_scope, child) in enumerate(
        zip(child_scopes, children, strict=True),
        start=1,
    ):
        claim = runs.claim(
            child_scope,
            worker_id=f"contract-child-worker-{index}",
            run_id=child.run.id,
        )
        _require(claim is not None, "child run could not be claimed")
        runs.start_step(child_scope, claim, "contract")
        if index == 1:
            paused = runs.pause_for_approval(
                child_scope,
                claim,
                "contract",
                approval_id="contract-child-approval",
                payload={"child": index},
            )
            _require(
                paused.status is RunStatus.WAITING_FOR_APPROVAL,
                "child approval pause was not durable",
            )
            runs.resume(
                child_scope,
                child.run.id,
                actor="contract-approver",
                reason="contract child approved",
            )
            claim = runs.claim(
                child_scope,
                worker_id="contract-child-worker-1-resumed",
                run_id=child.run.id,
            )
            _require(claim is not None, "approved child could not resume")
            runs.start_step(child_scope, claim, "contract")
        else:
            retrying = runs.fail_step(
                child_scope,
                claim,
                "contract",
                reason="contract retry",
                retryable=True,
            )
            _require(
                retrying.status is RunStatus.WAITING_FOR_RETRY,
                "retryable child failure did not pause durably",
            )
            runs.resume(
                child_scope,
                child.run.id,
                actor="contract-recovery",
                reason="contract retry is ready",
            )
            claim = runs.claim(
                child_scope,
                worker_id="contract-child-worker-2-retried",
                run_id=child.run.id,
            )
            _require(claim is not None, "retryable child could not resume")
            runs.start_step(child_scope, claim, "contract")
        progress = runs.record_child_progress(
            child_scope,
            claim,
            sequence=1,
            payload={"completed": index, "total": 2},
            idempotency_key=f"progress-{index}",
        )
        _require(
            progress.sequence == 1,
            "child progress was not committed in order",
        )
        runs.complete_step(
            child_scope,
            claim,
            "contract",
            result={"child": index},
        )
        runs.complete(
            child_scope,
            claim,
            result={"child": index, "status": "completed"},
        )
    aggregated = runs.aggregate_children(
        scope,
        actor="contract",
        idempotency_key="contract-aggregate",
    )
    repeated = runs.aggregate_children(
        scope,
        actor="contract",
        idempotency_key="contract-aggregate",
    )
    _require(
        aggregated.run.status is RunStatus.COMPLETED
        and repeated.aggregation_revision == 1,
        "parent aggregation was not idempotent",
    )
    _require(
        all(child.terminal_evidence for child in aggregated.children),
        "parent aggregation omitted terminal child evidence",
    )
    delivery_claim = runs.claim_parent_completion(
        scope,
        worker_id="contract-delivery-a",
        parent_run_id=scope.run_id,
    )
    _require(delivery_claim is not None, "parent completion was not enqueued")
    unknown = runs.mark_parent_completion_unknown(
        scope,
        delivery_claim,
        reason="contract disconnected after delivery",
    )
    _require(
        unknown.status is ParentCompletionStatus.UNKNOWN,
        "ambiguous parent completion was not quarantined",
    )
    _require(
        runs.claim_parent_completion(
            scope,
            worker_id="contract-delivery-b",
            parent_run_id=scope.run_id,
        )
        is None,
        "ambiguous parent completion was blindly replayed",
    )
    runs.reconcile_parent_completion(
        scope,
        scope.run_id,
        delivered=False,
        actor="contract-operator",
        reason="contract target confirms no delivery",
    )
    retry_claim = runs.claim_parent_completion(
        scope,
        worker_id="contract-delivery-b",
        parent_run_id=scope.run_id,
    )
    _require(retry_claim is not None, "reconciled parent completion was not retryable")
    delivered = runs.complete_parent_completion(scope, retry_claim)
    _require(
        delivered.status is ParentCompletionStatus.DELIVERED,
        "parent completion was not delivered",
    )
    _require(
        runs.claim_parent_completion(
            scope,
            worker_id="contract-delivery-c",
            parent_run_id=scope.run_id,
        )
        is None,
        "delivered parent completion was emitted twice",
    )

    child_cancel_scope = replace(
        scope,
        run_id=f"{scope.run_id}-child-cancel-parent",
    )
    runs.submit_parent(
        child_cancel_scope,
        _contract_parent_submission(child_cancel_scope.run_id),
        policy=_contract_parent_policy(required_children=1, max_children=1),
        actor="contract",
    )
    cancelled_child_scope = child_cancel_scope.child(
        run_id=f"{child_cancel_scope.run_id}-child",
        agent_id="contract-child",
        agent_version="published-cancelled",
        grants=frozenset(),
    )
    cancelled_child = runs.submit_child(
        child_cancel_scope,
        cancelled_child_scope,
        _contract_child_submission(
            f"{child_cancel_scope.run_id}-child-key",
            model_calls=1,
            tokens=100,
        ),
        definition_revision=cancelled_child_scope.agent_version,
        actor="contract",
    )
    cancelled_child = runs.request_child_cancellation(
        child_cancel_scope,
        cancelled_child.run.id,
        actor="contract",
        reason="contract child cancelled",
    )
    _require(
        cancelled_child.run.status is RunStatus.CANCELLED,
        "queued child cancellation was not terminal",
    )
    failed_parent = runs.aggregate_children(
        child_cancel_scope,
        actor="contract",
        idempotency_key="contract-child-cancel-aggregate",
    )
    _require(
        failed_parent.run.status is RunStatus.FAILED,
        "cancelled child did not propagate to the parent aggregate",
    )

    parent_cancel_scope = replace(
        scope,
        run_id=f"{scope.run_id}-parent-cancel",
    )
    runs.submit_parent(
        parent_cancel_scope,
        _contract_parent_submission(parent_cancel_scope.run_id),
        policy=_contract_parent_policy(required_children=1, max_children=1),
        actor="contract",
    )
    active_child_scope = parent_cancel_scope.child(
        run_id=f"{parent_cancel_scope.run_id}-child",
        agent_id="contract-child",
        agent_version="published-active",
        grants=frozenset(),
    )
    active_child = runs.submit_child(
        parent_cancel_scope,
        active_child_scope,
        _contract_child_submission(
            f"{parent_cancel_scope.run_id}-child-key",
            model_calls=1,
            tokens=100,
        ),
        definition_revision=active_child_scope.agent_version,
        actor="contract",
    )
    active_claim = runs.claim(
        active_child_scope,
        worker_id="contract-active-child",
        run_id=active_child.run.id,
    )
    _require(active_claim is not None, "active cancellation child was not claimed")
    runs.start_step(active_child_scope, active_claim, "contract")
    cancelling_parent = runs.request_parent_cancellation(
        parent_cancel_scope,
        actor="contract",
        reason="contract parent cancelled",
    )
    _require(
        cancelling_parent.run.cancellation_requested
        and cancelling_parent.children[0].run.cancellation_requested,
        "parent cancellation did not propagate to its active child",
    )
    runs.cancel(
        active_child_scope,
        active_child.run.id,
        actor="contract-active-child",
        reason="contract parent cancellation observed",
        claim=active_claim,
    )
    cancelled_parent = runs.aggregate_children(
        parent_cancel_scope,
        actor="contract",
        idempotency_key="contract-parent-cancel-aggregate",
    )
    _require(
        cancelled_parent.run.status is RunStatus.CANCELLED,
        "settled parent cancellation did not terminalize the parent",
    )
    return HostedContractReport(
        (
            "bounded_fanout",
            "child_scope_isolation",
            "parentless_child_scope_isolation",
            "parentless_queue_child_isolation",
            "child_approval_resume",
            "child_retry_resume",
            "ordered_progress",
            "terminal_evidence",
            "idempotent_parent_aggregation",
            "parent_delivery_reconciliation",
            "exactly_once_parent_delivery",
            "child_cancellation_propagation",
            "parent_cancellation_propagation",
        )
    )


async def assert_async_parent_child_run_contract(
    runs: Any,
    *,
    scope: ExecutionScope,
) -> HostedContractReport:
    """Native async equivalent of :func:`assert_parent_child_run_contract`."""
    policy = _contract_parent_policy(required_children=2, max_children=2)
    parent = await runs.submit_parent(
        scope,
        _contract_parent_submission(scope.run_id),
        policy=policy,
        actor="contract",
    )
    _require(
        parent.run.status is RunStatus.WAITING_FOR_CHILDREN,
        "async parent run entered the worker queue",
    )
    child_scopes = (
        scope.child(
            run_id=f"{scope.run_id}-child-1",
            agent_id="contract-child",
            agent_version="published-1",
            grants=frozenset(),
        ),
        scope.child(
            run_id=f"{scope.run_id}-child-2",
            agent_id="contract-child",
            agent_version="published-2",
            grants=frozenset(),
        ),
    )
    children = []
    for index, child_scope in enumerate(child_scopes, start=1):
        children.append(
            await runs.submit_child(
                scope,
                child_scope,
                _contract_child_submission(
                    f"{scope.run_id}-child-key-{index}",
                    model_calls=2,
                    tokens=200,
                ),
                definition_revision=f"published-{index}",
                actor="contract",
            )
        )
    await _must_reject_async(
        lambda: runs.get_child(child_scopes[0], children[1].run.id),
        "async child scope could inspect a sibling run",
    )
    parentless_child_scope = replace(
        child_scopes[0],
        actor_id="contract-other-actor",
        parent_run_id=None,
    )
    await _must_reject_async(
        lambda: runs.get(parentless_child_scope, children[0].run.id),
        "async parentless scope could inspect a linked child run",
    )
    ordinary_scope = replace(
        child_scopes[0],
        run_id=f"{scope.run_id}-ordinary",
        parent_run_id=None,
    )
    await runs.submit(
        ordinary_scope,
        _contract_child_submission(
            f"{scope.run_id}-ordinary-key",
            model_calls=1,
            tokens=100,
        ),
        actor="contract",
    )
    ordinary_claim = await runs.claim(
        ordinary_scope,
        worker_id="contract-ordinary-worker",
    )
    _require(
        ordinary_claim is not None
        and ordinary_claim.run_id == ordinary_scope.run_id,
        "async parentless queue claim was blocked by a linked child",
    )
    await runs.start_step(ordinary_scope, ordinary_claim, "contract")
    await runs.complete_step(
        ordinary_scope,
        ordinary_claim,
        "contract",
        result={"status": "completed"},
    )
    await runs.complete(
        ordinary_scope,
        ordinary_claim,
        result={"status": "completed"},
    )
    for index, (child_scope, child) in enumerate(
        zip(child_scopes, children, strict=True),
        start=1,
    ):
        claim = await runs.claim(
            child_scope,
            worker_id=f"contract-child-worker-{index}",
            run_id=child.run.id,
        )
        _require(claim is not None, "async child run could not be claimed")
        await runs.start_step(child_scope, claim, "contract")
        if index == 1:
            paused = await runs.pause_for_approval(
                child_scope,
                claim,
                "contract",
                approval_id="contract-async-child-approval",
                payload={"child": index},
            )
            _require(
                paused.status is RunStatus.WAITING_FOR_APPROVAL,
                "async child approval pause was not durable",
            )
            await runs.resume(
                child_scope,
                child.run.id,
                actor="contract-approver",
                reason="contract child approved",
            )
            claim = await runs.claim(
                child_scope,
                worker_id="contract-child-worker-1-resumed",
                run_id=child.run.id,
            )
            _require(claim is not None, "async approved child could not resume")
            await runs.start_step(child_scope, claim, "contract")
        else:
            retrying = await runs.fail_step(
                child_scope,
                claim,
                "contract",
                reason="contract retry",
                retryable=True,
            )
            _require(
                retrying.status is RunStatus.WAITING_FOR_RETRY,
                "async retryable child failure did not pause durably",
            )
            await runs.resume(
                child_scope,
                child.run.id,
                actor="contract-recovery",
                reason="contract retry is ready",
            )
            claim = await runs.claim(
                child_scope,
                worker_id="contract-child-worker-2-retried",
                run_id=child.run.id,
            )
            _require(claim is not None, "async retryable child could not resume")
            await runs.start_step(child_scope, claim, "contract")
        await runs.record_child_progress(
            child_scope,
            claim,
            sequence=1,
            payload={"completed": index, "total": 2},
            idempotency_key=f"progress-{index}",
        )
        await runs.complete_step(
            child_scope,
            claim,
            "contract",
            result={"child": index},
        )
        await runs.complete(
            child_scope,
            claim,
            result={"child": index, "status": "completed"},
        )
    aggregated = await runs.aggregate_children(
        scope,
        actor="contract",
        idempotency_key="contract-aggregate",
    )
    _require(
        aggregated.run.status is RunStatus.COMPLETED,
        "async parent aggregation did not complete",
    )
    claim = await runs.claim_parent_completion(
        scope,
        worker_id="contract-delivery",
        parent_run_id=scope.run_id,
    )
    _require(claim is not None, "async parent completion was not enqueued")
    delivered = await runs.complete_parent_completion(scope, claim)
    _require(
        delivered.status is ParentCompletionStatus.DELIVERED,
        "async parent completion was not delivered",
    )

    child_cancel_scope = replace(
        scope,
        run_id=f"{scope.run_id}-child-cancel-parent",
    )
    await runs.submit_parent(
        child_cancel_scope,
        _contract_parent_submission(child_cancel_scope.run_id),
        policy=_contract_parent_policy(required_children=1, max_children=1),
        actor="contract",
    )
    cancelled_child_scope = child_cancel_scope.child(
        run_id=f"{child_cancel_scope.run_id}-child",
        agent_id="contract-child",
        agent_version="published-cancelled",
        grants=frozenset(),
    )
    cancelled_child = await runs.submit_child(
        child_cancel_scope,
        cancelled_child_scope,
        _contract_child_submission(
            f"{child_cancel_scope.run_id}-child-key",
            model_calls=1,
            tokens=100,
        ),
        definition_revision=cancelled_child_scope.agent_version,
        actor="contract",
    )
    cancelled_child = await runs.request_child_cancellation(
        child_cancel_scope,
        cancelled_child.run.id,
        actor="contract",
        reason="contract child cancelled",
    )
    _require(
        cancelled_child.run.status is RunStatus.CANCELLED,
        "async queued child cancellation was not terminal",
    )
    failed_parent = await runs.aggregate_children(
        child_cancel_scope,
        actor="contract",
        idempotency_key="contract-child-cancel-aggregate",
    )
    _require(
        failed_parent.run.status is RunStatus.FAILED,
        "async child cancellation did not propagate to the parent aggregate",
    )

    parent_cancel_scope = replace(
        scope,
        run_id=f"{scope.run_id}-parent-cancel",
    )
    await runs.submit_parent(
        parent_cancel_scope,
        _contract_parent_submission(parent_cancel_scope.run_id),
        policy=_contract_parent_policy(required_children=1, max_children=1),
        actor="contract",
    )
    active_child_scope = parent_cancel_scope.child(
        run_id=f"{parent_cancel_scope.run_id}-child",
        agent_id="contract-child",
        agent_version="published-active",
        grants=frozenset(),
    )
    active_child = await runs.submit_child(
        parent_cancel_scope,
        active_child_scope,
        _contract_child_submission(
            f"{parent_cancel_scope.run_id}-child-key",
            model_calls=1,
            tokens=100,
        ),
        definition_revision=active_child_scope.agent_version,
        actor="contract",
    )
    active_claim = await runs.claim(
        active_child_scope,
        worker_id="contract-active-child",
        run_id=active_child.run.id,
    )
    _require(
        active_claim is not None,
        "async active cancellation child was not claimed",
    )
    await runs.start_step(active_child_scope, active_claim, "contract")
    cancelling_parent = await runs.request_parent_cancellation(
        parent_cancel_scope,
        actor="contract",
        reason="contract parent cancelled",
    )
    _require(
        cancelling_parent.run.cancellation_requested
        and cancelling_parent.children[0].run.cancellation_requested,
        "async parent cancellation did not propagate to its active child",
    )
    await runs.cancel(
        active_child_scope,
        active_child.run.id,
        actor="contract-active-child",
        reason="contract parent cancellation observed",
        claim=active_claim,
    )
    cancelled_parent = await runs.aggregate_children(
        parent_cancel_scope,
        actor="contract",
        idempotency_key="contract-parent-cancel-aggregate",
    )
    _require(
        cancelled_parent.run.status is RunStatus.CANCELLED,
        "async settled parent cancellation did not terminalize the parent",
    )
    return HostedContractReport(
        (
            "async_bounded_fanout",
            "async_child_scope_isolation",
            "async_parentless_child_scope_isolation",
            "async_parentless_queue_child_isolation",
            "async_child_approval_resume",
            "async_child_retry_resume",
            "async_ordered_progress",
            "async_terminal_evidence",
            "async_idempotent_parent_aggregation",
            "async_exactly_once_parent_delivery",
            "async_child_cancellation_propagation",
            "async_parent_cancellation_propagation",
        )
    )


def assert_gateway_store_contract(
    store: Any,
    *,
    target: GatewayRunTarget,
) -> HostedContractReport:
    """Exercise portable inbox, outbox, stale-claim, and reconciliation rules."""
    envelope = _contract_envelope()
    first = store.ingest(
        envelope,
        profile_id="contract",
        run_target=target,
    )
    duplicate = store.ingest(
        envelope,
        profile_id="contract",
        run_target=target,
    )
    _require(first.created, "gateway did not create the first inbox record")
    _require(not duplicate.created, "gateway did not deduplicate the inbox record")
    _require(
        duplicate.record.id == first.record.id,
        "duplicate gateway input changed ownership",
    )
    claim = store.claim_execution(global_limit=1, profile_limit=1)
    _require(claim is not None, "gateway did not claim queued input")
    outbound = _contract_outbound(first.record.id)
    _require(
        store.complete_execution(
            first.record.id,
            claim.execution_token,
            (outbound,),
        ),
        "gateway rejected its active execution claim",
    )
    _require(
        not store.complete_execution(
            first.record.id,
            "stale-worker",
            (outbound,),
        ),
        "gateway accepted a stale execution claim",
    )
    delivery = store.claim_delivery()
    _require(delivery is not None, "gateway did not claim durable delivery")
    _require(
        store.record_delivery(
            delivery.id,
            delivery.delivery_token,
            DeliveryReceipt(
                envelope_id=delivery.id,
                state=DeliveryState.UNKNOWN,
                attempt=delivery.attempt_count,
            ),
        ),
        "gateway rejected its active delivery claim",
    )
    _require(
        store.claim_delivery() is None,
        "gateway blindly resent an ambiguous provider outcome",
    )
    _require(
        store.reconcile_delivery(
            delivery.id,
            DeliveryReceipt(
                envelope_id=delivery.id,
                state=DeliveryState.DELIVERED,
                attempt=delivery.attempt_count,
            ),
        ),
        "gateway could not reconcile an ambiguous provider outcome",
    )
    return HostedContractReport(
        (
            "inbox_deduplication",
            "stale_claim_rejection",
            "durable_outbox",
            "delivery_reconciliation",
        )
    )


async def assert_async_gateway_store_contract(
    store: Any,
    *,
    target: GatewayRunTarget,
) -> HostedContractReport:
    """Native async equivalent of :func:`assert_gateway_store_contract`."""
    envelope = _contract_envelope()
    first = await store.ingest(
        envelope,
        profile_id="contract",
        run_target=target,
    )
    duplicate = await store.ingest(
        envelope,
        profile_id="contract",
        run_target=target,
    )
    _require(first.created and not duplicate.created, "async inbox deduplication failed")
    claim = await store.claim_execution(global_limit=1, profile_limit=1)
    _require(claim is not None, "async gateway did not claim queued input")
    outbound = _contract_outbound(first.record.id)
    _require(
        await store.complete_execution(
            first.record.id,
            claim.execution_token,
            (outbound,),
        ),
        "async gateway rejected its active execution claim",
    )
    delivery = await store.claim_delivery()
    _require(delivery is not None, "async gateway did not claim durable delivery")
    _require(
        await store.record_delivery(
            delivery.id,
            delivery.delivery_token,
            DeliveryReceipt(
                envelope_id=delivery.id,
                state=DeliveryState.UNKNOWN,
                attempt=delivery.attempt_count,
            ),
        ),
        "async gateway rejected its active delivery claim",
    )
    _require(
        await store.claim_delivery() is None,
        "async gateway blindly resent an ambiguous provider outcome",
    )
    _require(
        await store.reconcile_delivery(
            delivery.id,
            DeliveryReceipt(
                envelope_id=delivery.id,
                state=DeliveryState.DELIVERED,
                attempt=delivery.attempt_count,
            ),
        ),
        "async gateway reconciliation failed",
    )
    return HostedContractReport(
        (
            "async_inbox_deduplication",
            "async_durable_outbox",
            "async_delivery_reconciliation",
        )
    )


def _serialize_response(response: ScriptedResponse) -> str:
    if isinstance(response, str):
        return response
    if is_dataclass(response):
        response = asdict(response)
    if isinstance(response, Mapping):
        return json.dumps(dict(response), sort_keys=True)
    raise TypeError("Scripted responses must be strings, mappings, or AgentAction dataclasses")


def _contract_scopes(
    first_scope: ExecutionScope,
    second_scope: ExecutionScope,
) -> None:
    _require(
        first_scope.tenant_id != second_scope.tenant_id
        or first_scope.workspace_id != second_scope.workspace_id,
        "contract scopes must cross a tenant or workspace boundary",
    )


def _contract_submission(
    *,
    idempotency_key: str = "hosted-contract-run",
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=idempotency_key,
        input_digest="sha256:hosted-contract-input",
        definition_digest="sha256:hosted-contract-definition",
        steps=(StepDefinition(id="contract", name="Contract step"),),
    )


def _contract_parent_submission(run_id: str) -> RunSubmission:
    return RunSubmission(
        idempotency_key=f"{run_id}-parent-key",
        input_digest=f"sha256:{run_id}-parent-input",
        definition_digest=f"sha256:{run_id}-parent-definition",
        steps=(StepDefinition(id="orchestrate", name="Aggregate children"),),
    )


def _contract_child_submission(
    idempotency_key: str,
    *,
    model_calls: int,
    tokens: int,
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=idempotency_key,
        input_digest=f"sha256:{idempotency_key}-input",
        definition_digest=f"sha256:{idempotency_key}-definition",
        steps=(
            StepDefinition(
                id="contract",
                name="Contract child step",
                retry_policy=RetryPolicy(
                    initial_delay_seconds=0,
                    max_delay_seconds=0,
                ),
            ),
        ),
        budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=model_calls,
            max_tool_calls=model_calls,
            max_tokens=tokens,
        ).to_dict(),
    )


def _contract_parent_policy(
    *,
    required_children: int,
    max_children: int,
) -> ParentRunPolicy:
    return ParentRunPolicy(
        required_children=required_children,
        max_children=max_children,
        budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=4,
            max_tool_calls=4,
            max_tokens=400,
        ),
    )


def _contract_approval() -> ApprovalSubmission:
    return ApprovalSubmission(
        step_id="contract",
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
        policy_version="contract-policy-1",
        preview={"record_id": "contract-record"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _contract_approval_validation(
    scope: ExecutionScope,
) -> ApprovalValidation:
    return ApprovalValidation(
        scope=scope,
        tool_name="contract_write",
        tool_version="1.0.0",
        schema_version="1",
        arguments_digest="sha256:contract-arguments",
        policy_version="contract-policy-1",
    )


def _contract_event(scope: ExecutionScope) -> AgentEvent:
    return AgentEvent(
        name="run.queued",
        conversation_id=scope.conversation_id or scope.run_id,
        execution_scope=scope,
        payload=RunLifecyclePayload(
            status="queued",
            action="contract",
        ),
    )


def _contract_envelope() -> InboundEnvelope:
    return InboundEnvelope(
        event_id="hosted-contract-event",
        idempotency_key="hosted-contract-event",
        identity=ChannelIdentity("contract", "primary", "actor"),
        destination_id="destination",
        parts=(TextPart("contract input"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
    )


def _contract_outbound(inbox_id: str) -> OutboundEnvelope:
    return OutboundEnvelope(
        envelope_id=f"contract-reply-{inbox_id}",
        profile_id="contract",
        conversation_id="contract-conversation",
        target=DeliveryTarget("contract", "primary", "destination"),
        text="contract output",
    )


def _must_reject(call: Any, message: str) -> None:
    try:
        call()
    except (KeyError, LookupError, PermissionError, ValueError):
        return
    raise HostedContractError(message)


async def _must_reject_async(call: Any, message: str) -> None:
    try:
        result = call()
        if inspect.isawaitable(result):
            await result
    except (KeyError, LookupError, PermissionError, ValueError):
        return
    raise HostedContractError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise HostedContractError(message)


__all__ = [
    "HostedContractError",
    "HostedContractReport",
    "ScriptedLLMClient",
    "assert_async_durable_execution_contract",
    "assert_async_gateway_store_contract",
    "assert_async_hosted_services_contract",
    "assert_async_parent_child_run_contract",
    "assert_durable_execution_contract",
    "assert_gateway_store_contract",
    "assert_hosted_services_contract",
    "assert_parent_child_run_contract",
]
