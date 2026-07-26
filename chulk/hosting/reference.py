"""Credential-free in-memory hosted services used by examples and contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from typing import Any
from uuid import uuid4

from chulk.approvals import (
    AsyncInMemoryApprovalStore,
    InMemoryApprovalStore,
)
from chulk.core.state import TurnState
from chulk.execution import ExecutionSessionRequest
from chulk.hosting.scope import ExecutionScope
from chulk.hosting.services import (
    AsyncRuntimeServices,
    ResourceOwnership,
    RuntimeServices,
    ServiceBinding,
    SessionRuntimeServices,
    SkillRuntimeServices,
)
from chulk.hosting.sinks import InMemoryEventSink, safe_audit_payload
from chulk.media import MediaProcessorRegistry
from chulk.plugins import PluginAuditReport
from chulk.runs import AsyncInMemoryRunStore, InMemoryRunStore
from chulk.sessions import (
    ConversationRecord,
    ConversationSummaryRecord,
    MessageRecord,
    SessionSearchPage,
)
from chulk.sessions.sqlite_store import _turn_from_dict
from chulk.skills.registry import (
    Skill,
    SkillRouteDecision,
    SkillRoutingResult,
    SkillSelection,
)
from chulk.tools.policy import ToolPolicyHooks
from chulk.tracing.artifacts import ArtifactRead, ArtifactRecord
from chulk.usage import (
    BudgetReservation,
    ExactCost,
    ReservationState,
    ResourceKind,
    RunBudget,
    UsageDimensions,
    UsageEntry,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_text() -> str:
    return _now().isoformat()


class InMemoryServiceHub:
    """Share isolated in-memory service state across hosted scopes."""

    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStore] = {}
        self._memory: dict[str, InMemoryMemoryService] = {}
        self._artifacts: dict[str, InMemoryArtifactService] = {}
        self._traces: dict[str, InMemoryTraceService] = {}
        self._audit: dict[str, InMemoryAuditService] = {}
        self._runs: dict[str, InMemoryRunStore] = {}
        self._approvals: dict[str, InMemoryApprovalStore] = {}
        self._events: dict[str, InMemoryEventSink] = {}
        self._async_runs: dict[str, AsyncInMemoryRunStore] = {}
        self._async_approvals: dict[str, AsyncInMemoryApprovalStore] = {}

    def services(
        self,
        *,
        policy_hooks: ToolPolicyHooks | None = None,
    ) -> RuntimeServices:
        """Return a complete sync bundle whose factories bind to scope."""

        def sessions(scope: ExecutionScope) -> SessionRuntimeServices:
            store = self._sessions.setdefault(
                scope.key,
                InMemorySessionStore(scope),
            )
            return SessionRuntimeServices(
                store=store,
                search=InMemorySessionSearch(store),
            )

        def artifacts(scope: ExecutionScope) -> InMemoryArtifactService:
            return self._artifacts.setdefault(
                scope.key,
                InMemoryArtifactService(scope),
            )

        def traces(scope: ExecutionScope) -> InMemoryTraceService:
            return self._traces.setdefault(
                scope.key,
                InMemoryTraceService(scope, artifacts(scope)),
            )

        def runs(scope: ExecutionScope) -> InMemoryRunStore:
            return self._runs.setdefault(scope.key, InMemoryRunStore())

        def approvals(scope: ExecutionScope) -> InMemoryApprovalStore:
            return self._approvals.setdefault(
                scope.key,
                InMemoryApprovalStore(runs(scope)),
            )

        def host_factory(factory: Any) -> ServiceBinding[Any]:
            return ServiceBinding.scoped(
                factory,
                ownership=ResourceOwnership.HOST,
            )

        return RuntimeServices(
            memory=host_factory(
                lambda scope: self._memory.setdefault(
                    scope.key,
                    InMemoryMemoryService(scope),
                )
            ),
            sessions=host_factory(sessions),
            skills=host_factory(
                lambda scope: SkillRuntimeServices(
                    registry=InMemorySkillService(scope)
                )
            ),
            traces=host_factory(traces),
            artifacts=host_factory(artifacts),
            usage=host_factory(InMemoryUsageService),
            audit=host_factory(
                lambda scope: self._audit.setdefault(
                    scope.key,
                    InMemoryAuditService(scope),
                )
            ),
            execution=host_factory(InMemoryExecutionBackend),
            plugins=host_factory(InMemoryPluginService),
            content=host_factory(InMemoryContentService),
            media=host_factory(
                lambda _scope: MediaProcessorRegistry()
            ),
            tool_policy=ServiceBinding.host(
                policy_hooks or ToolPolicyHooks()
            ),
            runs=host_factory(runs),
            approvals=host_factory(approvals),
            events=host_factory(
                lambda scope: self._events.setdefault(
                    scope.key,
                    InMemoryEventSink(scope),
                )
            ),
        )

    def async_services(
        self,
        *,
        policy_hooks: ToolPolicyHooks | None = None,
    ) -> AsyncRuntimeServices:
        """Return the corresponding bundle for ``AsyncHostedRuntime``."""
        services = self.services(policy_hooks=policy_hooks)
        def async_runs(scope: ExecutionScope) -> AsyncInMemoryRunStore:
            sync = self._runs.setdefault(scope.key, InMemoryRunStore())
            return self._async_runs.setdefault(
                scope.key,
                AsyncInMemoryRunStore(sync),
            )

        def async_approvals(
            scope: ExecutionScope,
        ) -> AsyncInMemoryApprovalStore:
            sync_runs = self._runs.setdefault(scope.key, InMemoryRunStore())
            sync = self._approvals.setdefault(
                scope.key,
                InMemoryApprovalStore(sync_runs),
            )
            return self._async_approvals.setdefault(
                scope.key,
                AsyncInMemoryApprovalStore(sync),
            )

        return AsyncRuntimeServices(
            memory=services.memory,
            sessions=services.sessions,
            skills=services.skills,
            traces=services.traces,
            artifacts=services.artifacts,
            usage=services.usage,
            audit=services.audit,
            execution=services.execution,
            plugins=services.plugins,
            content=services.content,
            media=services.media,
            tool_policy=services.tool_policy,
            runs=ServiceBinding.scoped(
                async_runs,
                ownership=ResourceOwnership.HOST,
            ),
            approvals=ServiceBinding.scoped(
                async_approvals,
                ownership=ResourceOwnership.HOST,
            ),
            events=services.events,
        )

    def trace_events(self, scope: ExecutionScope) -> tuple[dict[str, Any], ...]:
        trace = self._traces.get(scope.key)
        return tuple(trace.events) if trace is not None else ()

    def audit_events(self, scope: ExecutionScope) -> tuple[dict[str, Any], ...]:
        audit = self._audit.get(scope.key)
        return tuple(audit.events) if audit is not None else ()

    def public_events(self, scope: ExecutionScope) -> tuple[Any, ...]:
        sink = self._events.get(scope.key)
        return tuple(sink.events) if sink is not None else ()


class InMemoryMemoryService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.namespace = f"scope:{scope.key}"
        self._records: dict[str, Any] = {}

    def profile_memories(self, limit: int = 50) -> list[Any]:
        return list(self._records.values())[:limit]

    def search_memory(self, query: str, limit: int = 5) -> list[Any]:
        terms = query.casefold().split()
        return [
            record
            for record in self._records.values()
            if any(term in record.content.casefold() for term in terms)
        ][:limit]

    def get_memory(
        self,
        memory_id: str,
        *,
        include_archived: bool = False,
    ) -> Any:
        return self._records.get(memory_id)


class InMemorySessionStore:
    """Minimal durable-session contract backed by scope-owned dictionaries."""

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self._conversations: dict[str, ConversationRecord] = {}
        self._messages: dict[str, list[MessageRecord]] = {}
        self._turns: dict[str, dict[str, dict[str, Any]]] = {}
        self._summaries: dict[str, ConversationSummaryRecord] = {}

    def create_conversation(
        self,
        conversation_id: str,
        *,
        provider: str,
        model: str,
        trace_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationRecord:
        self._assert_conversation(conversation_id)
        now = _now_text()
        existing = self._conversations.get(conversation_id)
        record = ConversationRecord(
            id=conversation_id,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            provider=provider,
            model=model,
            trace_path=trace_path,
            status=existing.status if existing is not None else "active",
            metadata=dict(metadata or {}),
            turn_count=len(self._turns.get(conversation_id, {})),
        )
        self._conversations[conversation_id] = record
        return record

    def get_conversation(self, conversation_id_or_prefix: str) -> ConversationRecord:
        self._assert_conversation(conversation_id_or_prefix)
        try:
            return self._conversations[conversation_id_or_prefix]
        except KeyError as exc:
            raise ValueError(
                f"No hosted session found for id: {conversation_id_or_prefix}"
            ) from exc

    def load_turns(self, conversation_id: str) -> list[TurnState]:
        self._assert_conversation(conversation_id)
        return [
            _turn_from_dict(payload)
            for payload in self._turns.get(conversation_id, {}).values()
        ]

    def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None:
        self._assert_conversation(conversation_id)
        turn_id = str(turn.get("turn_id") or "").strip()
        if turn_id:
            self._turns.setdefault(conversation_id, {})[turn_id] = dict(turn)
            self._touch(conversation_id)

    def load_latest_summary(
        self,
        conversation_id: str,
    ) -> ConversationSummaryRecord | None:
        self._assert_conversation(conversation_id)
        return self._summaries.get(conversation_id)

    def save_conversation_summary(
        self,
        conversation_id: str,
        *,
        content: str,
        source_message_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSummaryRecord:
        self._assert_conversation(conversation_id)
        now = _now_text()
        existing = self._summaries.get(conversation_id)
        record = ConversationSummaryRecord(
            id=existing.id if existing is not None else f"summary_{uuid4().hex}",
            conversation_id=conversation_id,
            content=content,
            source_message_count=source_message_count,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._summaries[conversation_id] = record
        return record

    def save_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        turn_id: str | None = None,
        message_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        self._assert_conversation(conversation_id)
        if not content.strip():
            return
        messages = self._messages.setdefault(conversation_id, [])
        key = message_key or f"message_{uuid4().hex}"
        if any(record.id == key for record in messages):
            return
        messages.append(
            MessageRecord(
                id=key,
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=role,
                content=content.strip(),
                ordinal=len(messages) + 1,
                created_at=created_at or _now_text(),
                metadata=dict(metadata or {}),
            )
        )
        self._touch(conversation_id)

    def load_recent_messages(
        self,
        conversation_id: str,
        limit: int,
        *,
        after_ordinal: int = 0,
    ) -> list[dict[str, str]]:
        records = self.list_messages(
            conversation_id,
            limit=limit,
            after_ordinal=after_ordinal,
        )
        return [{"role": item.role, "content": item.content} for item in records]

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        after_ordinal: int = 0,
    ) -> list[MessageRecord]:
        self._assert_conversation(conversation_id)
        return [
            item
            for item in self._messages.get(conversation_id, [])
            if item.ordinal > after_ordinal
        ][-limit:]

    def save_model_request(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None:
        self._assert_conversation(conversation_id)

    def save_model_response(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None:
        self._assert_conversation(conversation_id)

    def save_tool_call(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None:
        self._assert_conversation(conversation_id)

    def save_tool_observation_bundle(
        self,
        conversation_id: str,
        *,
        turn_id: str,
        observation_index: int,
        tool_name: str,
        content: str,
        output_metadata: dict[str, Any],
        action_context: str | None,
        turn: dict[str, Any] | None,
    ) -> None:
        self._assert_conversation(conversation_id)
        if turn is not None:
            self.save_turn_snapshot(conversation_id, turn)

    def max_observation_index(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> int:
        self._assert_conversation(conversation_id)
        return 0

    def save_terminal_turn_bundle(
        self,
        conversation_id: str,
        *,
        turn_id: str,
        content: str,
        message_key: str,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        self.save_message(
            conversation_id,
            turn_id=turn_id,
            role="assistant",
            content=content,
            message_key=message_key,
            metadata=metadata,
        )
        self.save_turn_snapshot(conversation_id, turn)
        status = str(turn.get("status") or "completed")
        self.update_conversation_status(conversation_id, status)
        return True

    def update_conversation_status(
        self,
        conversation_id: str,
        status: str,
    ) -> None:
        record = self.get_conversation(conversation_id)
        self._conversations[conversation_id] = replace(
            record,
            status=status,
            updated_at=_now_text(),
            turn_count=len(self._turns.get(conversation_id, {})),
        )

    def load_terminal_turn_message(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> dict[str, str] | None:
        for item in reversed(self._messages.get(conversation_id, [])):
            if item.turn_id == turn_id and item.role == "assistant":
                return {"content": item.content, "kind": "final"}
        return None

    def load_uncheckpointed_hosted_mcp_requests(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        checkpointed_request_count: int,
    ) -> list[dict[str, Any]]:
        return []

    def load_tool_calls_without_observations(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]:
        return []

    def _assert_conversation(self, conversation_id: str) -> None:
        expected = self.scope.conversation_id
        if expected is not None and conversation_id != expected:
            raise PermissionError("cross-scope conversation access denied")

    def _touch(self, conversation_id: str) -> None:
        record = self._conversations.get(conversation_id)
        if record is not None:
            self._conversations[conversation_id] = replace(
                record,
                updated_at=_now_text(),
                turn_count=len(self._turns.get(conversation_id, {})),
            )


class InMemorySessionSearch:
    def __init__(self, store: InMemorySessionStore) -> None:
        self.store = store
        self.profile_id = store.scope.actor_id or "hosted"

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        return SessionSearchPage(query=query, hits=())


class InMemorySkillService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self._skills: dict[str, Skill] = {}
        self.last_routing_result = SkillRoutingResult((), ())

    def load_metadata(self) -> None:
        return None

    def register(self, skill: Skill, *, replace: bool = False) -> None:
        if skill.name in self._skills and not replace:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def clear(self) -> None:
        self._skills = {}
        self.last_routing_result = SkillRoutingResult((), ())

    def configure_environment(self, **kwargs: Any) -> None:
        return None

    def restrict_to(self, names: list[str]) -> None:
        allowed = set(names)
        self._skills = {
            name: skill
            for name, skill in self._skills.items()
            if name in allowed
        }

    def list_visible_skills(self) -> list[Any]:
        return list(self._skills.values())

    def load_selected_skills(
        self,
        _user_request: str,
        *,
        pinned_names: tuple[str, ...] | list[str] = (),
        limit: int | None = None,
    ) -> list[SkillSelection]:
        selected: list[SkillSelection] = []
        decisions: list[SkillRouteDecision] = []
        for name in pinned_names:
            skill = self._skills.get(name)
            if skill is None:
                decisions.append(
                    SkillRouteDecision(
                        skill_name=name,
                        status="rejected",
                        stage="explicit",
                        reason="not_found",
                    )
                )
                continue
            if limit is not None and len(selected) >= limit:
                decisions.append(
                    SkillRouteDecision(
                        skill_name=name,
                        status="omitted",
                        stage="budget",
                        reason="skill_count_limit",
                    )
                )
                continue
            selected.append(
                SkillSelection(
                    skill=skill,
                    score=100,
                    matched_keywords=[name],
                    reason="pinned",
                    stage="explicit",
                )
            )
            decisions.append(
                SkillRouteDecision(
                    skill_name=name,
                    status="selected",
                    stage="explicit",
                    reason="pinned",
                    score=100,
                    matched_keywords=(name,),
                    digest=skill.digest,
                )
            )
        self.last_routing_result = SkillRoutingResult(
            tuple(selected),
            tuple(decisions),
            tuple(pinned_names),
        )
        return selected

    def get_skill(
        self,
        name: str,
        *,
        visible_only: bool = False,
    ) -> Skill | None:
        del visible_only
        return self._skills.get(name)

    def load_content(self, name: str) -> str:
        skill = self._skills[name]
        return skill.loaded_content or ""


class InMemoryArtifactService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self._records: dict[str, tuple[ArtifactRecord, str]] = {}

    def write(self, label: str, content: str) -> ArtifactRecord:
        artifact_id = f"art_{uuid4().hex}"
        encoded = content.encode("utf-8")
        record = ArtifactRecord(
            artifact_id=artifact_id,
            conversation_id=self.scope.conversation_id or self.scope.run_id,
            filename=artifact_id,
            label=label,
            char_count=len(content),
            byte_count=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            created_at=_now_text(),
        )
        self._records[artifact_id] = (record, content)
        return record

    def read(self, artifact_id: str, **kwargs: Any) -> ArtifactRead:
        record, content = self._records[artifact_id]
        encoded = content.encode("utf-8")
        return ArtifactRead(
            artifact_id=artifact_id,
            mode="head_tail",
            content=content,
            byte_count=len(encoded),
            total_byte_count=len(encoded),
            sha256=record.sha256,
            ranges=((0, len(encoded)),),
            truncated=False,
        )


class InMemoryTraceService:
    path = None

    def __init__(
        self,
        scope: ExecutionScope,
        artifacts: InMemoryArtifactService,
    ) -> None:
        self.scope = scope
        self.artifact_store = artifacts
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("trace service is closed")
        self.events.append(
            {
                "type": event_type,
                "turn_id": turn_id,
                "payload": dict(payload or {}),
                "scope": self.scope.to_dict(),
            }
        )

    def activate(self) -> None:
        return None

    def write_artifact(self, name: str, content: str) -> dict[str, Any]:
        return self.artifact_store.write(name, content).reference()

    def close(self) -> None:
        self.closed = True


class InMemoryAuditService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        scope: ExecutionScope,
    ) -> None:
        self.scope.assert_same_authority(scope)
        self.events.append(
            {
                "type": event_type,
                "payload": safe_audit_payload(payload),
                "scope": scope.to_dict(),
            }
        )


class InMemoryUsageService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self._reservations: dict[tuple[str, int, str], BudgetReservation] = {}

    def reserve_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
        **kwargs: Any,
    ) -> BudgetReservation:
        return self._reserve(turn_id, request_index, ResourceKind.MODEL)

    def commit_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
        purpose: str,
        **kwargs: Any,
    ) -> tuple[UsageEntry, ...]:
        reservation = self._reservations.pop(
            (turn_id, request_index, ResourceKind.MODEL.value)
        )
        return (self._entry(reservation, purpose=purpose),)

    def release_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
    ) -> BudgetReservation | None:
        return self._reservations.pop(
            (turn_id, request_index, ResourceKind.MODEL.value),
            None,
        )

    def reserve_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        **kwargs: Any,
    ) -> BudgetReservation:
        return self._reserve(turn_id, tool_call_index, ResourceKind.TOOL)

    def commit_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        tool_name: str,
        **kwargs: Any,
    ) -> tuple[UsageEntry, ...]:
        reservation = self._reservations.pop(
            (turn_id, tool_call_index, ResourceKind.TOOL.value)
        )
        return (self._entry(reservation, purpose=tool_name),)

    def release_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        **kwargs: Any,
    ) -> BudgetReservation | None:
        return self._reservations.pop(
            (turn_id, tool_call_index, ResourceKind.TOOL.value),
            None,
        )

    def _reserve(
        self,
        turn_id: str,
        index: int,
        kind: ResourceKind,
    ) -> BudgetReservation:
        reservation_id = f"reservation_{uuid4().hex}"
        reservation = BudgetReservation(
            id=reservation_id,
            idempotency_key=f"{turn_id}:{index}:{kind.value}",
            source_event_id=f"{turn_id}:{index}:{kind.value}",
            resource_kind=kind,
            dimensions=UsageDimensions(
                profile_id=self.scope.actor_id or "hosted",
                channel=self.scope.channel_id,
                conversation_id=self.scope.conversation_id,
                turn_id=turn_id,
            ),
            budget=RunBudget(),
            state=ReservationState.ACTIVE,
            reserved_model_calls=1 if kind is ResourceKind.MODEL else 0,
            reserved_tool_calls=1 if kind is ResourceKind.TOOL else 0,
            reserved_tokens=0,
            reserved_cost=ExactCost(Decimal("0"), pricing_known=True),
            created_at=_now(),
        )
        self._reservations[(turn_id, index, kind.value)] = reservation
        return reservation

    def _entry(
        self,
        reservation: BudgetReservation,
        *,
        purpose: str,
    ) -> UsageEntry:
        occurred_at = _now()
        return UsageEntry(
            id=f"usage_{uuid4().hex}",
            resource_kind=reservation.resource_kind,
            source_event_id=reservation.source_event_id,
            dimensions=reservation.dimensions,
            occurred_at=occurred_at,
            billing_period=occurred_at.strftime("%Y-%m"),
            purpose=purpose,
        )


class InMemoryExecutionBackend:
    name = "hosted-in-memory"

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.closed = False

    def open_session(self, request: ExecutionSessionRequest) -> Any:
        return _InMemoryExecutionSession()

    async def open_session_async(self, request: ExecutionSessionRequest) -> Any:
        return self.open_session(request)

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


class _InMemoryExecutionSession:
    def __init__(self) -> None:
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        if name.endswith("_async"):
            async def blocked_async(*args: Any, **kwargs: Any) -> Any:
                raise PermissionError(
                    "the in-memory reference backend does not expose execution tools"
                )

            return blocked_async
        raise PermissionError(
            "the in-memory reference backend does not expose execution tools"
        )

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


class InMemoryPluginService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.profile_id = scope.actor_id or "default"

    def verify_startup(self) -> PluginAuditReport:
        return PluginAuditReport(profile_id=self.profile_id)


class InMemoryContentService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.profile_id = scope.actor_id or "hosted"
        self._content: dict[str, bytes] = {}

    def sweep_expired(self) -> int:
        return 0
