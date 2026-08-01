"""Credential-free in-memory hosted services used by examples and contract tests."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
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
    AsyncServiceBinding,
    ResourceOwnership,
    RuntimeServices,
    ServiceBinding,
    SessionRuntimeServices,
    SkillRuntimeServices,
)
from chulk.hosting.sinks import (
    AsyncInMemoryEventSink,
    InMemoryEventSink,
    safe_audit_payload,
)
from chulk.llm import LLMCost, LLMUsage
from chulk.media import MediaProcessorRegistry
from chulk.memory.constants import PROFILE_MEMORY_TAGS
from chulk.memory.markdown import parse_markdown_memory_line
from chulk.memory.models import MemoryProposalRecord, MemoryRecord
from chulk.memory.retrieval import resolve_profile_conflicts
from chulk.memory.security import ensure_memory_payload_safe
from chulk.memory.store import select_recent_conversation_messages
from chulk.plugins import PluginAuditReport
from chulk.redaction import redact_text
from chulk.runs import AsyncInMemoryRunStore, InMemoryRunStore
from chulk.sessions import (
    ConversationRecord,
    ConversationSummaryRecord,
    MessageRecord,
    SessionHit,
    SessionMessage,
    SessionSearchPage,
    SessionWindow,
)
from chulk.sessions.search import (
    MAX_SESSION_SEARCH_LIMIT,
    MAX_SESSION_WINDOW_LIMIT,
    MAX_SESSION_WINDOW_RADIUS,
    _bounded_int,
    _decode_search_cursor,
    _decode_window_cursor,
    _encode_cursor,
    _message_is_sensitive,
    _parse_query,
    _shape_hash,
    _snippet,
    _window_message_visible,
)
from chulk.sessions.sqlite_store import (
    _message_is_search_eligible,
    _turn_from_dict,
)
from chulk.skills.registry import (
    Skill,
    SkillRouteDecision,
    SkillRoutingResult,
    SkillSelection,
)
from chulk.tools.policy import ToolPolicyHooks
from chulk.tracing.artifacts import ArtifactRead, ArtifactRecord
from chulk.usage import (
    MAX_EXPORT_ENTRIES,
    BudgetReservation,
    ExactCost,
    ReservationState,
    ResourceKind,
    RunBudget,
    UsageAggregate,
    UsageDimensions,
    UsageEntry,
    UsageGroupBy,
    UsagePage,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_text() -> str:
    return _now().isoformat()


def _usage_group_key(
    entry: UsageEntry,
    group_by: UsageGroupBy,
) -> str:
    if group_by is UsageGroupBy.RESOURCE_KIND:
        return entry.resource_kind.value
    if group_by is UsageGroupBy.MODEL:
        return entry.model or "unknown"
    if group_by is UsageGroupBy.TOOL_SERVICE:
        return entry.tool_or_service or entry.purpose
    field_name = {
        UsageGroupBy.PROFILE: "profile_id",
        UsageGroupBy.CHANNEL: "channel",
        UsageGroupBy.CONVERSATION: "conversation_id",
        UsageGroupBy.GOAL: "goal_id",
        UsageGroupBy.JOB: "job_id",
        UsageGroupBy.CHILD_TASK: "child_task_id",
    }[group_by]
    return getattr(entry.dimensions, field_name) or "unassigned"


def _encode_usage_cursor(occurred_at: datetime, entry_id: str) -> str:
    payload = json.dumps(
        {
            "occurred_at": occurred_at.isoformat(),
            "id": entry_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_usage_cursor(value: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode((value + padding).encode()).decode()
        )
        occurred_at = datetime.fromisoformat(str(payload["occurred_at"]))
        entry_id = str(payload["id"])
        if occurred_at.tzinfo is None:
            raise ValueError("usage cursor timestamp must be timezone-aware")
    except (
        binascii.Error,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("invalid usage cursor") from exc
    if not entry_id:
        raise ValueError("invalid usage cursor")
    return occurred_at.astimezone(timezone.utc), entry_id


class _AsyncServiceAdapter:
    """Expose one in-memory reference service through async-only methods."""

    def __init__(self, service: object) -> None:
        self._service = service

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._service, name)
        if not callable(value):
            return value

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            result = value(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return invoke


async def _async_value(value: Any) -> Any:
    return value


class InMemoryServiceHub:
    """Share isolated in-memory service state across hosted scopes."""

    def __init__(self) -> None:
        self._sessions: dict[str, InMemorySessionStore] = {}
        self._memory: dict[str, InMemoryMemoryService] = {}
        self._artifacts: dict[str, InMemoryArtifactService] = {}
        self._traces: dict[str, InMemoryTraceService] = {}
        self._audit: dict[str, InMemoryAuditService] = {}
        self._usage: dict[str, InMemoryUsageService] = {}
        self._runs: dict[str, InMemoryRunStore] = {}
        self._approvals: dict[str, InMemoryApprovalStore] = {}
        self._events: dict[str, InMemoryEventSink] = {}
        self._async_events: dict[str, AsyncInMemoryEventSink] = {}
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
            usage=host_factory(
                lambda scope: self._usage.setdefault(
                    scope.key,
                    InMemoryUsageService(scope),
                )
            ),
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

        async def memory(scope: ExecutionScope) -> object:
            return _AsyncServiceAdapter(services.memory.resolve(scope))

        async def sessions(
            scope: ExecutionScope,
        ) -> SessionRuntimeServices:
            resolved = services.sessions.resolve(scope)
            return SessionRuntimeServices(
                store=_AsyncServiceAdapter(resolved.store),
                search=_AsyncServiceAdapter(resolved.search),
            )

        async def skills(scope: ExecutionScope) -> SkillRuntimeServices:
            resolved = services.skills.resolve(scope)
            return SkillRuntimeServices(
                registry=_AsyncServiceAdapter(resolved.registry),
                lifecycle_store=(
                    _AsyncServiceAdapter(resolved.lifecycle_store)
                    if resolved.lifecycle_store is not None
                    else None
                ),
                lifecycle=resolved.lifecycle,
                learning_proposals=(
                    _AsyncServiceAdapter(resolved.learning_proposals)
                    if resolved.learning_proposals is not None
                    else None
                ),
                learning_reviewer=(
                    _AsyncServiceAdapter(resolved.learning_reviewer)
                    if resolved.learning_reviewer is not None
                    else None
                ),
            )

        async def adapt(
            binding: ServiceBinding[Any],
            scope: ExecutionScope,
        ) -> object:
            return _AsyncServiceAdapter(binding.resolve(scope))

        async def async_runs(
            scope: ExecutionScope,
        ) -> AsyncInMemoryRunStore:
            sync = self._runs.setdefault(scope.key, InMemoryRunStore())
            return self._async_runs.setdefault(
                scope.key,
                AsyncInMemoryRunStore(sync),
            )

        async def async_approvals(
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
            memory=AsyncServiceBinding.scoped(
                memory,
                ownership=ResourceOwnership.HOST,
            ),
            sessions=AsyncServiceBinding.scoped(
                sessions,
                ownership=ResourceOwnership.HOST,
            ),
            skills=AsyncServiceBinding.scoped(
                skills,
                ownership=ResourceOwnership.HOST,
            ),
            traces=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.traces, scope),
                ownership=ResourceOwnership.HOST,
            ),
            artifacts=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.artifacts, scope),
                ownership=ResourceOwnership.HOST,
            ),
            usage=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.usage, scope),
                ownership=ResourceOwnership.HOST,
            ),
            audit=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.audit, scope),
                ownership=ResourceOwnership.HOST,
            ),
            execution=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.execution, scope),
                ownership=ResourceOwnership.HOST,
            ),
            plugins=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.plugins, scope),
                ownership=ResourceOwnership.HOST,
            ),
            content=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.content, scope),
                ownership=ResourceOwnership.HOST,
            ),
            media=AsyncServiceBinding.scoped(
                lambda scope: adapt(services.media, scope),
                ownership=ResourceOwnership.HOST,
            ),
            tool_policy=services.tool_policy,
            runs=AsyncServiceBinding.scoped(
                async_runs,
                ownership=ResourceOwnership.HOST,
            ),
            approvals=AsyncServiceBinding.scoped(
                async_approvals,
                ownership=ResourceOwnership.HOST,
            ),
            events=AsyncServiceBinding.scoped(
                lambda scope: _async_value(
                    self._async_events.setdefault(
                    scope.key,
                    AsyncInMemoryEventSink(scope),
                    )
                ),
                ownership=ResourceOwnership.HOST,
            ),
        )

    def trace_events(self, scope: ExecutionScope) -> tuple[dict[str, Any], ...]:
        trace = self._traces.get(scope.key)
        return tuple(trace.events) if trace is not None else ()

    def audit_events(self, scope: ExecutionScope) -> tuple[dict[str, Any], ...]:
        audit = self._audit.get(scope.key)
        return tuple(audit.events) if audit is not None else ()

    def public_events(self, scope: ExecutionScope) -> tuple[Any, ...]:
        sink = self._events.get(scope.key) or self._async_events.get(scope.key)
        return tuple(sink.events) if sink is not None else ()

    def active_usage_reservations(
        self,
        scope: ExecutionScope,
    ) -> tuple[BudgetReservation, ...]:
        service = self._usage.get(scope.key)
        if service is None:
            return ()
        return tuple(service._reservations.values())


class InMemoryMemoryService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.namespace = f"scope:{scope.key}"
        self._records: dict[str, MemoryRecord] = {}
        self._proposals: dict[str, MemoryProposalRecord] = {}

    def profile_memories(self, limit: int = 50) -> list[Any]:
        eligible = [
            record
            for record in self._records.values()
            if record.archived_at is None
            and set(record.tags) & PROFILE_MEMORY_TAGS
        ]
        return resolve_profile_conflicts(eligible)[:limit]

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        *,
        include_archived: bool = False,
    ) -> list[Any]:
        terms = query.casefold().split()
        return [
            record
            for record in self._records.values()
            if (include_archived or record.archived_at is None)
            and any(term in record.content.casefold() for term in terms)
        ][:limit]

    def list_memories(
        self,
        limit: int = 50,
        *,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        return [
            record
            for record in self._records.values()
            if include_archived or record.archived_at is None
        ][:limit]

    def delete_memory(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def update_memory(
        self,
        memory_id: str,
        **updates: Any,
    ) -> bool:
        record = self._records.get(memory_id)
        if record is None:
            return False
        values = {
            key: value
            for key, value in updates.items()
            if value is not None
        }
        values["updated_at"] = _now_text()
        updated = replace(record, **values)
        ensure_memory_payload_safe(
            content=updated.content,
            tags=updated.tags,
            metadata=updated.metadata,
            source=updated.source,
        )
        self._records[memory_id] = updated
        return True

    def summarize_memories(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> str:
        records = (
            self.search_memory(query, limit=limit)
            if query
            else self.list_memories(limit=limit)
        )
        if not records:
            return "No memories found."
        return "\n".join(f"- {record.content}" for record in records)

    def archive_memory(self, memory_id: str) -> bool:
        record = self._records.get(memory_id)
        if record is None or record.archived_at is not None:
            return False
        self._records[memory_id] = replace(
            record,
            archived_at=_now_text(),
            updated_at=_now_text(),
        )
        return True

    def restore_memory(self, memory_id: str) -> bool:
        record = self._records.get(memory_id)
        if record is None or record.archived_at is None:
            return False
        self._records[memory_id] = replace(
            record,
            archived_at=None,
            updated_at=_now_text(),
        )
        return True

    def compact_memories(self) -> int:
        return 0

    def import_markdown(self, path: Any) -> list[str]:
        parsed_memories: list[tuple[str, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_markdown_memory_line(line)
            if parsed is None:
                continue
            content, tags = parsed
            ensure_memory_payload_safe(
                content=content,
                tags=tags,
                metadata={"path": str(path)},
                source="memory_md",
            )
            parsed_memories.append((content, tags))
        memory_ids = []
        for content, tags in parsed_memories:
            memory_ids.append(
                self.save_memory(
                    content,
                    tags=tags,
                    metadata={"path": str(path)},
                    source="memory_md",
                    confidence=0.8,
                )
            )
        return memory_ids

    def export_markdown(
        self,
        path: Any,
        *,
        include_archived: bool = False,
    ) -> int:
        records = self.list_memories(
            limit=max(len(self._records), 1),
            include_archived=include_archived,
        )
        path.write_text(
            "".join(f"- {record.content}\n" for record in records),
            encoding="utf-8",
        )
        return len(records)

    def get_memory(
        self,
        memory_id: str,
        *,
        include_archived: bool = False,
    ) -> Any:
        record = self._records.get(memory_id)
        if (
            record is not None
            and record.archived_at is not None
            and not include_archived
        ):
            return None
        return record

    def save_memory(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
        source: str = "manual",
        confidence: float = 1.0,
        **_kwargs: Any,
    ) -> str:
        ensure_memory_payload_safe(
            content=content,
            tags=tags or (),
            metadata=metadata or {},
            source=source,
        )
        memory_id = f"memory_{uuid4().hex}"
        now = _now_text()
        self._records[memory_id] = MemoryRecord(
            id=memory_id,
            content=content,
            created_at=now,
            updated_at=now,
            tags=list(tags or ()),
            metadata=dict(metadata or {}),
            importance=importance,
            source=source,
            confidence=confidence,
            namespace=self.namespace,
        )
        return memory_id

    def create_memory_proposal(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
        source: str = "manual_review",
        confidence: float = 1.0,
        evidence: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        **_kwargs: Any,
    ) -> str:
        ensure_memory_payload_safe(
            content=content,
            tags=tags or (),
            metadata=metadata or {},
            source=source,
            evidence=evidence,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        proposal_id = f"memory_proposal_{uuid4().hex}"
        self._proposals[proposal_id] = MemoryProposalRecord(
            id=proposal_id,
            content=content,
            tags=list(tags or ()),
            metadata=dict(metadata or {}),
            importance=importance,
            source=source,
            confidence=confidence,
            evidence=evidence,
            conversation_id=conversation_id,
            turn_id=turn_id,
            status="pending",
            created_at=_now_text(),
            namespace=self.namespace,
        )
        return proposal_id

    def list_memory_proposals(
        self,
        *,
        status: str | None = "pending",
        **_kwargs: Any,
    ) -> list[MemoryProposalRecord]:
        return [
            proposal
            for proposal in self._proposals.values()
            if status is None or proposal.status == status
        ]

    def approve_memory_proposal(
        self,
        proposal_id: str,
    ) -> MemoryProposalRecord:
        proposal = self._proposals[proposal_id]
        if proposal.status != "pending":
            return proposal
        ensure_memory_payload_safe(
            content=proposal.content,
            tags=proposal.tags,
            metadata=proposal.metadata,
            source=proposal.source,
            evidence=proposal.evidence,
            conversation_id=proposal.conversation_id,
            turn_id=proposal.turn_id,
        )
        memory_id = self.save_memory(
            proposal.content,
            tags=proposal.tags,
            metadata=proposal.metadata,
            importance=proposal.importance,
            source=proposal.source,
            confidence=proposal.confidence,
        )
        reviewed = replace(
            proposal,
            status="approved",
            reviewed_at=_now_text(),
            accepted_memory_id=memory_id,
        )
        self._proposals[proposal_id] = reviewed
        return reviewed

    def reject_memory_proposal(
        self,
        proposal_id: str,
    ) -> MemoryProposalRecord:
        proposal = self._proposals[proposal_id]
        if proposal.status != "pending":
            return proposal
        reviewed = replace(
            proposal,
            status="rejected",
            reviewed_at=_now_text(),
        )
        self._proposals[proposal_id] = reviewed
        return reviewed


class InMemorySessionStore:
    """Minimal durable-session contract backed by scope-owned dictionaries."""

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self._conversations: dict[str, ConversationRecord] = {}
        self._messages: dict[str, list[MessageRecord]] = {}
        self._turns: dict[str, dict[str, dict[str, Any]]] = {}
        self._summaries: dict[str, ConversationSummaryRecord] = {}
        self._tool_calls: dict[
            str,
            dict[tuple[str, str, int], dict[str, Any]],
        ] = {}
        self._observations: dict[
            str,
            dict[tuple[str, int], dict[str, Any]],
        ] = {}

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
        self._assert_conversation(conversation_id)
        messages = [
            {"role": item.role, "content": item.content}
            for item in self._messages.get(conversation_id, [])
            if item.ordinal > after_ordinal
            and item.metadata.get("prompt_excluded") is not True
        ]
        return select_recent_conversation_messages(
            messages,
            max_messages=max(1, min(limit, 500)),
        )

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
        turn_id = str(payload.get("turn_id") or "").strip()
        raw_iteration = payload.get("iteration")
        tool_name = str(
            payload.get("tool_name")
            or payload.get("resolved_tool_name")
            or ""
        ).strip()
        if (
            not turn_id
            or not isinstance(raw_iteration, int)
            or isinstance(raw_iteration, bool)
            or raw_iteration < 1
            or not tool_name
        ):
            return
        phase = str(payload.get("phase") or "execution")
        self._tool_calls.setdefault(conversation_id, {})[
            (turn_id, phase, raw_iteration)
        ] = dict(payload)
        turn = payload.get("turn")
        if isinstance(turn, dict) and str(turn.get("turn_id") or "") == turn_id:
            self.save_turn_snapshot(conversation_id, turn)
        else:
            self._touch(conversation_id)

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
        if (
            not isinstance(observation_index, int)
            or isinstance(observation_index, bool)
            or observation_index < 1
        ):
            raise ValueError("observation_index must be a positive integer")
        clean_content = content.strip()
        if not clean_content:
            return
        observation_key = (turn_id, observation_index)
        observations = self._observations.setdefault(conversation_id, {})
        inserted = observation_key not in observations
        observations.setdefault(
            observation_key,
            {
                "tool_name": tool_name,
                "content": clean_content,
                "output_metadata": dict(output_metadata),
            },
        )
        if isinstance(action_context, str) and action_context.strip():
            self.save_message(
                conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=action_context,
                message_key=f"{turn_id}:tool_action:{observation_index}",
                metadata={
                    "tool_name": tool_name,
                    "internal": True,
                    "event": "tool_observation",
                    "observation_index": observation_index,
                },
            )
        self.save_message(
            conversation_id,
            turn_id=turn_id,
            role="observation",
            content=clean_content,
            message_key=f"{turn_id}:observation:{observation_index}",
            metadata={
                "tool_name": tool_name,
                "observation_index": observation_index,
            },
        )
        if inserted and turn is not None:
            self.save_turn_snapshot(conversation_id, turn)
        else:
            self._touch(conversation_id)

    def max_observation_index(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> int:
        self._assert_conversation(conversation_id)
        return max(
            (
                index
                for stored_turn_id, index in self._observations.get(
                    conversation_id,
                    {},
                )
                if stored_turn_id == turn_id
            ),
            default=0,
        )

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
        self._assert_conversation(conversation_id)
        calls: list[dict[str, Any]] = []
        for (stored_turn_id, phase, iteration), payload in sorted(
            self._tool_calls.get(conversation_id, {}).items(),
            key=lambda item: item[0][2],
        ):
            if stored_turn_id != turn_id:
                continue
            calls.append(
                {
                    "tool_name": str(
                        payload.get("tool_name")
                        or payload.get("resolved_tool_name")
                        or "tool"
                    ),
                    "arguments": dict(payload.get("arguments") or {}),
                    "iteration": iteration,
                    "phase": phase,
                    "started_at": str(
                        payload.get("started_at") or ""
                    ),
                    "ended_at": payload.get("ended_at"),
                    "success": payload.get("success"),
                }
            )

        observed_identities: set[tuple[str, int]] = set()
        legacy_observation_tools: list[str] = []
        for (stored_turn_id, _index), observation in self._observations.get(
            conversation_id,
            {},
        ).items():
            if stored_turn_id != turn_id:
                continue
            metadata = observation["output_metadata"]
            if metadata.get("synthetic") is True:
                continue
            identity = metadata.get("tool_call_identity")
            if isinstance(identity, dict):
                identity_phase = identity.get("phase")
                identity_iteration = identity.get("iteration")
                if (
                    isinstance(identity_phase, str)
                    and identity_phase
                    and isinstance(identity_iteration, int)
                    and not isinstance(identity_iteration, bool)
                    and identity_iteration > 0
                ):
                    observed_identities.add(
                        (identity_phase, identity_iteration)
                    )
                    continue
            legacy_observation_tools.append(str(observation["tool_name"]))

        unmatched = [
            call
            for call in calls
            if (str(call["phase"]), int(call["iteration"]))
            not in observed_identities
        ]
        for observed_tool_name in legacy_observation_tools:
            matching_index = next(
                (
                    index
                    for index, call in enumerate(unmatched)
                    if call["tool_name"] == observed_tool_name
                ),
                None,
            )
            if matching_index is not None:
                unmatched.pop(matching_index)
        return unmatched

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
        self.redactor = redact_text

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        clean_query, terms = _parse_query(query)
        clean_limit = _bounded_int(
            "session search limit",
            limit,
            maximum=MAX_SESSION_SEARCH_LIMIT,
        )
        shape = _shape_hash("search", self.profile_id, clean_query)
        offset = (
            _decode_search_cursor(cursor, shape=shape)
            if cursor
            else 0
        )
        candidates: list[tuple[str, MessageRecord]] = []
        for conversation_id, records in self.store._messages.items():
            for record in records:
                if not _message_is_search_eligible(
                    record.role,
                    record.metadata,
                ):
                    continue
                if not any(
                    term in record.content.casefold()
                    for term in terms
                ):
                    continue
                candidates.append((conversation_id, record))
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].ordinal,
                item[1].id,
            )
        )
        candidates.sort(
            key=lambda item: item[1].created_at,
            reverse=True,
        )
        candidates = candidates[:10_000]
        page_records = candidates[
            offset : offset + clean_limit + 1
        ]
        has_more = len(page_records) > clean_limit
        page_records = page_records[:clean_limit]
        hits = tuple(
            SessionHit(
                message_id=record.id,
                conversation_id=conversation_id,
                ordinal=record.ordinal,
                role=record.role,
                snippet=self.redactor(
                    _snippet(record.content, terms)
                ),
                created_at=record.created_at,
                turn_id=record.turn_id,
            )
            for conversation_id, record in page_records
        )
        next_cursor = (
            _encode_cursor(
                {
                    "kind": "search",
                    "shape": shape,
                    "offset": offset + len(hits),
                }
            )
            if has_more
            else None
        )
        return SessionSearchPage(
            query=clean_query,
            hits=hits,
            next_cursor=next_cursor,
        )

    def read_window(
        self,
        conversation_id: str,
        *,
        ordinal: int,
        before: int = 3,
        after: int = 3,
        limit: int = 20,
        cursor: str | None = None,
        include_sensitive: bool = False,
    ) -> SessionWindow:
        self.store._assert_conversation(conversation_id)
        anchor = _bounded_int(
            "session ordinal",
            ordinal,
            maximum=2_147_483_647,
        )
        clean_before = _bounded_int(
            "session window before",
            before,
            maximum=MAX_SESSION_WINDOW_RADIUS,
            minimum=0,
        )
        clean_after = _bounded_int(
            "session window after",
            after,
            maximum=MAX_SESSION_WINDOW_RADIUS,
            minimum=0,
        )
        clean_limit = _bounded_int(
            "session window limit",
            limit,
            maximum=MAX_SESSION_WINDOW_LIMIT,
        )
        lower = max(1, anchor - clean_before)
        upper = anchor + clean_after
        shape = _shape_hash(
            "window",
            self.profile_id,
            conversation_id,
            str(anchor),
            str(clean_before),
            str(clean_after),
            str(include_sensitive),
        )
        next_ordinal = (
            _decode_window_cursor(
                cursor,
                shape=shape,
                conversation_id=conversation_id,
            )
            if cursor
            else lower
        )
        records = [
            record
            for record in self.store._messages.get(conversation_id, ())
            if next_ordinal <= record.ordinal <= upper
            and _window_message_visible(
                record.role,
                record.metadata,
                include_sensitive=include_sensitive,
            )
        ][: clean_limit + 1]
        has_more = len(records) > clean_limit
        records = records[:clean_limit]
        messages = tuple(
            SessionMessage(
                message_id=record.id,
                conversation_id=record.conversation_id,
                ordinal=record.ordinal,
                role=record.role,
                content=(
                    record.content
                    if include_sensitive
                    else self.redactor(record.content)
                ),
                created_at=record.created_at,
                turn_id=record.turn_id,
                sensitive=_message_is_sensitive(record.metadata),
            )
            for record in records
        )
        next_cursor = (
            _encode_cursor(
                {
                    "kind": "window",
                    "shape": shape,
                    "conversation_id": conversation_id,
                    "next_ordinal": messages[-1].ordinal + 1,
                }
            )
            if has_more and messages
            else None
        )
        return SessionWindow(
            conversation_id=conversation_id,
            anchor_ordinal=anchor,
            messages=messages,
            next_cursor=next_cursor,
        )


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
        self._media_reservations: dict[str, BudgetReservation] = {}
        self._entries: list[UsageEntry] = []

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
        usage: LLMUsage | None = None,
        cost: LLMCost | None = None,
        fallback_attempts: object = None,
        **_kwargs: Any,
    ) -> tuple[UsageEntry, ...]:
        reservation = self._reservations.pop(
            (turn_id, request_index, ResourceKind.MODEL.value)
        )
        attempts = (
            tuple(fallback_attempts)
            if isinstance(fallback_attempts, (list, tuple))
            else ()
        )
        entries = tuple(
            self._model_entry(
                reservation,
                purpose=purpose,
                usage=_attempt_usage(attempt),
                cost=_attempt_cost(attempt),
                provider=_optional_text(_attempt_value(attempt, "provider")),
                model=_optional_text(_attempt_value(attempt, "model")),
                model_profile_id=_optional_text(
                    _attempt_value(attempt, "model_profile_id")
                ),
                source_event_id=(
                    f"{reservation.source_event_id}:attempt:{index}"
                ),
                metadata={
                    "attempt": index,
                    "success": bool(_attempt_value(attempt, "success")),
                    "error_code": _optional_text(
                        _attempt_value(attempt, "error_code")
                    ),
                },
            )
            for index, attempt in enumerate(attempts, start=1)
        ) or (
            self._model_entry(
                reservation,
                purpose=purpose,
                usage=usage,
                cost=cost,
                provider=cost.provider if cost is not None else None,
                model=cost.model if cost is not None else None,
            ),
        )
        self._entries.extend(entries)
        return entries

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
        entry = self._entry(
            reservation,
            purpose="agent_tool",
            units={"tool_calls": Decimal(1)},
            cost=ExactCost(Decimal(0), pricing_known=True),
            tool_or_service=tool_name,
            metadata={
                "tool_call_index": tool_call_index,
                "attempt": kwargs.get("attempt"),
                "success": kwargs.get("success"),
                "failure_kind": kwargs.get("failure_kind"),
            },
        )
        self._entries.append(entry)
        return (entry,)

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

    def reserve_media_transform(
        self,
        *,
        turn_id: str,
        operation_index: int,
        **kwargs: Any,
    ) -> BudgetReservation:
        network_access = bool(kwargs.get("network_access"))
        pricing_per_unit = kwargs.get("pricing_per_unit")
        units = int(kwargs.get("units", 1))
        if operation_index < 1 or units < 0:
            raise ValueError("media usage quantities cannot be negative")
        reserved_cost = (
            ExactCost(
                Decimal(str(pricing_per_unit)) * Decimal(units),
                pricing_known=True,
                estimated=True,
            )
            if pricing_per_unit is not None
            else ExactCost(
                Decimal(0) if not network_access else None,
                pricing_known=not network_access,
            )
        )
        reservation = self._reserve(
            turn_id,
            operation_index,
            ResourceKind.MEDIA,
            reserved_cost=reserved_cost,
        )
        self._media_reservations[reservation.id] = reservation
        return reservation

    def commit_media_transform(
        self,
        reservation: BudgetReservation,
        *,
        processor: str,
        **kwargs: Any,
    ) -> tuple[UsageEntry, ...]:
        self._media_reservations.pop(reservation.id, None)
        for key, value in tuple(self._reservations.items()):
            if value.id == reservation.id:
                self._reservations.pop(key, None)
        unit_name = str(kwargs.get("unit_name") or "request")
        entry = self._entry(
            reservation,
            purpose="media_transform",
            units={
                "requests": Decimal(1),
                "bytes": Decimal(int(kwargs.get("byte_length", 0))),
                unit_name: Decimal(int(kwargs.get("units", 1))),
            },
            cost=reservation.reserved_cost,
            provider=_optional_text(kwargs.get("provider")),
            tool_or_service=processor,
            metadata={
                "content_ref": kwargs.get("content_ref"),
                "network_access": bool(kwargs.get("network_access")),
                "retains_data": bool(kwargs.get("retains_data")),
            },
        )
        self._entries.append(entry)
        return (entry,)

    def release_media_transform(
        self,
        reservation: BudgetReservation,
    ) -> BudgetReservation | None:
        released = self._media_reservations.pop(reservation.id, None)
        for key, value in tuple(self._reservations.items()):
            if value.id == reservation.id:
                self._reservations.pop(key, None)
        return released

    def query(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resource_kind: ResourceKind | None = None,
        channel: str | None = None,
        conversation_id: str | None = None,
        goal_id: str | None = None,
        job_id: str | None = None,
        child_task_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> UsagePage:
        for name, boundary in (("start", start), ("end", end)):
            if boundary is not None and boundary.tzinfo is None:
                raise ValueError(
                    f"usage query {name} must be timezone-aware"
                )
        if start is not None and end is not None and start >= end:
            raise ValueError(
                "usage query start must be earlier than end"
            )
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("usage query limit must be an integer")
        if limit < 1 or limit > MAX_EXPORT_ENTRIES:
            raise ValueError(
                "usage query limit must be between 1 and 10000"
            )
        cursor_key = (
            _decode_usage_cursor(cursor)
            if cursor is not None
            else None
        )
        entries = sorted(
            (
                entry
                for entry in self._entries
                if (start is None or entry.occurred_at >= start)
                and (end is None or entry.occurred_at < end)
                and (
                    resource_kind is None
                    or entry.resource_kind is ResourceKind(resource_kind)
                )
                and (
                    channel is None
                    or entry.dimensions.channel == channel
                )
                and (
                    conversation_id is None
                    or entry.dimensions.conversation_id == conversation_id
                )
                and (
                    goal_id is None
                    or entry.dimensions.goal_id == goal_id
                )
                and (
                    job_id is None
                    or entry.dimensions.job_id == job_id
                )
                and (
                    child_task_id is None
                    or entry.dimensions.child_task_id == child_task_id
                )
                and (
                    cursor_key is None
                    or (entry.occurred_at, entry.id) > cursor_key
                )
            ),
            key=lambda entry: (entry.occurred_at, entry.id),
        )
        has_more = len(entries) > limit
        page_entries = tuple(entries[:limit])
        next_cursor = (
            _encode_usage_cursor(
                page_entries[-1].occurred_at,
                page_entries[-1].id,
            )
            if has_more and page_entries
            else None
        )
        return UsagePage(
            entries=page_entries,
            next_cursor=next_cursor,
        )

    def group(
        self,
        group_by: UsageGroupBy,
        **kwargs: Any,
    ) -> tuple[UsageAggregate, ...]:
        group_by = UsageGroupBy(group_by)
        query_kwargs = dict(kwargs)
        query_kwargs.setdefault("limit", MAX_EXPORT_ENTRIES)
        page = self.query(**query_kwargs)
        if page.next_cursor is not None:
            raise ValueError(
                "usage aggregation exceeds the bounded query limit; "
                "narrow the range"
            )
        entries = page.entries
        grouped: dict[str, list[UsageEntry]] = {}
        for entry in entries:
            key = _usage_group_key(entry, group_by)
            grouped.setdefault(key, []).append(entry)
        aggregates = []
        for key, records in sorted(grouped.items()):
            currencies = {record.cost.currency for record in records}
            known_cost = sum(
                (
                    record.cost.amount
                    for record in records
                    if record.cost.amount is not None
                ),
                start=Decimal(0),
            )
            unknown_cost_entries = sum(
                not record.cost.pricing_known
                or record.cost.amount is None
                for record in records
            )
            currency = (
                next(iter(currencies))
                if len(currencies) == 1
                else "MIXED"
            )
            aggregates.append(
                UsageAggregate(
                    key=key,
                    entry_count=len(records),
                    model_calls=sum(
                        int(
                            record.units.get(
                                "model_calls",
                                Decimal(0),
                            )
                        )
                        for record in records
                    ),
                    tool_calls=sum(
                        int(
                            record.units.get(
                                "tool_calls",
                                Decimal(0),
                            )
                        )
                        for record in records
                    ),
                    total_tokens=int(
                        sum(
                            record.units.get(
                                "total_tokens",
                                Decimal(0),
                            )
                            for record in records
                        )
                    ),
                    cost=ExactCost(
                        known_cost if len(currencies) == 1 else None,
                        currency=currency,
                        pricing_known=(
                            unknown_cost_entries == 0
                            and len(currencies) == 1
                        ),
                        estimated=any(
                            record.cost.estimated
                            for record in records
                        ),
                        reported=all(
                            record.cost.reported
                            for record in records
                        ),
                    ),
                    unknown_cost_entries=unknown_cost_entries,
                )
            )
        return tuple(aggregates)

    def _reserve(
        self,
        turn_id: str,
        index: int,
        kind: ResourceKind,
        *,
        reserved_cost: ExactCost | None = None,
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
            reserved_cost=reserved_cost
            or ExactCost(Decimal("0"), pricing_known=True),
            created_at=_now(),
        )
        self._reservations[(turn_id, index, kind.value)] = reservation
        return reservation

    def _entry(
        self,
        reservation: BudgetReservation,
        *,
        purpose: str,
        source_event_id: str | None = None,
        units: Mapping[str, Decimal] | None = None,
        cost: ExactCost | None = None,
        provider: str | None = None,
        model: str | None = None,
        tool_or_service: str | None = None,
        model_profile_id: str | None = None,
        usage_estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> UsageEntry:
        occurred_at = _now()
        return UsageEntry(
            id=f"usage_{uuid4().hex}",
            resource_kind=reservation.resource_kind,
            source_event_id=source_event_id or reservation.source_event_id,
            dimensions=reservation.dimensions,
            occurred_at=occurred_at,
            billing_period=occurred_at.strftime("%Y-%m"),
            purpose=purpose,
            units=units or {},
            cost=cost or ExactCost(None),
            provider=provider,
            model=model,
            tool_or_service=tool_or_service,
            model_profile_id=model_profile_id,
            usage_estimated=usage_estimated,
            metadata=metadata or {},
        )

    def _model_entry(
        self,
        reservation: BudgetReservation,
        *,
        purpose: str,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        provider: str | None = None,
        model: str | None = None,
        model_profile_id: str | None = None,
        source_event_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> UsageEntry:
        units = {
            "model_calls": Decimal(1),
            "input_tokens": Decimal(
                usage.input_tokens if usage is not None else 0
            ),
            "output_tokens": Decimal(
                usage.output_tokens if usage is not None else 0
            ),
            "total_tokens": Decimal(
                usage.total_tokens if usage is not None else 0
            ),
            "cached_input_tokens": Decimal(
                usage.cached_input_tokens if usage is not None else 0
            ),
            "cache_hit_input_tokens": Decimal(
                usage.cache_hit_input_tokens if usage is not None else 0
            ),
            "cache_write_input_tokens": Decimal(
                usage.cache_write_input_tokens if usage is not None else 0
            ),
            "cache_miss_input_tokens": Decimal(
                usage.cache_miss_input_tokens if usage is not None else 0
            ),
            "reasoning_tokens": Decimal(
                usage.reasoning_tokens if usage is not None else 0
            ),
        }
        exact_cost = (
            ExactCost(
                cost.amount,
                currency=cost.currency,
                pricing_known=cost.pricing_known,
                estimated=cost.estimated,
            )
            if cost is not None
            else ExactCost(None)
        )
        return self._entry(
            reservation,
            purpose=purpose,
            source_event_id=source_event_id,
            units=units,
            cost=exact_cost,
            provider=provider,
            model=model,
            model_profile_id=model_profile_id,
            usage_estimated=usage.estimated if usage is not None else False,
            metadata=metadata,
        )


def _attempt_value(attempt: object, name: str) -> object:
    if isinstance(attempt, Mapping):
        return attempt.get(name)
    return getattr(attempt, name, None)


def _attempt_usage(attempt: object) -> LLMUsage | None:
    value = _attempt_value(attempt, "usage")
    if isinstance(value, LLMUsage):
        return value
    if isinstance(value, Mapping):
        return LLMUsage(**dict(value))
    return None


def _attempt_cost(attempt: object) -> LLMCost | None:
    value = _attempt_value(attempt, "cost")
    if isinstance(value, LLMCost):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        for name in (
            "amount",
            "input_cost",
            "cached_input_cost",
            "cache_write_input_cost",
            "output_cost",
        ):
            if payload.get(name) is not None:
                payload[name] = Decimal(str(payload[name]))
        return LLMCost(**payload)
    return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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

    def list(self) -> tuple[Any, ...]:
        return ()

    def audit(self) -> PluginAuditReport:
        return self.verify_startup()


class InMemoryContentService:
    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.profile_id = scope.actor_id or "hosted"
        self._content: dict[str, bytes] = {}

    def sweep_expired(self) -> int:
        return 0
