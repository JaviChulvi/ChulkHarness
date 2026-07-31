"""One review queue for legacy memory and governed learning proposals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from collections.abc import Callable
from typing import Any

from chulk.memory.models import MemoryProposalRecord
from chulk.memory.security import ensure_memory_payload_safe
from chulk.memory.sqlite_store import SQLiteMemoryStore
from chulk.skills.lifecycle import SkillLifecycleManager
from chulk.skills.lifecycle_models import (
    LearningProposalKind,
    LearningProposalRecord,
    LearningProposalStatus,
)
from chulk.skills.lifecycle_store import SQLiteSkillLifecycleStore
from chulk.skills.manifest import split_skill_front_matter
from chulk.storage import sqlite_connection


class AutomaticLearningBlocked(RuntimeError):
    """A proposal requires explicit host review."""


@dataclass(frozen=True, slots=True)
class LearningProposalDraft:
    """Validated reviewer output before durable storage."""

    kind: LearningProposalKind
    rationale: str
    target_name: str | None = None
    evidence_turn_ids: tuple[str, ...] = ()
    source_trace: str | None = None
    content: str | None = None
    diff: str | None = None
    required_capabilities: tuple[str, ...] = ()
    confidence: float = 1.0
    verification_steps: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


class LearningProposalService:
    """Adapt old memory proposals and new proposals into one durable queue."""

    def __init__(
        self,
        *,
        memory_store: SQLiteMemoryStore,
        lifecycle_store: SQLiteSkillLifecycleStore,
        lifecycle_manager: SkillLifecycleManager,
        automatic_approval_enabled: bool = False,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.lifecycle_store = lifecycle_store
        self.lifecycle_manager = lifecycle_manager
        self.automatic_approval_enabled = automatic_approval_enabled
        self.event_callback = event_callback
        if memory_store.db_path.resolve() != lifecycle_store.db_path.resolve():
            raise ValueError(
                "unified proposals require memory and lifecycle state "
                "in the same SQLite database"
            )

    def create(
        self,
        draft: LearningProposalDraft,
        *,
        reviewer_model: str | None = None,
        cost: str | None = None,
        reviewer_metadata: dict[str, Any] | None = None,
        review_run_id: str | None = None,
        review_token_count: int = 0,
        review_cost_amount: Decimal = Decimal(0),
    ) -> LearningProposalRecord:
        return self.create_many(
            (draft,),
            reviewer_model=reviewer_model,
            cost=cost,
            reviewer_metadata=reviewer_metadata,
            review_run_id=review_run_id,
            review_token_count=review_token_count,
            review_cost_amount=review_cost_amount,
        )[0]

    def create_many(
        self,
        drafts: tuple[LearningProposalDraft, ...],
        *,
        reviewer_model: str | None = None,
        cost: str | None = None,
        reviewer_metadata: dict[str, Any] | None = None,
        review_run_id: str | None = None,
        review_token_count: int = 0,
        review_cost_amount: Decimal = Decimal(0),
    ) -> tuple[LearningProposalRecord, ...]:
        """Persist a validated reviewer batch in one transaction."""
        for draft in drafts:
            _ensure_draft_safe(draft)
        records = self.lifecycle_store.create_proposals(
            tuple(
                {
                    "kind": draft.kind,
                    "rationale": draft.rationale,
                    "target_name": draft.target_name,
                    "evidence_turn_ids": draft.evidence_turn_ids,
                    "source_trace": draft.source_trace,
                    "content": draft.content,
                    "diff": draft.diff,
                    "required_capabilities": draft.required_capabilities,
                    "confidence": draft.confidence,
                    "verification_steps": draft.verification_steps,
                    "reviewer_model": reviewer_model,
                    "cost": cost,
                    "metadata": {
                        **dict(draft.metadata or {}),
                        **dict(reviewer_metadata or {}),
                    },
                }
                for draft in drafts
            ),
            review_run_id=review_run_id,
            review_token_count=review_token_count,
            review_cost_amount=review_cost_amount,
        )
        for record in records:
            self._emit(record, action="created")
        return records

    def list(
        self,
        *,
        status: LearningProposalStatus | str | None = LearningProposalStatus.PENDING,
        limit: int = 100,
    ) -> tuple[LearningProposalRecord, ...]:
        normalized_status = (
            None if status is None else LearningProposalStatus(status)
        )
        legacy_status = (
            None if normalized_status is None else normalized_status.value
        )
        if legacy_status == LearningProposalStatus.FAILED.value:
            legacy: list[MemoryProposalRecord] = []
        else:
            legacy = self.memory_store.list_memory_proposals(
                status=legacy_status
            )
        governed = self.lifecycle_store.list_proposals(
            status=normalized_status,
            limit=limit,
        )
        merged = [
            *(_legacy_record(item) for item in legacy),
            *governed,
        ]
        merged.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return tuple(merged[:_limit(limit)])

    def get(self, proposal_id: str) -> LearningProposalRecord:
        legacy = self.memory_store.get_memory_proposal(proposal_id)
        if legacy is not None:
            return _legacy_record(legacy)
        return self.lifecycle_store.get_proposal(proposal_id)

    def approve(
        self,
        proposal_id: str,
        *,
        approved_by: str,
        automatic: bool = False,
        granted_capabilities: tuple[str, ...] = (),
    ) -> LearningProposalRecord:
        if not approved_by.strip():
            raise ValueError("approved_by cannot be empty")
        legacy = self.memory_store.get_memory_proposal(proposal_id)
        if legacy is not None:
            if automatic:
                self._ensure_automatic_allowed(
                    _legacy_record(legacy),
                    granted_capabilities=granted_capabilities,
                )
            approved = _legacy_record(
                self.memory_store.approve_memory_proposal(proposal_id)
            )
            self._emit(approved, action="approved")
            return approved

        proposal = self.lifecycle_store.get_proposal(proposal_id)
        if proposal.status is not LearningProposalStatus.PENDING:
            return proposal
        if automatic:
            self._ensure_automatic_allowed(
                proposal,
                granted_capabilities=granted_capabilities,
            )
        if proposal.kind.value.startswith("skill_"):
            approved = self.lifecycle_manager.approve(
                proposal_id,
                approved_by=approved_by,
            )
        else:
            approved = self._approve_memory(
                proposal,
                approved_by=approved_by,
            )
        self._emit(approved, action="approved")
        return approved

    def reject(
        self,
        proposal_id: str,
        *,
        rejected_by: str,
    ) -> LearningProposalRecord:
        if not rejected_by.strip():
            raise ValueError("rejected_by cannot be empty")
        legacy = self.memory_store.get_memory_proposal(proposal_id)
        if legacy is not None:
            rejected = _legacy_record(
                self.memory_store.reject_memory_proposal(proposal_id)
            )
            self._emit(rejected, action="rejected")
            return rejected
        proposal = self.lifecycle_store.get_proposal(proposal_id)
        if proposal.kind.value.startswith("skill_"):
            rejected = self.lifecycle_manager.reject(
                proposal_id,
                rejected_by=rejected_by,
            )
        else:
            rejected = self.lifecycle_store.transition_proposal(
                proposal_id,
                status=LearningProposalStatus.REJECTED,
                reviewed_by=rejected_by,
            )
        self._emit(rejected, action="rejected")
        return rejected

    def _emit(
        self,
        proposal: LearningProposalRecord,
        *,
        action: str,
    ) -> None:
        if self.event_callback is None:
            return
        self.event_callback(
            "learning_proposal_changed",
            {
                "proposal_id": proposal.id,
                "kind": proposal.kind.value,
                "status": proposal.status.value,
                "action": action,
                "target_name": proposal.target_name,
                "reviewed_by": proposal.reviewed_by,
                "accepted_memory_id": proposal.accepted_memory_id,
                "applied_revision_id": proposal.applied_revision_id,
            },
        )

    def _approve_memory(
        self,
        proposal: LearningProposalRecord,
        *,
        approved_by: str,
    ) -> LearningProposalRecord:
        assert proposal.content is not None
        memory_options = _memory_options(proposal.metadata)
        ensure_memory_payload_safe(
            content=proposal.content,
            tags=memory_options["tags"],
            metadata=memory_options["metadata"],
            source=memory_options["source"],
        )
        with sqlite_connection(self.lifecycle_store.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT kind, target_name, content, metadata_json, confidence,
                    status
                FROM learning_proposals
                WHERE id = ? AND profile_id = ?
                """,
                (proposal.id, self.lifecycle_store.profile_id),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"learning proposal {proposal.id!r} does not exist"
                )
            if row["status"] != LearningProposalStatus.PENDING.value:
                return self.lifecycle_store.get_proposal(proposal.id)
            if (
                row["kind"] != proposal.kind.value
                or row["target_name"] != proposal.target_name
                or row["content"] != proposal.content
                or json.loads(str(row["metadata_json"])) != proposal.metadata
                or float(row["confidence"]) != proposal.confidence
            ):
                raise RuntimeError(
                    "learning proposal changed before approval"
                )
            if proposal.kind is LearningProposalKind.MEMORY_CREATE:
                memory_id = self.memory_store.save_memory_in_connection(
                    conn,
                    proposal.content,
                    tags=memory_options["tags"],
                    metadata={
                        **memory_options["metadata"],
                        "proposal_id": proposal.id,
                    },
                    importance=memory_options["importance"],
                    source=memory_options["source"],
                    confidence=proposal.confidence,
                    dedupe=False,
                )
            elif proposal.kind is LearningProposalKind.MEMORY_UPDATE:
                if proposal.target_name is None:
                    raise ValueError(
                        "memory_update requires a target memory id"
                    )
                updated = self.memory_store.update_memory_in_connection(
                    conn,
                    proposal.target_name,
                    content=proposal.content,
                    tags=memory_options["tags"],
                    metadata=memory_options["metadata"],
                    importance=memory_options["importance"],
                    source=memory_options["source"],
                    confidence=proposal.confidence,
                )
                if not updated:
                    raise KeyError(
                        f"memory {proposal.target_name!r} does not exist"
                    )
                memory_id = proposal.target_name
            else:
                raise ValueError("proposal is not a memory change")
            conn.execute(
                """
                UPDATE learning_proposals
                SET status = 'approved', reviewed_at = ?, reviewed_by = ?,
                    accepted_memory_id = ?, error = NULL
                WHERE id = ? AND profile_id = ? AND status = 'pending'
                """,
                (
                    _utc_now(),
                    approved_by.strip(),
                    memory_id,
                    proposal.id,
                    self.lifecycle_store.profile_id,
                ),
            )
        return self.lifecycle_store.get_proposal(proposal.id)

    def _ensure_automatic_allowed(
        self,
        proposal: LearningProposalRecord,
        *,
        granted_capabilities: tuple[str, ...],
    ) -> None:
        if not self.automatic_approval_enabled:
            raise AutomaticLearningBlocked(
                "automatic learning approval is disabled"
            )
        if bool(proposal.metadata.get("external_source")):
            raise AutomaticLearningBlocked(
                "external-source proposals require explicit host review"
            )
        granted = set(granted_capabilities)
        if set(proposal.required_capabilities) - granted:
            raise AutomaticLearningBlocked(
                "capability-increasing proposals require explicit host review"
            )
        if not proposal.kind.value.startswith("skill_"):
            return
        if proposal.content is not None:
            try:
                manifest_data, _body = split_skill_front_matter(
                    proposal.content
                )
            except ValueError:
                manifest_data = {}
            if str(manifest_data.get("source", "")).lower() in {
                "external",
                "remote",
                "registry",
            } or str(manifest_data.get("trust", "")).lower() in {
                "external",
                "untrusted",
            }:
                raise AutomaticLearningBlocked(
                    "external-source proposals require explicit host review"
                )
        base_capabilities: set[str] = set()
        if proposal.kind is LearningProposalKind.SKILL_PATCH:
            assert proposal.target_name is not None
            scope = str(proposal.metadata.get("scope", "project"))
            current = self.lifecycle_store.get_skill(
                proposal.target_name,
                scope=scope,
            )
            revision = self.lifecycle_store.get_revision(
                current.active_revision_id
            )
            raw_capabilities = revision.manifest.get(
                "required_capabilities",
                [],
            )
            if isinstance(raw_capabilities, list):
                base_capabilities = {
                    str(item) for item in raw_capabilities
                }
        if set(proposal.required_capabilities) - base_capabilities:
            raise AutomaticLearningBlocked(
                "capability-increasing proposals require explicit host review"
            )


def _legacy_record(proposal: MemoryProposalRecord) -> LearningProposalRecord:
    status = LearningProposalStatus(proposal.status)
    evidence_turn_ids = (proposal.turn_id,) if proposal.turn_id else ()
    source_trace = proposal.conversation_id
    return LearningProposalRecord(
        id=proposal.id,
        profile_id=proposal.namespace,
        kind=LearningProposalKind.MEMORY_CREATE,
        target_name=None,
        rationale=proposal.evidence or "Legacy memory proposal.",
        evidence_turn_ids=evidence_turn_ids,
        source_trace=source_trace,
        content=proposal.content,
        diff=None,
        required_capabilities=(),
        confidence=proposal.confidence,
        verification_steps=(),
        reviewer_model=None,
        cost=None,
        status=status,
        created_at=proposal.created_at,
        reviewed_at=proposal.reviewed_at,
        accepted_memory_id=proposal.accepted_memory_id,
        metadata={
            "legacy_memory_proposal": True,
            "memory": {
                "tags": list(proposal.tags),
                "metadata": dict(proposal.metadata),
                "importance": proposal.importance,
                "source": proposal.source,
            },
        },
    )


def _memory_options(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("memory", {})
    if not isinstance(value, dict):
        raise ValueError("proposal memory metadata must be an object")
    tags = value.get("tags", [])
    memory_metadata = value.get("metadata", {})
    importance = value.get("importance", 1)
    source = value.get("source", "learning_review")
    if not isinstance(tags, list) or not all(
        isinstance(item, str) for item in tags
    ):
        raise ValueError("proposal memory tags must be strings")
    if not isinstance(memory_metadata, dict):
        raise ValueError("proposal memory metadata must be an object")
    if isinstance(importance, bool) or not isinstance(importance, int):
        raise ValueError("proposal memory importance must be an integer")
    if not isinstance(source, str):
        raise ValueError("proposal memory source must be a string")
    return {
        "tags": tags,
        "metadata": memory_metadata,
        "importance": importance,
        "source": source,
    }


def _ensure_draft_safe(draft: LearningProposalDraft) -> None:
    ensure_memory_payload_safe(
        rationale=draft.rationale,
        metadata=draft.metadata or {},
    )
    if draft.content is None:
        return
    if draft.kind in {
        LearningProposalKind.MEMORY_CREATE,
        LearningProposalKind.MEMORY_UPDATE,
    }:
        ensure_memory_payload_safe(content=draft.content)
        return
    try:
        _metadata, body = split_skill_front_matter(draft.content)
    except ValueError:
        body = draft.content
    ensure_memory_payload_safe(skill_body=body)


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return min(value, 1_000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AutomaticLearningBlocked",
    "LearningProposalDraft",
    "LearningProposalService",
]
