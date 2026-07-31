"""Explicit memory-mode decisions and proposal routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from chulk.capabilities import MemoryMode
from chulk.memory.models import MemoryExtractionCandidate, MemoryProposalRecord
from chulk.memory.security import MemorySecretError, ensure_memory_payload_safe


class _MemoryPolicyStore(Protocol):
    def save_memory(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
        source: str = "manual",
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        dedupe: bool = True,
    ) -> str: ...

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
    ) -> str: ...

    def list_memory_proposals(
        self,
        *,
        status: str | None = "pending",
    ) -> list[MemoryProposalRecord]: ...

    def approve_memory_proposal(self, proposal_id: str) -> MemoryProposalRecord: ...

    def reject_memory_proposal(self, proposal_id: str) -> MemoryProposalRecord: ...


@dataclass(frozen=True)
class MemoryPolicyResult:
    accepted_memory_ids: tuple[str, ...] = ()
    proposal_ids: tuple[str, ...] = ()


class MemoryPolicy:
    """Apply one memory mode to retrieval and proposed writes."""

    def __init__(self, store: _MemoryPolicyStore, mode: MemoryMode | str) -> None:
        self.store = store
        self.mode = MemoryMode(mode)

    @property
    def retrieval_enabled(self) -> bool:
        return self.mode is not MemoryMode.OFF

    def handle_candidates(
        self,
        candidates: list[MemoryExtractionCandidate],
        *,
        conversation_id: str | None,
        turn_id: str | None,
        evidence: str | None,
    ) -> MemoryPolicyResult:
        if self.mode in {MemoryMode.OFF, MemoryMode.READ_ONLY}:
            return MemoryPolicyResult()
        if self.mode is MemoryMode.MANUAL:
            proposal_ids = []
            for candidate in candidates:
                try:
                    proposal_ids.append(
                        self.store.create_memory_proposal(
                            candidate.content,
                            tags=candidate.tags,
                            metadata=candidate.metadata,
                            importance=candidate.importance,
                            source=candidate.source,
                            confidence=candidate.confidence,
                            evidence=evidence,
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                        )
                    )
                except MemorySecretError:
                    continue
            return MemoryPolicyResult(proposal_ids=tuple(proposal_ids))

        memory_ids = []
        for candidate in candidates:
            try:
                memory_ids.append(
                    self.store.save_memory(
                        candidate.content,
                        tags=candidate.tags,
                        metadata=candidate.metadata,
                        importance=candidate.importance,
                        source=candidate.source,
                        confidence=candidate.confidence,
                    )
                )
            except MemorySecretError:
                continue
        return MemoryPolicyResult(accepted_memory_ids=tuple(memory_ids))

    def propose_explicit(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        importance: int = 1,
        source: str = "user_explicit",
        confidence: float = 1.0,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> MemoryPolicyResult:
        ensure_memory_payload_safe(content=content, tags=tags or [], metadata=metadata or {}, source=source)
        candidate = MemoryExtractionCandidate(
            content=content,
            tags=tags or [],
            metadata=metadata or {},
            importance=importance,
            source=source,
            confidence=confidence,
        )
        return self.handle_candidates(
            [candidate],
            conversation_id=conversation_id,
            turn_id=turn_id,
            evidence="explicit tool request",
        )

    def list_pending(self) -> list[MemoryProposalRecord]:
        return self.store.list_memory_proposals(status="pending")

    def approve(self, proposal_id: str) -> MemoryProposalRecord:
        return self.store.approve_memory_proposal(proposal_id)

    def reject(self, proposal_id: str) -> MemoryProposalRecord:
        return self.store.reject_memory_proposal(proposal_id)


__all__ = ["MemoryPolicy", "MemoryPolicyResult"]
