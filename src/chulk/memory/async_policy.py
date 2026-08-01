"""Async memory-mode decisions for hosted service boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from chulk.capabilities import MemoryMode
from chulk.hosting.async_utils import call_async_service
from chulk.memory.models import MemoryExtractionCandidate
from chulk.memory.policy import MemoryPolicyResult
from chulk.memory.security import MemorySecretError, ensure_memory_payload_safe


class AsyncMemoryPolicy:
    """Apply one memory mode without calling sync persistence methods."""

    def __init__(self, store: object, mode: MemoryMode | str) -> None:
        self.store = store
        self.mode = MemoryMode(mode)

    @property
    def retrieval_enabled(self) -> bool:
        return self.mode is not MemoryMode.OFF

    async def handle_candidates(
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
            proposal_ids: list[str] = []
            for candidate in candidates:
                try:
                    ensure_memory_payload_safe(
                        content=candidate.content,
                        tags=candidate.tags,
                        metadata=candidate.metadata,
                        source=candidate.source,
                        evidence=evidence,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                    )
                    proposal_ids.append(
                        await call_async_service(
                            self.store,
                            "create_memory_proposal",
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

        memory_ids: list[str] = []
        for candidate in candidates:
            try:
                ensure_memory_payload_safe(
                    content=candidate.content,
                    tags=candidate.tags,
                    metadata=candidate.metadata,
                    source=candidate.source,
                )
                memory_ids.append(
                    await call_async_service(
                        self.store,
                        "save_memory",
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

    async def list_pending(self) -> list[object]:
        return await call_async_service(
            self.store,
            "list_memory_proposals",
            status="pending",
        )

    async def approve(self, proposal_id: str) -> object:
        for proposal in await self.list_pending():
            values = (
                proposal
                if isinstance(proposal, Mapping)
                else {
                    name: getattr(proposal, name, None)
                    for name in (
                        "id",
                        "content",
                        "tags",
                        "metadata",
                        "source",
                        "evidence",
                        "conversation_id",
                        "turn_id",
                    )
                }
            )
            if values.get("id") != proposal_id:
                continue
            ensure_memory_payload_safe(
                content=values.get("content"),
                tags=values.get("tags") or (),
                metadata=values.get("metadata") or {},
                source=values.get("source"),
                evidence=values.get("evidence"),
                conversation_id=values.get("conversation_id"),
                turn_id=values.get("turn_id"),
            )
            break
        return await call_async_service(
            self.store,
            "approve_memory_proposal",
            proposal_id,
        )

    async def reject(self, proposal_id: str) -> object:
        return await call_async_service(
            self.store,
            "reject_memory_proposal",
            proposal_id,
        )


__all__ = ["AsyncMemoryPolicy"]
