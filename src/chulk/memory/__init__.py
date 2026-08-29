"""Memory primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chulk._lazy import public_dir, resolve_export

if TYPE_CHECKING:
    from chulk.memory.async_policy import AsyncMemoryPolicy
    from chulk.memory.extraction import extract_memory_candidates, route_memory_candidates
    from chulk.memory.models import (
        DEFAULT_MEMORY_NAMESPACE,
        MemoryExtractionCandidate,
        MemoryProposalRecord,
        MemoryRecord,
        MemoryRetentionPolicy,
        normalize_memory_namespace,
    )
    from chulk.memory.policy import MemoryPolicy, MemoryPolicyResult
    from chulk.memory.retrieval import text_to_embedding
    from chulk.memory.security import MemorySecretError
    from chulk.memory.sqlite_store import SQLiteMemoryStore, select_memories_for_prompt
    from chulk.memory.store import ConversationMemory, Memory, new_memory

__all__ = [
    "ConversationMemory",
    "AsyncMemoryPolicy",
    "DEFAULT_MEMORY_NAMESPACE",
    "Memory",
    "MemoryExtractionCandidate",
    "MemoryPolicy",
    "MemoryPolicyResult",
    "MemoryProposalRecord",
    "MemoryRecord",
    "MemoryRetentionPolicy",
    "normalize_memory_namespace",
    "MemorySecretError",
    "SQLiteMemoryStore",
    "extract_memory_candidates",
    "new_memory",
    "route_memory_candidates",
    "select_memories_for_prompt",
    "text_to_embedding",
]


_EXPORT_MODULES = (
    "chulk.memory.async_policy",
    "chulk.memory.models",
    "chulk.memory.extraction",
    "chulk.memory.policy",
    "chulk.memory.retrieval",
    "chulk.memory.security",
    "chulk.memory.sqlite_store",
    "chulk.memory.store",
)


if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        return resolve_export(
            name,
            public_names=__all__,
            owner_modules=_EXPORT_MODULES,
            namespace=globals(),
        )

    def __dir__() -> list[str]:
        return public_dir(__all__, globals())
