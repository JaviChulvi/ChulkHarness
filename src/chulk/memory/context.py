"""Per-turn memory extraction and prompt selection ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from chulk.capabilities import MemoryMode
from chulk.hosting.async_utils import call_async_service
from chulk.memory.constants import PROFILE_MEMORY_TAGS
from chulk.memory.extraction import extract_memory_candidates, route_memory_candidates
from chulk.memory.policy import MemoryPolicy, MemoryPolicyResult
from chulk.memory.sqlite_store import select_memories_for_prompt


class MemoryContextService:
    """Own durable-memory extraction and per-turn prompt selection."""

    def __init__(
        self,
        *,
        state: Any,
        store: Any = None,
        policy: MemoryPolicy | None = None,
        async_store: Any = None,
        async_policy: Any = None,
        trace: Callable[[str, dict], None],
    ) -> None:
        self.state = state
        self.store = store
        self.policy = policy
        self.async_store = async_store
        self.async_policy = async_policy
        self.trace = trace
        self.profile_memories: list[Any] = []
        self.relevant_memories: list[Any] = []

    def extract(self, user_message: str) -> None:
        self.state.extracted_memory_ids = []
        if self.store is None or self.policy is None:
            return
        if self.policy.mode in {MemoryMode.OFF, MemoryMode.READ_ONLY}:
            return
        result = route_memory_candidates(
            user_message,
            self.policy,
            conversation_id=self.state.conversation_id,
            turn_id=self.state.current_turn_id,
        )
        self._record_extraction(result, getattr(self.store, "namespace", None))

    async def extract_async(self, user_message: str) -> None:
        self.state.extracted_memory_ids = []
        policy = self.async_policy
        if policy is None:
            await asyncio.to_thread(self.extract, user_message)
            return
        if policy.mode in {MemoryMode.OFF, MemoryMode.READ_ONLY}:
            return
        result = await policy.handle_candidates(
            extract_memory_candidates(user_message),
            conversation_id=self.state.conversation_id,
            turn_id=self.state.current_turn_id,
            evidence=user_message,
        )
        self._record_extraction(result, getattr(self.async_store, "namespace", None))

    def select(self, user_message: str) -> None:
        self._clear_selection()
        if (
            self.store is None
            or self.policy is None
            or not self.policy.retrieval_enabled
        ):
            return
        namespace = getattr(self.store, "namespace", None)
        self._trace_search_started(user_message, namespace)
        profile, relevant = select_memories_for_prompt(self.store, user_message)
        self._record_selection(profile, relevant, namespace)

    async def select_async(self, user_message: str) -> None:
        self._clear_selection()
        store = self.async_store
        policy = self.async_policy
        if store is None or policy is None:
            await asyncio.to_thread(self.select, user_message)
            return
        if not policy.retrieval_enabled:
            return
        namespace = getattr(store, "namespace", None)
        self._trace_search_started(user_message, namespace)
        profile = await call_async_service(store, "profile_memories", limit=5)
        relevant = await call_async_service(store, "search_memory", user_message, limit=5)
        profile_ids = {memory.id for memory in profile}
        self._record_selection(
            list(profile),
            [memory for memory in relevant if memory.id not in profile_ids],
            namespace,
        )

    def restore(self, turn: Any) -> None:
        self._clear_restored()
        if (
            self.store is not None
            and self.policy is not None
            and self.policy.retrieval_enabled
        ):
            for memory_id in turn.loaded_memory_ids:
                memory = self.store.get_memory(memory_id, include_archived=True)
                self._append_restored(memory)
        elif self.policy is not None and not self.policy.retrieval_enabled:
            self._clear_loaded(turn)

    async def restore_async(self, turn: Any) -> None:
        self._clear_restored()
        store = self.async_store
        policy = self.async_policy
        if store is not None and policy is not None and policy.retrieval_enabled:
            for memory_id in turn.loaded_memory_ids:
                memory = await call_async_service(
                    store,
                    "get_memory",
                    memory_id,
                    include_archived=True,
                )
                self._append_restored(memory)
        elif policy is not None and not policy.retrieval_enabled:
            self._clear_loaded(turn)

    def _record_extraction(self, result: MemoryPolicyResult, namespace: Any) -> None:
        self.state.extracted_memory_ids = list(result.accepted_memory_ids)
        if result.accepted_memory_ids or result.proposal_ids:
            policy = self.async_policy or self.policy
            assert policy is not None
            self.trace(
                "memory_extraction_completed",
                {
                    "turn_id": self.state.current_turn_id,
                    "memory_ids": list(result.accepted_memory_ids),
                    "proposal_ids": list(result.proposal_ids),
                    "memory_mode": policy.mode.value,
                    "memory_namespace": namespace,
                },
            )
        for proposal_id in result.proposal_ids:
            self.trace(
                "learning_proposal_changed",
                {
                    "turn_id": self.state.current_turn_id,
                    "proposal_id": proposal_id,
                    "kind": "memory_create",
                    "status": "pending",
                    "action": "created",
                    "target_name": None,
                },
            )

    def _clear_selection(self) -> None:
        self._clear_restored()
        self.state.loaded_memory_ids = []

    def _clear_restored(self) -> None:
        self.profile_memories = []
        self.relevant_memories = []

    def _append_restored(self, memory: Any) -> None:
        if memory is None:
            return
        destination = (
            self.profile_memories
            if set(memory.tags) & PROFILE_MEMORY_TAGS
            else self.relevant_memories
        )
        destination.append(memory)

    def _clear_loaded(self, turn: Any) -> None:
        self.state.loaded_memory_ids = []
        turn.loaded_memory_ids = []

    def _trace_search_started(self, query: str, namespace: Any) -> None:
        self.trace(
            "memory_search_started",
            {
                "turn_id": self.state.current_turn_id,
                "query": query,
                "memory_namespace": namespace,
            },
        )

    def _record_selection(
        self,
        profile: list[Any],
        relevant: list[Any],
        namespace: Any,
    ) -> None:
        self.profile_memories = list(profile)
        self.relevant_memories = list(relevant)
        self.state.loaded_memory_ids = [memory.id for memory in [*profile, *relevant]]
        self.trace(
            "memory_search_completed",
            {
                "turn_id": self.state.current_turn_id,
                "profile_memory_ids": [memory.id for memory in profile],
                "relevant_memory_ids": [memory.id for memory in relevant],
                "loaded_memory_ids": self.state.loaded_memory_ids,
                "memory_namespace": namespace,
            },
        )
