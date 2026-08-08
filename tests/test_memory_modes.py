"""Tests for SDK memory modes and durable proposal review."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

import pytest

from chulk import Agent, AgentConfig, Capabilities, MemoryMode, MemoryProposalStatus
from chulk.llm import LLMClient


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [json.dumps({"type": "final_answer", "content": "done"})]

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _agent(root, mode: str, responses: list[str] | None = None) -> Agent:
    return Agent(
        config=AgentConfig(project_root=root, permission_profile="workspace-write"),
        capabilities=Capabilities(files="off", memory=mode, utilities=False),
        llm=FakeLLM(responses),
        skills=[],
    )


def test_off_mode_disables_tools_retrieval_and_inferred_writes(tmp_path):
    facade = _agent(tmp_path, "off")

    result = facade.run_result("Please remember that project alpha uses SQLite.")

    assert facade.tool_registry.list_tools() == []
    assert result.loaded_memory_ids == ()
    assert facade.runtime.memory_store.list_memories() == []
    assert facade.list_memory_proposals() == ()


def test_read_only_mode_retrieves_without_running_memory_extraction(
    tmp_path, monkeypatch
):
    facade = _agent(tmp_path, "read-only")
    existing_id = facade.runtime.memory_store.save_memory("Project alpha uses SQLite", tags=["project"])

    def fail_extraction(*_args, **_kwargs):
        raise AssertionError("read-only memory must not run candidate extraction")

    monkeypatch.setattr(
        "chulk.core.agent.route_memory_candidates",
        fail_extraction,
    )

    result = facade.run_result("Remember that project beta uses Postgres. What database does alpha use?")

    assert existing_id in result.loaded_memory_ids
    contents = [memory.content for memory in facade.runtime.memory_store.list_memories()]
    assert contents == ["Project alpha uses SQLite"]
    assert facade.list_memory_proposals() == ()


@pytest.mark.asyncio
async def test_async_read_only_mode_skips_candidate_extraction(tmp_path, monkeypatch):
    facade = _agent(tmp_path, "read-only")

    class ReadOnlyPolicy:
        mode = MemoryMode.READ_ONLY

    facade.runtime.async_memory_policy = ReadOnlyPolicy()

    def fail_extraction(*_args, **_kwargs):
        raise AssertionError("read-only memory must not run candidate extraction")

    monkeypatch.setattr(
        "chulk.core.agent.extract_memory_candidates",
        fail_extraction,
    )

    await facade.runtime._extract_long_term_memories_async(
        "Remember that project beta uses Postgres."
    )

    assert facade.runtime.state.extracted_memory_ids == []
    facade.close()


def test_manual_mode_persists_proposals_across_restart_and_approves(tmp_path):
    facade = _agent(tmp_path, "manual")
    facade.run("Please remember that project alpha uses SQLite.")
    proposals = facade.list_memory_proposals()

    assert len(proposals) == 1
    assert proposals[0].status is MemoryProposalStatus.PENDING
    assert proposals[0].namespace == "default"
    assert facade.runtime.memory_store.list_memories() == []
    proposal_id = proposals[0].id
    facade.close()

    restarted = _agent(tmp_path, "manual")
    assert restarted.list_memory_proposals()[0].id == proposal_id
    approved = restarted.approve_memory_proposal(proposal_id)

    assert approved.status is MemoryProposalStatus.APPROVED
    assert approved.accepted_memory_id is not None
    assert [memory.content for memory in restarted.runtime.memory_store.list_memories()] == [
        "project alpha uses SQLite"
    ]


def test_concurrent_proposal_approval_persists_exactly_one_memory(tmp_path, monkeypatch):
    facade = _agent(tmp_path, "manual")
    facade.run("Please remember that project alpha uses SQLite.")
    proposal_id = facade.list_memory_proposals()[0].id
    store = facade.runtime.memory_store
    race = Barrier(2)

    def force_original_race_window(content: str, *, threshold: float = 0.90):
        race.wait(timeout=2)
        return None

    monkeypatch.setattr(store, "find_duplicate_memory", force_original_race_window)
    with ThreadPoolExecutor(max_workers=2) as executor:
        approvals = list(executor.map(lambda _: store.approve_memory_proposal(proposal_id), range(2)))

    memories = store.list_memories()
    assert len(memories) == 1
    assert approvals[0].accepted_memory_id == approvals[1].accepted_memory_id == memories[0].id


def test_manual_mode_rejects_without_persisting(tmp_path):
    facade = _agent(tmp_path, "manual")
    facade.run("Please remember that project alpha uses SQLite.")
    proposal = facade.list_memory_proposals()[0]

    rejected = facade.reject_memory_proposal(proposal.id)

    assert rejected.status is MemoryProposalStatus.REJECTED
    assert facade.list_memory_proposals() == ()
    assert facade.runtime.memory_store.list_memories() == []


def test_automatic_mode_persists_inferred_memory_without_proposal(tmp_path):
    facade = _agent(tmp_path, "automatic")

    result = facade.run_result("Please remember that project alpha uses SQLite.")

    assert len(result.loaded_memory_ids) == 1
    assert facade.list_memory_proposals() == ()
    assert [memory.content for memory in facade.runtime.memory_store.list_memories()] == [
        "project alpha uses SQLite"
    ]


def test_manual_explicit_save_tool_creates_review_proposal(tmp_path):
    facade = _agent(
        tmp_path,
        "manual",
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "save_memory",
                    "arguments_json": json.dumps({"content": "User prefers concise answers"}),
                }
            ),
            json.dumps({"type": "final_answer", "content": "proposed"}),
        ],
    )

    result = facade.run_result("Save this preference")

    assert result.tool_calls[0].success is True
    assert result.tool_calls[0].metadata["review_required"] is True
    assert facade.runtime.memory_store.list_memories() == []
    proposal = facade.list_memory_proposals()[0]
    assert proposal.content == "User prefers concise answers"
    assert proposal.conversation_id == facade.conversation_id
    assert proposal.turn_id == result.turn_id
