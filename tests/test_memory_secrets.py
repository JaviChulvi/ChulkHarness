"""Tests for credential protection at durable memory boundaries."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from chulk.memory import (
    AsyncMemoryPolicy,
    MemoryExtractionCandidate,
    MemoryPolicy,
    SQLiteMemoryStore,
)
from chulk.memory.security import MemorySecretError
from chulk.tools.memory import (
    async_import_memories_tool,
    async_save_memory_tool,
    async_update_memory_tool,
)
from chulk.tools.registry import ToolRegistry


class _PoisonAsyncMemoryStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        async def invoke(*_args, **_kwargs):
            self.calls.append(name)
            raise AssertionError(f"unsafe hosted memory call: {name}")

        return invoke


@pytest.mark.parametrize(
    "content",
    [
        "OPENAI_API_KEY=sk-proj-FakeCredentialValue123456789",
        "Database password: correct-horse-battery-staple",
        "Authorization: Bearer FakeBearerCredential123456789",
        "GitHub token github_pat_FakeCredentialValue123456789",
        "Database URL is postgresql://user:FakePassword123@localhost/app",
    ],
)
def test_save_memory_rejects_credential_like_content_without_echoing_it(tmp_path, content):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    with pytest.raises(MemorySecretError) as exc_info:
        store.save_memory(content, tags=["preference"])

    assert content not in str(exc_info.value)
    assert "FakeCredentialValue" not in str(exc_info.value)
    assert store.list_memories() == []


def test_save_memory_rejects_secrets_nested_in_metadata(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    secret = "FakeOpaqueCredentialValue123456789"

    with pytest.raises(MemorySecretError) as exc_info:
        store.save_memory(
            "User prefers concise answers.",
            tags=["preference"],
            metadata={"provider": {"accessToken": secret}},
        )

    assert secret not in str(exc_info.value)
    assert store.list_memories() == []


def test_update_memory_rejects_secret_and_preserves_existing_record(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = store.save_memory("User prefers concise answers.", tags=["preference"])
    secret = "sk-proj-FakeUpdateCredential123456789"

    with pytest.raises(MemorySecretError) as exc_info:
        store.update_memory(memory_id, content=f"OPENAI_API_KEY={secret}")

    memory = store.get_memory(memory_id)
    assert secret not in str(exc_info.value)
    assert memory is not None
    assert memory.content == "User prefers concise answers."


def test_import_preflights_all_memories_before_writing_any_rows(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    secret = "sk-proj-FakeImportCredential123456789"
    markdown = tmp_path / "MEMORY.md"
    markdown.write_text(
        "- [preference] User prefers concise answers.\n"
        f"- [project] OPENAI_API_KEY={secret}\n",
        encoding="utf-8",
    )

    with pytest.raises(MemorySecretError) as exc_info:
        store.import_markdown(markdown)

    assert secret not in str(exc_info.value)
    assert store.list_memories() == []


def test_create_proposal_rejects_secret_in_evidence(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    secret = "FakeBearerCredential123456789"

    with pytest.raises(MemorySecretError) as exc_info:
        store.create_memory_proposal(
            "User prefers concise answers.",
            tags=["preference"],
            evidence=f"Authorization: Bearer {secret}",
        )

    assert secret not in str(exc_info.value)
    assert store.list_memory_proposals() == []


def test_approval_rechecks_legacy_pending_proposal_before_saving(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    proposal_id = store.create_memory_proposal("User prefers concise answers.")
    secret = "sk-proj-FakeLegacyCredential123456789"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE memory_proposals SET content = ? WHERE id = ?",
            (f"OPENAI_API_KEY={secret}", proposal_id),
        )

    with pytest.raises(MemorySecretError) as exc_info:
        store.approve_memory_proposal(proposal_id)

    proposal = store.get_memory_proposal(proposal_id)
    assert secret not in str(exc_info.value)
    assert store.list_memories() == []
    assert proposal is not None
    assert proposal.status == "pending"
    assert proposal.accepted_memory_id is None


@pytest.mark.parametrize(
    "content",
    [
        "User prefers environment variables for API keys.",
        "User uses a password manager.",
        "Keep token budgets under 4,000 tokens.",
        "OPENAI_API_KEY should be configured before startup.",
        "The password is strong and stored in a password manager.",
    ],
)
def test_normal_preference_and_workflow_memories_remain_allowed(tmp_path, content):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    memory_id = store.save_memory(content, tags=["persona", "preference", "workflow"])

    memory = store.get_memory(memory_id)
    assert memory is not None
    assert memory.content == content


@pytest.mark.parametrize("mode", ["automatic", "manual"])
def test_inferred_secret_candidates_are_skipped_without_aborting_the_turn(tmp_path, mode):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    policy = MemoryPolicy(store, mode)
    secret = "sk-proj-FakeInferredCredential123456789"
    candidate = MemoryExtractionCandidate(
        content=f"OPENAI_API_KEY={secret}",
        tags=["explicit", "user"],
    )

    result = policy.handle_candidates(
        [candidate],
        conversation_id="conversation-id",
        turn_id="turn-id",
        evidence=f"Remember OPENAI_API_KEY={secret}",
    )

    assert result.accepted_memory_ids == ()
    assert result.proposal_ids == ()
    assert store.list_memories() == []
    assert store.list_memory_proposals() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_factory", "arguments"),
    [
        (
            async_save_memory_tool,
            {
                "content": (
                    "OPENAI_API_KEY="
                    "sk-proj-FakeHostedSaveCredential123456789"
                )
            },
        ),
        (
            async_update_memory_tool,
            {
                "memory_id": "memory-1",
                "metadata": {
                    "provider": {
                        "accessToken": (
                            "FakeHostedUpdateCredential123456789"
                        )
                    }
                },
            },
        ),
    ],
)
async def test_async_memory_tools_reject_secrets_before_hosted_write(
    tool_factory,
    arguments,
):
    store = _PoisonAsyncMemoryStore()
    registry = ToolRegistry()
    registry.register(tool_factory(store))

    result = await registry.run_async(registry.list_tools()[0].name, arguments)

    assert not result.success
    assert result.error is not None
    assert "FakeHosted" not in result.error
    assert store.calls == []


@pytest.mark.asyncio
async def test_async_memory_import_preflights_before_hosted_write(tmp_path):
    secret = "sk-proj-FakeHostedImportCredential123456789"
    markdown = tmp_path / "MEMORY.md"
    markdown.write_text(
        "- [preference] User prefers concise answers.\n"
        f"- [project] OPENAI_API_KEY={secret}\n",
        encoding="utf-8",
    )
    store = _PoisonAsyncMemoryStore()
    registry = ToolRegistry()
    registry.register(async_import_memories_tool(store, tmp_path))

    result = await registry.run_async(
        "import_memories",
        {"path": "MEMORY.md"},
    )

    assert not result.success
    assert result.error is not None
    assert secret not in result.error
    assert store.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["automatic", "manual"])
async def test_async_memory_policy_rejects_secrets_before_hosted_write(mode):
    store = _PoisonAsyncMemoryStore()
    policy = AsyncMemoryPolicy(store, mode)
    secret = "sk-proj-FakeHostedPolicyCredential123456789"

    result = await policy.handle_candidates(
        [
            MemoryExtractionCandidate(
                content=f"OPENAI_API_KEY={secret}",
                tags=["explicit", "user"],
            )
        ],
        conversation_id="conversation-id",
        turn_id="turn-id",
        evidence=f"Remember OPENAI_API_KEY={secret}",
    )

    assert result.accepted_memory_ids == ()
    assert result.proposal_ids == ()
    assert store.calls == []


@pytest.mark.asyncio
async def test_async_memory_policy_rechecks_proposal_before_hosted_approval():
    secret = "sk-proj-FakeHostedApprovalCredential123456789"

    class ProposalStore:
        def __init__(self) -> None:
            self.approve_called = False

        async def list_memory_proposals(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="proposal-1",
                    content=f"OPENAI_API_KEY={secret}",
                    tags=[],
                    metadata={},
                    source="manual_review",
                    evidence="legacy proposal",
                    conversation_id="conversation-id",
                    turn_id="turn-id",
                )
            ]

        async def approve_memory_proposal(self, _proposal_id):
            self.approve_called = True
            raise AssertionError("unsafe hosted proposal approval")

    store = ProposalStore()
    policy = AsyncMemoryPolicy(store, "manual")

    with pytest.raises(MemorySecretError) as exc_info:
        await policy.approve("proposal-1")

    assert secret not in str(exc_info.value)
    assert not store.approve_called


@pytest.mark.asyncio
async def test_async_memory_policy_rechecks_mapping_proposal_before_approval():
    secret = "sk-proj-FakeHostedMappingCredential123456789"

    class ProposalStore:
        def __init__(self) -> None:
            self.approve_called = False

        async def list_memory_proposals(self, **_kwargs):
            return [
                {
                    "id": "proposal-1",
                    "content": f"OPENAI_API_KEY={secret}",
                    "tags": [],
                    "metadata": {},
                    "source": "manual_review",
                    "evidence": "legacy proposal",
                    "conversation_id": "conversation-id",
                    "turn_id": "turn-id",
                }
            ]

        async def approve_memory_proposal(self, _proposal_id):
            self.approve_called = True
            raise AssertionError("unsafe hosted proposal approval")

    store = ProposalStore()
    policy = AsyncMemoryPolicy(store, "manual")

    with pytest.raises(MemorySecretError) as exc_info:
        await policy.approve("proposal-1")

    assert secret not in str(exc_info.value)
    assert not store.approve_called
