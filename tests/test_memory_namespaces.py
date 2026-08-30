from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from chulk import Agent, AgentConfig, Capabilities, MemoryMode
from chulk.memory import SQLiteMemoryStore
from chulk.testing import ScriptedLLMClient
from chulk.tools.memory import (
    archive_memory,
    delete_memory,
    list_memories,
    restore_memory,
    search_memory,
    summarize_memories,
    update_memory,
)


def _stores(tmp_path):
    path = tmp_path / "shared.sqlite"
    return (
        SQLiteMemoryStore(path, namespace="tenant:alpha"),
        SQLiteMemoryStore(path, namespace="tenant:beta"),
    )


def test_scoped_retrieval_covers_fts_like_vector_tags_profile_and_dedupe(tmp_path) -> None:
    alpha, beta = _stores(tmp_path)
    alpha_id = alpha.save_memory(
        "Shared keyword with alpha-only context.",
        tags=["preference", "shared"],
        embedding=[1.0, 0.0],
    )
    beta_id = beta.save_memory(
        "Shared keyword with beta-only context.",
        tags=["preference", "shared"],
        embedding=[1.0, 0.0],
    )
    alpha_duplicate = alpha.save_memory(
        "Shared keyword with alpha-only context.",
        tags=["workflow"],
    )

    assert alpha_id != beta_id
    assert alpha_duplicate == alpha_id
    assert {item.id for item in alpha.search_memory("shared keyword")} == {alpha_id}
    assert {item.id for item in beta.search_memory("shared keyword")} == {beta_id}
    assert {item.id for item in alpha.search_by_tags(["shared"])} == {alpha_id}
    assert {item.id for item in beta.search_by_tags(["shared"])} == {beta_id}
    assert {item.id for item in alpha.profile_memories()} == {alpha_id}
    assert {item.id for item in beta.profile_memories()} == {beta_id}
    assert {
        item.id
        for item in alpha.search_memory_by_embedding(
            [1.0, 0.0],
            limit=10,
        )
    } == {alpha_id}
    assert {
        item.id
        for item in beta.search_memory_by_embedding(
            [1.0, 0.0],
            limit=10,
        )
    } == {beta_id}

    alpha.fts_enabled = False
    assert [item.id for item in alpha.search_memory("alpha-only")] == [alpha_id]
    assert SQLiteMemoryStore(alpha.db_path).search_memory("shared keyword") == []


def test_scoped_mutations_and_maintenance_cannot_cross_namespace(tmp_path) -> None:
    alpha, beta = _stores(tmp_path)
    alpha_id = alpha.save_memory("Alpha mutable record.", importance=6)
    beta_id = beta.save_memory("Beta mutable record.", importance=6)

    assert beta.get_memory(alpha_id) is None
    assert beta.update_memory(alpha_id, content="cross-scope update") is False
    assert beta.archive_memory(alpha_id) is False
    assert beta.restore_memory(alpha_id) is False
    assert beta.delete_memory(alpha_id) is False
    assert alpha.get_memory(alpha_id).content == "Alpha mutable record."

    assert alpha.archive_memory(alpha_id)
    assert beta.restore_memory(alpha_id) is False
    assert alpha.restore_memory(alpha_id)
    with sqlite3.connect(alpha.db_path) as conn:
        conn.execute(
            """
            UPDATE memories
            SET updated_at = '2000-01-01T00:00:00+00:00',
                last_accessed_at = '2000-01-01T00:00:00+00:00'
            WHERE id IN (?, ?)
            """,
            (alpha_id, beta_id),
        )

    assert alpha.decay_importance(days_since_accessed=1, amount=2) == 1
    assert alpha.get_memory(alpha_id).importance == 4
    assert beta.get_memory(beta_id).importance == 6
    with sqlite3.connect(alpha.db_path) as conn:
        conn.execute(
            "UPDATE memories SET updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
            (alpha_id,),
        )
    assert alpha.archive_memories_older_than(1) == 1
    assert alpha.get_memory(alpha_id) is None
    assert beta.get_memory(beta_id) is not None
    assert alpha.delete_memory(beta_id) is False
    assert beta.delete_memory(beta_id) is True


def test_scoped_proposal_review_never_crosses_namespace(tmp_path) -> None:
    alpha, beta = _stores(tmp_path)
    alpha_proposal = alpha.create_memory_proposal("Alpha proposal.")
    beta_proposal = beta.create_memory_proposal("Beta proposal.")

    assert beta.get_memory_proposal(alpha_proposal) is None
    assert alpha.get_memory_proposal(beta_proposal) is None
    with pytest.raises(KeyError, match="Unknown memory proposal"):
        beta.approve_memory_proposal(alpha_proposal)
    with pytest.raises(KeyError, match="Unknown memory proposal"):
        alpha.reject_memory_proposal(beta_proposal)

    approved = alpha.approve_memory_proposal(alpha_proposal)
    rejected = beta.reject_memory_proposal(beta_proposal)

    assert approved.namespace == "tenant:alpha"
    assert rejected.namespace == "tenant:beta"
    assert approved.accepted_memory_id is not None
    assert alpha.get_memory(approved.accepted_memory_id).namespace == "tenant:alpha"
    assert beta.get_memory(approved.accepted_memory_id) is None
    assert [item.id for item in alpha.list_memory_proposals(status=None)] == [
        alpha_proposal
    ]
    assert [item.id for item in beta.list_memory_proposals(status=None)] == [
        beta_proposal
    ]


def test_scoped_import_export_and_compaction_are_local(tmp_path) -> None:
    alpha, beta = _stores(tmp_path)
    alpha_markdown = tmp_path / "alpha.md"
    beta_markdown = tmp_path / "beta.md"
    alpha_markdown.write_text("- [project] Alpha imported memory.\n", encoding="utf-8")
    beta_markdown.write_text("- [project] Beta imported memory.\n", encoding="utf-8")

    alpha.import_markdown(alpha_markdown)
    beta.import_markdown(beta_markdown)
    alpha_export = tmp_path / "alpha-export.md"
    beta_export = tmp_path / "beta-export.md"
    alpha.export_markdown(alpha_export)
    beta.export_markdown(beta_export)

    assert "Alpha imported memory" in alpha_export.read_text(encoding="utf-8")
    assert "Beta imported memory" not in alpha_export.read_text(encoding="utf-8")
    assert "Beta imported memory" in beta_export.read_text(encoding="utf-8")
    assert "Alpha imported memory" not in beta_export.read_text(encoding="utf-8")

    alpha.save_memory("Alpha compact candidate exact.", dedupe=False)
    alpha.save_memory("Alpha compact candidate exact.", dedupe=False)
    beta_candidate = beta.save_memory("Alpha compact candidate exact.", dedupe=False)
    assert alpha.compact_memories() == 1
    assert beta.get_memory(beta_candidate) is not None


def test_scoped_concurrent_writers_share_one_database_without_mixing(tmp_path) -> None:
    alpha, beta = _stores(tmp_path)

    def write(store: SQLiteMemoryStore, prefix: str, index: int) -> str:
        return store.save_memory(
            f"{prefix} concurrent memory {index}",
            dedupe=False,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(write, store, prefix, index)
            for store, prefix in ((alpha, "alpha"), (beta, "beta"))
            for index in range(20)
        ]
        ids = [future.result() for future in futures]

    assert len(set(ids)) == 40
    assert len(alpha.list_memories(limit=100)) == 20
    assert len(beta.list_memories(limit=100)) == 20
    assert all(item.namespace == "tenant:alpha" for item in alpha.list_memories(limit=100))
    assert all(item.namespace == "tenant:beta" for item in beta.list_memories(limit=100))


def test_scoped_tools_and_prompt_selection_use_the_bound_store(tmp_path) -> None:
    path = tmp_path / "shared.sqlite"
    alpha = SQLiteMemoryStore(path, namespace="tenant:alpha")
    beta = SQLiteMemoryStore(path, namespace="tenant:beta")
    alpha_id = alpha.save_memory(
        "Alpha prompt marker about namespace isolation.",
        tags=["preference"],
    )
    beta_id = beta.save_memory(
        "Beta prompt marker about namespace isolation.",
        tags=["preference"],
    )

    assert "Alpha prompt marker" in search_memory(
        {"query": "namespace isolation"},
        alpha,
    ).observation
    assert "Beta prompt marker" not in list_memories({}, alpha).observation
    assert "Beta prompt marker" not in summarize_memories({}, alpha).observation
    assert update_memory(
        {"memory_id": beta_id, "content": "crossed"},
        alpha,
    ).error == "not_found"
    assert archive_memory({"memory_id": beta_id}, alpha).error == "not_found"
    assert restore_memory({"memory_id": beta_id}, alpha).error == "not_found"
    assert delete_memory({"memory_id": beta_id}, alpha).error == "not_found"

    llm = ScriptedLLMClient(
        [{"type": "final_answer", "content": "Scoped answer."}]
    )
    with Agent(
        config=AgentConfig(
            project_root=tmp_path,
            store_path=path,
            memory_namespace="tenant:alpha",
        ),
        llm=llm,
        tools=[],
        skills=[],
        capabilities=Capabilities.read_only().with_memory(MemoryMode.READ_ONLY),
    ) as agent:
        assert agent.run("Explain namespace isolation") == "Scoped answer."
        assert agent.runtime.memory_context.store.namespace == "tenant:alpha"

    prompt = llm.call_log[0]["messages"][0]["content"]
    assert alpha_id in agent.state.loaded_memory_ids
    assert beta_id not in agent.state.loaded_memory_ids
    assert "Alpha prompt marker" in prompt
    assert "Beta prompt marker" not in prompt
