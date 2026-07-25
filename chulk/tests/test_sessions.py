"""Tests for durable conversation sessions."""

import json
from pathlib import Path
import sqlite3

import chulk.main as main_module
import chulk.runtime as runtime_module
import chulk.sessions.sqlite_store as session_store_module
import pytest
from chulk import AgentHandle
from chulk.config import load_config
from chulk.cli.terminal import TerminalUI
from chulk.core import Agent as CoreAgent
from chulk.core.context import ContextBudget, TurnContextSection
from chulk.core.events import TraceEvent
from chulk.core.state import (
    AgentState,
    ObservationRecord,
    Plan,
    PlanStep,
    ToolCallRecord,
    TurnState,
)
from chulk.llm import LLMCapabilities, LLMClient
from chulk.main import create_agent, main
from chulk.memory import ConversationMemory, MemoryPolicy, SQLiteMemoryStore
from chulk.sessions import SQLiteSessionStore, SessionRecorder
from chulk.skills import SkillRegistry
from chulk.tools import Tool, ToolExecutionContext, ToolRegistry, ToolResult


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [json.dumps({"type": "final_answer", "content": "ok"})]
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class RecordingPromptHistory:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.added: list[str] = []

    def replace(self, messages) -> None:
        self.items = [
            message.content if hasattr(message, "content") else message["content"]
            for message in messages
            if (message.role if hasattr(message, "role") else message["role"]) == "user"
            and not ((message.metadata if hasattr(message, "metadata") else message.get("metadata")) or {}).get("internal")
        ]

    def add(self, prompt: str) -> None:
        self.added.append(prompt.strip())


def test_session_store_saves_messages_and_turn_snapshots(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    store.create_conversation("conversation-1", provider="test", model="mock", trace_path="traces/conversation-1.jsonl")
    store.save_message("conversation-1", role="user", content="hello", turn_id="turn-1", message_key="turn-1:user")
    store.save_message(
        "conversation-1",
        role="assistant",
        content="hi back",
        turn_id="turn-1",
        message_key="turn-1:assistant",
    )

    turn = TurnState(user_message="hello", turn_id="turn-1")
    turn.reflection_count = 1
    turn.reflections.append({"attempt": 1, "approved": True, "reason": "ok", "feedback": None})
    turn.complete("hi back")
    store.save_turn_snapshot("conversation-1", turn.to_dict())

    conversations = store.list_conversations()
    messages = store.load_recent_messages("conversation-1", limit=10)
    turns = store.load_turns("conversation-1")

    assert conversations[0].id == "conversation-1"
    assert conversations[0].turn_count == 1
    assert messages == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi back"}]
    assert turns[0].turn_id == "turn-1"
    assert turns[0].final_answer == "hi back"
    assert turns[0].reflection_count == 1
    assert turns[0].reflections[0]["approved"] is True


def test_session_recorder_persists_tool_action_before_observation(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    recorder = SessionRecorder(
        store,
        "conversation-1",
        provider="test",
        model="mock",
    )
    turn = TurnState(user_message="Use lookup.", turn_id="turn-1")
    recorder.callback(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})

    recorder.callback(
        TraceEvent.TOOL_OBSERVATION,
        {
            "turn_id": turn.turn_id,
            "tool_name": "lookup",
            "tool_action_context": (
                '<executed_tool_action>\n{"tool_name":"lookup","arguments_json":"{}"}'
                "\n</executed_tool_action>"
            ),
            "observation": "Lookup completed.",
            "output_metadata": {},
        },
    )

    assert store.load_recent_messages("conversation-1", limit=10) == [
        {
            "role": "assistant",
            "content": (
                '<executed_tool_action>\n{"tool_name":"lookup","arguments_json":"{}"}'
                "\n</executed_tool_action>"
            ),
        },
        {"role": "observation", "content": "Lookup completed."},
    ]
    rendered_history = TerminalUI(color_enabled=False).history(
        store.list_messages("conversation-1", limit=10)
    )
    assert "executed_tool_action" not in rendered_history
    assert "Lookup completed." in rendered_history


def test_session_recorders_atomically_dedupe_replayed_tool_observations(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "store.sqlite"
    first_store = SQLiteSessionStore(path)
    first_recorder = SessionRecorder(
        first_store,
        "conversation-1",
        provider="test",
        model="mock",
    )
    turn = TurnState(user_message="Use lookup.", turn_id="turn-1")
    first_recorder.callback(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})
    turn.observations.append(
        ObservationRecord(tool_name="lookup", content="First lookup completed.")
    )
    first_payload = {
        "turn_id": turn.turn_id,
        "observation_index": 1,
        "tool_name": "lookup",
        "tool_action_context": "Executed lookup with query one.",
        "observation": "First lookup completed.",
        "output_metadata": {},
        "turn": turn.to_dict(),
    }

    original_insert_message = session_store_module._insert_message

    def fail_before_observation_message(*args, **kwargs):
        if kwargs.get("role") == "observation":
            raise RuntimeError("simulated crash")
        return original_insert_message(*args, **kwargs)

    monkeypatch.setattr(session_store_module, "_insert_message", fail_before_observation_message)
    with pytest.raises(RuntimeError, match="simulated crash"):
        first_recorder.callback(TraceEvent.TOOL_OBSERVATION, first_payload)

    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM conversation_observations").fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM conversation_messages WHERE role != 'user'"
        ).fetchone()[0] == 0

    monkeypatch.setattr(session_store_module, "_insert_message", original_insert_message)
    first_recorder.callback(TraceEvent.TOOL_OBSERVATION, first_payload)
    first_recorder.callback(TraceEvent.TOOL_OBSERVATION, first_payload)
    second_store = SQLiteSessionStore(path)
    second_recorder = SessionRecorder(
        second_store,
        "conversation-1",
        provider="test",
        model="mock",
    )
    turn.observations.append(
        ObservationRecord(tool_name="lookup", content="Second lookup completed.")
    )
    second_recorder.callback(
        TraceEvent.TOOL_OBSERVATION,
        {
            "turn_id": turn.turn_id,
            "tool_name": "lookup",
            "tool_action_context": "Executed lookup with query two.",
            "observation": "Second lookup completed.",
            "output_metadata": {},
            "turn": turn.to_dict(),
        },
    )
    second_recorder.callback(TraceEvent.TOOL_OBSERVATION, first_payload)

    with sqlite3.connect(path) as conn:
        observation_keys = [
            row[0]
            for row in conn.execute(
                "SELECT observation_key FROM conversation_observations ORDER BY observation_key"
            )
        ]
        message_keys = [
            row[0]
            for row in conn.execute(
                """
                SELECT message_key
                FROM conversation_messages
                WHERE role != 'user'
                ORDER BY ordinal
                """
            )
        ]

    assert observation_keys == ["turn-1:observation:1", "turn-1:observation:2"]
    assert message_keys == [
        "turn-1:tool_action:1",
        "turn-1:observation:1",
        "turn-1:tool_action:2",
        "turn-1:observation:2",
    ]
    restored_turn = SQLiteSessionStore(path).load_turns("conversation-1")[0]
    assert [record.content for record in restored_turn.observations] == [
        "First lookup completed.",
        "Second lookup completed.",
    ]


def test_session_recorder_checkpoints_approved_and_plan_step_progress(tmp_path):
    path = tmp_path / "store.sqlite"
    first_store = SQLiteSessionStore(path)
    first_recorder = SessionRecorder(
        first_store,
        "conversation-1",
        provider="test",
        model="mock",
    )
    plan = Plan(
        summary="Execute two durable steps.",
        steps=[
            PlanStep(id="1", title="Complete work", description="Finish the first step."),
            PlanStep(
                id="2",
                title="Blocked work",
                description="Record a blocked second step.",
                depends_on=["1"],
            ),
        ],
    )
    turn = TurnState(user_message="Run the durable plan.", turn_id="turn-1")
    turn.wait_for_plan_approval(plan)
    first_recorder.callback(
        TraceEvent.PLAN_CREATED,
        {"turn_id": turn.turn_id, "plan": plan.to_dict(), "turn": turn.to_dict()},
    )
    turn.approve_plan()
    first_recorder.callback(
        TraceEvent.PLAN_APPROVED,
        {"turn_id": turn.turn_id, "plan": plan.to_dict(), "turn": turn.to_dict()},
    )

    approved_turn = SQLiteSessionStore(path).load_turns("conversation-1")[0]
    assert approved_turn.plan_approved is True
    assert approved_turn.active_plan is not None
    assert approved_turn.active_plan.status() == "approved"

    second_store = SQLiteSessionStore(path)
    second_recorder = SessionRecorder(
        second_store,
        "conversation-1",
        provider="test",
        model="mock",
    )
    first_step, second_step = plan.steps
    first_step.mark("in_progress")
    second_recorder.callback(
        TraceEvent.PLAN_STEP_STARTED,
        {
            "turn_id": turn.turn_id,
            "step": first_step.to_dict(),
            "plan": plan.to_dict(),
            "turn": turn.to_dict(),
        },
    )
    started_turn = SQLiteSessionStore(path).load_turns("conversation-1")[0]
    assert started_turn.active_plan is not None
    assert started_turn.active_plan.steps[0].status == "in_progress"

    first_step.add_evidence("The first step completed.", tool_name="plan_step_update")
    first_step.mark("completed")
    completed_payload = {
        "turn_id": turn.turn_id,
        "step": first_step.to_dict(),
        "plan": plan.to_dict(),
        "turn": turn.to_dict(),
    }
    second_recorder.callback(TraceEvent.PLAN_STEP_COMPLETED, completed_payload)
    second_recorder.callback(TraceEvent.PLAN_STEP_COMPLETED, completed_payload)
    completed_turn = SQLiteSessionStore(path).load_turns("conversation-1")[0]
    assert completed_turn.active_plan is not None
    assert completed_turn.active_plan.steps[0].status == "completed"
    assert completed_turn.active_plan.steps[0].evidence[-1].content == "The first step completed."

    second_step.mark("in_progress")
    second_recorder.callback(
        TraceEvent.PLAN_STEP_STARTED,
        {
            "turn_id": turn.turn_id,
            "step": second_step.to_dict(),
            "plan": plan.to_dict(),
            "turn": turn.to_dict(),
        },
    )
    second_step.block("A durable blocker was recorded.")
    second_recorder.callback(
        TraceEvent.PLAN_STEP_BLOCKED,
        {
            "turn_id": turn.turn_id,
            "step": second_step.to_dict(),
            "plan": plan.to_dict(),
            "turn": turn.to_dict(),
        },
    )

    blocked_turn = SQLiteSessionStore(path).load_turns("conversation-1")[0]
    assert blocked_turn.active_plan is not None
    assert [step.status for step in blocked_turn.active_plan.steps] == ["completed", "blocked"]
    assert blocked_turn.active_plan.steps[1].blocked_reason == "A durable blocker was recorded."
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM conversation_turns").fetchone()[0] == 1


def test_session_recorder_atomically_persists_terminal_message_and_turn(tmp_path):
    final_turn = TurnState(user_message="finish", turn_id="turn-final")
    final_turn.complete("finished")

    rejected_plan = Plan(
        summary="Reject this plan.",
        steps=[PlanStep(id="1", title="Work", description="Do work.")],
    )
    rejected_turn = TurnState(user_message="reject", turn_id="turn-rejected")
    rejected_turn.wait_for_plan_approval(rejected_plan)
    rejected_turn.reject_plan("Plan rejected. No tools were run.")

    failed_turn = TurnState(user_message="fail", turn_id="turn-failed")
    failed_turn.fail("failed safely")
    blocked_turn = TurnState(user_message="block", turn_id="turn-blocked")
    blocked_turn.block("blocked safely")

    cases = [
        (
            TraceEvent.FINAL_ANSWER,
            final_turn,
            {"content": "finished"},
            "completed",
            "final",
        ),
        (
            TraceEvent.PLAN_REJECTED,
            rejected_turn,
            {},
            "plan_rejected",
            "plan_rejected",
        ),
        (
            TraceEvent.TURN_FAILED,
            failed_turn,
            {"message": "failed safely", "status": "failed"},
            "failed",
            "failed",
        ),
        (
            TraceEvent.TURN_FAILED,
            blocked_turn,
            {"message": "blocked safely", "status": "blocked"},
            "blocked",
            "failed",
        ),
        (
            TraceEvent.PLAN_STEP_BLOCKED,
            blocked_turn,
            {},
            "blocked",
            "failed",
        ),
    ]

    for index, (event_type, turn, payload, status, message_kind) in enumerate(cases):
        store = SQLiteSessionStore(tmp_path / f"terminal-{index}.sqlite")
        recorder = SessionRecorder(
            store,
            f"conversation-{index}",
            provider="test",
            model="mock",
        )
        recorder.callback(
            event_type,
            {"turn_id": turn.turn_id, "turn": turn.to_dict(), **payload},
        )

        restored_turn = store.load_turns(f"conversation-{index}")[0]
        terminal_message = store.load_terminal_turn_message(
            f"conversation-{index}",
            turn.turn_id,
        )
        assert restored_turn.status == status
        assert store.get_conversation(f"conversation-{index}").status == status
        assert terminal_message is not None
        assert terminal_message["kind"] == message_kind


def test_final_answer_checkpoint_survives_crash_before_turn_finished(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "durable final"})]
        ),
    )
    recorder_callback = agent.session_recorder.callback

    def crash_before_turn_finished(event_type, payload):
        if event_type == TraceEvent.TURN_FINISHED:
            raise RuntimeError("simulated crash before turn snapshot")
        recorder_callback(event_type, payload)

    agent.event_callback = crash_before_turn_finished

    with pytest.raises(RuntimeError, match="simulated crash"):
        agent.run_turn("finish once")

    store = SQLiteSessionStore(config.store_path)
    restored_turn = store.load_turns(agent.state.conversation_id)[-1]
    assert restored_turn.status == "completed"
    assert restored_turn.final_answer == "durable final"

    resumed_llm = FakeLLMClient()
    resumed_agent = create_agent(
        config,
        lambda _config: resumed_llm,
        conversation_id=agent.state.conversation_id,
    )
    assert resumed_agent.has_resumable_plan() is False
    assert resumed_agent.approve_plan() == "No plan is waiting for approval."
    assert resumed_llm.requests == []


def test_session_store_round_trips_turn_context_metadata(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    store.create_conversation("conversation-1", provider="test", model="mock")
    turn = TurnState(
        user_message="hello",
        turn_id="turn-context",
        context_sections=[
            TurnContextSection(
                id="src-1",
                title="Source",
                source="drive://src-1",
                content="Source text.",
                metadata={"score": 0.92},
            )
        ],
        prompt_profile="polp-search",
        locale="es-ES",
        extension_metadata={"confidence": 0.8},
        tool_context_metadata={"org_id": "org-1"},
    )
    store.save_turn_snapshot("conversation-1", turn.to_dict())

    restored_turn = store.load_turns("conversation-1")[0]

    assert restored_turn.context_sections[0].id == "src-1"
    assert restored_turn.context_sections[0].metadata == {"score": 0.92}
    assert restored_turn.prompt_profile == "polp-search"
    assert restored_turn.locale == "es-ES"
    assert restored_turn.extension_metadata == {"confidence": 0.8}
    assert restored_turn.tool_context_metadata == {"org_id": "org-1"}


def test_session_store_saves_summary_and_loads_unsummarized_messages(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    store.create_conversation("conversation-1", provider="test", model="mock")
    store.save_message("conversation-1", role="user", content="old question", message_key="m1")
    store.save_message(
        "conversation-1",
        role="assistant",
        content="Plan display for the user only.",
        message_key="plan-display",
        metadata={"prompt_excluded": True},
    )
    store.save_message("conversation-1", role="assistant", content="old answer", message_key="m2")
    store.save_message("conversation-1", role="user", content="latest question", message_key="m3")

    store.save_conversation_summary(
        "conversation-1",
        content="Old question and answer were about context compaction.",
        source_message_count=2,
    )

    summary = store.load_latest_summary("conversation-1")
    assert summary is not None
    messages = store.load_recent_messages(
        "conversation-1",
        limit=10,
        after_ordinal=summary.metadata["source_message_ordinal"],
    )

    assert summary.content == "Old question and answer were about context compaction."
    assert summary.source_message_count == 2
    assert summary.metadata["source_message_ordinal"] == 3
    assert messages == [{"role": "user", "content": "latest question"}]
    assert [message.content for message in store.list_messages("conversation-1")] == [
        "old question",
        "Plan display for the user only.",
        "old answer",
        "latest question",
    ]


def test_session_store_restores_pending_plan_turn(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    store.create_conversation("conversation-1", provider="test", model="mock")

    plan = Plan(
        summary="Change the code.",
        steps=[
            PlanStep(
                id="1",
                title="Edit runtime",
                description="Update the runtime code.",
                acceptance_criteria=["Runtime is updated."],
            )
        ],
    )
    turn = TurnState(user_message="plan this", turn_id="turn-1")
    turn.wait_for_plan_approval(plan)
    store.save_turn_snapshot("conversation-1", turn.to_dict())

    restored_turn = store.load_turns("conversation-1")[0]

    assert restored_turn.status == "waiting_for_approval"
    assert restored_turn.active_plan is not None
    assert restored_turn.active_plan.summary == "Change the code."
    assert restored_turn.active_plan.steps[0].title == "Edit runtime"
    assert restored_turn.active_plan.steps[0].acceptance_criteria == ["Runtime is updated."]


def test_pending_plan_restores_skills_memories_and_current_dependencies(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review workflow.\n---\n"
        "# Review\n\nCheck the result carefully.\n",
        encoding="utf-8",
    )
    skill_registry = SkillRegistry(skills_dir)
    skill_registry.load_metadata()
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    profile_id = memory_store.save_memory(
        "Prefer exact verification.",
        tags=["preference"],
    )
    relevant_id = memory_store.save_memory(
        "The pending change targets the runtime.",
        tags=["project"],
    )
    plan = Plan(
        summary="Use the restored execution context.",
        steps=[
            PlanStep(
                id="1",
                title="Run the context tool",
                description="Use the current host dependency.",
            )
        ],
    )
    turn = TurnState(
        user_message="plan this",
        turn_id="turn-pending",
        loaded_skill_names=["review"],
        loaded_memory_ids=[profile_id, relevant_id],
        tool_context_metadata={"org_id": "persisted-org"},
    )
    turn.wait_for_plan_approval(plan)
    state = AgentState(
        turns=[turn],
        active_plan=plan,
        pending_plan_turn_id=turn.turn_id,
    )
    memory = ConversationMemory()
    memory.add_user_message("plan this")
    captured_contexts = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="use_dependency",
            description="Use a host dependency.",
            args_schema={"type": "object", "properties": {}},
            callable=lambda _arguments, context: captured_contexts.append(context)
            or ToolResult("use_dependency", True, "dependency used"),
            accepts_context=True,
        )
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "use_dependency",
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The dependency was used.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "done"}),
        ]
    )
    agent = CoreAgent(
        llm,
        state=state,
        memory=memory,
        memory_store=memory_store,
        skill_registry=skill_registry,
        tool_registry=registry,
        default_tool_context=ToolExecutionContext(
            metadata={"runtime": "current"},
            deps={"token": "live"},
        ),
    )

    assert agent.approve_plan() == "done"

    system_prompt = llm.requests[0][0]["content"]
    assert "Review workflow." in system_prompt
    assert "Check the result carefully." in system_prompt
    assert "Prefer exact verification." in system_prompt
    assert "pending change targets the runtime" in system_prompt
    assert len(captured_contexts) == 1
    assert captured_contexts[0].metadata["org_id"] == "persisted-org"
    assert captured_contexts[0].metadata["runtime"] == "current"
    assert captured_contexts[0].deps == {"token": "live"}


def test_resumable_plan_does_not_restore_memories_when_retrieval_is_off(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = memory_store.save_memory(
        "ARCHIVED_MEMORY_MARKER must stay out of the prompt.",
        tags=["project"],
    )
    memory_store.archive_memory(memory_id)
    plan = Plan(
        summary="Finish restored work.",
        steps=[
            PlanStep(
                id="1",
                title="Finish",
                description="Return the final answer.",
            )
        ],
    )
    turn = TurnState(
        user_message="finish this",
        turn_id="turn-memory-off",
        loaded_memory_ids=[memory_id],
    )
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].mark("completed")
    state = AgentState(
        turns=[turn],
        active_plan=plan,
        loaded_memory_ids=[memory_id],
    )
    memory = ConversationMemory()
    memory.add_user_message(turn.user_message)
    llm = FakeLLMClient(
        [json.dumps({"type": "final_answer", "content": "done without memory"})]
    )

    agent = CoreAgent(
        llm,
        state=state,
        memory=memory,
        memory_store=memory_store,
        memory_policy=MemoryPolicy(memory_store, "off"),
    )

    assert agent.approve_plan() == "done without memory"
    assert "ARCHIVED_MEMORY_MARKER" not in llm.requests[0][0]["content"]
    assert agent.state.loaded_memory_ids == []
    assert agent.state.turns[-1].loaded_memory_ids == []


def test_session_store_preserves_blocked_plan_status(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    store.create_conversation("conversation-1", provider="test", model="mock")

    plan = Plan(
        summary="Run risky work.",
        steps=[
            PlanStep(
                id="1",
                title="Run tool",
                description="Run a tool that can fail.",
            )
        ],
    )
    plan.approve()
    plan.steps[0].block("Tool failed with deliberate_failure.")
    turn = TurnState(user_message="run this", turn_id="turn-1", active_plan=plan, plan_approved=True)
    turn.block("Plan step blocked: Run tool. Tool failed with deliberate_failure.")

    store.save_turn_snapshot("conversation-1", turn.to_dict())
    restored_turn = store.load_turns("conversation-1")[0]
    conversation = store.get_conversation("conversation-1")

    assert conversation.status == "blocked"
    assert restored_turn.status == "blocked"
    assert restored_turn.active_plan is not None
    assert restored_turn.active_plan.status() == "blocked"
    assert restored_turn.active_plan.steps[0].blocked_reason == "Tool failed with deliberate_failure."


def test_create_agent_continues_latest_approved_plan_after_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-approved-plan"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    plan = Plan(
        summary="Continue durable approved work.",
        steps=[
            PlanStep(
                id="1",
                title="Finish durable work",
                description="Complete the work after restart.",
            )
        ],
    )
    turn = TurnState(user_message="run the durable work", turn_id="turn-approved")
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].mark("in_progress")
    store.save_message(
        conversation_id,
        turn_id=turn.turn_id,
        role="user",
        content=turn.user_message,
        message_key=f"{turn.turn_id}:user",
    )
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "Durable work completed after restart.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "resumed work complete"}),
        ]
    )
    agent = create_agent(
        config,
        lambda _config: llm,
        conversation_id=conversation_id,
    )

    assert agent.has_pending_plan() is False
    assert agent.has_resumable_plan() is True
    assert "plan      resumable" in TerminalUI(color_enabled=False).status(config, agent)
    assert agent.run_turn("start unrelated work") == (
        "An approved plan is waiting to continue. Use /approve to resume it or "
        "/reject to cancel it before starting a new turn."
    )
    assert llm.requests == []

    result = AgentHandle(agent).approve_result()

    assert result.content == "resumed work complete"
    assert result.status == "completed"
    assert result.plan is not None
    assert result.plan.status == "completed"
    assert len(agent.state.turns) == 1
    assert agent.has_resumable_plan() is False


def test_restored_approved_plan_can_be_cancelled_without_resuming(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-cancel-resumed-plan"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    plan = Plan(
        summary="Continue approved work.",
        steps=[
            PlanStep(id="1", title="Completed mutation", description="Already done."),
            PlanStep(
                id="2",
                title="Remaining mutation",
                description="Continue once.",
                depends_on=["1"],
            ),
        ],
    )
    turn = TurnState(user_message="run approved work", turn_id="turn-cancel-resume")
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].mark("completed")
    plan.steps[1].mark("in_progress")
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    llm = FakeLLMClient()
    agent = create_agent(
        config,
        lambda _config: llm,
        conversation_id=conversation_id,
    )

    result = AgentHandle(agent).reject_result()
    message = result.content

    assert message == (
        "Approved plan cancelled. No further steps will run; any work already "
        "completed was not rolled back."
    )
    assert llm.requests == []
    assert result.status == "cancelled"
    restored_plan = agent.state.turns[-1].active_plan
    assert restored_plan is not None
    assert [step.status for step in restored_plan.steps] == ["completed", "in_progress"]
    assert agent.state.turns[-1].status == "cancelled"
    assert agent.has_resumable_plan() is False
    assert store.get_conversation(conversation_id).status == "cancelled"
    assert agent.run_turn("start new work") == "ok"


def test_create_agent_does_not_resume_older_approved_plan(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-newer-turn"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    old_plan = Plan(
        summary="Old approved work.",
        steps=[PlanStep(id="1", title="Old work", description="Do old work.")],
    )
    old_turn = TurnState(
        user_message="old request",
        turn_id="turn-old",
        started_at="2026-01-01T00:00:00+00:00",
    )
    old_turn.wait_for_plan_approval(old_plan)
    old_turn.approve_plan()
    store.save_turn_snapshot(conversation_id, old_turn.to_dict())
    latest_turn = TurnState(
        user_message="new request",
        turn_id="turn-new",
        started_at="2026-01-02T00:00:00+00:00",
    )
    latest_turn.complete("new request complete")
    store.save_turn_snapshot(conversation_id, latest_turn.to_dict())

    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(),
        conversation_id=conversation_id,
    )

    assert agent.has_resumable_plan() is False
    assert agent.state.active_plan is None


@pytest.mark.parametrize("response_recorded", [False, True])
def test_restart_blocks_uncheckpointed_hosted_mcp_request(
    monkeypatch,
    tmp_path,
    response_recorded,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = f"conversation-hosted-mcp-{response_recorded}"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    plan = Plan(
        summary="Perform a remote mutation.",
        steps=[
            PlanStep(
                id="1",
                title="Mutate remote state",
                description="Run the hosted operation once.",
            )
        ],
    )
    turn = TurnState(user_message="mutate remotely", turn_id="turn-hosted-mcp")
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].mark("in_progress")
    turn.model_request_count = 1
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    store.save_model_request(
        conversation_id,
        {
            "turn_id": turn.turn_id,
            "request_index": 2,
            "hosted_mcp_enabled": True,
            "hosted_mcp_server_labels": ["remote"],
        },
    )
    if response_recorded:
        store.save_model_response(
            conversation_id,
            {
                "turn_id": turn.turn_id,
                "request_index": 2,
                "content": "provider response arrived before checkpoint",
            },
        )
    assert len(
        store.load_uncheckpointed_hosted_mcp_requests(
            conversation_id,
            turn.turn_id,
            checkpointed_request_count=1,
        )
    ) == 1
    assert store.load_uncheckpointed_hosted_mcp_requests(
        conversation_id,
        turn.turn_id,
        checkpointed_request_count=2,
    ) == []
    llm = FakeLLMClient()

    agent = create_agent(
        config,
        lambda _config: llm,
        conversation_id=conversation_id,
    )

    restored_turn = agent.state.turns[-1]
    assert restored_turn.status == "blocked"
    assert "may have executed a remote operation" in (restored_turn.final_answer or "")
    assert agent.has_resumable_plan() is False
    assert agent.approve_plan() == "No plan is waiting for approval."
    assert llm.requests == []


def test_restart_terminalizes_in_progress_turn_with_blocked_plan(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-blocked-plan-checkpoint"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    plan = Plan(
        summary="Run one fallible step.",
        steps=[PlanStep(id="1", title="Run mutation", description="Run once.")],
    )
    turn = TurnState(user_message="run it", turn_id="turn-blocked-checkpoint")
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].block("Retry limit exhausted.")
    store.save_turn_snapshot(conversation_id, turn.to_dict())

    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(),
        conversation_id=conversation_id,
    )

    restored_turn = agent.state.turns[-1]
    assert restored_turn.status == "blocked"
    assert restored_turn.final_answer == (
        "Plan step blocked: Run mutation. Retry limit exhausted."
    )
    assert agent.has_resumable_plan() is False
    assert any(
        message.content == restored_turn.final_answer
        for message in store.list_messages(conversation_id)
    )


def test_create_agent_reconciles_legacy_terminal_messages_without_replay(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    cases = [
        ("final", "completed", "legacy final answer", "active"),
        (
            "plan_rejected",
            "plan_rejected",
            "Plan rejected. No tools were run.",
            "plan_rejected",
        ),
        ("failed", "failed", "legacy turn failed", "failed"),
    ]

    for index, (message_kind, expected_status, content, conversation_status) in enumerate(cases):
        conversation_id = f"conversation-legacy-{message_kind}"
        turn_id = f"turn-legacy-{message_kind}"
        store.create_conversation(
            conversation_id,
            provider=config.llm_provider,
            model=config.model,
        )
        plan = Plan(
            summary=f"Legacy {message_kind} plan.",
            steps=[
                PlanStep(
                    id="1",
                    title="Legacy work",
                    description="Complete legacy work once.",
                )
            ],
        )
        turn = TurnState(
            user_message=f"legacy request {index}",
            turn_id=turn_id,
        )
        turn.wait_for_plan_approval(plan)
        if message_kind != "plan_rejected":
            turn.approve_plan()
            plan.steps[0].mark(
                "completed" if message_kind == "final" else "in_progress"
            )
        store.save_message(
            conversation_id,
            turn_id=turn_id,
            role="user",
            content=turn.user_message,
            message_key=f"{turn_id}:user",
        )
        store.save_turn_snapshot(conversation_id, turn.to_dict())
        store.save_message(
            conversation_id,
            turn_id=turn_id,
            role="assistant",
            content=content,
            message_key=f"{turn_id}:assistant:{message_kind}",
        )
        if conversation_status != "active":
            store.update_conversation_status(conversation_id, conversation_status)
        llm = FakeLLMClient()

        agent = create_agent(
            config,
            lambda _config, client=llm: client,
            conversation_id=conversation_id,
        )

        restored_turn = agent.state.turns[-1]
        assert restored_turn.status == expected_status
        assert restored_turn.final_answer == content
        assert agent.has_pending_plan() is False
        assert agent.has_resumable_plan() is False
        assert agent.approve_plan() == "No plan is waiting for approval."
        assert llm.requests == []
        assert store.get_conversation(conversation_id).status == expected_status


def test_tool_intent_is_durable_before_callable_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    observed_intents = []

    def inspect_intent(_arguments):
        with sqlite3.connect(config.store_path) as conn:
            conn.row_factory = sqlite3.Row
            tool_call = conn.execute(
                "SELECT success, ended_at FROM conversation_tool_calls"
            ).fetchone()
            turn_row = conn.execute(
                "SELECT turn_json FROM conversation_turns"
            ).fetchone()
        turn_payload = json.loads(turn_row["turn_json"])
        observed_intents.append(
            (
                tool_call["success"],
                tool_call["ended_at"],
                turn_payload["tool_calls"][-1]["success"],
                turn_payload["tool_calls"][-1]["ended_at"],
            )
        )
        return ToolResult("inspect_intent", True, "intent was durable")

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "inspect_intent",
                    "arguments_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "done"}),
        ]
    )
    agent = create_agent(
        config,
        lambda _config: llm,
        tool_specs=[
            Tool(
                name="inspect_intent",
                description="Inspect the durable tool intent.",
                args_schema={"type": "object", "properties": {}},
                callable=inspect_intent,
            )
        ],
    )

    assert agent.run_turn("inspect the intent checkpoint") == "done"
    assert observed_intents == [(None, None, None, None)]
    assert SQLiteSessionStore(config.store_path).load_tool_calls_without_observations(
        agent.state.conversation_id,
        agent.state.turns[-1].turn_id,
    ) == []


def test_restart_blocks_unresolved_non_plan_tool_intent(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-unresolved-turn"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    turn = TurnState(user_message="mutate outside a plan", turn_id="turn-unresolved")
    turn.tool_call_count = 1
    turn.tool_calls.append(
        ToolCallRecord(
            tool_name="external_mutation",
            arguments={"value": "once"},
            iteration=1,
        )
    )
    store.save_message(
        conversation_id,
        turn_id=turn.turn_id,
        role="user",
        content=turn.user_message,
        message_key=f"{turn.turn_id}:user",
    )
    store.save_turn_snapshot(conversation_id, turn.to_dict())

    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(),
        conversation_id=conversation_id,
    )

    restored_turn = agent.state.turns[-1]
    assert restored_turn.status == "blocked"
    assert restored_turn.active_plan is None
    assert "will not replay it automatically" in (restored_turn.final_answer or "")
    assert SQLiteSessionStore(config.store_path).get_conversation(conversation_id).status == "blocked"


def test_unresolved_tool_recovery_persists_status_and_message_atomically(
    monkeypatch,
    tmp_path,
):
    store = SQLiteSessionStore(tmp_path / "atomic-recovery.sqlite")
    conversation_id = "conversation-atomic-recovery"
    store.create_conversation(conversation_id, provider="test", model="mock")
    turn = TurnState(user_message="mutate once", turn_id="turn-atomic-recovery")
    store.save_turn_snapshot(conversation_id, turn.to_dict())

    monkeypatch.setattr(
        store,
        "save_turn_snapshot",
        lambda *_args, **_kwargs: pytest.fail("recovery must not save the snapshot separately"),
    )
    monkeypatch.setattr(
        store,
        "save_message",
        lambda *_args, **_kwargs: pytest.fail("recovery must not save the message separately"),
    )

    runtime_module._block_unresolved_tool_intent(
        store,
        conversation_id,
        turn,
        [{"tool_name": "external_mutation", "iteration": 1}],
    )

    restored_turn = store.load_turns(conversation_id)[0]
    assert restored_turn.status == "blocked"
    assert any(
        message.content == restored_turn.final_answer
        for message in store.list_messages(conversation_id)
    )


def test_restart_blocks_completed_call_without_observation_missing_from_legacy_snapshot(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-legacy-unresolved-turn"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    turn = TurnState(user_message="run two mutations", turn_id="turn-legacy")
    completed_call = ToolCallRecord(
        tool_name="first_mutation",
        arguments={},
        iteration=1,
    )
    completed_call.finish(ToolResult("first_mutation", True, "done"))
    turn.tool_call_count = 2
    turn.tool_calls.append(completed_call)
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    second_call = ToolCallRecord(
        tool_name="second_mutation",
        arguments={},
        iteration=2,
    )
    second_call.finish(ToolResult("second_mutation", True, "mutation completed"))
    store.save_tool_call(
        conversation_id,
        {**second_call.to_dict(), "turn_id": turn.turn_id},
    )
    with sqlite3.connect(config.store_path) as conn:
        assert conn.execute(
            "SELECT success FROM conversation_tool_calls WHERE iteration = 2"
        ).fetchone()[0] == 1

    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(),
        conversation_id=conversation_id,
    )

    restored_turn = agent.state.turns[-1]
    assert restored_turn.status == "blocked"
    assert "second_mutation (iteration 2)" in (restored_turn.final_answer or "")


def test_restart_blocks_unresolved_tool_intent_even_if_call_row_completed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-unresolved-intent"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    plan = Plan(
        summary="Perform one external mutation.",
        steps=[
            PlanStep(
                id="1",
                title="Mutate external state",
                description="Run the mutation exactly once.",
            )
        ],
    )
    turn = TurnState(user_message="mutate once", turn_id="turn-unresolved")
    turn.wait_for_plan_approval(plan)
    turn.approve_plan()
    plan.steps[0].mark("in_progress")
    record = ToolCallRecord(
        tool_name="external_mutation",
        arguments={"value": "once"},
        iteration=1,
        plan_step_id="1",
    )
    turn.tool_call_count = 1
    turn.tool_calls.append(record)
    store.save_message(
        conversation_id,
        turn_id=turn.turn_id,
        role="user",
        content=turn.user_message,
        message_key=f"{turn.turn_id}:user",
    )
    store.save_tool_call(
        conversation_id,
        {**record.to_dict(), "turn_id": turn.turn_id, "turn": turn.to_dict()},
    )
    completed_call = record.to_dict()
    completed_call.update(
        {
            "turn_id": turn.turn_id,
            "ended_at": "2026-01-01T00:00:01+00:00",
            "success": True,
        }
    )
    store.save_tool_call(
        conversation_id,
        completed_call,
    )
    with sqlite3.connect(config.store_path) as conn:
        assert conn.execute(
            "SELECT success FROM conversation_tool_calls"
        ).fetchone()[0] == 1
    mutations = []
    agent = create_agent(
        config,
        lambda _config: FakeLLMClient(),
        conversation_id=conversation_id,
        tool_specs=[
            Tool(
                name="external_mutation",
                description="Mutate external state.",
                args_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                callable=lambda arguments: mutations.append(arguments)
                or ToolResult("external_mutation", True, "mutated"),
            )
        ],
    )

    restored_turn = agent.state.turns[-1]
    assert restored_turn.status == "blocked"
    assert restored_turn.active_plan is not None
    assert restored_turn.active_plan.steps[0].status == "blocked"
    assert "will not replay it automatically" in (restored_turn.final_answer or "")
    assert agent.has_resumable_plan() is False
    assert agent.approve_plan() == "No plan is waiting for approval."
    assert mutations == []
    assert SQLiteSessionStore(config.store_path).get_conversation(conversation_id).status == "blocked"
    assert any(
        "will not replay it automatically" in message.content
        for message in SQLiteSessionStore(config.store_path).list_messages(conversation_id)
    )


def test_create_agent_resumes_short_term_history(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    first_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "stored in session"})])
    first_agent = create_agent(config, lambda _config: first_llm)

    first_agent.run_turn("remember this short term detail")

    second_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "used resumed history"})])
    second_agent = create_agent(config, lambda _config: second_llm, conversation_id=first_agent.state.conversation_id)

    assert second_agent.state.last_usage_report is not None
    assert second_agent.state.last_usage_report["request_count"] == 1
    second_agent.run_turn("what detail did I mention?")

    resumed_prompt_messages = second_llm.requests[0]
    assert any(message["content"] == "remember this short term detail" for message in resumed_prompt_messages)
    assert any(message["content"] == "stored in session" for message in resumed_prompt_messages)
    assert second_agent.trace_logger.path == tmp_path / "traces" / f"{first_agent.state.conversation_id}.jsonl"


def test_create_agent_resumes_conversation_summary_without_covered_raw_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    llm = FakeLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "first answer"}),
            "Summary: first turn established the compaction approach.",
            json.dumps({"type": "final_answer", "content": "second answer"}),
        ]
    )
    first_agent = create_agent(config, lambda _config: llm)

    first_agent.run_turn("old context " + ("x" * 2500))
    first_agent.context_budget = ContextBudget(max_prompt_tokens=1200, response_reserve_tokens=0)
    first_agent.run_turn("latest question")

    resumed_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "resumed"})])
    resumed_agent = create_agent(config, lambda _config: resumed_llm, conversation_id=first_agent.state.conversation_id)
    resumed_agent.run_turn("continue")

    resumed_prompt = resumed_llm.requests[0][0]["content"]
    resumed_payload = json.dumps(resumed_llm.requests[0])

    assert "Summary: first turn established the compaction approach." in resumed_prompt
    assert "old context" not in resumed_payload
    assert resumed_agent.memory.summary_message_count == 2


def test_create_agent_resumes_summary_across_prompt_excluded_display_message(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-summary-plan-display"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    store.save_message(
        conversation_id,
        role="user",
        content="OLD_COVERED_QUESTION",
        message_key="covered-user",
    )
    store.save_message(
        conversation_id,
        role="assistant",
        content="Plan display that must remain user-visible only.",
        message_key="plan-display",
        metadata={"prompt_excluded": True},
    )
    store.save_message(
        conversation_id,
        role="assistant",
        content="covered answer",
        message_key="covered-assistant",
    )
    store.save_message(
        conversation_id,
        role="user",
        content="uncovered question",
        message_key="uncovered-user",
    )
    store.save_conversation_summary(
        conversation_id,
        content="The covered exchange established durable context.",
        source_message_count=2,
    )
    llm = FakeLLMClient(
        [json.dumps({"type": "final_answer", "content": "used summary"})]
    )

    agent = create_agent(
        config,
        lambda _config: llm,
        conversation_id=conversation_id,
    )
    assert agent.memory.recent() == [
        {"role": "user", "content": "uncovered question"}
    ]

    assert agent.run_turn("continue") == "used summary"
    request_payload = json.dumps(llm.requests[0])
    assert "The covered exchange established durable context." in request_payload
    assert "uncovered question" in request_payload
    assert "OLD_COVERED_QUESTION" not in request_payload
    assert "covered answer" not in request_payload
    assert "Plan display that must remain user-visible only." not in request_payload


def test_create_agent_requires_model_token_capabilities(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_MODEL", "unknown-model")
    config = load_config()

    try:
        create_agent(config, lambda _config: FakeLLMClient())
    except ValueError as exc:
        assert "No token capability metadata configured for openai/unknown-model" in str(exc)
    else:
        raise AssertionError("Expected unknown model capability metadata to fail")


def test_create_agent_registers_mcp_bridge_tools_for_local_provider(monkeypatch, tmp_path):
    (tmp_path / ".chulk").mkdir()
    (tmp_path / ".chulk" / "mcp.json").write_text(
        json.dumps({"servers": [{"label": "docs", "server_url": "https://mcp.example.com"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "local")
    calls = []

    def fake_create_bridge_tools(servers):
        calls.append([server.label for server in servers])
        return [
            Tool(
                name="mcp_docs_search_docs",
                description="Bridge docs search.",
                args_schema={"type": "object", "properties": {}, "additionalProperties": False},
                callable=lambda _arguments: ToolResult("mcp_docs_search_docs", True, "ok"),
                requires_confirmation=True,
            )
        ]

    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", fake_create_bridge_tools)
    config = load_config()

    agent = create_agent(config, lambda _config: FakeLLMClient(), tool_specs=[])

    assert agent.trace_logger.path.exists() is False
    assert SQLiteSessionStore(config.store_path).list_conversations() == []
    agent.run_turn("inspect MCP configuration")

    events = [json.loads(line) for line in agent.trace_logger.path.read_text(encoding="utf-8").splitlines()]
    assert calls == [["docs"]]
    assert agent.mcp_bridge_tool_names == ["mcp_docs_search_docs"]
    assert agent.tool_registry.get("mcp_docs_search_docs").requires_confirmation is True
    assert [event["type"] for event in events[:4]] == [
        "session_started",
        "mcp_config_loaded",
        "mcp_tool_discovery_completed",
        "turn_started",
    ]
    assert events[1]["payload"]["provider_path"] == "bridge"
    assert events[2]["payload"]["bridge_required"] is True


def test_create_agent_uses_hosted_mcp_without_bridge_for_openai_only(monkeypatch, tmp_path):
    (tmp_path / ".chulk").mkdir()
    (tmp_path / ".chulk" / "mcp.json").write_text(
        json.dumps({"servers": [{"label": "docs", "server_url": "https://mcp.example.com"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setattr(runtime_module, "create_mcp_bridge_tools", lambda servers: calls.append(servers) or [])
    config = load_config()

    class HostedOpenAIFakeLLM(FakeLLMClient):
        capabilities = LLMCapabilities(
            supports_native_tool_calling=True,
            supports_hosted_mcp_tools=True,
        )

    agent = create_agent(config, lambda _config: HostedOpenAIFakeLLM(), tool_specs=[])

    assert calls == []
    assert [server.label for server in agent.mcp_servers] == ["docs"]
    assert agent.mcp_bridge_tool_names == []


def test_agent_persists_model_tool_observation_and_final_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "calculator",
                    "arguments_json": json.dumps({"expression": "2 + 2"}),
                }
            ),
            json.dumps({"type": "final_answer", "content": "The result is 4."}),
        ]
    )
    llm.provider = "openai"
    llm.model = "gpt-4.1-mini"
    agent = create_agent(config, lambda _config: llm)

    agent.run_turn("what is 2 + 2?")

    with sqlite3.connect(config.store_path) as conn:
        conn.row_factory = sqlite3.Row
        request_count = conn.execute("SELECT count(*) AS count FROM conversation_model_requests").fetchone()["count"]
        model_request = conn.execute(
            "SELECT request_json, usage_json, cost_json FROM conversation_model_requests ORDER BY request_index LIMIT 1"
        ).fetchone()
        tool_call = conn.execute("SELECT * FROM conversation_tool_calls").fetchone()
        observation = conn.execute("SELECT * FROM conversation_observations").fetchone()
        turn = conn.execute("SELECT * FROM conversation_turns").fetchone()

    assert request_count == 2
    request_payload = json.loads(model_request["request_json"])
    usage_payload = json.loads(model_request["usage_json"])
    cost_payload = json.loads(model_request["cost_json"])
    turn_payload = json.loads(turn["turn_json"])
    assert request_payload["context_report"]["estimated_tokens"] > 0
    assert "max_output_tokens" not in request_payload
    assert usage_payload["estimated"] is True
    assert usage_payload["total_tokens"] > 0
    assert cost_payload["pricing_known"] is True
    assert cost_payload["amount"] is not None
    assert turn_payload["model_usage_totals"]["request_count"] == 2
    assert turn_payload["model_usage_totals"]["cost"]["pricing_known"] is True
    assert tool_call["tool_name"] == "calculator"
    assert tool_call["success"] == 1
    assert "4" in observation["content"]
    assert turn["final_answer"] == "The result is 4."


def test_create_agent_resumes_pending_plan_and_approves(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    plan_response = json.dumps(
        {
            "type": "plan",
            "content": None,
            "tool_name": None,
            "arguments_json": "{}",
            "plan_json": json.dumps(
                {
                    "summary": "Make a small change.",
                    "steps": [
                        {
                            "id": "1",
                            "title": "Update code",
                            "description": "Implement the requested change.",
                            "status": "pending",
                        }
                    ],
                }
            ),
        }
    )
    first_agent = create_agent(config, lambda _config: FakeLLMClient([plan_response]))

    first_agent.run_planned_turn("plan a change")

    session_store = SQLiteSessionStore(config.store_path)
    stored_plan_messages = session_store.load_recent_messages(
        first_agent.state.conversation_id,
        limit=10,
    )
    assert stored_plan_messages == [{"role": "user", "content": "plan a change"}]
    visible_history = TerminalUI(color_enabled=False).history(
        session_store.list_messages(first_agent.state.conversation_id, limit=10)
    )
    assert "Make a small change." in visible_history
    assert any(
        "Use /approve to execute this plan" in message.content
        for message in session_store.list_messages(
            first_agent.state.conversation_id,
            limit=10,
        )
    )

    resumed_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The small change was completed.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "approved work complete"}),
        ]
    )
    resumed_agent = create_agent(config, lambda _config: resumed_llm, conversation_id=first_agent.state.conversation_id)
    response = resumed_agent.approve_plan()

    assert resumed_agent.has_pending_plan() is False
    assert response == "approved work complete"
    approved_messages = resumed_llm.requests[0]
    approved_prompt = approved_messages[0]["content"]
    assert "Planning: approved for this turn." in approved_prompt
    assert "Make a small change." in approved_prompt
    assert not any("Use /approve to execute this plan" in message["content"] for message in approved_messages)
    assert not any("User approved the plan" in message["content"] for message in approved_messages)


def test_plan_display_is_excluded_from_live_execution_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(
                        {
                            "summary": "Keep display text out of model history.",
                            "steps": [
                                {
                                    "id": "1",
                                    "title": "Finish work",
                                    "description": "Complete the planned work.",
                                    "status": "pending",
                                }
                            ],
                        }
                    ),
                }
            ),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The work completed.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "done"}),
        ]
    )
    agent = create_agent(config, lambda _config: llm)

    plan_display = agent.run_planned_turn("plan this work")

    assert "Use /approve to execute this plan" in plan_display
    assert agent.memory.recent() == [{"role": "user", "content": "plan this work"}]
    assert agent.approve_plan() == "done"
    execution_requests = llm.requests[1:]
    assert execution_requests
    assert not any(
        "Use /approve to execute this plan" in message["content"]
        for request in execution_requests
        for message in request
    )


def test_resumed_low_history_limit_keeps_turn_anchor_and_tool_result(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHULK_HISTORY_LIMIT", "1")
    config = load_config()
    store = SQLiteSessionStore(config.store_path)
    conversation_id = "conversation-low-history"
    turn_id = "turn-low-history"
    store.create_conversation(
        conversation_id,
        provider=config.llm_provider,
        model=config.model,
    )
    turn = TurnState(user_message="Inspect the current state.", turn_id=turn_id)
    turn.wait_for_plan_approval(
        Plan(
            summary="Finish the inspected work.",
            steps=[
                PlanStep(
                    id="1",
                    title="Finish work",
                    description="Complete the work using the inspection result.",
                )
            ],
        )
    )
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    action_context = (
        '<executed_tool_action>\n{"tool_name":"lookup","arguments_json":"{}"}'
        "\n</executed_tool_action>"
    )
    store.save_message(
        conversation_id,
        role="user",
        content=turn.user_message,
        turn_id=turn_id,
        message_key=f"{turn_id}:user",
    )
    store.save_message(
        conversation_id,
        role="assistant",
        content=action_context,
        turn_id=turn_id,
        message_key=f"{turn_id}:tool_action:1",
    )
    store.save_message(
        conversation_id,
        role="observation",
        content="Lookup completed.",
        turn_id=turn_id,
        message_key=f"{turn_id}:observation:1",
    )
    resumed_llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The inspected work was completed.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "Work complete."}),
        ]
    )

    resumed_agent = create_agent(
        config,
        lambda _config: resumed_llm,
        conversation_id=conversation_id,
    )

    assert resumed_agent.approve_plan() == "Work complete."
    resumed_history = resumed_llm.requests[0][1:]
    assert resumed_history == [
        {"role": "user", "content": turn.user_message},
        {"role": "assistant", "content": action_context},
        {"role": "observation", "content": "Lookup completed."},
    ]


def test_cli_lists_resumes_and_shows_history(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    first_exit = main(
        ["--once", "first persisted message"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "first persisted answer"})]
        ),
    )
    store = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite")
    session_id = store.list_conversations()[0].id
    interactive_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "resumed answer"})])
    inputs = iter(["/sessions", f"/resume {session_id[:8]}", "/history", "continue", "/q"])

    second_exit = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: interactive_llm,
    )

    output = capsys.readouterr().out
    resumed_request_messages = interactive_llm.requests[0]

    assert first_exit == 0
    assert second_exit == 0
    assert "Sessions" in output
    assert "resumed session" in output
    assert "History" in output
    assert "first persisted message" in output
    assert "first persisted answer" in output
    assert any(message["content"] == "first persisted message" for message in resumed_request_messages)
    assert any(message["content"] == "first persisted answer" for message in resumed_request_messages)


def test_cli_resume_flag_starts_in_existing_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    first_exit = main(
        ["--once", "remember startup resume"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "remembered"})]
        ),
    )
    store = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite")
    session_id = store.list_conversations()[0].id
    resumed_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "resumed directly"})])
    inputs = iter(["continue directly", "/q"])

    second_exit = main(
        ["--resume", session_id[:8]],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: resumed_llm,
    )

    output = capsys.readouterr().out

    assert first_exit == 0
    assert second_exit == 0
    assert "resumed directly" in output
    assert any(message["content"] == "remember startup resume" for message in resumed_llm.requests[0])
    assert len(store.list_conversations()) == 1


def test_cli_continue_resumes_latest_nonempty_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    main(
        ["--once", "latest session marker"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "stored"})]
        ),
    )
    continued_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "continued latest"})])
    inputs = iter(["continue", "/q"])

    exit_code = main(
        ["--continue"],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: continued_llm,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "continued latest" in output
    assert any(message["content"] == "latest session marker" for message in continued_llm.requests[0])


def test_cli_continue_without_session_fails_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    exit_code = main(["--continue"], llm_client_factory=lambda _config: FakeLLMClient())

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "No persisted session is available to continue" in captured.err


def test_cli_resume_reloads_arrow_key_prompt_history(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    first_exit = main(
        ["--once", "persisted prompt for arrows"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "persisted answer"})]
        ),
    )
    store = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite")
    session_id = store.list_conversations()[0].id
    prompt_history = RecordingPromptHistory()
    monkeypatch.setattr(main_module.PromptHistory, "create", lambda enabled=True: prompt_history)
    inputs = iter([f"/resume {session_id}", "/q"])

    second_exit = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: FakeLLMClient(),
    )

    capsys.readouterr()

    assert first_exit == 0
    assert second_exit == 0
    assert prompt_history.items == ["persisted prompt for arrows"]
    assert f"/resume {session_id}" in prompt_history.added
    assert "/q" not in prompt_history.added


def test_cli_context_command_shows_latest_prompt_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["hello context", "/context", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "context ready"})]
        ),
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Context" in output
    assert "estimated" in output
    assert "sections" in output
    assert "Conversation history" in output
def test_adapter_cursor_is_durable_and_never_regresses(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")

    assert store.get_adapter_cursor("telegram") is None
    assert store.save_adapter_cursor("telegram", 42) == 42
    assert store.save_adapter_cursor("telegram", 20) == 42
    assert SQLiteSessionStore(tmp_path / "store.sqlite").get_adapter_cursor("telegram") == 42


def test_session_search_index_backfills_and_tracks_only_eligible_messages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")
    store.save_message(
        "conversation-1",
        role="user",
        content="eligible user evidence",
        message_key="eligible-user",
    )
    store.save_message(
        "conversation-1",
        role="assistant",
        content="eligible assistant evidence",
        message_key="eligible-assistant",
    )
    store.save_message(
        "conversation-1",
        role="assistant",
        content="hidden internal prompt",
        message_key="internal",
        metadata={"internal": True},
    )
    store.save_message(
        "conversation-1",
        role="user",
        content="sensitive customer token",
        message_key="sensitive",
        metadata={"sensitive": True},
    )
    store.save_message(
        "conversation-1",
        role="observation",
        content="raw tool output",
        message_key="observation",
    )

    if not store.fts_enabled:
        pytest.skip("SQLite build does not provide FTS5")
    with store._connect() as conn:
        indexed = conn.execute(
            "SELECT content FROM session_messages_fts ORDER BY rowid"
        ).fetchall()

    assert [row["content"] for row in indexed] == [
        "eligible user evidence",
        "eligible assistant evidence",
    ]

    with store._connect() as conn:
        conn.execute("DELETE FROM session_messages_fts")
    assert store.rebuild_search_index() == 2
    with store._connect() as conn:
        rebuilt = conn.execute(
            "SELECT content FROM session_messages_fts ORDER BY rowid"
        ).fetchall()
    assert [row["content"] for row in rebuilt] == [
        "eligible user evidence",
        "eligible assistant evidence",
    ]


def test_session_search_index_backfills_messages_written_before_store_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.sqlite"
    initial = SQLiteSessionStore(path)
    initial.create_conversation("conversation-1", provider="test", model="mock")
    with initial._connect() as conn:
        conn.execute("DELETE FROM session_messages_fts")
        conn.execute(
            """
            INSERT INTO conversation_messages (
                id, conversation_id, turn_id, role, content, ordinal,
                message_key, created_at, metadata
            )
            VALUES (
                'legacy-message', 'conversation-1', 'turn-1', 'user',
                'legacy indexed phrase', 1, 'legacy-key',
                '2026-07-25T10:00:00+00:00', '{}'
            )
            """
        )

    reopened = SQLiteSessionStore(path)
    if not reopened.fts_enabled:
        pytest.skip("SQLite build does not provide FTS5")
    with reopened._connect() as conn:
        row = conn.execute(
            """
            SELECT message_id
            FROM session_messages_fts
            WHERE session_messages_fts MATCH 'legacy'
            """
        ).fetchone()

    assert row["message_id"] == "legacy-message"
