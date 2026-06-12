"""Tests for durable conversation sessions."""

import json
import sqlite3

import src.main as main_module
from src.config import load_config
from src.core.state import Plan, PlanStep, TurnState
from src.llm import LLMClient
from src.main import create_agent, main
from src.sessions import SQLiteSessionStore


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


def test_create_agent_resumes_short_term_history(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    config = load_config()
    first_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "stored in session"})])
    first_agent = create_agent(config, lambda _config: first_llm)

    first_agent.run_turn("remember this short term detail")

    second_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "used resumed history"})])
    second_agent = create_agent(config, lambda _config: second_llm, conversation_id=first_agent.state.conversation_id)

    second_agent.run_turn("what detail did I mention?")

    resumed_prompt_messages = second_llm.requests[0]
    assert any(message["content"] == "remember this short term detail" for message in resumed_prompt_messages)
    assert any(message["content"] == "stored in session" for message in resumed_prompt_messages)
    assert second_agent.trace_logger.path == tmp_path / "traces" / f"{first_agent.state.conversation_id}.jsonl"


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
    agent = create_agent(config, lambda _config: llm)

    agent.run_turn("what is 2 + 2?")

    with sqlite3.connect(config.store_path) as conn:
        conn.row_factory = sqlite3.Row
        request_count = conn.execute("SELECT count(*) AS count FROM conversation_model_requests").fetchone()["count"]
        request_json = conn.execute(
            "SELECT request_json FROM conversation_model_requests ORDER BY request_index LIMIT 1"
        ).fetchone()["request_json"]
        tool_call = conn.execute("SELECT * FROM conversation_tool_calls").fetchone()
        observation = conn.execute("SELECT * FROM conversation_observations").fetchone()
        turn = conn.execute("SELECT * FROM conversation_turns").fetchone()

    assert request_count == 2
    request_payload = json.loads(request_json)
    assert request_payload["context_report"]["estimated_tokens"] > 0
    assert request_payload["max_output_tokens"] > 0
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

    resumed_llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "approved work complete"})])
    resumed_agent = create_agent(config, lambda _config: resumed_llm, conversation_id=first_agent.state.conversation_id)
    response = resumed_agent.approve_plan()

    assert resumed_agent.has_pending_plan() is False
    assert response == "approved work complete"
    approved_prompt = resumed_llm.requests[0][0]["content"]
    assert "Planning: approved for this turn." in approved_prompt
    assert "Make a small change." in approved_prompt


def test_cli_lists_resumes_and_shows_history(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    first_exit = main(
        ["--once", "first persisted message"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "first persisted answer"})]
        ),
    )
    store = SQLiteSessionStore(tmp_path / "src" / "store.sqlite")
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


def test_cli_resume_reloads_arrow_key_prompt_history(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    first_exit = main(
        ["--once", "persisted prompt for arrows"],
        llm_client_factory=lambda _config: FakeLLMClient(
            [json.dumps({"type": "final_answer", "content": "persisted answer"})]
        ),
    )
    store = SQLiteSessionStore(tmp_path / "src" / "store.sqlite")
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
