"""Tests for session transcript export."""

import json
from pathlib import Path
import re

from chulk.core.state import ObservationRecord, ToolCallRecord, TurnState
from chulk.llm import LLMClient
from chulk.main import main
from chulk.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SQLiteSessionStore,
    export_session,
    render_json_transcript,
    render_markdown_transcript,
)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _seed_conversation_with_tool_call(store: SQLiteSessionStore, conversation_id: str = "conversation-1") -> None:
    store.create_conversation(conversation_id, provider="openai", model="gpt-4.1-mini")
    turn = TurnState(user_message="what is 2 + 2?", turn_id="turn-1")
    turn.tool_calls.append(
        ToolCallRecord(
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
            iteration=1,
            resolved_tool_name="calculator",
            success=True,
        )
    )
    turn.observations.append(ObservationRecord(tool_name="calculator", content="4"))
    turn.complete("The result is 4.")
    store.save_turn_snapshot(conversation_id, turn.to_dict())
    store.save_conversation_summary(conversation_id, content="User asked a math question.", source_message_count=1)


def test_render_markdown_transcript_includes_tool_call_arguments_and_result(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store)
    record = store.get_conversation("conversation-1")
    turns = store.load_turns("conversation-1")
    summary = store.load_latest_summary("conversation-1")

    markdown = render_markdown_transcript(record, turns, summary)

    assert "what is 2 + 2?" in markdown
    assert "`calculator`" in markdown
    assert '"expression": "2 + 2"' in markdown
    assert "4" in markdown
    assert "The result is 4." in markdown
    assert "User asked a math question." in markdown
    assert "{" not in markdown.split("result:")[1].split("```")[1]


def test_render_json_transcript_has_session_and_messages_shape(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store)
    record = store.get_conversation("conversation-1")
    turns = store.load_turns("conversation-1")
    summary = store.load_latest_summary("conversation-1")

    payload = render_json_transcript(record, turns, summary)

    assert payload["session"]["id"] == "conversation-1"
    assert payload["session"]["provider"] == "openai"
    assert payload["session"]["summary"]["content"] == "User asked a math question."
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["user", "tool_call", "assistant"]
    tool_call_message = payload["messages"][1]
    assert tool_call_message["tool_name"] == "calculator"
    assert tool_call_message["arguments"] == {"expression": "2 + 2"}
    assert tool_call_message["result"] == "4"


def test_export_session_writes_markdown_to_default_path(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store)
    runtime_dir = tmp_path / ".chulk"

    written_path = export_session(store, "conversation-1", runtime_dir=runtime_dir)

    assert written_path.parent == runtime_dir / "exports"
    assert written_path.name.startswith("conversation-1-")
    assert written_path.suffix == ".md"
    assert "The result is 4." in written_path.read_text(encoding="utf-8")


def test_export_session_supports_json_format_and_explicit_output_path(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store)
    out_path = tmp_path / "custom" / "transcript.json"

    written_path = export_session(
        store,
        "conversation-1",
        runtime_dir=tmp_path / ".chulk",
        format="json",
        output_path=out_path,
    )

    assert written_path == out_path
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["session"]["id"] == "conversation-1"
    assert payload["messages"][1]["tool_name"] == "calculator"


def test_export_session_accepts_unique_id_prefix(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store, conversation_id="conversation-abcdef")

    written_path = export_session(store, "conversation-a", runtime_dir=tmp_path / ".chulk")

    assert written_path.exists()


def test_export_session_raises_for_missing_session(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")

    try:
        export_session(store, "does-not-exist", runtime_dir=tmp_path / ".chulk")
        assert False, "expected SessionNotFoundError"
    except SessionNotFoundError:
        pass


def test_export_session_raises_for_ambiguous_prefix(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store, conversation_id="conversation-one")
    _seed_conversation_with_tool_call(store, conversation_id="conversation-two")

    try:
        export_session(store, "conversation-", runtime_dir=tmp_path / ".chulk")
        assert False, "expected AmbiguousSessionError"
    except AmbiguousSessionError:
        pass


def test_export_session_rejects_unknown_format(tmp_path):
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _seed_conversation_with_tool_call(store)

    try:
        export_session(store, "conversation-1", runtime_dir=tmp_path / ".chulk", format="yaml")
        assert False, "expected ValueError"
    except ValueError:
        pass


class ToolTurnFakeLLM(LLMClient):
    def __init__(self) -> None:
        self.responses = [
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

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.responses.pop(0)


def test_interactive_export_command_writes_markdown_transcript(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["what is 2 + 2?", "/export", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: ToolTurnFakeLLM(),
    )

    output = strip_ansi(capsys.readouterr().out)
    exports_dir = tmp_path / ".chulk" / "exports"
    exported_files = list(exports_dir.glob("*.md"))

    assert exit_code == 0
    assert "exported session to" in output
    assert len(exported_files) == 1
    assert "The result is 4." in exported_files[0].read_text(encoding="utf-8")


def test_interactive_export_command_supports_json_format_and_out_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    out_file = tmp_path / "transcript.json"
    inputs = iter(["what is 2 + 2?", f"/export --format json --out {out_file}", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: ToolTurnFakeLLM(),
    )

    output = strip_ansi(capsys.readouterr().out)
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert f"exported session to {out_file}" in output
    assert payload["messages"][1]["tool_name"] == "calculator"


def test_interactive_export_command_reports_unknown_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    inputs = iter(["hello", "/export does-not-exist", "/q"])

    exit_code = main(
        [],
        input_func=lambda _prompt: next(inputs),
        llm_client_factory=lambda _config: ToolTurnFakeLLM(),
    )

    output = strip_ansi(capsys.readouterr().out)

    assert exit_code == 0
    assert "No session found for id" in output
    assert "Traceback" not in output


def test_non_interactive_export_flag_defaults_to_most_recent_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    exit_code = main(["--once", "what is 2 + 2?"], llm_client_factory=lambda _config: ToolTurnFakeLLM())
    assert exit_code == 0
    capsys.readouterr()

    session = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite").list_conversations()[0]

    export_exit_code = main(["--export"])
    output = capsys.readouterr().out.strip()

    assert export_exit_code == 0
    written_path = Path(output)
    assert written_path.exists()
    assert session.id in written_path.name
    assert "The result is 4." in written_path.read_text(encoding="utf-8")


def test_non_interactive_export_flag_supports_explicit_session_and_out(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    exit_code = main(["--once", "what is 2 + 2?"], llm_client_factory=lambda _config: ToolTurnFakeLLM())
    assert exit_code == 0
    capsys.readouterr()

    session = SQLiteSessionStore(tmp_path / "chulk" / "store.sqlite").list_conversations()[0]
    out_file = tmp_path / "out.json"

    export_exit_code = main(["--export", session.id, "--format", "json", "--out", str(out_file)])
    output = capsys.readouterr().out.strip()

    assert export_exit_code == 0
    assert output == str(out_file)
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["session"]["id"] == session.id


def test_non_interactive_export_flag_reports_error_for_unknown_session(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    exit_code = main(["--export", "does-not-exist"])
    output = capsys.readouterr().out.strip()

    assert exit_code == 1
    assert output.startswith("error:")
    assert "Traceback" not in output
