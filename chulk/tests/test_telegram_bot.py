from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import chulk.telegram.bot as telegram_bot_module
from chulk.config import load_config
from chulk.sessions import SessionRecorder, SQLiteSessionStore
from chulk.telegram.bot import TELEGRAM_CHAT_METADATA_KEY, TelegramAgentBot
from chulk.telegram.client import TelegramAttachment, TelegramUpdate
from chulk.telegram.config import TelegramConfig


class FakeClient:
    def __init__(self, updates: tuple[TelegramUpdate, ...] = ()) -> None:
        self.updates = updates
        self.next_offset = 100
        self.sent: list[tuple[int, str]] = []
        self.actions: list[tuple[int, str]] = []
        self.requested_offsets: list[int | None] = []
        self.commands_registered = 0
        self.downloads: list[tuple[str, int]] = []

    def get_updates(self, *, offset: int | None, timeout_seconds: int):
        assert timeout_seconds == 1
        self.requested_offsets.append(offset)
        return self.updates

    def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    def set_commands(self, _commands=()) -> None:
        self.commands_registered += 1

    def download_file(self, file_id: str, *, max_bytes: int) -> bytes:
        self.downloads.append((file_id, max_bytes))
        return b"media"


class FakeAgent:
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    async def run(self, message: str, **kwargs: object) -> str:
        self.calls.append(("run", (message, kwargs)))
        return f"answer: {message}"

    async def plan(self, message: str) -> str:
        self.calls.append(("plan", message))
        return f"plan: {message}"

    async def approve(self) -> str:
        self.calls.append(("approve", None))
        return "approved"

    async def reject(self) -> str:
        self.calls.append(("reject", None))
        return "rejected"

    async def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path):
    return load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            "CHULK_GEMINI_API_KEY": "fake",
        }
    )


def _update(text: str, *, user_id: int = 7, chat_type: str = "private") -> TelegramUpdate:
    return TelegramUpdate(
        update_id=1,
        chat_id=9,
        user_id=user_id,
        text=text,
        chat_type=chat_type,
    )


def _bot(tmp_path: Path, client: FakeClient, agents: list[FakeAgent]) -> TelegramAgentBot:
    def factory(chat_id: int, conversation_id: str | None) -> FakeAgent:
        agent = FakeAgent(conversation_id or f"new-{chat_id}-{len(agents)}")
        agents.append(agent)
        return agent

    return TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
            poll_timeout_seconds=1,
            scheduling_enabled=True,
        ),
        client=client,  # type: ignore[arg-type]
        agent_factory=factory,
    )


class FakeMediaProcessor:
    def process(self, attachment, data: bytes, *, instruction: str) -> str:
        assert attachment.file_id == "voice-1"
        assert data == b"media"
        assert instruction == "Summarize"
        return "transcribed words"


@pytest.mark.asyncio
async def test_bot_ignores_unauthorized_users_and_rejects_group_chats(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)

    await bot.handle_update(_update("hello", user_id=99))
    await bot.handle_update(_update("hello", chat_type="group"))

    assert agents == []
    assert client.sent == [(9, "For safety, this bot only works in private chats.")]
    assert client.actions == []


@pytest.mark.asyncio
async def test_bot_routes_messages_and_commands_to_one_chat_agent(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)

    await bot.handle_update(_update("hello"))
    await bot.handle_update(_update("/plan deploy safely"))
    await bot.handle_update(_update("/approve"))
    await bot.handle_update(_update("/status"))

    assert len(agents) == 1
    assert agents[0].calls[0][0] == "run"
    message, kwargs = agents[0].calls[0][1]
    assert message == "hello"
    assert kwargs["extension_metadata"] == {
        "source": "telegram",
        "telegram_chat_id": 9,
    }
    assert agents[0].calls[1:] == [("plan", "deploy safely"), ("approve", None)]
    assert client.sent[-1][1] == "Provider: gemini\nModel: gemini-test\nConversation: new-9-0"
    assert client.actions == [(9, "typing")] * 4


@pytest.mark.asyncio
async def test_new_closes_cached_agent_and_starts_new_conversation(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)

    await bot.handle_update(_update("hello"))
    await bot.handle_update(_update("/new"))

    assert len(agents) == 2
    assert agents[0].closed is True
    assert "Started a new conversation" in client.sent[-1][1]


@pytest.mark.asyncio
async def test_bot_processes_media_before_normal_agent_turn(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)
    bot._media_processor = FakeMediaProcessor()
    update = TelegramUpdate(
        update_id=4,
        chat_id=9,
        user_id=7,
        text="Summarize",
        chat_type="private",
        attachment=TelegramAttachment("voice-1", "voice", "audio/ogg"),
    )

    await bot.handle_update(update)

    message, _kwargs = agents[0].calls[0][1]
    assert message == "Summarize"
    context_sections = _kwargs["context_sections"]
    assert len(context_sections) == 1
    assert context_sections[0].content == "transcribed words"
    assert context_sections[0].source == "telegram_attachment"
    assert context_sections[0].metadata["trusted"] is False
    assert client.downloads == [("voice-1", 10 * 1024 * 1024)]


@pytest.mark.asyncio
async def test_bot_rejects_iwork_package_before_download(tmp_path: Path) -> None:
    client = FakeClient()
    bot = _bot(tmp_path, client, [])
    bot._media_processor = FakeMediaProcessor()
    update = TelegramUpdate(
        update_id=5,
        chat_id=9,
        user_id=7,
        text="Summarize",
        chat_type="private",
        attachment=TelegramAttachment(
            "pages-1",
            "document",
            "application/octet-stream",
            "proposal.pages",
        ),
    )

    await bot.handle_update(update)

    assert "Pages files are recognized" in client.sent[-1][1]
    assert "Export" in client.sent[-1][1]
    assert client.downloads == []


@pytest.mark.asyncio
async def test_bot_resumes_conversation_mapped_in_sqlite(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)
    bot.session_store.create_conversation(
        "existing-conversation",
        provider="gemini",
        model="gemini-test",
        metadata={TELEGRAM_CHAT_METADATA_KEY: 9},
    )

    await bot.handle_update(_update("continue"))

    assert agents[0].conversation_id == "existing-conversation"


@pytest.mark.asyncio
async def test_poll_once_advances_offset_and_processes_updates(tmp_path: Path) -> None:
    client = FakeClient((_update("hello"),))
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)

    await bot.poll_once()

    assert bot._offset == 100
    assert client.sent == [(9, "answer: hello")]
    assert bot.session_store.get_adapter_cursor("telegram") == 100

    restarted_client = FakeClient()
    restarted_bot = _bot(tmp_path, restarted_client, [])
    await restarted_bot.poll_once()
    assert restarted_client.requested_offsets == [100]


@pytest.mark.asyncio
async def test_bot_registers_telegram_command_menu(tmp_path: Path) -> None:
    client = FakeClient()
    bot = _bot(tmp_path, client, [])

    await bot._prepare_with_retry()

    assert client.commands_registered == 1


@pytest.mark.asyncio
async def test_bot_executes_and_delivers_due_scheduled_job(tmp_path: Path) -> None:
    client = FakeClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)
    job = bot.schedule_store.create(
        adapter="telegram",
        destination_id="9",
        prompt="scheduled research",
        next_run_at=datetime.now(timezone.utc),
    )

    await bot.run_due_jobs_once()

    assert agents[0].calls[0][0] == "run"
    message, kwargs = agents[0].calls[0][1]
    assert message == "scheduled research"
    assert kwargs["extension_metadata"]["scheduled_job_id"] == job.id
    assert client.sent == [(9, "answer: scheduled research")]
    assert bot.schedule_store.get(job.id).status == "completed"


@pytest.mark.asyncio
async def test_bot_lists_and_cancels_only_chat_scheduled_jobs(tmp_path: Path) -> None:
    client = FakeClient()
    bot = _bot(tmp_path, client, [])
    job = bot.schedule_store.create(
        adapter="telegram",
        destination_id="9",
        prompt="remind me",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )

    await bot.handle_update(_update("/reminders"))
    assert job.id[:8] in client.sent[-1][1]
    await bot.handle_update(_update(f"/cancel {job.id[:8]}"))
    assert "Cancelled" in client.sent[-1][1]
    assert bot.schedule_store.get(job.id).status == "cancelled"


@pytest.mark.asyncio
async def test_default_agent_adds_only_bounded_web_network_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def close(self) -> None:
            pass

    monkeypatch.setattr(telegram_bot_module, "AsyncAgent", CapturingAgent)
    client = FakeClient()
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
            tavily_api_key="search-secret",
            web_search_max_results=3,
            scheduling_enabled=True,
        ),
        client=client,  # type: ignore[arg-type]
    )

    agent = bot._default_agent_factory(9, None)
    try:
        names = {tool.name for tool in captured["tools"]}
        assert names == {
            "calculator",
            "read_file",
            "list_files",
            "search_files",
            "search_memory",
            "list_memories",
            "summarize_memories",
            "schedule_task",
            "current_time",
            "list_scheduled_tasks",
            "cancel_scheduled_task",
            "web_search",
        }
        capabilities = captured["capabilities"]
        assert capabilities.network is True
        assert capabilities.shell is False
    finally:
        await agent.close()


def test_session_metadata_persists_telegram_chat_mapping(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    SessionRecorder(
        store,
        "telegram-conversation",
        provider="gemini",
        model="gemini-test",
        metadata={TELEGRAM_CHAT_METADATA_KEY: 9},
    )

    conversation = store.find_conversation_by_metadata(TELEGRAM_CHAT_METADATA_KEY, 9)

    assert conversation is not None
    assert conversation.id == "telegram-conversation"


def test_scheduling_is_not_added_when_adapter_does_not_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(telegram_bot_module, "AsyncAgent", CapturingAgent)
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    bot._default_agent_factory(9, None)

    assert bot.schedule_store is None
    names = {tool.name for tool in captured["tools"]}
    assert "schedule_task" not in names
    assert "list_scheduled_tasks" not in names
