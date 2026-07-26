from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import chulk.telegram.bot as telegram_bot_module
from chulk import MemoryMode
from chulk.media import UserInput
from chulk.config import load_config
from chulk.gateway import SQLiteGatewayLedger, SQLiteGatewayRouter
from chulk.profiles import ProfileRuntimeFactory
from chulk.sessions import SessionRecorder, SQLiteSessionStore
from chulk.telegram.bot import TELEGRAM_CHAT_METADATA_KEY, TelegramAgentBot
from chulk.telegram.client import TelegramAttachment, TelegramError, TelegramUpdate
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
        self.ignored_updates: tuple[tuple[int, str], ...] = ()

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

    async def run_input(self, user_input: UserInput, **kwargs: object) -> str:
        self.calls.append(("run_input", (user_input, kwargs)))
        return f"answer: {user_input.textual_projection()}"

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


def _update(
    text: str,
    *,
    update_id: int = 1,
    user_id: int = 7,
    chat_type: str = "private",
) -> TelegramUpdate:
    return TelegramUpdate(
        update_id=update_id,
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
    def __init__(self) -> None:
        self.close_count = 0

    def process(self, attachment, data: bytes, *, instruction: str) -> str:
        assert attachment.file_id == "voice-1"
        assert data == b"media"
        assert instruction == "Summarize"
        return "transcribed words"

    def close(self) -> None:
        self.close_count += 1


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

    typed_input, _kwargs = agents[0].calls[0][1]
    assert agents[0].calls[0][0] == "run_input"
    assert typed_input.textual_projection().startswith("Summarize")
    media = typed_input.media_parts[0].media
    assert media.mime_type == "audio/ogg"
    assert media.provenance == "telegram:voice-1"
    assert "media" not in typed_input.textual_projection()
    assert client.downloads == [("voice-1", 10 * 1024 * 1024)]


@pytest.mark.asyncio
async def test_bot_closes_owned_media_processor_once(tmp_path: Path) -> None:
    processor = FakeMediaProcessor()
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
        client=FakeClient(),  # type: ignore[arg-type]
        media_processor=processor,
        owns_media_processor=True,
    )

    await bot.close()
    await bot.close()

    assert processor.close_count == 1


@pytest.mark.asyncio
async def test_media_deadline_is_sanitized_and_releases_chat_lock(tmp_path: Path) -> None:
    class BlockingAgent(FakeAgent):
        async def run_input(self, user_input: UserInput, **kwargs: object) -> str:
            await asyncio.sleep(0.05)
            return "too late"

    client = FakeClient()
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            "CHULK_GEMINI_API_KEY": "fake",
            "CHULK_LLM_TIMEOUT_SECONDS": "0.01",
        }
    )
    bot = TelegramAgentBot(
        config=config,
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
        client=client,  # type: ignore[arg-type]
        agent_factory=lambda chat_id, conversation_id: BlockingAgent(
            conversation_id or f"new-{chat_id}"
        ),
    )
    update = TelegramUpdate(
        update_id=5,
        chat_id=9,
        user_id=7,
        text="Summarize",
        chat_type="private",
        attachment=TelegramAttachment("voice-1", "voice", "audio/ogg"),
    )

    await bot.handle_update(update)

    assert client.sent[-1] == (
        9,
        "The configured media provider could not interpret that attachment.",
    )
    assert bot._chat_locks[9].locked() is False


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
    status = bot.gateway_ledger.adapter_status("telegram", "primary")
    assert status is not None
    assert status.cursor == "100"

    restarted_client = FakeClient()
    restarted_bot = _bot(tmp_path, restarted_client, [])
    await restarted_bot.poll_once()
    assert restarted_client.requested_offsets == [100]


@pytest.mark.asyncio
async def test_poll_retry_after_send_failure_does_not_repeat_agent_execution(
    tmp_path: Path,
) -> None:
    class FailingOnceClient(FakeClient):
        def __init__(self) -> None:
            super().__init__((_update("change remote state"),))
            self.fail_next_send = True

        def send_message(self, chat_id: int, text: str) -> None:
            if self.fail_next_send:
                self.fail_next_send = False
                raise TelegramError("send failed")
            super().send_message(chat_id, text)

    client = FailingOnceClient()
    agents: list[FakeAgent] = []
    bot = _bot(tmp_path, client, agents)

    with pytest.raises(TelegramError, match="send failed"):
        await bot.poll_once()
    assert len(agents[0].calls) == 1
    assert bot._offset is None
    assert bot.update_ledger.get(adapter="telegram", update_id=1).status == "executed"

    await bot.poll_once()

    assert len(agents[0].calls) == 1
    assert client.sent == [(9, "answer: change remote state")]
    assert bot.update_ledger.get(adapter="telegram", update_id=1).status == "delivered"
    assert bot._offset == 100


@pytest.mark.asyncio
async def test_restart_delivers_recorded_response_without_reexecuting_update(
    tmp_path: Path,
) -> None:
    class AlwaysFailingClient(FakeClient):
        def send_message(self, chat_id: int, text: str) -> None:
            raise TelegramError("offline")

    update = _update("one execution")
    first_agents: list[FakeAgent] = []
    first_bot = _bot(tmp_path, AlwaysFailingClient((update,)), first_agents)
    with pytest.raises(TelegramError, match="offline"):
        await first_bot.poll_once()
    assert len(first_agents[0].calls) == 1

    second_agents: list[FakeAgent] = []
    second_client = FakeClient((update,))
    restarted = _bot(tmp_path, second_client, second_agents)
    await restarted.poll_once()

    assert second_agents == []
    assert second_client.sent == [(9, "answer: one execution")]
    assert restarted.update_ledger.get(adapter="telegram", update_id=1).status == "delivered"


@pytest.mark.asyncio
async def test_multipart_retry_resumes_after_last_delivery_checkpoint(tmp_path: Path) -> None:
    class LongAgent(FakeAgent):
        async def run(self, message: str, **kwargs: object) -> str:
            self.calls.append(("run", (message, kwargs)))
            return "x" * 5_000

    class SecondPartFailsOnceClient(FakeClient):
        def __init__(self) -> None:
            super().__init__((_update("long"),))
            self.send_attempts = 0

        def send_message(self, chat_id: int, text: str) -> None:
            self.send_attempts += 1
            if self.send_attempts == 2:
                raise TelegramError("second part failed")
            super().send_message(chat_id, text)

    agents: list[LongAgent] = []

    def factory(chat_id: int, conversation_id: str | None) -> LongAgent:
        agent = LongAgent(conversation_id or f"long-{chat_id}")
        agents.append(agent)
        return agent

    client = SecondPartFailsOnceClient()
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
            poll_timeout_seconds=1,
        ),
        client=client,  # type: ignore[arg-type]
        agent_factory=factory,
    )

    with pytest.raises(TelegramError, match="second part failed"):
        await bot.poll_once()
    await bot.poll_once()

    assert len(agents) == 1
    assert len(agents[0].calls) == 1
    assert [len(text) for _chat_id, text in client.sent] == [4096, 904]
    assert client.send_attempts == 3


@pytest.mark.asyncio
async def test_concurrent_bot_does_not_execute_or_advance_an_active_update(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingAgent(FakeAgent):
        async def run(self, message: str, **kwargs: object) -> str:
            self.calls.append(("run", (message, kwargs)))
            started.set()
            await release.wait()
            return "done"

    first_agents: list[BlockingAgent] = []

    def blocking_factory(chat_id: int, conversation_id: str | None) -> BlockingAgent:
        agent = BlockingAgent(conversation_id or f"blocking-{chat_id}")
        first_agents.append(agent)
        return agent

    update = _update("only once")
    first = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
            poll_timeout_seconds=1,
        ),
        client=FakeClient((update,)),  # type: ignore[arg-type]
        agent_factory=blocking_factory,
    )
    second_agents: list[FakeAgent] = []
    second = _bot(tmp_path, FakeClient((update,)), second_agents)

    first_poll = asyncio.create_task(first.poll_once())
    await started.wait()
    await second.poll_once()

    assert second_agents == []
    assert second._offset is None
    release.set()
    await first_poll
    assert len(first_agents[0].calls) == 1


@pytest.mark.asyncio
async def test_poll_retains_unauthorized_update_as_ignored(tmp_path: Path) -> None:
    client = FakeClient((_update("no", user_id=99),))
    bot = _bot(tmp_path, client, [])

    await bot.poll_once()

    record = bot.update_ledger.get(adapter="telegram", update_id=1)
    assert record is not None
    assert record.status == "ignored"
    assert bot._offset == 100


@pytest.mark.asyncio
async def test_poll_retains_unsupported_update_as_ignored(tmp_path: Path) -> None:
    client = FakeClient()
    client.ignored_updates = ((4, "9"),)
    bot = _bot(tmp_path, client, [])

    await bot.poll_once()

    record = bot.update_ledger.get(adapter="telegram", update_id=4)
    assert record is not None
    assert record.status == "ignored"
    assert record.destination_id == "9"


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
async def test_scheduler_loop_survives_a_recoverable_iteration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = _bot(tmp_path, FakeClient(), [])
    calls = 0

    async def flaky_iteration() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        raise asyncio.CancelledError

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(bot, "run_due_jobs_once", flaky_iteration)
    monkeypatch.setattr(telegram_bot_module.asyncio, "sleep", no_sleep)

    with pytest.raises(asyncio.CancelledError):
        await bot._scheduler_loop()

    assert calls == 2


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
        assert captured["memory_namespace"] == "telegram:chat:9"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_multi_user_safe_mode_excludes_memory_tools_and_prompt_selection(
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
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7, 8}),
            long_term_memory_enabled=False,
        ),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    agent = bot._default_agent_factory(9, None)
    try:
        names = {tool.name for tool in captured["tools"]}
        assert {
            "search_memory",
            "list_memories",
            "summarize_memories",
        }.isdisjoint(names)
        assert captured["capabilities"].memory is MemoryMode.OFF
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_multi_user_agents_receive_distinct_nondefault_memory_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class CapturingAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.append(dict(kwargs))

        async def close(self) -> None:
            pass

    monkeypatch.setattr(telegram_bot_module, "AsyncAgent", CapturingAgent)
    bot = TelegramAgentBot(
        config=_config(tmp_path),
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7, 8}),
        ),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    first = bot._default_agent_factory(9, None)
    second = bot._default_agent_factory(10, None)
    try:
        assert [item["memory_namespace"] for item in captured] == [
            "telegram:chat:9",
            "telegram:chat:10",
        ]
        for item in captured:
            assert item["capabilities"].memory is MemoryMode.READ_ONLY
            assert {"search_memory", "list_memories", "summarize_memories"} <= {
                tool.name for tool in item["tools"]
            }
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_owner_route_selects_an_isolated_profile_without_startup_overwrite(
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
    config = _config(tmp_path)
    profile_factory = ProfileRuntimeFactory(config)
    profile_factory.profile_store.create_profile(
        "work",
        project_root=tmp_path,
    )
    control_path = config.runtime_dir / "control.sqlite"
    ledger = SQLiteGatewayLedger(control_path)
    router = SQLiteGatewayRouter(control_path)
    router.add_route(
        adapter="telegram",
        account_id="primary",
        principal_id="7",
        profile_id="work",
    )
    bot = TelegramAgentBot(
        config=config,
        telegram_config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
        client=FakeClient(),  # type: ignore[arg-type]
        gateway_ledger=ledger,
        gateway_router=router,
        profile_runtime_factory=profile_factory,
    )

    selected = router.list_routes()[0]
    assert selected.profile_id == "work"
    agent = bot._default_agent_factory_for_profile("work", 9, None)
    try:
        assert captured["config"].profile_id == "work"
        assert captured["memory_namespace"] == "profile:work:telegram:chat:9"
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
