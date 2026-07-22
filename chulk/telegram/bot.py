"""Telegram message loop backed by durable Chulk conversations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import Protocol

from chulk import AsyncAgent, Capabilities, FileAccess, MemoryMode, Tools
from chulk.config import Config
from chulk.sessions import SQLiteSessionStore
from chulk.telegram.client import TelegramClient, TelegramError, TelegramUpdate
from chulk.telegram.config import TelegramConfig
from chulk.tools import PermissionDecision, PermissionRequest
from chulk.tools.permissions import PermissionDecisionRecord
from chulk.tools.web_search import tavily_search_tool


LOGGER = logging.getLogger(__name__)
TELEGRAM_CHAT_METADATA_KEY = "telegram_chat_id"


class TelegramAgent(Protocol):
    """Agent operations consumed by Telegram commands."""

    @property
    def conversation_id(self) -> str: ...

    async def run(self, message: str, **kwargs: object) -> str: ...

    async def plan(self, message: str) -> str: ...

    async def approve(self) -> str: ...

    async def reject(self) -> str: ...

    async def close(self) -> None: ...


AgentFactory = Callable[[int, str | None], TelegramAgent]


class TelegramAgentBot:
    """Authorize Telegram messages and route them to per-chat Chulk agents."""

    def __init__(
        self,
        *,
        config: Config,
        telegram_config: TelegramConfig,
        client: TelegramClient,
        session_store: SQLiteSessionStore | None = None,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self.config = config
        self.telegram_config = telegram_config
        self.client = client
        self.session_store = session_store or SQLiteSessionStore(config.store_path)
        self._agent_factory = agent_factory or self._default_agent_factory
        self._agents: dict[int, TelegramAgent] = {}
        self._offset: int | None = None

    async def run_forever(self) -> None:
        """Poll until cancelled, retrying sanitized transport failures."""
        try:
            while True:
                try:
                    await self.poll_once()
                except TelegramError as exc:
                    LOGGER.warning("Telegram polling failed: %s", exc)
                    await asyncio.sleep(self.telegram_config.retry_delay_seconds)
        finally:
            await self.close()

    async def poll_once(self) -> None:
        """Fetch and process one batch of updates."""
        updates = await asyncio.to_thread(
            self.client.get_updates,
            offset=self._offset,
            timeout_seconds=self.telegram_config.poll_timeout_seconds,
        )
        self._offset = self.client.next_offset
        for update in updates:
            await self.handle_update(update)

    async def handle_update(self, update: TelegramUpdate) -> None:
        """Handle one authorized private text message."""
        if update.user_id not in self.telegram_config.allowed_user_ids:
            LOGGER.warning("Ignored Telegram message from unauthorized user id %s", update.user_id)
            return
        if update.chat_type != "private":
            await self._send(update.chat_id, "For safety, this bot only works in private chats.")
            return

        try:
            response = await self._dispatch(update.chat_id, update.text.strip())
        except Exception as exc:
            LOGGER.error("Telegram agent request failed (%s)", type(exc).__name__)
            response = "The agent could not complete that request. Check the server logs and try again."
        await self._send(update.chat_id, response)

    async def close(self) -> None:
        """Close all cached agent runtimes."""
        agents = tuple(self._agents.values())
        self._agents.clear()
        for agent in agents:
            await agent.close()

    async def _dispatch(self, chat_id: int, text: str) -> str:
        command, arguments = _parse_command(text)
        if command in {"/start", "/help"}:
            return _help_text()
        if command == "/new":
            old_agent = self._agents.pop(chat_id, None)
            if old_agent is not None:
                await old_agent.close()
            agent = self._create_agent(chat_id, None)
            return f"Started a new conversation ({agent.conversation_id[:8]})."
        agent = self._agent_for_chat(chat_id)
        if command == "/status":
            return (
                f"Provider: {self.config.llm_provider}\n"
                f"Model: {self.config.model}\n"
                f"Conversation: {agent.conversation_id[:8]}"
            )
        if command == "/plan":
            if not arguments:
                return "Usage: /plan <request>"
            return await agent.plan(arguments)
        if command == "/approve":
            return await agent.approve()
        if command == "/reject":
            return await agent.reject()
        if command is not None:
            return "Unknown command. Use /help to see available commands."
        if not text:
            return "Send a text message for the agent."
        return await agent.run(
            text,
            extension_metadata={"source": "telegram", "telegram_chat_id": chat_id},
        )

    def _agent_for_chat(self, chat_id: int) -> TelegramAgent:
        cached = self._agents.get(chat_id)
        if cached is not None:
            return cached
        conversation = self.session_store.find_conversation_by_metadata(
            TELEGRAM_CHAT_METADATA_KEY,
            chat_id,
        )
        return self._create_agent(chat_id, conversation.id if conversation is not None else None)

    def _create_agent(self, chat_id: int, conversation_id: str | None) -> TelegramAgent:
        agent = self._agent_factory(chat_id, conversation_id)
        self._agents[chat_id] = agent
        return agent

    def _default_agent_factory(self, chat_id: int, conversation_id: str | None) -> TelegramAgent:
        tool_specs: list[object] = [
            Tools.calculator,
            Tools.read_file,
            Tools.list_files,
            Tools.search_files,
            Tools.search_memory,
            Tools.list_memories,
            Tools.summarize_memories,
        ]
        if self.telegram_config.tavily_api_key is not None:
            tool_specs.append(
                tavily_search_tool(
                    self.telegram_config.tavily_api_key,
                    max_results=self.telegram_config.web_search_max_results,
                )
            )
        return AsyncAgent(
            config=self.config,
            tools=tool_specs,
            capabilities=Capabilities(
                files=FileAccess.READ,
                shell=False,
                memory=MemoryMode.READ_ONLY,
                network=self.telegram_config.tavily_api_key is not None,
                external_services=False,
                utilities=True,
            ),
            permission_callback=_telegram_permission_callback,
            conversation_id=conversation_id,
            conversation_metadata={TELEGRAM_CHAT_METADATA_KEY: chat_id},
        )

    async def _send(self, chat_id: int, text: str) -> None:
        await asyncio.to_thread(self.client.send_message, chat_id, text)


def _parse_command(text: str) -> tuple[str | None, str]:
    if not text.startswith("/"):
        return None, ""
    raw_command, _, arguments = text.partition(" ")
    command = raw_command.split("@", 1)[0].lower()
    return command, arguments.strip()


def _help_text() -> str:
    return (
        "Send any text to talk with the Chulk agent.\n\n"
        "/new — start a new conversation\n"
        "/status — show provider, model, and conversation\n"
        "/plan <request> — propose an approval plan\n"
        "/approve — approve the pending plan\n"
        "/reject — reject the pending plan\n"
        "/help — show this help"
    )


def _telegram_permission_callback(
    request: PermissionRequest,
    _record: PermissionDecisionRecord,
) -> PermissionDecision:
    """Allow only the adapter's bounded search network action."""
    if request.tool_name == "web_search":
        return PermissionDecision.ALLOW
    return PermissionDecision.DENY


__all__ = ["TELEGRAM_CHAT_METADATA_KEY", "TelegramAgentBot"]
