"""Telegram message loop backed by durable Chulk conversations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
import logging
from typing import Protocol

from chulk import AsyncAgent, Capabilities, FileAccess, MemoryMode, Tools
from chulk.config import Config
from chulk.core.context import TurnContextSection
from chulk.sessions import SQLiteSessionStore
from chulk.telegram.client import TelegramClient, TelegramError, TelegramUpdate
from chulk.telegram.config import TelegramConfig
from chulk.telegram.media import (
    TelegramMediaError,
    TelegramMediaProcessor,
    attachment_context,
)
from chulk.tools import PermissionDecision, PermissionRequest
from chulk.tools.permissions import PermissionDecisionRecord
from chulk.tools.web_search import tavily_search_tool


LOGGER = logging.getLogger(__name__)
TELEGRAM_CHAT_METADATA_KEY = "telegram_chat_id"
TELEGRAM_CURSOR_NAME = "telegram"
TELEGRAM_TYPING_REFRESH_SECONDS = 4.0


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
        media_processor: TelegramMediaProcessor | None = None,
    ) -> None:
        self.config = config
        self.telegram_config = telegram_config
        self.client = client
        self.session_store = session_store or SQLiteSessionStore(config.store_path)
        self._agent_factory = agent_factory or self._default_agent_factory
        self._media_processor = media_processor
        self._agents: dict[int, TelegramAgent] = {}
        self._offset = self.session_store.get_adapter_cursor(TELEGRAM_CURSOR_NAME)

    async def run_forever(self) -> None:
        """Poll until cancelled, retrying sanitized transport failures."""
        try:
            await self._prepare_with_retry()
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
        for update in updates:
            if self._offset is not None and update.update_id < self._offset:
                continue
            await self.handle_update(update)
            self._save_offset(update.update_id + 1)
        if self.client.next_offset is not None:
            self._save_offset(self.client.next_offset)

    async def handle_update(self, update: TelegramUpdate) -> None:
        """Handle one authorized private text message."""
        if update.user_id not in self.telegram_config.allowed_user_ids:
            LOGGER.warning("Ignored Telegram message from unauthorized user id %s", update.user_id)
            return
        if update.chat_type != "private":
            await self._send(update.chat_id, "For safety, this bot only works in private chats.")
            return

        await self._send_typing_once(update.chat_id)
        typing_task = asyncio.create_task(self._refresh_typing(update.chat_id))
        try:
            try:
                text = update.text.strip()
                context_sections: list[TurnContextSection] | None = None
                if update.attachment is not None:
                    text, context = await self._process_attachment(update)
                    context_sections = [context]
                response = await self._dispatch(
                    update.chat_id,
                    text,
                    context_sections=context_sections,
                )
            except TelegramMediaError as exc:
                LOGGER.warning("Telegram media request rejected (%s)", type(exc).__name__)
                response = str(exc)
            except Exception as exc:
                LOGGER.error("Telegram agent request failed (%s)", type(exc).__name__)
                response = "The agent could not complete that request. Check the server logs and try again."
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task
        await self._send(update.chat_id, response)

    async def _process_attachment(
        self,
        update: TelegramUpdate,
    ) -> tuple[str, TurnContextSection]:
        attachment = update.attachment
        if attachment is None:
            raise TelegramMediaError("No supported attachment was found in this message.")
        if self._media_processor is None:
            raise TelegramMediaError(
                "Attachment processing is unavailable for the configured model provider."
            )
        try:
            data = await asyncio.to_thread(
                self.client.download_file,
                attachment.file_id,
                max_bytes=self.telegram_config.max_attachment_bytes,
            )
            extracted = await asyncio.to_thread(
                self._media_processor.process,
                attachment,
                data,
                instruction=update.text.strip(),
            )
        except TelegramError as exc:
            if "size limit" in str(exc):
                raise TelegramMediaError(
                    "That attachment is larger than the configured download limit."
                ) from exc
            raise TelegramMediaError(
                "Telegram could not download that attachment. Please try sending it again."
            ) from exc
        except TelegramMediaError:
            raise
        except Exception as exc:
            raise TelegramMediaError(
                "The configured media provider could not interpret that attachment."
            ) from exc
        instruction = update.text.strip() or "Respond to the attachment."
        return instruction, attachment_context(attachment, extracted)

    async def close(self) -> None:
        """Close all cached agent runtimes."""
        agents = tuple(self._agents.values())
        self._agents.clear()
        for agent in agents:
            await agent.close()

    async def _dispatch(
        self,
        chat_id: int,
        text: str,
        *,
        context_sections: list[TurnContextSection] | None = None,
    ) -> str:
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
            context_sections=context_sections,
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

    async def _prepare_with_retry(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.client.set_commands)
                return
            except TelegramError as exc:
                LOGGER.warning("Telegram command registration failed: %s", exc)
                await asyncio.sleep(self.telegram_config.retry_delay_seconds)

    async def _send_typing_once(self, chat_id: int) -> None:
        try:
            await asyncio.to_thread(self.client.send_chat_action, chat_id, "typing")
        except TelegramError as exc:
            LOGGER.warning("Telegram typing indicator failed: %s", exc)

    async def _refresh_typing(self, chat_id: int) -> None:
        while True:
            await asyncio.sleep(TELEGRAM_TYPING_REFRESH_SECONDS)
            await self._send_typing_once(chat_id)

    def _save_offset(self, offset: int) -> None:
        self._offset = self.session_store.save_adapter_cursor(TELEGRAM_CURSOR_NAME, offset)


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
