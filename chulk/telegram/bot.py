"""Telegram message loop backed by durable Chulk conversations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import logging
from typing import Protocol

from chulk import AsyncAgent, Capabilities, FileAccess, MemoryMode, Tools
from chulk.config import Config
from chulk.core.context import TurnContextSection
from chulk.scheduling import SQLiteScheduleStore
from chulk.scheduling.tools import format_jobs, scheduled_job_tools
from chulk.sessions import SQLiteSessionStore
from chulk.telegram.client import (
    TELEGRAM_COMMANDS,
    TELEGRAM_SCHEDULING_COMMANDS,
    TelegramClient,
    TelegramError,
    TelegramUpdate,
    split_message,
)
from chulk.telegram.config import TelegramConfig
from chulk.telegram.ledger import SQLiteAdapterUpdateLedger
from chulk.telegram.media import (
    TelegramMediaError,
    TelegramMediaProcessor,
    attachment_context,
    validate_attachment,
)
from chulk.tools import PermissionDecision, PermissionRequest
from chulk.tools.permissions import PermissionDecisionRecord
from chulk.tools.web_search import tavily_search_tool


LOGGER = logging.getLogger(__name__)
TELEGRAM_CHAT_METADATA_KEY = "telegram_chat_id"
TELEGRAM_CURSOR_NAME = "telegram"
TELEGRAM_TYPING_REFRESH_SECONDS = 4.0
SCHEDULE_LEASE_SECONDS = 120
SCHEDULE_LEASE_RENEW_SECONDS = 40.0
UPDATE_EXECUTION_LEASE_SECONDS = 300
UPDATE_EXECUTION_RENEW_SECONDS = 100.0
UPDATE_LEDGER_RETENTION_DAYS = 30


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
        schedule_store: SQLiteScheduleStore | None = None,
        media_processor: TelegramMediaProcessor | None = None,
        update_ledger: SQLiteAdapterUpdateLedger | None = None,
    ) -> None:
        self.config = config
        self.telegram_config = telegram_config
        self.client = client
        self.session_store = session_store or SQLiteSessionStore(config.store_path)
        self.schedule_store = (
            schedule_store or SQLiteScheduleStore(config.store_path)
            if telegram_config.scheduling_enabled
            else None
        )
        self._agent_factory = agent_factory or self._default_agent_factory
        self._media_processor = media_processor
        self.update_ledger = update_ledger or SQLiteAdapterUpdateLedger(config.store_path)
        self._agents: dict[int, TelegramAgent] = {}
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._offset = self.session_store.get_adapter_cursor(TELEGRAM_CURSOR_NAME)

    async def run_forever(self) -> None:
        """Poll until cancelled, retrying sanitized transport failures."""
        scheduler_task: asyncio.Task[None] | None = None
        try:
            await self._prepare_with_retry()
            if self.schedule_store is not None:
                scheduler_task = asyncio.create_task(self._scheduler_loop())
            while True:
                try:
                    await self.poll_once()
                except TelegramError as exc:
                    LOGGER.warning("Telegram polling failed: %s", exc)
                    await asyncio.sleep(self.telegram_config.retry_delay_seconds)
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await scheduler_task
            await self.close()

    async def poll_once(self) -> None:
        """Fetch and process one batch of updates."""
        updates = await asyncio.to_thread(
            self.client.get_updates,
            offset=self._offset,
            timeout_seconds=self.telegram_config.poll_timeout_seconds,
        )
        for update_id, destination_id in getattr(self.client, "ignored_updates", ()):
            await asyncio.to_thread(
                self.update_ledger.ignore,
                adapter=TELEGRAM_CURSOR_NAME,
                update_id=update_id,
                destination_id=destination_id,
            )
        all_recorded = True
        for update in updates:
            if self._offset is not None and update.update_id < self._offset:
                continue
            if not await self._handle_polled_update(update):
                all_recorded = False
                break
            self._save_offset(update.update_id + 1)
        if all_recorded and self.client.next_offset is not None:
            self._save_offset(self.client.next_offset)
        await asyncio.to_thread(
            self.update_ledger.purge_terminal,
            before=datetime.now(timezone.utc) - timedelta(days=UPDATE_LEDGER_RETENTION_DAYS),
        )

    async def handle_update(self, update: TelegramUpdate) -> None:
        """Handle one update directly without changing the polling ledger."""
        response = await self._response_for_update(update)
        if response is not None:
            await self._send(update.chat_id, response)

    async def _handle_polled_update(self, update: TelegramUpdate) -> bool:
        """Execute a polled update once and retry only its durable response."""
        if update.user_id not in self.telegram_config.allowed_user_ids:
            await asyncio.to_thread(
                self.update_ledger.ignore,
                adapter=TELEGRAM_CURSOR_NAME,
                update_id=update.update_id,
                destination_id=str(update.chat_id),
            )
            LOGGER.warning("Ignored Telegram message from unauthorized user id %s", update.user_id)
            return True

        claim = await asyncio.to_thread(
            self.update_ledger.begin_execution,
            adapter=TELEGRAM_CURSOR_NAME,
            update_id=update.update_id,
            destination_id=str(update.chat_id),
            lease_seconds=UPDATE_EXECUTION_LEASE_SECONDS,
        )
        if claim.should_execute:
            assert claim.execution_token is not None
            renewal_task = asyncio.create_task(
                self._renew_update_execution(update.update_id, claim.execution_token)
            )
            try:
                response = await self._response_for_update(update)
                assert response is not None
                recorded = await asyncio.to_thread(
                    self.update_ledger.record_response,
                    adapter=TELEGRAM_CURSOR_NAME,
                    update_id=update.update_id,
                    execution_token=claim.execution_token,
                    response_parts=split_message(response),
                )
            finally:
                renewal_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renewal_task
            if not recorded:
                LOGGER.warning("Telegram update execution lost its durable claim")
                return False
        elif claim.record.status == "processing":
            return False
        elif claim.record.status in {"delivered", "ignored"}:
            return True

        return await self._deliver_polled_response(update.update_id)

    async def _response_for_update(self, update: TelegramUpdate) -> str | None:
        if update.user_id not in self.telegram_config.allowed_user_ids:
            LOGGER.warning("Ignored Telegram message from unauthorized user id %s", update.user_id)
            return None
        if update.chat_type != "private":
            return "For safety, this bot only works in private chats."
        await self._send_typing_once(update.chat_id)
        typing_task = asyncio.create_task(self._refresh_typing(update.chat_id))
        try:
            try:
                async with self._chat_lock(update.chat_id):
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
        return response

    async def _renew_update_execution(self, update_id: int, execution_token: str) -> None:
        while True:
            await asyncio.sleep(UPDATE_EXECUTION_RENEW_SECONDS)
            renewed = await asyncio.to_thread(
                self.update_ledger.renew_execution,
                adapter=TELEGRAM_CURSOR_NAME,
                update_id=update_id,
                execution_token=execution_token,
                lease_seconds=UPDATE_EXECUTION_LEASE_SECONDS,
            )
            if not renewed:
                LOGGER.warning("Telegram update execution renewal lost its claim")
                return

    async def _deliver_polled_response(self, update_id: int) -> bool:
        claimed = await asyncio.to_thread(
            self.update_ledger.claim_delivery,
            adapter=TELEGRAM_CURSOR_NAME,
            update_id=update_id,
        )
        if claimed is None:
            current = await asyncio.to_thread(
                self.update_ledger.get,
                adapter=TELEGRAM_CURSOR_NAME,
                update_id=update_id,
            )
            return current is not None and current.status in {"delivered", "ignored"}
        assert claimed.delivery_token is not None
        try:
            for part_index in range(
                claimed.next_response_part,
                len(claimed.response_parts),
            ):
                await self._send(
                    int(claimed.destination_id),
                    claimed.response_parts[part_index],
                )
                checkpointed = await asyncio.to_thread(
                    self.update_ledger.mark_response_part_delivered,
                    adapter=TELEGRAM_CURSOR_NAME,
                    update_id=update_id,
                    delivery_token=claimed.delivery_token,
                    expected_part=part_index,
                )
                if not checkpointed:
                    LOGGER.warning("Telegram response delivery lost its durable claim")
                    return False
        except TelegramError as exc:
            await asyncio.to_thread(
                self.update_ledger.release_delivery,
                adapter=TELEGRAM_CURSOR_NAME,
                update_id=update_id,
                delivery_token=claimed.delivery_token,
                error=type(exc).__name__,
            )
            raise
        return True

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
        attachment = validate_attachment(attachment)
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
        if command == "/reminders":
            if self.schedule_store is None:
                return "Scheduling is disabled for this Telegram agent."
            return format_jobs(
                self.schedule_store.list(adapter="telegram", destination_id=str(chat_id)),
                timezone_name=self.telegram_config.timezone,
            )
        if command == "/cancel":
            if self.schedule_store is None:
                return "Scheduling is disabled for this Telegram agent."
            if not arguments:
                return "Usage: /cancel <task-id>"
            jobs = self.schedule_store.list(adapter="telegram", destination_id=str(chat_id))
            matches = [job for job in jobs if job.id.startswith(arguments)]
            if len(matches) != 1:
                return "Task id is missing or ambiguous. Use /reminders."
            self.schedule_store.cancel(
                matches[0].id,
                adapter="telegram",
                destination_id=str(chat_id),
            )
            return f"Cancelled scheduled task {matches[0].id[:8]}."
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
        ]
        if self.telegram_config.long_term_memory_enabled:
            tool_specs.extend(
                (
                    Tools.search_memory,
                    Tools.list_memories,
                    Tools.summarize_memories,
                )
            )
        if self.schedule_store is not None:
            tool_specs.extend(
                scheduled_job_tools(
                    self.schedule_store,
                    adapter="telegram",
                    destination_id=str(chat_id),
                    timezone_name=self.telegram_config.timezone,
                )
            )
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
                memory=(
                    MemoryMode.READ_ONLY
                    if self.telegram_config.long_term_memory_enabled
                    else MemoryMode.OFF
                ),
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
                commands = TELEGRAM_COMMANDS
                if self.schedule_store is not None:
                    commands += TELEGRAM_SCHEDULING_COMMANDS
                await asyncio.to_thread(self.client.set_commands, commands)
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

    def _chat_lock(self, chat_id: int) -> asyncio.Lock:
        return self._chat_locks.setdefault(chat_id, asyncio.Lock())

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self.run_due_jobs_once()
            except Exception as exc:
                LOGGER.error("Telegram scheduler iteration failed (%s)", type(exc).__name__)
            await asyncio.sleep(self.telegram_config.scheduler_poll_seconds)

    async def run_due_jobs_once(self) -> None:
        """Claim and execute one bounded batch of due Telegram jobs."""
        if self.schedule_store is None:
            return
        store = self.schedule_store
        jobs = await asyncio.to_thread(
            store.claim_due,
            adapter="telegram",
            lease_seconds=SCHEDULE_LEASE_SECONDS,
        )
        for job in jobs:
            if job.claim_token is None:
                LOGGER.error("Scheduled Telegram job has no claim token")
                continue
            renewal_task = asyncio.create_task(
                self._renew_schedule_lease(job.id, job.claim_token)
            )
            try:
                chat_id = int(job.destination_id)
                async with self._chat_lock(chat_id):
                    response = await self._agent_for_chat(chat_id).run(
                        job.prompt,
                        extension_metadata={
                            "source": "telegram_schedule",
                            "telegram_chat_id": chat_id,
                            "scheduled_job_id": job.id,
                        },
                    )
                await self._send(chat_id, response)
                completed = await asyncio.to_thread(store.complete, job.id, job.claim_token)
                if not completed:
                    LOGGER.warning("Scheduled Telegram job completion lost its claim")
            except Exception as exc:
                LOGGER.error("Scheduled Telegram job failed (%s)", type(exc).__name__)
                await asyncio.to_thread(
                    store.fail,
                    job.id,
                    job.claim_token,
                    type(exc).__name__,
                )
            finally:
                renewal_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renewal_task

    async def _renew_schedule_lease(self, job_id: str, claim_token: str) -> None:
        if self.schedule_store is None:
            return
        while True:
            await asyncio.sleep(SCHEDULE_LEASE_RENEW_SECONDS)
            renewed = await asyncio.to_thread(
                self.schedule_store.renew_lease,
                job_id,
                claim_token,
                lease_seconds=SCHEDULE_LEASE_SECONDS,
            )
            if not renewed:
                LOGGER.warning("Scheduled Telegram job lease renewal lost its claim")
                return


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
        "/reminders — list active scheduled tasks\n"
        "/cancel <task-id> — cancel a scheduled task\n"
        "/help — show this help"
    )


def _telegram_permission_callback(
    request: PermissionRequest,
    _record: PermissionDecisionRecord,
) -> PermissionDecision:
    """Allow only the adapter's bounded search network action."""
    if request.tool_name in {"web_search", "schedule_task", "cancel_scheduled_task"}:
        return PermissionDecision.ALLOW
    return PermissionDecision.DENY


__all__ = ["TELEGRAM_CHAT_METADATA_KEY", "TelegramAgentBot"]
