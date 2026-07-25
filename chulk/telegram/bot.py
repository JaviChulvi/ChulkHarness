"""Telegram message loop backed by durable Chulk conversations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
import logging
from pathlib import Path
from typing import Protocol

from chulk import AsyncAgent, Capabilities, FileAccess, MemoryMode, Tools
from chulk.config import Config
from chulk.core.context import TurnContextSection
from chulk.gateway import (
    ChannelCommandSpec,
    DeliveryTarget,
    GatewayLimits,
    GatewayRuntime,
    InboundEnvelope,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    SQLiteGatewayRouter,
    TextPart,
    adopt_legacy_telegram_state,
    parse_channel_command,
    shared_command_help,
)
from chulk.llm.lifecycle import aclose_resources
from chulk.profiles import ProfileRuntimeFactory
from chulk.scheduling import SQLiteScheduleStore
from chulk.scheduling.tools import format_jobs, scheduled_job_tools
from chulk.sessions import SQLiteSessionStore
from chulk.telegram.client import (
    TELEGRAM_COMMANDS,
    TELEGRAM_SCHEDULING_COMMANDS,
    TelegramAttachment,
    TelegramClient,
    TelegramError,
    TelegramUpdate,
    split_message,
)
from chulk.telegram.config import TelegramConfig
from chulk.telegram.adapter import TelegramChannelAdapter
from chulk.telegram.gateway_ledger import TelegramGatewayLedgerView
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
        gateway_ledger: SQLiteGatewayLedger | None = None,
        gateway_router: SQLiteGatewayRouter | None = None,
        control_db_path: Path | str | None = None,
        profile_runtime_factory: ProfileRuntimeFactory | None = None,
        owns_media_processor: bool = False,
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
        self._custom_agent_factory = agent_factory
        self._profile_runtime_factory = profile_runtime_factory
        self._profile_configs: dict[str, Config] = {config.profile_id: config}
        self._session_stores: dict[str, SQLiteSessionStore] = {
            config.profile_id: self.session_store
        }
        self._schedule_stores: dict[str, SQLiteScheduleStore] = {}
        if self.schedule_store is not None:
            self._schedule_stores[config.profile_id] = self.schedule_store
        self._media_processor = media_processor
        self._owns_media_processor = owns_media_processor
        legacy_ledger = update_ledger or SQLiteAdapterUpdateLedger(config.store_path)
        control_path = (
            gateway_ledger.db_path
            if gateway_ledger is not None
            else (
                config.runtime_dir / "control.sqlite"
                if control_db_path is None
                else control_db_path
            )
        )
        adopt_legacy_telegram_state(
            control_db_path=control_path,
            profile_db_path=legacy_ledger.db_path,
            profile_id=config.profile_id,
        )
        self.gateway_ledger = gateway_ledger or SQLiteGatewayLedger(control_path)
        self.gateway_router = gateway_router or SQLiteGatewayRouter(control_path)
        existing_routes = self.gateway_router.list_routes(include_disabled=True)
        for user_id in telegram_config.allowed_user_ids:
            if not any(
                route.adapter == TELEGRAM_CURSOR_NAME
                and route.account_id == "primary"
                and route.principal_id == str(user_id)
                and route.destination_id is None
                and route.thread_id is None
                for route in existing_routes
            ):
                self.gateway_router.add_route(
                    adapter=TELEGRAM_CURSOR_NAME,
                    account_id="primary",
                    principal_id=str(user_id),
                    profile_id=config.profile_id,
                )
        self.channel_adapter = TelegramChannelAdapter(
            client=client,
            config=telegram_config,
            ledger=self.gateway_ledger,
            defer_cursor=True,
        )
        self.gateway_runtime = GatewayRuntime(
            ledger=self.gateway_ledger,
            router=self.gateway_router,
            adapters=(self.channel_adapter,),
            executor=self._execute_gateway_envelope,
            limits=GatewayLimits(global_concurrency=4, profile_concurrency=1),
            propagate_delivery_errors=True,
            delivery_retry_delay_seconds=0,
        )
        self.update_ledger = TelegramGatewayLedgerView(self.gateway_ledger)
        self._agents: dict[tuple[str, int], TelegramAgent] = {}
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._gateway_stop_requested = False
        adapter_status = self.gateway_ledger.adapter_status(
            TELEGRAM_CURSOR_NAME,
            "primary",
        )
        self._offset = (
            int(adapter_status.cursor)
            if adapter_status is not None and adapter_status.cursor is not None
            else None
        )

    async def run_forever(self) -> None:
        """Poll until cancelled, retrying sanitized transport failures."""
        scheduler_task: asyncio.Task[None] | None = None
        try:
            await self._prepare_with_retry()
            if self.schedule_store is not None:
                scheduler_task = asyncio.create_task(self._scheduler_loop())
            while True:
                try:
                    owned_adapter = await self.poll_once()
                    if self._gateway_stop_requested:
                        return
                    if not owned_adapter:
                        await asyncio.sleep(
                            self.telegram_config.retry_delay_seconds
                        )
                except TelegramError as exc:
                    LOGGER.warning("Telegram polling failed: %s", exc)
                    if self._gateway_stop_requested:
                        return
                    await asyncio.sleep(self.telegram_config.retry_delay_seconds)
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await scheduler_task
            await self.close()

    async def poll_once(self) -> bool:
        """Fetch and process one batch of updates."""
        try:
            envelopes = await self.channel_adapter.poll_once()
        except RuntimeError as exc:
            if "already running" in str(exc):
                return False
            raise
        try:
            all_complete = True
            for envelope in envelopes:
                event_id = int(envelope.event_id)
                if self._offset is not None and event_id < self._offset:
                    continue
                if (
                    envelope.scope.value == "group"
                    and envelope.identity.principal_id
                    in {str(value) for value in self.telegram_config.allowed_user_ids}
                ):
                    routes = self.gateway_router.list_routes(include_disabled=True)
                    exact_group = next(
                        (
                            route
                            for route in routes
                            if route.adapter == TELEGRAM_CURSOR_NAME
                            and route.account_id == "primary"
                            and route.principal_id
                            == envelope.identity.principal_id
                            and route.destination_id == envelope.destination_id
                            and route.thread_id is None
                        ),
                        None,
                    )
                    principal_route = next(
                        (
                            route
                            for route in routes
                            if route.enabled
                            and route.adapter == TELEGRAM_CURSOR_NAME
                            and route.account_id == "primary"
                            and route.principal_id
                            == envelope.identity.principal_id
                            and route.destination_id is None
                            and route.thread_id is None
                        ),
                        None,
                    )
                    if exact_group is None:
                        self.gateway_router.add_route(
                            adapter=TELEGRAM_CURSOR_NAME,
                            account_id="primary",
                            principal_id=envelope.identity.principal_id,
                            destination_id=envelope.destination_id,
                            profile_id=(
                                principal_route.profile_id
                                if principal_route is not None
                                else self.config.profile_id
                            ),
                        )
                await self.gateway_runtime.accept(
                    self.channel_adapter,
                    envelope,
                )
                await self.gateway_runtime.run_once()
                record = self.gateway_ledger.find_inbox(
                    adapter=TELEGRAM_CURSOR_NAME,
                    account_id="primary",
                    idempotency_key=envelope.idempotency_key,
                )
                if record is None or not self.gateway_ledger.inbox_complete(record.id):
                    all_complete = False
                    break
                await self.channel_adapter.commit_acknowledgements()
                self._refresh_offset()
            if all_complete and self.client.next_offset is not None:
                if not envelopes:
                    await self.gateway_runtime.run_once()
                await self.channel_adapter.commit_cursor(self.client.next_offset)
                self._refresh_offset()
        finally:
            status = self.gateway_ledger.adapter_status(
                TELEGRAM_CURSOR_NAME,
                "primary",
            )
            if status is not None and status.stop_requested:
                self._gateway_stop_requested = True
            await self.channel_adapter.close()
        return True

    async def handle_update(self, update: TelegramUpdate) -> None:
        """Handle one update directly without changing the polling ledger."""
        response = await self._response_for_update(
            update,
            profile_id=self.config.profile_id,
        )
        if response is not None:
            await self._send(update.chat_id, response)

    async def _response_for_update(
        self,
        update: TelegramUpdate,
        *,
        gateway_authorized: bool = False,
        profile_id: str | None = None,
    ) -> str | None:
        if (
            not gateway_authorized
            and update.user_id not in self.telegram_config.allowed_user_ids
        ):
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
                        profile_id=profile_id or self.config.profile_id,
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
            extracted = await asyncio.wait_for(
                asyncio.to_thread(
                    self._media_processor.process,
                    attachment,
                    data,
                    instruction=update.text.strip(),
                ),
                timeout=self.config.llm_timeout_seconds,
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
        await self.channel_adapter.close()
        agents = tuple(self._agents.values())
        self._agents.clear()
        for agent in agents:
            await agent.close()
        if self._owns_media_processor and self._media_processor is not None:
            await aclose_resources((self._media_processor,))
            self._owns_media_processor = False

    async def _execute_gateway_envelope(
        self,
        profile_id: str,
        envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        update = _telegram_update_from_envelope(envelope)
        response = await self._response_for_update(
            update,
            gateway_authorized=True,
            profile_id=profile_id,
        )
        if response is None:
            raise RuntimeError("authorized Telegram execution produced no response")
        conversation = self._session_store_for(profile_id).find_conversation_by_metadata(
            TELEGRAM_CHAT_METADATA_KEY,
            update.chat_id,
        )
        conversation_id = (
            conversation.id
            if conversation is not None
            else f"telegram:{update.chat_id}"
        )
        parts = split_message(response)
        return tuple(
            OutboundEnvelope(
                profile_id=profile_id,
                conversation_id=conversation_id,
                target=DeliveryTarget(
                    TELEGRAM_CURSOR_NAME,
                    "primary",
                    str(update.chat_id),
                ),
                text=part,
                reply_to_event_id=envelope.event_id,
                sequence=index,
                final=index == len(parts) - 1,
            )
            for index, part in enumerate(parts)
        )

    async def _dispatch(
        self,
        chat_id: int,
        text: str,
        *,
        profile_id: str,
        context_sections: list[TurnContextSection] | None = None,
    ) -> str:
        runtime_config = self._config_for_profile(profile_id)
        schedule_store = self._schedule_store_for(profile_id)
        parsed = parse_channel_command(text)
        command = f"/{parsed.name}" if parsed is not None else None
        arguments = parsed.arguments if parsed is not None else ""
        if command in {"/start", "/help"}:
            return _help_text()
        if command == "/new":
            old_agent = self._agents.pop((profile_id, chat_id), None)
            if old_agent is not None:
                await old_agent.close()
            agent = self._create_agent(profile_id, chat_id, None)
            return f"Started a new conversation ({agent.conversation_id[:8]})."
        agent = self._agent_for_chat(profile_id, chat_id)
        if command == "/status":
            return (
                f"Provider: {runtime_config.llm_provider}\n"
                f"Model: {runtime_config.model}\n"
                f"Conversation: {agent.conversation_id[:8]}"
            )
        if command == "/stop":
            stopped = self._agents.pop((profile_id, chat_id), None)
            if stopped is None:
                return "No active conversation work to stop."
            await stopped.close()
            return f"Stopped conversation {stopped.conversation_id[:8]}."
        if command == "/model":
            if arguments:
                return "Model switching is not available in this channel."
            model_profile_id = (
                self._profile_runtime_factory.resolve(
                    profile_id
                ).profile.model_profile_id
                if self._profile_runtime_factory is not None
                else "default"
            )
            return (
                f"Model profile: {model_profile_id}\n"
                f"Provider: {runtime_config.llm_provider}\n"
                f"Model: {runtime_config.model}"
            )
        if command == "/skills":
            from chulk.skills import SkillRegistry

            registry = SkillRegistry(
                runtime_config.skills_dir,
                skills_dirs=runtime_config.skills_dirs,
                max_skills=runtime_config.max_skills_per_turn,
                max_content_chars=runtime_config.max_skill_content_chars,
            )
            registry.load_metadata()
            names = [skill.name for skill in registry.list_skills()]
            return (
                "Available skills: " + ", ".join(names)
                if names
                else "No skills are available."
            )
        if command == "/memory":
            profile = (
                self._profile_runtime_factory.resolve(profile_id).profile
                if self._profile_runtime_factory is not None
                else None
            )
            namespace = (
                profile.memory_namespace
                if profile is not None and profile.memory_namespace
                else f"telegram:chat:{chat_id}"
            )
            access = (
                "read-only"
                if self.telegram_config.long_term_memory_enabled
                else "off"
            )
            return f"Memory namespace: {namespace}\nAccess: {access}"
        if command == "/agents":
            return "No delegated agents are active in this conversation."
        if command == "/jobs":
            if schedule_store is None:
                return "Scheduling is disabled for this channel."
            return format_jobs(
                schedule_store.list(
                    adapter="telegram",
                    destination_id=str(chat_id),
                ),
                timezone_name=self.telegram_config.timezone,
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
            if schedule_store is None:
                return "Scheduling is disabled for this Telegram agent."
            return format_jobs(
                schedule_store.list(adapter="telegram", destination_id=str(chat_id)),
                timezone_name=self.telegram_config.timezone,
            )
        if command == "/cancel":
            if schedule_store is None:
                return "Scheduling is disabled for this Telegram agent."
            if not arguments:
                return "Usage: /cancel <task-id>"
            jobs = schedule_store.list(adapter="telegram", destination_id=str(chat_id))
            matches = [job for job in jobs if job.id.startswith(arguments)]
            if len(matches) != 1:
                return "Task id is missing or ambiguous. Use /reminders."
            schedule_store.cancel(
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

    def _agent_for_chat(self, profile_id: str, chat_id: int) -> TelegramAgent:
        key = (profile_id, chat_id)
        cached = self._agents.get(key)
        if cached is not None:
            return cached
        conversation = self._session_store_for(profile_id).find_conversation_by_metadata(
            TELEGRAM_CHAT_METADATA_KEY,
            chat_id,
        )
        return self._create_agent(
            profile_id,
            chat_id,
            conversation.id if conversation is not None else None,
        )

    def _create_agent(
        self,
        profile_id: str,
        chat_id: int,
        conversation_id: str | None,
    ) -> TelegramAgent:
        if profile_id == self.config.profile_id:
            agent = self._agent_factory(chat_id, conversation_id)
        elif self._custom_agent_factory is not None:
            raise RuntimeError(
                "the injected Telegram agent factory cannot serve another profile"
            )
        else:
            agent = self._default_agent_factory_for_profile(
                profile_id,
                chat_id,
                conversation_id,
            )
        self._agents[(profile_id, chat_id)] = agent
        return agent

    def _default_agent_factory(self, chat_id: int, conversation_id: str | None) -> TelegramAgent:
        return self._default_agent_factory_for_profile(
            self.config.profile_id,
            chat_id,
            conversation_id,
        )

    def _default_agent_factory_for_profile(
        self,
        profile_id: str,
        chat_id: int,
        conversation_id: str | None,
    ) -> TelegramAgent:
        runtime_config = self._config_for_profile(profile_id)
        schedule_store = self._schedule_store_for(profile_id)
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
        if schedule_store is not None:
            tool_specs.extend(
                scheduled_job_tools(
                    schedule_store,
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
            config=runtime_config,
            memory_namespace=(
                f"telegram:chat:{chat_id}"
                if profile_id == self.config.profile_id
                else f"profile:{profile_id}:telegram:chat:{chat_id}"
            ),
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

    def _config_for_profile(self, profile_id: str) -> Config:
        cached = self._profile_configs.get(profile_id)
        if cached is not None:
            return cached
        if self._profile_runtime_factory is None:
            raise RuntimeError(
                f"gateway route selected unavailable profile {profile_id!r}"
            )
        resolved = self._profile_runtime_factory.resolve(profile_id)
        self._profile_configs[profile_id] = resolved.config
        return resolved.config

    def _session_store_for(self, profile_id: str) -> SQLiteSessionStore:
        cached = self._session_stores.get(profile_id)
        if cached is not None:
            return cached
        store = SQLiteSessionStore(self._config_for_profile(profile_id).store_path)
        self._session_stores[profile_id] = store
        return store

    def _schedule_store_for(self, profile_id: str) -> SQLiteScheduleStore | None:
        if not self.telegram_config.scheduling_enabled:
            return None
        cached = self._schedule_stores.get(profile_id)
        if cached is not None:
            return cached
        store = SQLiteScheduleStore(self._config_for_profile(profile_id).store_path)
        self._schedule_stores[profile_id] = store
        return store

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

    def _refresh_offset(self) -> None:
        status = self.gateway_ledger.adapter_status(TELEGRAM_CURSOR_NAME, "primary")
        self._offset = (
            int(status.cursor)
            if status is not None and status.cursor is not None
            else None
        )

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
        for profile_id, store in tuple(self._schedule_stores.items()):
            await self._run_due_jobs_for_profile(profile_id, store)

    async def _run_due_jobs_for_profile(
        self,
        profile_id: str,
        store: SQLiteScheduleStore,
    ) -> None:
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
                self._renew_schedule_lease(store, job.id, job.claim_token)
            )
            try:
                chat_id = int(job.destination_id)
                async with self._chat_lock(chat_id):
                    response = await self._agent_for_chat(profile_id, chat_id).run(
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

    async def _renew_schedule_lease(
        self,
        store: SQLiteScheduleStore,
        job_id: str,
        claim_token: str,
    ) -> None:
        while True:
            await asyncio.sleep(SCHEDULE_LEASE_RENEW_SECONDS)
            renewed = await asyncio.to_thread(
                store.renew_lease,
                job_id,
                claim_token,
                lease_seconds=SCHEDULE_LEASE_SECONDS,
            )
            if not renewed:
                LOGGER.warning("Scheduled Telegram job lease renewal lost its claim")
                return


def _telegram_update_from_envelope(envelope: InboundEnvelope) -> TelegramUpdate:
    text = next(
        (part.text for part in envelope.parts if isinstance(part, TextPart)),
        "",
    )
    attachment_value = envelope.extensions.get("attachment")
    attachment = None
    if isinstance(attachment_value, Mapping):
        attachment = TelegramAttachment(
            file_id=str(attachment_value["file_id"]),
            kind=str(attachment_value["kind"]),
            mime_type=str(attachment_value["mime_type"]),
            file_name=(
                str(attachment_value["file_name"])
                if attachment_value.get("file_name") is not None
                else None
            ),
        )
    return TelegramUpdate(
        update_id=int(envelope.event_id),
        chat_id=int(envelope.destination_id),
        user_id=int(envelope.identity.principal_id),
        text=text,
        chat_type=str(envelope.extensions.get("chat_type", "private")),
        attachment=attachment,
    )


def _help_text() -> str:
    return shared_command_help(
        additional=(
            ChannelCommandSpec("plan", "Propose an approval plan", "/plan <request>"),
            ChannelCommandSpec("approve", "Approve the pending plan", "/approve"),
            ChannelCommandSpec("reject", "Reject the pending plan", "/reject"),
            ChannelCommandSpec("reminders", "List scheduled tasks", "/reminders"),
            ChannelCommandSpec(
                "cancel",
                "Cancel a scheduled task",
                "/cancel <task-id>",
            ),
        )
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
