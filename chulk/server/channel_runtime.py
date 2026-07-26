"""Shared conversation execution for WebSocket and external channel adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from chulk.gateway import (
    DeliveryTarget,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
    conversation_key_for,
    parse_channel_command,
    shared_command_help,
    shared_command_spec,
)
from chulk.memory import SQLiteMemoryStore
from chulk.scheduling import SQLiteScheduleStore
from chulk.sessions import SQLiteSessionStore, SessionNotFoundError
from chulk.skills import SkillRegistry

from chulk.server.dispatcher import ConversationDispatcher


CHANNEL_CONVERSATION_METADATA_KEY = "gateway_conversation_key"


@dataclass(frozen=True, slots=True)
class ChannelExecutionResult:
    text: str
    conversation_id: str
    status: str = "completed"
    error: str | None = None
    command_id: str | None = None
    channel_command: str | None = None


class ChannelConversationExecutor:
    """Execute channel messages without moving orchestration into adapters."""

    def __init__(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

    async def __call__(
        self,
        profile_id: str,
        envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        text = "\n".join(
            part.text for part in envelope.parts if isinstance(part, TextPart)
        ).strip()
        requested = envelope.extensions.get("conversation_id")
        requested_id = (
            str(requested).strip()
            if isinstance(requested, str) and requested.strip()
            else None
        )
        parsed = parse_channel_command(text)
        try:
            if parsed is not None:
                result = await self._execute_command(
                    profile_id,
                    envelope,
                    parsed.name,
                    parsed.arguments,
                    requested_id,
                )
            else:
                conversation_id = await self._resolve_conversation(
                    profile_id,
                    envelope,
                    requested_id=requested_id,
                    create=True,
                )
                mode = envelope.extensions.get("mode", "run")
                try:
                    command = await self.dispatcher.submit_and_wait(
                        profile_id,
                        conversation_id,
                        text,
                        mode="plan" if mode == "plan" else "run",
                        source=f"gateway:{envelope.identity.adapter}",
                        idempotency_key=envelope.idempotency_key,
                    )
                except asyncio.CancelledError:
                    await self.dispatcher.cancel(profile_id, conversation_id)
                    raise
                payload = command.result or {}
                answer = payload.get("content")
                if not isinstance(answer, str):
                    answer = (
                        command.error
                        or f"Command ended with status {command.status}."
                    )
                result = ChannelExecutionResult(
                    text=answer,
                    conversation_id=conversation_id,
                    status=command.status,
                    error=command.error,
                    command_id=command.id,
                )
        except SessionNotFoundError:
            result = ChannelExecutionResult(
                text="Conversation not found. Use /new to start another one.",
                conversation_id=requested_id or "unknown",
                status="failed",
                error="conversation_not_found",
            )
        return (self._outbound(envelope, profile_id, result),)

    async def _execute_command(
        self,
        profile_id: str,
        envelope: InboundEnvelope,
        name: str,
        arguments: str,
        requested_id: str | None,
    ) -> ChannelExecutionResult:
        if name in {"help", "start"}:
            return ChannelExecutionResult(
                shared_command_help(),
                requested_id or "none",
                channel_command=name,
            )
        if shared_command_spec(name) is None:
            return ChannelExecutionResult(
                f"Unknown command /{name}. Use /help to list channel commands.",
                requested_id or "none",
                status="failed",
                error="unknown_channel_command",
                channel_command=name,
            )
        if name == "new":
            created = await self.dispatcher.create_conversation(
                profile_id,
                metadata=self._metadata(envelope),
            )
            conversation_id = str(created["id"])
            return ChannelExecutionResult(
                f"Started a new conversation ({conversation_id[:8]}).",
                conversation_id,
                channel_command=name,
            )
        conversation_id = await self._resolve_conversation(
            profile_id,
            envelope,
            requested_id=requested_id,
            create=False,
        )
        resolved = self.dispatcher.runtime_factory.resolve(profile_id)
        profile = resolved.profile
        config = resolved.config
        if name == "status":
            current = conversation_id if conversation_id != "none" else "none"
            return ChannelExecutionResult(
                (
                    f"Profile: {profile.id}\n"
                    f"Provider: {config.llm_provider}\n"
                    f"Model: {config.model}\n"
                    f"Conversation: {current}"
                ),
                conversation_id,
                channel_command=name,
            )
        if name == "stop":
            cancelled = (
                await self.dispatcher.cancel(profile_id, conversation_id)
                if conversation_id != "none"
                else False
            )
            return ChannelExecutionResult(
                (
                    f"Stop requested for conversation {conversation_id[:8]}."
                    if cancelled
                    else "No active conversation work to stop."
                ),
                conversation_id,
                channel_command=name,
            )
        if name == "model":
            if arguments:
                text = "Model switching is not available in this channel."
            else:
                text = (
                    f"Model profile: {profile.model_profile_id}\n"
                    f"Provider: {config.llm_provider}\n"
                    f"Model: {config.model}"
                )
            return ChannelExecutionResult(
                text,
                conversation_id,
                channel_command=name,
            )
        if name == "skills":
            registry = SkillRegistry(
                config.skills_dir,
                skills_dirs=config.skills_dirs,
                max_skills=config.max_skills_per_turn,
                max_content_chars=config.max_skill_content_chars,
            )
            registry.load_metadata()
            skills = registry.list_skills()
            if profile.allowed_skills is not None:
                allowed = set(profile.allowed_skills)
                skills = [skill for skill in skills if skill.name in allowed]
            text = (
                "Available skills: " + ", ".join(item.name for item in skills)
                if skills
                else "No skills are available."
            )
            return ChannelExecutionResult(
                text,
                conversation_id,
                channel_command=name,
            )
        if name == "memory":
            namespace = profile.memory_namespace or "default"
            count = len(
                SQLiteMemoryStore(
                    config.store_path,
                    namespace=profile.memory_namespace,
                ).list_memories(limit=50)
            )
            return ChannelExecutionResult(
                f"Memory namespace: {namespace}\nStored memories shown: {count}",
                conversation_id,
                channel_command=name,
            )
        if name == "agents":
            return ChannelExecutionResult(
                "No delegated agents are active in this conversation.",
                conversation_id,
                channel_command=name,
            )
        if name == "jobs":
            records = SQLiteScheduleStore(
                config.store_path,
                profile_id=profile.id,
            ).list(
                adapter=envelope.identity.adapter,
                destination_id=envelope.destination_id,
            )
            if not records:
                text = "No scheduled jobs for this channel."
            else:
                text = "Scheduled jobs:\n" + "\n".join(
                    f"- {item.id[:8]} · {item.status} · {item.next_run_at.isoformat()}"
                    for item in records[:20]
                )
            return ChannelExecutionResult(
                text,
                conversation_id,
                channel_command=name,
            )
        raise AssertionError(f"unhandled shared command: {name}")

    async def _resolve_conversation(
        self,
        profile_id: str,
        envelope: InboundEnvelope,
        *,
        requested_id: str | None,
        create: bool,
    ) -> str:
        if requested_id is not None:
            await self.dispatcher.get_conversation(profile_id, requested_id)
            return requested_id
        resolved = self.dispatcher.runtime_factory.resolve(profile_id)
        key = _channel_conversation_key(envelope)
        existing = SQLiteSessionStore(
            resolved.config.store_path
        ).find_conversation_by_metadata(CHANNEL_CONVERSATION_METADATA_KEY, key)
        if existing is not None:
            return existing.id
        if not create:
            return "none"
        created = await self.dispatcher.create_conversation(
            profile_id,
            metadata=self._metadata(envelope),
        )
        return str(created["id"])

    @staticmethod
    def _metadata(envelope: InboundEnvelope) -> dict[str, Any]:
        return {
            "channel": envelope.identity.adapter,
            CHANNEL_CONVERSATION_METADATA_KEY: _channel_conversation_key(envelope),
        }

    @staticmethod
    def _outbound(
        envelope: InboundEnvelope,
        profile_id: str,
        result: ChannelExecutionResult,
    ) -> OutboundEnvelope:
        return OutboundEnvelope(
            profile_id=profile_id,
            conversation_id=result.conversation_id,
            target=DeliveryTarget(
                envelope.identity.adapter,
                envelope.identity.account_id,
                envelope.destination_id,
                thread_id=envelope.thread_id,
            ),
            text=result.text,
            reply_to_event_id=envelope.event_id,
            extensions={
                "command_id": result.command_id,
                "channel_command": result.channel_command,
                "status": result.status,
                "error": result.error,
            },
        )


def _channel_conversation_key(envelope: InboundEnvelope) -> str:
    if envelope.identity.adapter == "websocket":
        return "\x1f".join(
            (
                envelope.identity.adapter,
                envelope.destination_id,
                envelope.thread_id or "",
            )
        )
    return conversation_key_for(envelope)


__all__ = [
    "CHANNEL_CONVERSATION_METADATA_KEY",
    "ChannelConversationExecutor",
    "ChannelExecutionResult",
]
