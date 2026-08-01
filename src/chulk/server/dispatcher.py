"""Shared FIFO conversation execution for REST and channel adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal
from uuid import uuid4

from chulk._sdk.events import EventDispatcher
from chulk._sdk.results import run_result_from_runtime
from chulk.core import Agent as CoreAgent
from chulk.events import AgentEvent, EventName
from chulk.profiles import ProfileRuntimeFactory
from chulk.results import RunResult
from chulk.server.journal import PublicEventJournal
from chulk.server.permissions import DurablePermissionBroker, PermissionBroker
from chulk.storage import sqlite_connection


CommandMode = Literal["run", "plan"]
AgentBuilder = Callable[[str, str | None, Mapping[str, Any] | None], CoreAgent]


class ConversationBackpressureError(RuntimeError):
    """Raised when the bounded conversation submission queue is full."""


class ConversationCommandNotFoundError(LookupError):
    """Raised when a command does not belong to the requested conversation."""


class ControlDecisionConflictError(RuntimeError):
    """Raised when a plan decision races or reuses an idempotency key."""


@dataclass(frozen=True, slots=True)
class ConversationCommand:
    id: str
    profile_id: str
    conversation_id: str
    source: str
    mode: str
    message: str
    idempotency_key: str
    status: str
    result: Mapping[str, Any] | None
    error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

    def to_dict(self, *, include_message: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "profile_id": self.profile_id,
            "conversation_id": self.conversation_id,
            "source": self.source,
            "mode": self.mode,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "result": dict(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if include_message:
            value["message"] = self.message
        return value


@dataclass
class _ConversationWorker:
    profile_id: str
    conversation_id: str
    agent: CoreAgent
    journal: PublicEventJournal
    broker: PermissionBroker | DurablePermissionBroker
    queue: asyncio.Queue[str]
    control_lock: asyncio.Lock
    task: asyncio.Task[None] | None = None
    active_task: asyncio.Task[RunResult] | None = None


class ConversationDispatcher:
    """Serialize every source through one profile-owned conversation queue."""

    def __init__(
        self,
        runtime_factory: ProfileRuntimeFactory,
        *,
        agent_builder: AgentBuilder | None = None,
        max_pending_per_conversation: int = 100,
    ) -> None:
        if max_pending_per_conversation < 1:
            raise ValueError("max_pending_per_conversation must be greater than zero")
        self.runtime_factory = runtime_factory
        self.profile_store = runtime_factory.profile_store
        self.agent_builder = agent_builder or self._default_agent_builder
        self.max_pending_per_conversation = max_pending_per_conversation
        self._workers: dict[tuple[str, str], _ConversationWorker] = {}
        self._waiters: dict[str, asyncio.Future[ConversationCommand]] = {}
        self._closed = False

    async def create_conversation(
        self,
        profile_id: str,
        *,
        conversation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or restore one profile-owned conversation runtime."""
        if self._closed:
            raise RuntimeError("conversation dispatcher is closed")
        agent = self.agent_builder(
            profile_id,
            conversation_id,
            {
                **dict(metadata or {}),
                "profile_id": profile_id,
                "source": "control_server",
            },
        )
        worker = self._install_worker(profile_id, agent)
        return self._conversation_snapshot(worker.agent)

    async def get_conversation(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        worker = self._worker(profile_id, conversation_id)
        return self._conversation_snapshot(worker.agent)

    async def submit(
        self,
        profile_id: str,
        conversation_id: str,
        message: str,
        *,
        mode: CommandMode = "run",
        source: str = "api",
        idempotency_key: str | None = None,
    ) -> ConversationCommand:
        """Durably enqueue work before returning to its source."""
        if self._closed:
            raise RuntimeError("conversation dispatcher is closed")
        worker = self._worker(profile_id, conversation_id)
        command, inserted = self._enqueue(
            worker,
            message=message,
            mode=mode,
            source=source,
            idempotency_key=idempotency_key or uuid4().hex,
        )
        if inserted:
            loop = asyncio.get_running_loop()
            self._waiters[command.id] = loop.create_future()
            worker.queue.put_nowait(command.id)
            self._ensure_worker_task(worker)
        return command

    async def submit_and_wait(
        self,
        profile_id: str,
        conversation_id: str,
        message: str,
        *,
        mode: CommandMode = "run",
        source: str = "gateway",
        idempotency_key: str | None = None,
    ) -> ConversationCommand:
        command = await self.submit(
            profile_id,
            conversation_id,
            message,
            mode=mode,
            source=source,
            idempotency_key=idempotency_key,
        )
        if command.status in {"completed", "failed", "cancelled", "uncertain"}:
            return command
        waiter = self._waiters.get(command.id)
        if waiter is None:
            while True:
                current = self.get_command(
                    profile_id,
                    conversation_id,
                    command.id,
                )
                if current.status not in {"queued", "running"}:
                    return current
                await asyncio.sleep(0.05)
        return await asyncio.shield(waiter)

    def get_command(
        self,
        profile_id: str,
        conversation_id: str,
        command_id: str,
    ) -> ConversationCommand:
        path = self._store_path(profile_id)
        with sqlite_connection(path) as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation_commands
                WHERE id = ? AND profile_id = ? AND conversation_id = ?
                """,
                (command_id, profile_id, conversation_id),
            ).fetchone()
        if row is None:
            raise ConversationCommandNotFoundError(
                "command does not belong to this conversation"
            )
        return _command(row)

    async def approve_plan(
        self,
        profile_id: str,
        conversation_id: str,
        turn_id: str,
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        worker = self._worker(profile_id, conversation_id)
        return await self._apply_plan_decision(
            worker,
            turn_id=turn_id,
            action="approve",
            idempotency_key=idempotency_key,
            operation=worker.agent.approve_plan_async,
        )

    async def reject_plan(
        self,
        profile_id: str,
        conversation_id: str,
        turn_id: str,
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        worker = self._worker(profile_id, conversation_id)

        async def reject() -> str:
            return worker.agent.reject_plan()

        return await self._apply_plan_decision(
            worker,
            turn_id=turn_id,
            action="reject",
            idempotency_key=idempotency_key,
            operation=reject,
        )

    async def cancel(self, profile_id: str, conversation_id: str) -> bool:
        worker = self._worker(profile_id, conversation_id)
        cancelled = worker.active_task is not None and not worker.active_task.done()
        if cancelled and worker.active_task is not None:
            worker.active_task.cancel()
        permission_count = worker.broker.cancel_pending()
        queued_count = self._cancel_queued(worker)
        return cancelled or permission_count > 0 or queued_count > 0

    def journal(self, profile_id: str, conversation_id: str) -> PublicEventJournal:
        return self._worker(profile_id, conversation_id).journal

    def permissions(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> PermissionBroker | DurablePermissionBroker:
        return self._worker(profile_id, conversation_id).broker

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Task[Any]] = []
        for worker in self._workers.values():
            worker.broker.cancel_pending(reason="control server stopped")
            if worker.active_task is not None:
                worker.active_task.cancel()
                tasks.append(worker.active_task)
            if worker.task is not None:
                worker.task.cancel()
                tasks.append(worker.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(worker.agent.aclose() for worker in self._workers.values()),
            return_exceptions=True,
        )
        self._workers.clear()

    def _default_agent_builder(
        self,
        profile_id: str,
        conversation_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> CoreAgent:
        return self.runtime_factory.create_agent(
            profile_id,
            conversation_id=conversation_id,
            conversation_metadata=dict(metadata or {}),
        )

    def _worker(self, profile_id: str, conversation_id: str) -> _ConversationWorker:
        key = (profile_id, conversation_id)
        worker = self._workers.get(key)
        if worker is not None:
            return worker
        agent = self.agent_builder(profile_id, conversation_id, None)
        return self._install_worker(profile_id, agent)

    def _install_worker(
        self,
        profile_id: str,
        agent: CoreAgent,
    ) -> _ConversationWorker:
        conversation_id = agent.state.conversation_id
        key = (profile_id, conversation_id)
        existing = self._workers.get(key)
        if existing is not None:
            agent.close()
            return existing
        path = self._store_path(profile_id)
        journal = PublicEventJournal(path, profile_id=profile_id)
        if (
            agent.execution_scope is not None
            and agent.run_store is not None
            and agent.approval_store is not None
        ):
            broker: PermissionBroker | DurablePermissionBroker = (
                DurablePermissionBroker(
                    agent.approval_store,
                    agent.run_store,
                    scope=agent.execution_scope,
                )
            )
        else:
            broker = PermissionBroker(
                path,
                profile_id=profile_id,
                conversation_id=conversation_id,
                turn_id=lambda: agent.state.current_turn_id,
                journal=journal,
            )
        agent.permission_callback = broker.callback

        def publish(event: AgentEvent) -> None:
            if event.name in {
                EventName.PERMISSION_REQUESTED.value,
                EventName.PERMISSION_RESOLVED.value,
            }:
                return
            journal.append(event)

        EventDispatcher(agent, on_event=publish)
        worker = _ConversationWorker(
            profile_id=profile_id,
            conversation_id=conversation_id,
            agent=agent,
            journal=journal,
            broker=broker,
            queue=asyncio.Queue(maxsize=self.max_pending_per_conversation),
            control_lock=asyncio.Lock(),
        )
        self._workers[key] = worker
        self._recover_commands(worker)
        self._recover_control_decisions(worker)
        return worker

    def _enqueue(
        self,
        worker: _ConversationWorker,
        *,
        message: str,
        mode: CommandMode,
        source: str,
        idempotency_key: str,
    ) -> tuple[ConversationCommand, bool]:
        path = self._store_path(worker.profile_id)
        now = _utc_now()
        command_id = uuid4().hex
        with sqlite_connection(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM conversation_commands
                WHERE conversation_id = ? AND idempotency_key = ?
                """,
                (worker.conversation_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["message"]) != message
                    or str(existing["mode"]) != mode
                    or str(existing["source"]) != source
                ):
                    raise ValueError(
                        "idempotency key was reused with a different command"
                    )
                return _command(existing), False
            pending = conn.execute(
                """
                SELECT COUNT(*) AS count FROM conversation_commands
                WHERE conversation_id = ? AND status IN ('queued', 'running')
                """,
                (worker.conversation_id,),
            ).fetchone()
            if int(pending["count"]) >= self.max_pending_per_conversation:
                raise ConversationBackpressureError(
                    "conversation queue reached its configured limit"
                )
            conn.execute(
                """
                INSERT INTO conversation_commands (
                    id, profile_id, conversation_id, source, mode, message,
                    idempotency_key, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    command_id,
                    worker.profile_id,
                    worker.conversation_id,
                    source,
                    mode,
                    message,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM conversation_commands WHERE id = ?",
                (command_id,),
            ).fetchone()
        assert row is not None
        return _command(row), True

    def _recover_commands(self, worker: _ConversationWorker) -> None:
        path = self._store_path(worker.profile_id)
        now = _utc_now()
        with sqlite_connection(path) as conn:
            conn.execute(
                """
                UPDATE conversation_commands
                SET status = 'uncertain',
                    error = 'server restarted during execution',
                    completed_at = ?, updated_at = ?
                WHERE conversation_id = ? AND status = 'running'
                """,
                (now, now, worker.conversation_id),
            )
            rows = conn.execute(
                """
                SELECT id FROM conversation_commands
                WHERE conversation_id = ? AND status = 'queued'
                ORDER BY created_at, id
                LIMIT ?
                """,
                (worker.conversation_id, self.max_pending_per_conversation),
            ).fetchall()
        for row in rows:
            worker.queue.put_nowait(str(row["id"]))
        if rows:
            self._ensure_worker_task(worker)

    def _ensure_worker_task(self, worker: _ConversationWorker) -> None:
        if worker.task is None or worker.task.done():
            worker.task = asyncio.create_task(self._work(worker))

    async def _work(self, worker: _ConversationWorker) -> None:
        while not self._closed:
            command_id = await worker.queue.get()
            try:
                command = self.get_command(
                    worker.profile_id,
                    worker.conversation_id,
                    command_id,
                )
                if command.status != "queued":
                    continue
                self._mark_running(worker, command_id)
                worker.active_task = asyncio.create_task(
                    self._execute(worker, command)
                )
                try:
                    result = await worker.active_task
                except asyncio.CancelledError:
                    completed = self._finish(
                        worker,
                        command_id,
                        status="cancelled",
                        error="conversation execution cancelled",
                    )
                except Exception as exc:
                    completed = self._finish(
                        worker,
                        command_id,
                        status="failed",
                        error=f"execution failed with {type(exc).__name__}",
                    )
                else:
                    completed = self._finish(
                        worker,
                        command_id,
                        status="completed",
                        result=result.to_dict(),
                    )
                waiter = self._waiters.pop(command_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(completed)
            finally:
                worker.active_task = None
                worker.queue.task_done()

    async def _execute(
        self,
        worker: _ConversationWorker,
        command: ConversationCommand,
    ) -> RunResult:
        if command.mode == "plan":
            await worker.agent.run_planned_turn_async(command.message)
        else:
            await worker.agent.run_turn_async(command.message)
        return run_result_from_runtime(worker.agent)

    async def _run_control(
        self,
        worker: _ConversationWorker,
        operation: Callable[[], Any],
    ) -> RunResult:
        if worker.active_task is not None and not worker.active_task.done():
            raise RuntimeError("conversation is currently executing")
        value = operation()
        if hasattr(value, "__await__"):
            await value
        return run_result_from_runtime(worker.agent)

    async def _apply_plan_decision(
        self,
        worker: _ConversationWorker,
        *,
        turn_id: str,
        action: str,
        idempotency_key: str,
        operation: Callable[[], Any],
    ) -> Mapping[str, Any]:
        async with worker.control_lock:
            existing = self._begin_control_decision(
                worker,
                turn_id=turn_id,
                action=action,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                return existing
            if worker.agent.state.current_turn_id != turn_id:
                self._finish_control_decision(
                    worker,
                    turn_id=turn_id,
                    status="failed",
                    error="turn is not the current conversation turn",
                )
                raise ValueError("turn is not the current conversation turn")
            try:
                result = await self._run_control(worker, operation)
            except BaseException as exc:
                self._finish_control_decision(
                    worker,
                    turn_id=turn_id,
                    status="uncertain",
                    error=f"plan decision failed with {type(exc).__name__}",
                )
                raise
            payload = result.to_dict()
            self._finish_control_decision(
                worker,
                turn_id=turn_id,
                status="completed",
                result=payload,
            )
            return payload

    def _begin_control_decision(
        self,
        worker: _ConversationWorker,
        *,
        turn_id: str,
        action: str,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None:
        path = self._store_path(worker.profile_id)
        now = _utc_now()
        with sqlite_connection(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM control_decisions
                WHERE conversation_id = ? AND target_type = 'plan'
                  AND target_id = ?
                """,
                (worker.conversation_id, turn_id),
            ).fetchone()
            if row is not None:
                if (
                    str(row["action"]) != action
                    or str(row["idempotency_key"]) != idempotency_key
                ):
                    raise ControlDecisionConflictError(
                        "plan already has a different control decision"
                    )
                if row["status"] != "completed" or row["result_json"] is None:
                    raise ControlDecisionConflictError(
                        f"plan decision is {row['status']}"
                    )
                result = json.loads(str(row["result_json"]))
                if not isinstance(result, dict):
                    raise ValueError("stored plan decision result is invalid")
                return result
            duplicate = conn.execute(
                """
                SELECT target_id FROM control_decisions
                WHERE profile_id = ? AND idempotency_key = ?
                """,
                (worker.profile_id, idempotency_key),
            ).fetchone()
            if duplicate is not None:
                raise ControlDecisionConflictError(
                    "idempotency key was used for another control decision"
                )
            conn.execute(
                """
                INSERT INTO control_decisions (
                    id, profile_id, conversation_id, target_type, target_id,
                    action, idempotency_key, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'plan', ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    uuid4().hex,
                    worker.profile_id,
                    worker.conversation_id,
                    turn_id,
                    action,
                    idempotency_key,
                    now,
                    now,
                ),
            )
        return None

    def _finish_control_decision(
        self,
        worker: _ConversationWorker,
        *,
        turn_id: str,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with sqlite_connection(self._store_path(worker.profile_id)) as conn:
            conn.execute(
                """
                UPDATE control_decisions
                SET status = ?, result_json = ?, error = ?, updated_at = ?
                WHERE conversation_id = ? AND target_type = 'plan'
                  AND target_id = ? AND status = 'pending'
                """,
                (
                    status,
                    (
                        json.dumps(dict(result), sort_keys=True)
                        if result is not None
                        else None
                    ),
                    error,
                    _utc_now(),
                    worker.conversation_id,
                    turn_id,
                ),
            )

    def _recover_control_decisions(self, worker: _ConversationWorker) -> None:
        with sqlite_connection(self._store_path(worker.profile_id)) as conn:
            conn.execute(
                """
                UPDATE control_decisions
                SET status = 'uncertain',
                    error = 'server restarted during plan decision',
                    updated_at = ?
                WHERE conversation_id = ? AND status = 'pending'
                """,
                (_utc_now(), worker.conversation_id),
            )

    def _mark_running(self, worker: _ConversationWorker, command_id: str) -> None:
        now = _utc_now()
        with sqlite_connection(self._store_path(worker.profile_id)) as conn:
            conn.execute(
                """
                UPDATE conversation_commands
                SET status = 'running', started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, command_id),
            )

    def _finish(
        self,
        worker: _ConversationWorker,
        command_id: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> ConversationCommand:
        now = _utc_now()
        with sqlite_connection(self._store_path(worker.profile_id)) as conn:
            conn.execute(
                """
                UPDATE conversation_commands
                SET status = ?, result_json = ?, error = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (
                    status,
                    json.dumps(dict(result), sort_keys=True) if result is not None else None,
                    error,
                    now,
                    now,
                    command_id,
                ),
            )
        return self.get_command(worker.profile_id, worker.conversation_id, command_id)

    def _cancel_queued(self, worker: _ConversationWorker) -> int:
        now = _utc_now()
        with sqlite_connection(self._store_path(worker.profile_id)) as conn:
            rows = conn.execute(
                """
                SELECT id FROM conversation_commands
                WHERE conversation_id = ? AND status = 'queued'
                """,
                (worker.conversation_id,),
            ).fetchall()
            conn.execute(
                """
                UPDATE conversation_commands
                SET status = 'cancelled', error = 'conversation cancelled',
                    completed_at = ?, updated_at = ?
                WHERE conversation_id = ? AND status = 'queued'
                """,
                (now, now, worker.conversation_id),
            )
        for row in rows:
            command_id = str(row["id"])
            waiter = self._waiters.pop(command_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(
                    self.get_command(
                        worker.profile_id,
                        worker.conversation_id,
                        command_id,
                    )
                )
        return len(rows)

    def _store_path(self, profile_id: str) -> Path:
        return self.runtime_factory.resolve(profile_id).config.store_path

    @staticmethod
    def _conversation_snapshot(agent: CoreAgent) -> dict[str, Any]:
        store = agent.session_store
        conversation = store.get_conversation(agent.state.conversation_id)
        return {
            "id": conversation.id,
            "profile_id": agent.profile_id,
            "title": conversation.title,
            "status": conversation.status,
            "provider": conversation.provider,
            "model": conversation.model,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "turn_count": conversation.turn_count,
        }


def _command(row: sqlite3.Row) -> ConversationCommand:
    result = json.loads(str(row["result_json"])) if row["result_json"] is not None else None
    if result is not None and not isinstance(result, dict):
        raise ValueError("stored command result is invalid")
    return ConversationCommand(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        conversation_id=str(row["conversation_id"]),
        source=str(row["source"]),
        mode=str(row["mode"]),
        message=str(row["message"]),
        idempotency_key=str(row["idempotency_key"]),
        status=str(row["status"]),
        result=result,
        error=str(row["error"]) if row["error"] is not None else None,
        created_at=str(row["created_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        completed_at=(
            str(row["completed_at"]) if row["completed_at"] is not None else None
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AgentBuilder",
    "CommandMode",
    "ConversationBackpressureError",
    "ConversationCommand",
    "ConversationCommandNotFoundError",
    "ConversationDispatcher",
    "ControlDecisionConflictError",
]
