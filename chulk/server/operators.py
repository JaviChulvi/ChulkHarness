"""Profile-owned operator reads and explicit review mutations."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from datetime import datetime
import json
from typing import Any

from chulk.children import ChildDeliveryStatus, ChildTaskStatus, ChildTaskStore
from chulk.goals import GoalService, GoalStatus, GoalStore
from chulk.memory import SQLiteMemoryStore
from chulk.profiles import ProfileRuntimeFactory
from chulk.scheduling import AutomationJobStatus, SQLiteScheduleStore
from chulk.sessions import SQLiteSessionStore
from chulk.skills import (
    LearningProposalService,
    SkillLifecycleManager,
    SkillRegistry,
    SQLiteSkillLifecycleStore,
)
from chulk.tracing.artifacts import TraceArtifactStore
from chulk.usage import ResourceKind, UsageGroupBy, UsageLedger


class OperatorService:
    """Expose bounded public views without returning raw trace or credential data."""

    def __init__(self, runtime_factory: ProfileRuntimeFactory) -> None:
        self.runtime_factory = runtime_factory

    def conversations(self, profile_id: str, *, limit: int = 20) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        records = SQLiteSessionStore(resolved.config.store_path).list_conversations(
            limit=limit
        )
        return {
            "conversations": [
                {
                    "id": item.id,
                    "title": item.title,
                    "status": item.status,
                    "provider": item.provider,
                    "model": item.model,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "turn_count": item.turn_count,
                }
                for item in records
            ],
            "next_cursor": None,
        }

    def jobs(
        self,
        profile_id: str,
        *,
        adapter: str | None = None,
        destination_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        include_terminal: bool = False,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        jobs = self._automation_store(profile_id).list(
            adapter=adapter,
            destination_id=destination_id,
            status=AutomationJobStatus(status) if status is not None else None,
            limit=1_000,
            include_terminal=include_terminal,
        )
        page, next_cursor = _page(
            jobs,
            cursor=cursor,
            limit=limit,
            kind="jobs",
            item_id=lambda item: item.id,
        )
        values = []
        for item in page:
            value = item.to_dict()
            value["prompt_preview"] = item.prompt[:500]
            value["prompt_truncated"] = len(item.prompt) > 500
            value.pop("prompt", None)
            values.append(value)
        return {"jobs": values, "next_cursor": next_cursor}

    def goals(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        service = self._goal_service(profile_id)
        goals = service.store.list(
            status=GoalStatus(status) if status is not None else None,
            limit=1_000,
        )
        page, next_cursor = _page(
            goals,
            cursor=cursor,
            limit=limit,
            kind="goals",
            item_id=lambda item: item.id,
        )
        return {
            "goals": [item.to_dict() for item in page],
            "next_cursor": next_cursor,
        }

    def goal(self, profile_id: str, goal_id: str) -> dict[str, Any]:
        service = self._goal_service(profile_id)
        goal = service.store.get(goal_id)
        return {
            "goal": goal.to_dict(),
            "events": [
                event.to_dict() for event in service.store.events(goal_id)
            ],
            "action_checkpoints": [
                item.to_dict()
                for item in service.store.action_checkpoints(goal_id)
            ],
        }

    def control_goal(
        self,
        profile_id: str,
        goal_id: str,
        *,
        action: str,
        revision: int,
        step_id: str | None = None,
        instruction: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        service = self._goal_service(profile_id)
        actor = "control-server"
        if action == "approve":
            goal = service.approve(
                goal_id,
                expected_revision=revision,
                approved_by=actor,
                reason=reason,
            )
        elif action == "run":
            goal = service.run(
                goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif action == "pause":
            goal = service.pause(
                goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif action == "resume":
            goal = service.resume(
                goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif action == "steer":
            goal = service.steer(
                goal_id,
                expected_revision=revision,
                instruction=_required(instruction, "instruction"),
                created_by=actor,
            )
        elif action == "approve_step":
            goal = service.approve_step(
                goal_id,
                _required(step_id, "step_id"),
                expected_revision=revision,
                approved_by=actor,
                reason=reason,
            )
        elif action == "skip_step":
            goal = service.skip_step(
                goal_id,
                _required(step_id, "step_id"),
                expected_revision=revision,
                reason=_required(reason, "reason"),
                approved_by=actor,
            )
        elif action == "retry_step":
            goal = service.retry_step(
                goal_id,
                _required(step_id, "step_id"),
                expected_revision=revision,
                actor=actor,
            )
        elif action == "cancel":
            goal = service.request_cancel(
                goal_id,
                expected_revision=revision,
                actor=actor,
            )
        else:
            raise ValueError(
                "goal action must be approve, run, pause, resume, steer, "
                "approve_step, skip_step, retry_step, or cancel"
            )
        return goal.to_dict()

    def child_tasks(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        goal_id: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        store = self._child_store(profile_id)
        tasks = store.list(
            status=ChildTaskStatus(status) if status is not None else None,
            goal_id=goal_id,
            parent_task_id=parent_task_id,
            limit=1_000,
        )
        page, next_cursor = _page(
            tasks,
            cursor=cursor,
            limit=limit,
            kind="tasks",
            item_id=lambda item: item.id,
        )
        return {
            "tasks": [item.to_dict() for item in page],
            "next_cursor": next_cursor,
        }

    def child_task(self, profile_id: str, task_id: str) -> dict[str, Any]:
        store = self._child_store(profile_id)
        task = store.get(task_id)
        return {
            "task": task.to_dict(),
            "events": [
                {
                    "id": event.id,
                    "revision": event.revision,
                    "kind": event.kind,
                    "actor": event.actor,
                    "payload": dict(event.payload),
                    "created_at": event.created_at.isoformat(),
                }
                for event in store.events(task_id)
            ],
            "attempts": [dict(item) for item in store.attempts(task_id)],
            "deliveries": [
                _child_delivery(item)
                for item in store.list_deliveries(
                    task_id=task_id,
                    limit=100,
                )
            ],
        }

    def control_child_task(
        self,
        profile_id: str,
        task_id: str,
        *,
        action: str,
        revision: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        store = self._child_store(profile_id)
        if action == "cancel":
            changed = store.request_cancel(
                task_id,
                expected_revision=revision,
                actor="control-server",
                reason=reason or "Cancellation requested by operator.",
            )
            task = next(item for item in changed if item.id == task_id)
        elif action == "retry":
            task = store.retry(
                task_id,
                expected_revision=revision,
                actor="control-server",
            )
        else:
            raise ValueError("child task action must be cancel or retry")
        return task.to_dict()

    def automation_job(self, profile_id: str, job_id: str) -> dict[str, Any]:
        store = self._automation_store(profile_id)
        return {
            "job": store.get(job_id).to_dict(),
            "runs": [item.to_dict() for item in store.runs(job_id)],
            "triggers": [item.to_dict() for item in store.triggers(job_id)],
        }

    def control_automation(
        self,
        profile_id: str,
        job_id: str,
        *,
        action: str,
        revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        store = self._automation_store(profile_id)
        operations = {
            "pause": store.pause,
            "resume": store.resume,
            "approve": store.approve,
            "run_now": store.run_now,
        }
        if action == "cancel":
            changed = store.cancel(
                job_id,
                expected_revision=revision,
                idempotency_key=idempotency_key,
                actor="control-server",
            )
            if not changed:
                raise ValueError("automation is already terminal")
            job = store.get(job_id)
        else:
            try:
                operation = operations[action]
            except KeyError as exc:
                raise ValueError(
                    "automation action must be pause, resume, approve, run_now, or cancel"
                ) from exc
            job = operation(
                job_id,
                expected_revision=revision,
                idempotency_key=idempotency_key,
                actor="control-server",
            )
        return job.to_dict()

    def create_automation_webhook(
        self,
        profile_id: str,
        job_id: str,
    ) -> dict[str, Any]:
        trigger, credential = self._automation_store(profile_id).create_webhook_trigger(
            job_id
        )
        return {"trigger": trigger.to_dict(), "credential": credential}

    def ingest_automation_webhook(
        self,
        profile_id: str,
        trigger_id: str,
        *,
        credential: str,
        event_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        envelope = self._automation_store(profile_id).ingest_webhook(
            trigger_id,
            token=credential,
            event_id=event_id,
            payload=payload,
        )
        return envelope.to_dict()

    def proposals(
        self,
        profile_id: str,
        *,
        status: str | None = "pending",
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        proposals = self._proposal_service(profile_id).list(
            status=status,
            limit=1_000,
        )
        page, next_cursor = _page(
            proposals,
            cursor=cursor,
            limit=limit,
            kind="proposals",
            item_id=lambda item: item.id,
        )
        return {
            "proposals": [item.to_dict() for item in page],
            "next_cursor": next_cursor,
        }

    def proposal(self, profile_id: str, proposal_id: str) -> dict[str, Any]:
        return self._proposal_service(profile_id).get(proposal_id).to_dict()

    def decide_proposal(
        self,
        profile_id: str,
        proposal_id: str,
        *,
        action: str,
    ) -> dict[str, Any]:
        service = self._proposal_service(profile_id)
        if action == "approve":
            result = service.approve(proposal_id, approved_by="control-server")
        elif action == "reject":
            result = service.reject(proposal_id, rejected_by="control-server")
        else:
            raise ValueError("proposal action must be approve or reject")
        return result.to_dict()

    def usage(
        self,
        profile_id: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resource_kind: str | None = None,
        channel: str | None = None,
        conversation_id: str | None = None,
        goal_id: str | None = None,
        job_id: str | None = None,
        child_task_id: str | None = None,
        group_by: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        ledger = UsageLedger(
            resolved.config.store_path,
            profile_id=profile_id,
        )
        kind = (
            ResourceKind(resource_kind) if resource_kind is not None else None
        )
        if group_by is not None:
            groups = ledger.group(
                UsageGroupBy(group_by),
                start=start,
                end=end,
                resource_kind=kind,
                channel=channel,
                conversation_id=conversation_id,
                goal_id=goal_id,
                job_id=job_id,
                child_task_id=child_task_id,
                limit=limit,
            )
            return {
                "group_by": group_by,
                "groups": [item.to_dict() for item in groups],
                "next_cursor": None,
            }
        page = ledger.query(
            start=start,
            end=end,
            resource_kind=kind,
            channel=channel,
            conversation_id=conversation_id,
            goal_id=goal_id,
            job_id=job_id,
            child_task_id=child_task_id,
            limit=limit,
            cursor=cursor,
        )
        return page.to_dict()

    def traces(
        self,
        profile_id: str,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        records = SQLiteSessionStore(resolved.config.store_path).list_conversations(
            limit=1_000
        )
        page, next_cursor = _page(
            records,
            cursor=cursor,
            limit=limit,
            kind="traces",
            item_id=lambda item: item.id,
        )
        return {
            "traces": [
                {
                    "conversation_id": item.id,
                    "status": item.status,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "available": (
                        resolved.config.traces_dir / f"{item.id}.jsonl"
                    ).is_file(),
                    "artifact_count": len(
                        TraceArtifactStore(
                            resolved.config.traces_dir,
                            item.id,
                        ).inventory()
                    ),
                }
                for item in page
            ],
            "next_cursor": next_cursor,
        }

    def trace(self, profile_id: str, conversation_id: str) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        conversation = SQLiteSessionStore(
            resolved.config.store_path
        ).get_conversation(conversation_id)
        artifacts = TraceArtifactStore(
            resolved.config.traces_dir,
            conversation_id,
        ).inventory()
        return {
            "trace": {
                "conversation_id": conversation.id,
                "status": conversation.status,
                "provider": conversation.provider,
                "model": conversation.model,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "turn_count": conversation.turn_count,
                "available": (
                    resolved.config.traces_dir / f"{conversation.id}.jsonl"
                ).is_file(),
                "artifact_count": len(artifacts),
            },
            "artifacts": artifacts,
        }

    def artifacts(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        SQLiteSessionStore(resolved.config.store_path).get_conversation(
            conversation_id
        )
        inventory = TraceArtifactStore(
            resolved.config.traces_dir,
            conversation_id,
        ).inventory()
        page, next_cursor = _page(
            inventory,
            cursor=cursor,
            limit=limit,
            kind=f"artifacts:{conversation_id}",
            item_id=lambda item: str(item["artifact_id"]),
        )
        return {"artifacts": list(page), "next_cursor": next_cursor}

    def read_artifact(
        self,
        profile_id: str,
        conversation_id: str,
        artifact_id: str,
        *,
        mode: str = "head_tail",
        offset: int = 0,
        max_bytes: int = 8_192,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        SQLiteSessionStore(resolved.config.store_path).get_conversation(
            conversation_id
        )
        return TraceArtifactStore(
            resolved.config.traces_dir,
            conversation_id,
        ).read(
            artifact_id,
            mode=mode,  # type: ignore[arg-type]
            offset=offset,
            max_bytes=max_bytes,
        ).to_dict()

    def _proposal_service(self, profile_id: str) -> LearningProposalService:
        resolved = self.runtime_factory.resolve(profile_id)
        config = resolved.config
        lifecycle_store = SQLiteSkillLifecycleStore(
            config.store_path,
            profile_id=profile_id,
        )
        profile_skills_dir = config.runtime_dir / "profile-skills"
        registry = SkillRegistry(
            config.skills_dir,
            skills_dirs=(*config.skills_dirs, profile_skills_dir),
            max_skills=config.max_skills_per_turn,
            max_content_chars=config.max_skill_content_chars,
        )
        manager = SkillLifecycleManager(
            lifecycle_store,
            project_skills_dir=config.skills_dir,
            profile_skills_dir=profile_skills_dir,
            project_lock_path=config.skills_dir.parent / "skills.lock",
            profile_lock_path=config.runtime_dir / "profile-skills.lock",
            registry=registry,
        )
        return LearningProposalService(
            memory_store=SQLiteMemoryStore(
                config.store_path,
                namespace=resolved.profile.memory_namespace,
            ),
            lifecycle_store=lifecycle_store,
            lifecycle_manager=manager,
        )

    def _automation_store(self, profile_id: str) -> SQLiteScheduleStore:
        resolved = self.runtime_factory.resolve(profile_id)
        return SQLiteScheduleStore(
            resolved.config.store_path,
            profile_id=profile_id,
        )

    def _goal_service(self, profile_id: str) -> GoalService:
        resolved = self.runtime_factory.resolve(profile_id)
        return GoalService(
            GoalStore(
                resolved.config.store_path,
                profile_id=profile_id,
            )
        )

    def _child_store(self, profile_id: str) -> ChildTaskStore:
        resolved = self.runtime_factory.resolve(profile_id)
        return ChildTaskStore(
            resolved.config.store_path,
            profile_id=profile_id,
        )


def parse_timestamp(value: str | None, *, field: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def integer_query(
    value: str | None,
    *,
    field: str,
    default: int,
    minimum: int = 0,
    maximum: int,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return parsed


def boolean_query(
    value: str | None,
    *,
    field: str,
    default: bool = False,
) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{field} must be true or false")


def _required(value: str | None, field: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{field} is required")
    return clean


def _child_delivery(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "task_id": item.task_id,
        "task_revision": item.task_revision,
        "status": ChildDeliveryStatus(item.status).value,
        "idempotency_key": item.idempotency_key,
        "attempts": item.attempts,
        "error": item.error,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "delivered_at": (
            item.delivered_at.isoformat()
            if item.delivered_at is not None
            else None
        ),
    }


def _page(
    items: Sequence[Any],
    *,
    cursor: str | None,
    limit: int,
    kind: str,
    item_id: Callable[[Any], str],
) -> tuple[tuple[Any, ...], str | None]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
        raise ValueError("operator page limit must be between 1 and 1000")
    start = 0
    if cursor is not None:
        after = _decode_cursor(cursor, kind=kind)
        for index, item in enumerate(items):
            if item_id(item) == after:
                start = index + 1
                break
        else:
            raise ValueError("operator cursor is no longer available")
    page = tuple(items[start : start + limit])
    next_cursor = (
        _encode_cursor(kind, item_id(page[-1]))
        if page and start + len(page) < len(items)
        else None
    )
    return page, next_cursor


def _encode_cursor(kind: str, after: str) -> str:
    raw = json.dumps(
        {"kind": kind, "after": after},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str, *, kind: str) -> str:
    if not value or len(value) > 1_024:
        raise ValueError("operator cursor is invalid")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("operator cursor is invalid") from exc
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError("operator cursor does not match this resource")
    after = payload.get("after")
    if not isinstance(after, str) or not after:
        raise ValueError("operator cursor is invalid")
    return after


__all__ = [
    "OperatorService",
    "boolean_query",
    "integer_query",
    "parse_timestamp",
]
