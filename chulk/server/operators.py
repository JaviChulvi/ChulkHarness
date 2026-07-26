"""Profile-owned operator reads and explicit review mutations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chulk.memory import SQLiteMemoryStore
from chulk.profiles import ProfileRuntimeFactory
from chulk.scheduling import SQLiteScheduleStore
from chulk.sessions import SQLiteSessionStore
from chulk.skills import (
    LearningProposalService,
    SkillLifecycleManager,
    SkillRegistry,
    SQLiteSkillLifecycleStore,
)
from chulk.tracing.artifacts import TraceArtifactStore
from chulk.usage import ResourceKind, UsageLedger


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
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        jobs = SQLiteScheduleStore(resolved.config.store_path).list(
            adapter=adapter,
            destination_id=destination_id,
        )
        values = []
        for item in jobs:
            value = item.to_dict()
            value["prompt_preview"] = item.prompt[:500]
            value["prompt_truncated"] = len(item.prompt) > 500
            value.pop("prompt", None)
            values.append(value)
        return {"jobs": values, "next_cursor": None}

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
    ) -> dict[str, Any]:
        proposals = self._proposal_service(profile_id).list(
            status=status,
            limit=limit,
        )
        return {
            "proposals": [item.to_dict() for item in proposals],
            "next_cursor": None,
        }

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
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        page = UsageLedger(
            resolved.config.store_path,
            profile_id=profile_id,
        ).query(
            start=start,
            end=end,
            resource_kind=(
                ResourceKind(resource_kind) if resource_kind is not None else None
            ),
            channel=channel,
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        )
        return page.to_dict()

    def traces(self, profile_id: str, *, limit: int = 20) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        records = SQLiteSessionStore(resolved.config.store_path).list_conversations(
            limit=limit
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
                for item in records
            ],
            "next_cursor": None,
        }

    def artifacts(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        resolved = self.runtime_factory.resolve(profile_id)
        SQLiteSessionStore(resolved.config.store_path).get_conversation(
            conversation_id
        )
        return {
            "artifacts": TraceArtifactStore(
                resolved.config.traces_dir,
                conversation_id,
            ).inventory()
        }

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


__all__ = ["OperatorService", "integer_query", "parse_timestamp"]
