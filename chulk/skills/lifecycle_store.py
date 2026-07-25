"""SQLite persistence for governed skill packages and learning proposals."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.skills.lifecycle_models import (
    LearningProposalKind,
    LearningProposalRecord,
    LearningProposalStatus,
    LearningReviewUsage,
    SkillLifecycleRecord,
    SkillLifecycleStatus,
    SkillRevisionRecord,
    SkillUsageKind,
)
from chulk.skills.manifest import SkillManifest


_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


class SQLiteSkillLifecycleStore:
    """Persist profile-owned skill history, counters, and review proposals."""

    def __init__(self, db_path: Path | str, *, profile_id: str) -> None:
        self.db_path = Path(db_path)
        self.profile_id = _normalize_profile_id(profile_id)
        initialize_sqlite_database(self.db_path)

    def create_proposal(
        self,
        *,
        kind: LearningProposalKind | str,
        rationale: str,
        target_name: str | None = None,
        evidence_turn_ids: tuple[str, ...] = (),
        source_trace: str | None = None,
        content: str | None = None,
        diff: str | None = None,
        required_capabilities: tuple[str, ...] = (),
        confidence: float = 1.0,
        verification_steps: tuple[str, ...] = (),
        reviewer_model: str | None = None,
        cost: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LearningProposalRecord:
        """Create one bounded pending proposal without applying it."""
        records = self.create_proposals(
            (
                {
                    "kind": kind,
                    "rationale": rationale,
                    "target_name": target_name,
                    "evidence_turn_ids": evidence_turn_ids,
                    "source_trace": source_trace,
                    "content": content,
                    "diff": diff,
                    "required_capabilities": required_capabilities,
                    "confidence": confidence,
                    "verification_steps": verification_steps,
                    "reviewer_model": reviewer_model,
                    "cost": cost,
                    "metadata": metadata,
                },
            )
        )
        return records[0]

    def create_proposals(
        self,
        proposals: tuple[Mapping[str, Any], ...],
        *,
        review_run_id: str | None = None,
        review_token_count: int = 0,
        review_cost_amount: Decimal = Decimal(0),
    ) -> tuple[LearningProposalRecord, ...]:
        """Validate and insert one reviewer batch atomically."""
        if not proposals and review_run_id is None:
            return ()
        values = tuple(
            _proposal_values(profile_id=self.profile_id, **dict(proposal))
            for proposal in proposals
        )
        actual = _review_usage(
            proposal_count=len(values),
            token_count=review_token_count,
            cost_amount=review_cost_amount,
        )
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if values:
                conn.executemany(
                    """
                    INSERT INTO learning_proposals (
                        id, profile_id, kind, target_name, rationale,
                        evidence_turn_ids_json, source_trace, content, diff,
                        required_capabilities_json, confidence,
                        verification_steps_json, reviewer_model, cost, status,
                        created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            if review_run_id is not None:
                cursor = conn.execute(
                    """
                    UPDATE learning_review_runs
                    SET status = 'completed', proposal_count = ?,
                        token_count = ?, cost_amount = ?, completed_at = ?,
                        error = NULL
                    WHERE id = ? AND profile_id = ? AND status = 'reserved'
                    """,
                    (
                        actual.proposal_count,
                        actual.token_count,
                        str(actual.cost_amount),
                        _utc_now(),
                        review_run_id,
                        self.profile_id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise KeyError(
                        f"learning review run {review_run_id!r} is not reserved"
                    )
        return tuple(self.get_proposal(str(value[0])) for value in values)

    def get_proposal(self, proposal_id: str) -> LearningProposalRecord:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM learning_proposals
                WHERE id = ? AND profile_id = ?
                """,
                (proposal_id, self.profile_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"learning proposal {proposal_id!r} does not exist")
        return _row_to_proposal(row)

    def list_proposals(
        self,
        *,
        status: LearningProposalStatus | str | None = LearningProposalStatus.PENDING,
        limit: int = 100,
    ) -> tuple[LearningProposalRecord, ...]:
        clean_limit = _limit(limit)
        query = "SELECT * FROM learning_proposals WHERE profile_id = ?"
        parameters: list[object] = [self.profile_id]
        if status is not None:
            query += " AND status = ?"
            parameters.append(LearningProposalStatus(status).value)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(clean_limit)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return tuple(_row_to_proposal(row) for row in rows)

    def count_proposals_since(self, occurred_at: str) -> int:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS proposal_count
                FROM learning_proposals
                WHERE profile_id = ? AND created_at >= ?
                """,
                (self.profile_id, occurred_at),
            ).fetchone()
        return int(row["proposal_count"]) if row is not None else 0

    def review_usage_since(
        self,
        occurred_at: str,
        *,
        currency: str = "USD",
    ) -> LearningReviewUsage:
        with sqlite_connection(self.db_path) as conn:
            return _review_usage_in_connection(
                conn,
                profile_id=self.profile_id,
                occurred_at=occurred_at,
                currency=_currency(currency),
            )

    def reserve_review_run(
        self,
        *,
        trigger: str,
        reviewer_model: str | None,
        proposal_count: int,
        token_count: int,
        cost_amount: Decimal,
        occurred_at: str,
        max_proposals: int,
        max_tokens: int,
        max_cost: Decimal | None,
        currency: str = "USD",
    ) -> str:
        """Atomically reserve one bounded reviewer call against daily quotas."""
        requested = _review_usage(
            proposal_count=proposal_count,
            token_count=token_count,
            cost_amount=cost_amount,
        )
        run_id = str(uuid4())
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            used = _review_usage_in_connection(
                conn,
                profile_id=self.profile_id,
                occurred_at=occurred_at,
                currency=_currency(currency),
            )
            if used.proposal_count + requested.proposal_count > max_proposals:
                raise ValueError("daily learning proposal quota exceeded")
            if used.token_count + requested.token_count > max_tokens:
                raise ValueError("daily learning reviewer token quota exceeded")
            if (
                max_cost is not None
                and used.cost_amount + requested.cost_amount > max_cost
            ):
                raise ValueError("daily learning reviewer cost quota exceeded")
            conn.execute(
                """
                INSERT INTO learning_review_runs (
                    id, profile_id, trigger, reviewer_model, status,
                    proposal_count, token_count, cost_amount, currency,
                    created_at
                ) VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self.profile_id,
                    _required_text(trigger, "trigger", max_chars=64),
                    _optional_text(
                        reviewer_model,
                        "reviewer_model",
                        max_chars=256,
                    ),
                    requested.proposal_count,
                    requested.token_count,
                    str(requested.cost_amount),
                    _currency(currency),
                    _utc_now(),
                ),
            )
        return run_id

    def finalize_review_run(
        self,
        run_id: str,
        *,
        proposal_count: int,
        token_count: int,
        cost_amount: Decimal,
        failed: bool = False,
        error: str | None = None,
    ) -> None:
        """Replace a reservation with actual reviewer consumption."""
        actual = _review_usage(
            proposal_count=proposal_count,
            token_count=token_count,
            cost_amount=cost_amount,
        )
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE learning_review_runs
                SET status = ?, proposal_count = ?, token_count = ?,
                    cost_amount = ?, completed_at = ?, error = ?
                WHERE id = ? AND profile_id = ? AND status = 'reserved'
                """,
                (
                    "failed" if failed else "completed",
                    actual.proposal_count,
                    actual.token_count,
                    str(actual.cost_amount),
                    _utc_now(),
                    _optional_text(error, "error", max_chars=4_000),
                    run_id,
                    self.profile_id,
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"learning review run {run_id!r} is not reserved")

    def transition_proposal(
        self,
        proposal_id: str,
        *,
        status: LearningProposalStatus | str,
        applied_revision_id: str | None = None,
        accepted_memory_id: str | None = None,
        error: str | None = None,
        reviewed_by: str | None = None,
    ) -> LearningProposalRecord:
        """Move a pending proposal to one terminal state idempotently."""
        next_status = LearningProposalStatus(status)
        if next_status is LearningProposalStatus.PENDING:
            raise ValueError("proposal transition requires a terminal status")
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status FROM learning_proposals
                WHERE id = ? AND profile_id = ?
                """,
                (proposal_id, self.profile_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"learning proposal {proposal_id!r} does not exist")
            if row["status"] == LearningProposalStatus.PENDING.value:
                conn.execute(
                    """
                    UPDATE learning_proposals
                    SET status = ?, reviewed_at = ?, reviewed_by = ?,
                        applied_revision_id = ?, accepted_memory_id = ?, error = ?
                    WHERE id = ? AND profile_id = ? AND status = 'pending'
                    """,
                    (
                        next_status.value,
                        _utc_now(),
                        _optional_text(
                            reviewed_by,
                            "reviewed_by",
                            max_chars=256,
                        ),
                        applied_revision_id,
                        accepted_memory_id,
                        _optional_text(error, "error", max_chars=4_000),
                        proposal_id,
                        self.profile_id,
                    ),
                )
        return self.get_proposal(proposal_id)

    def save_revision(
        self,
        *,
        manifest: SkillManifest,
        digest: str,
        package_files: Mapping[str, bytes],
        scope: str = "project",
        proposal_id: str | None = None,
        status: SkillLifecycleStatus | str = SkillLifecycleStatus.ACTIVE,
        increment_patch: bool = False,
    ) -> SkillRevisionRecord:
        """Save an immutable package snapshot and make it current."""
        clean_digest = _required_text(digest, "digest", max_chars=128)
        if not clean_digest.startswith("sha256:"):
            raise ValueError("skill digest must use sha256")
        encoded_package = _encode_package(package_files)
        revision_id = str(uuid4())
        now = _utc_now()
        lifecycle_status = SkillLifecycleStatus(status)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            revision_id = _save_revision_in_connection(
                conn,
                profile_id=self.profile_id,
                scope=_scope(scope),
                manifest=manifest,
                digest=clean_digest,
                encoded_package=encoded_package,
                proposal_id=proposal_id,
                status=lifecycle_status,
                increment_patch=increment_patch,
                revision_id=revision_id,
                now=now,
            )
        return self.get_revision(revision_id)

    def approve_skill_revision(
        self,
        proposal_id: str,
        *,
        manifest: SkillManifest,
        digest: str,
        package_files: Mapping[str, bytes],
        status: SkillLifecycleStatus | str = SkillLifecycleStatus.ACTIVE,
        increment_patch: bool = True,
        revision_id: str | None = None,
        reviewed_by: str,
        scope: str = "project",
    ) -> LearningProposalRecord:
        """Commit a validated revision and approve its proposal atomically."""
        clean_digest = _required_text(digest, "digest", max_chars=128)
        if not clean_digest.startswith("sha256:"):
            raise ValueError("skill digest must use sha256")
        encoded_package = _encode_package(package_files)
        lifecycle_status = SkillLifecycleStatus(status)
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            proposal = _pending_skill_proposal(
                conn,
                profile_id=self.profile_id,
                proposal_id=proposal_id,
            )
            if proposal.status is not LearningProposalStatus.PENDING:
                return proposal
            if proposal.target_name != manifest.name:
                raise ValueError(
                    "proposal target does not match validated skill manifest"
                )
            revision_id = _save_revision_in_connection(
                conn,
                profile_id=self.profile_id,
                scope=_scope(scope),
                manifest=manifest,
                digest=clean_digest,
                encoded_package=encoded_package,
                proposal_id=proposal_id,
                status=lifecycle_status,
                increment_patch=increment_patch,
                revision_id=revision_id or str(uuid4()),
                now=now,
            )
            conn.execute(
                """
                UPDATE learning_proposals
                SET status = 'approved', reviewed_at = ?, reviewed_by = ?,
                    applied_revision_id = ?, error = NULL
                WHERE id = ? AND profile_id = ? AND status = 'pending'
                """,
                (
                    now,
                    _required_text(
                        reviewed_by,
                        "reviewed_by",
                        max_chars=256,
                    ),
                    revision_id,
                    proposal_id,
                    self.profile_id,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM learning_proposals
                WHERE id = ? AND profile_id = ?
                """,
                (proposal_id, self.profile_id),
            ).fetchone()
        assert row is not None
        return _row_to_proposal(row)

    def find_revision(
        self,
        *,
        name: str,
        digest: str,
        scope: str = "project",
    ) -> SkillRevisionRecord | None:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM skill_package_revisions
                WHERE profile_id = ? AND scope = ? AND name = ? AND digest = ?
                """,
                (self.profile_id, _scope(scope), name, digest),
            ).fetchone()
        return _row_to_revision(row) if row is not None else None

    def approve_skill_archive(
        self,
        proposal_id: str,
        *,
        name: str,
        reviewed_by: str,
        scope: str = "project",
    ) -> LearningProposalRecord:
        """Archive a skill and approve the matching proposal atomically."""
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            proposal = _pending_skill_proposal(
                conn,
                profile_id=self.profile_id,
                proposal_id=proposal_id,
            )
            if proposal.status is not LearningProposalStatus.PENDING:
                return proposal
            if (
                proposal.kind is not LearningProposalKind.SKILL_ARCHIVE
                or proposal.target_name != name
            ):
                raise ValueError("proposal does not authorize this skill archive")
            current = conn.execute(
                """
                SELECT active_revision_id
                FROM skill_packages
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (self.profile_id, _scope(scope), name),
            ).fetchone()
            if current is None:
                raise KeyError(f"governed skill {name!r} does not exist")
            cursor = conn.execute(
                """
                UPDATE skill_packages
                SET status = 'archived', updated_at = ?
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (now, self.profile_id, _scope(scope), name),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"governed skill {name!r} does not exist")
            conn.execute(
                """
                UPDATE learning_proposals
                SET status = 'approved', reviewed_at = ?, reviewed_by = ?,
                    applied_revision_id = ?, error = NULL
                WHERE id = ? AND profile_id = ? AND status = 'pending'
                """,
                (
                    now,
                    _required_text(
                        reviewed_by,
                        "reviewed_by",
                        max_chars=256,
                    ),
                    str(current["active_revision_id"]),
                    proposal_id,
                    self.profile_id,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM learning_proposals
                WHERE id = ? AND profile_id = ?
                """,
                (proposal_id, self.profile_id),
            ).fetchone()
        assert row is not None
        return _row_to_proposal(row)

    def get_revision(self, revision_id: str) -> SkillRevisionRecord:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM skill_package_revisions
                WHERE id = ? AND profile_id = ?
                """,
                (revision_id, self.profile_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"skill revision {revision_id!r} does not exist")
        return _row_to_revision(row)

    def list_revisions(
        self,
        name: str,
        *,
        scope: str = "project",
        limit: int = 100,
    ) -> tuple[SkillRevisionRecord, ...]:
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM skill_package_revisions
                WHERE profile_id = ? AND scope = ? AND name = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (self.profile_id, _scope(scope), name, _limit(limit)),
            ).fetchall()
        return tuple(_row_to_revision(row) for row in rows)

    def get_skill(
        self,
        name: str,
        *,
        scope: str = "project",
    ) -> SkillLifecycleRecord:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM skill_packages
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (self.profile_id, _scope(scope), name),
            ).fetchone()
        if row is None:
            raise KeyError(f"governed skill {name!r} does not exist")
        return _row_to_skill(row)

    def list_skills(
        self,
        *,
        scope: str | None = None,
        limit: int = 1_000,
    ) -> tuple[SkillLifecycleRecord, ...]:
        query = "SELECT * FROM skill_packages WHERE profile_id = ?"
        parameters: list[object] = [self.profile_id]
        if scope is not None:
            query += " AND scope = ?"
            parameters.append(_scope(scope))
        query += " ORDER BY scope, name LIMIT ?"
        parameters.append(_limit(limit, maximum=10_000))
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return tuple(_row_to_skill(row) for row in rows)

    def set_skill_status(
        self,
        name: str,
        status: SkillLifecycleStatus | str,
        *,
        scope: str = "project",
    ) -> SkillLifecycleRecord:
        lifecycle_status = SkillLifecycleStatus(status)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE skill_packages
                SET status = ?, updated_at = ?
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (
                    lifecycle_status.value,
                    _utc_now(),
                    self.profile_id,
                    _scope(scope),
                    name,
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"governed skill {name!r} does not exist")
        return self.get_skill(name, scope=scope)

    def activate_revision(self, revision_id: str) -> SkillLifecycleRecord:
        revision = self.get_revision(revision_id)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE skill_packages
                SET version = ?, digest = ?, source = ?, trust = ?,
                    status = 'active', active_revision_id = ?,
                    patch_count = patch_count + 1, updated_at = ?
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (
                    revision.version,
                    revision.digest,
                    revision.source,
                    revision.trust,
                    revision.id,
                    _utc_now(),
                    self.profile_id,
                    revision.scope,
                    revision.name,
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"governed skill {revision.name!r} does not exist")
        return self.get_skill(revision.name, scope=revision.scope)

    def record_usage(
        self,
        *,
        name: str,
        version: str,
        digest: str,
        kind: SkillUsageKind | str,
        source_event_id: str,
        host_confirmed: bool = False,
        scope: str = "project",
    ) -> SkillLifecycleRecord:
        """Record one idempotent counter event against an exact skill version."""
        usage_kind = SkillUsageKind(kind)
        if usage_kind is SkillUsageKind.SUCCESS and not host_confirmed:
            raise ValueError("skill success requires host confirmation")
        clean_event_id = _required_text(
            source_event_id,
            "source_event_id",
            max_chars=256,
        )
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """
                SELECT version, digest FROM skill_packages
                WHERE profile_id = ? AND scope = ? AND name = ?
                """,
                (self.profile_id, _scope(scope), name),
            ).fetchone()
            if current is None:
                raise KeyError(f"governed skill {name!r} does not exist")
            if current["version"] != version or current["digest"] != digest:
                revision = conn.execute(
                    """
                    SELECT 1 FROM skill_package_revisions
                    WHERE profile_id = ? AND scope = ? AND name = ?
                        AND version = ? AND digest = ?
                    """,
                    (
                        self.profile_id,
                        _scope(scope),
                        name,
                        version,
                        digest,
                    ),
                ).fetchone()
                if revision is None:
                    raise ValueError(
                        "skill usage must identify a governed revision"
                    )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO skill_usage_events (
                    id, profile_id, scope, skill_name, skill_version, skill_digest,
                    kind, source_event_id, host_confirmed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    self.profile_id,
                    _scope(scope),
                    name,
                    version,
                    digest,
                    usage_kind.value,
                    clean_event_id,
                    int(host_confirmed),
                    _utc_now(),
                ),
            )
            if cursor.rowcount:
                column = {
                    SkillUsageKind.VIEW: "view_count",
                    SkillUsageKind.USE: "use_count",
                    SkillUsageKind.SUCCESS: "success_count",
                    SkillUsageKind.PATCH: "patch_count",
                }[usage_kind]
                conn.execute(
                    f"""
                    UPDATE skill_packages
                    SET {column} = {column} + 1, updated_at = ?
                    WHERE profile_id = ? AND scope = ? AND name = ?
                    """,
                    (_utc_now(), self.profile_id, _scope(scope), name),
                )
        return self.get_skill(name, scope=scope)


def _proposal_values(
    *,
    profile_id: str,
    kind: LearningProposalKind | str,
    rationale: str,
    target_name: str | None = None,
    evidence_turn_ids: tuple[str, ...] = (),
    source_trace: str | None = None,
    content: str | None = None,
    diff: str | None = None,
    required_capabilities: tuple[str, ...] = (),
    confidence: float = 1.0,
    verification_steps: tuple[str, ...] = (),
    reviewer_model: str | None = None,
    cost: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[object, ...]:
    proposal_kind = LearningProposalKind(kind)
    clean_rationale = _required_text(
        rationale,
        "rationale",
        max_chars=4_000,
    )
    clean_target = _optional_text(target_name, "target_name", max_chars=128)
    if (
        proposal_kind.value.startswith("skill_")
        or proposal_kind is LearningProposalKind.MEMORY_UPDATE
    ) and clean_target is None:
        raise ValueError(f"{proposal_kind.value} proposals require target_name")
    if proposal_kind in {
        LearningProposalKind.SKILL_CREATE,
        LearningProposalKind.SKILL_PATCH,
        LearningProposalKind.MEMORY_CREATE,
        LearningProposalKind.MEMORY_UPDATE,
    } and not (content and content.strip()):
        raise ValueError(f"{proposal_kind.value} proposals require content")
    if proposal_kind in {
        LearningProposalKind.SKILL_PATCH,
        LearningProposalKind.SKILL_ARCHIVE,
    } and not (diff and diff.strip()):
        raise ValueError(f"{proposal_kind.value} proposals require diff")
    clean_evidence = _text_tuple(
        evidence_turn_ids,
        "evidence_turn_ids",
        max_items=50,
        max_chars=128,
    )
    clean_capabilities = _text_tuple(
        required_capabilities,
        "required_capabilities",
        max_items=50,
        max_chars=128,
    )
    clean_steps = _text_tuple(
        verification_steps,
        "verification_steps",
        max_items=50,
        max_chars=1_000,
    )
    encoded_metadata = json.dumps(dict(metadata or {}), sort_keys=True)
    if len(encoded_metadata) > 50_000:
        raise ValueError("proposal metadata cannot exceed 50000 characters")
    return (
        str(uuid4()),
        profile_id,
        proposal_kind.value,
        clean_target,
        clean_rationale,
        json.dumps(clean_evidence),
        _optional_text(source_trace, "source_trace", max_chars=2_000),
        _optional_blob(content, "content", max_chars=500_000),
        _optional_blob(diff, "diff", max_chars=200_000),
        json.dumps(clean_capabilities),
        _confidence(confidence),
        json.dumps(clean_steps),
        _optional_text(reviewer_model, "reviewer_model", max_chars=256),
        _optional_text(cost, "cost", max_chars=128),
        LearningProposalStatus.PENDING.value,
        _utc_now(),
        encoded_metadata,
    )


def _row_to_proposal(row: sqlite3.Row) -> LearningProposalRecord:
    return LearningProposalRecord(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        kind=LearningProposalKind(str(row["kind"])),
        target_name=row["target_name"],
        rationale=str(row["rationale"]),
        evidence_turn_ids=_json_string_tuple(row["evidence_turn_ids_json"]),
        source_trace=row["source_trace"],
        content=row["content"],
        diff=row["diff"],
        required_capabilities=_json_string_tuple(
            row["required_capabilities_json"]
        ),
        confidence=float(row["confidence"]),
        verification_steps=_json_string_tuple(row["verification_steps_json"]),
        reviewer_model=row["reviewer_model"],
        cost=row["cost"],
        status=LearningProposalStatus(str(row["status"])),
        created_at=str(row["created_at"]),
        reviewed_at=row["reviewed_at"],
        reviewed_by=row["reviewed_by"],
        applied_revision_id=row["applied_revision_id"],
        accepted_memory_id=row["accepted_memory_id"],
        error=row["error"],
        metadata=_json_dict(row["metadata_json"]),
    )


def _save_revision_in_connection(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    scope: str,
    manifest: SkillManifest,
    digest: str,
    encoded_package: str,
    proposal_id: str | None,
    status: SkillLifecycleStatus,
    increment_patch: bool,
    revision_id: str,
    now: str,
) -> str:
    existing_revision = conn.execute(
        """
        SELECT id FROM skill_package_revisions
        WHERE profile_id = ? AND scope = ? AND name = ? AND digest = ?
        """,
        (profile_id, scope, manifest.name, digest),
    ).fetchone()
    if existing_revision is not None:
        revision_id = str(existing_revision["id"])
    else:
        conn.execute(
            """
            INSERT INTO skill_package_revisions (
                id, profile_id, scope, name, version, digest, source, trust,
                package_json, manifest_json, proposal_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                profile_id,
                scope,
                manifest.name,
                manifest.version,
                digest,
                manifest.source,
                manifest.trust,
                encoded_package,
                json.dumps(manifest.to_dict(), sort_keys=True),
                proposal_id,
                now,
            ),
        )
    conn.execute(
        """
        INSERT INTO skill_packages (
            profile_id, scope, name, version, digest, source, trust, status,
            active_revision_id, patch_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, scope, name) DO UPDATE SET
            version = excluded.version,
            digest = excluded.digest,
            source = excluded.source,
            trust = excluded.trust,
            status = excluded.status,
            active_revision_id = excluded.active_revision_id,
            patch_count = skill_packages.patch_count + ?,
            updated_at = CASE
                WHEN skill_packages.version = excluded.version
                    AND skill_packages.digest = excluded.digest
                    AND skill_packages.source = excluded.source
                    AND skill_packages.trust = excluded.trust
                    AND skill_packages.status = excluded.status
                    AND skill_packages.active_revision_id
                        = excluded.active_revision_id
                    AND ? = 0
                THEN skill_packages.updated_at
                ELSE excluded.updated_at
            END
        """,
        (
            profile_id,
            scope,
            manifest.name,
            manifest.version,
            digest,
            manifest.source,
            manifest.trust,
            status.value,
            revision_id,
            int(increment_patch),
            now,
            now,
            int(increment_patch),
            int(increment_patch),
        ),
    )
    return revision_id


def _pending_skill_proposal(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    proposal_id: str,
) -> LearningProposalRecord:
    row = conn.execute(
        """
        SELECT * FROM learning_proposals
        WHERE id = ? AND profile_id = ?
        """,
        (proposal_id, profile_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"learning proposal {proposal_id!r} does not exist")
    proposal = _row_to_proposal(row)
    if not proposal.kind.value.startswith("skill_"):
        raise ValueError("proposal is not a skill change")
    return proposal


def _row_to_revision(row: sqlite3.Row) -> SkillRevisionRecord:
    return SkillRevisionRecord(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        scope=str(row["scope"]),
        name=str(row["name"]),
        version=str(row["version"]),
        digest=str(row["digest"]),
        source=str(row["source"]),
        trust=str(row["trust"]),
        package_files=_decode_package(str(row["package_json"])),
        manifest=_json_dict(row["manifest_json"]),
        created_at=str(row["created_at"]),
        proposal_id=row["proposal_id"],
    )


def _row_to_skill(row: sqlite3.Row) -> SkillLifecycleRecord:
    return SkillLifecycleRecord(
        profile_id=str(row["profile_id"]),
        scope=str(row["scope"]),
        name=str(row["name"]),
        version=str(row["version"]),
        digest=str(row["digest"]),
        source=str(row["source"]),
        trust=str(row["trust"]),
        status=SkillLifecycleStatus(str(row["status"])),
        active_revision_id=str(row["active_revision_id"]),
        view_count=int(row["view_count"]),
        use_count=int(row["use_count"]),
        success_count=int(row["success_count"]),
        patch_count=int(row["patch_count"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _encode_package(files: Mapping[str, bytes]) -> str:
    if not files:
        raise ValueError("skill package snapshot cannot be empty")
    encoded: dict[str, str] = {}
    total_bytes = 0
    for path, content in sorted(files.items()):
        clean_path = _required_text(
            path,
            "package path",
            max_chars=1_000,
        ).replace("\\", "/")
        pure_path = PurePosixPath(clean_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise ValueError(f"unsafe package snapshot path: {clean_path}")
        if not isinstance(content, bytes):
            raise TypeError("skill package snapshot values must be bytes")
        total_bytes += len(content)
        if total_bytes > 5_000_000:
            raise ValueError("skill package snapshot cannot exceed 5000000 bytes")
        encoded[clean_path] = base64.b64encode(content).decode("ascii")
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"))


def _decode_package(value: str) -> dict[str, bytes]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("stored skill package snapshot is invalid")
    return {
        str(path): base64.b64decode(str(content), validate=True)
        for path, content in payload.items()
    }


def _json_string_tuple(value: object) -> tuple[str, ...]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        return ()
    return tuple(str(item) for item in payload)


def _json_dict(value: object) -> dict[str, Any]:
    payload = json.loads(str(value))
    return payload if isinstance(payload, dict) else {}


def _required_text(value: object, label: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    clean = value.strip()
    if len(clean) > max_chars:
        raise ValueError(f"{label} cannot exceed {max_chars} characters")
    return clean


def _optional_text(
    value: object,
    label: str,
    *,
    max_chars: int,
) -> str | None:
    if value is None:
        return None
    return _required_text(value, label, max_chars=max_chars)


def _optional_blob(
    value: object,
    label: str,
    *,
    max_chars: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    if len(value) > max_chars:
        raise ValueError(f"{label} cannot exceed {max_chars} characters")
    return value


def _text_tuple(
    values: tuple[str, ...],
    label: str,
    *,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if len(values) > max_items:
        raise ValueError(f"{label} cannot contain more than {max_items} entries")
    result: list[str] = []
    for value in values:
        clean = _required_text(value, label, max_chars=max_chars)
        if clean not in result:
            result.append(clean)
    return tuple(result)


def _confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be a number")
    clean = float(value)
    if not 0.0 <= clean <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return clean


def _limit(value: int, *, maximum: int = 1_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("limit must be a positive integer")
    return min(value, maximum)


def _review_usage(
    *,
    proposal_count: int,
    token_count: int,
    cost_amount: Decimal,
) -> LearningReviewUsage:
    for label, value in (
        ("proposal_count", proposal_count),
        ("token_count", token_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if (
        not isinstance(cost_amount, Decimal)
        or not cost_amount.is_finite()
        or cost_amount < 0
    ):
        raise ValueError("cost_amount must be a finite non-negative Decimal")
    return LearningReviewUsage(
        proposal_count=proposal_count,
        token_count=token_count,
        cost_amount=cost_amount,
    )


def _review_usage_in_connection(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    occurred_at: str,
    currency: str,
) -> LearningReviewUsage:
    rows = conn.execute(
        """
        SELECT proposal_count, token_count, cost_amount, currency
        FROM learning_review_runs
        WHERE profile_id = ? AND created_at >= ?
        """,
        (profile_id, occurred_at),
    ).fetchall()
    currencies = {str(row["currency"]).upper() for row in rows}
    if currencies - {currency}:
        raise ValueError(
            "stored learning review usage uses a different currency"
        )
    proposal_count = sum(int(row["proposal_count"]) for row in rows)
    token_count = sum(int(row["token_count"]) for row in rows)
    try:
        cost_amount = sum(
            (Decimal(str(row["cost_amount"])) for row in rows),
            Decimal(0),
        )
    except InvalidOperation as exc:
        raise ValueError("stored learning review cost is invalid") from exc
    return LearningReviewUsage(
        proposal_count=proposal_count,
        token_count=token_count,
        cost_amount=cost_amount,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_profile_id(value: str) -> str:
    normalized = value.strip().lower()
    if _PROFILE_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "profile id must start with a letter and contain only lowercase "
            "letters, digits, underscores, or hyphens"
        )
    return normalized


def _scope(value: str) -> str:
    if value not in {"project", "profile"}:
        raise ValueError("skill scope must be 'project' or 'profile'")
    return value


def _currency(value: str) -> str:
    clean = value.strip().upper()
    if not clean or len(clean) > 16:
        raise ValueError("currency must be a short non-empty code")
    return clean


__all__ = ["SQLiteSkillLifecycleStore"]
