"""SQLite-backed scheduled jobs with short execution leases."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from chulk.scheduling.models import ScheduledJob
from chulk.storage import initialize_sqlite_database, sqlite_connection

MAX_SCHEDULED_PROMPT_CHARS = 8_000


class SQLiteScheduleStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        initialize_sqlite_database(self.db_path)

    def create(
        self,
        *,
        adapter: str,
        destination_id: str,
        prompt: str,
        next_run_at: datetime,
        interval_seconds: int | None = None,
    ) -> ScheduledJob:
        if next_run_at.tzinfo is None:
            raise ValueError("next_run_at must include a timezone")
        if interval_seconds is not None and interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt is required")
        if len(clean_prompt) > MAX_SCHEDULED_PROMPT_CHARS:
            raise ValueError(
                f"prompt exceeds the {MAX_SCHEDULED_PROMPT_CHARS}-character limit"
            )
        if not adapter.strip() or not destination_id.strip():
            raise ValueError("adapter and destination_id are required")
        job_id = uuid4().hex
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scheduled_jobs (
                    id, adapter, destination_id, prompt, next_run_at,
                    interval_seconds, status, scheduled_for, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    job_id,
                    adapter,
                    destination_id,
                    clean_prompt,
                    _encode(next_run_at),
                    interval_seconds,
                    _encode(next_run_at),
                    _encode(now),
                    _encode(now),
                ),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> ScheduledJob:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM scheduled_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Scheduled job not found: {job_id}")
        return _row_to_job(row)

    def list(self, *, adapter: str, destination_id: str) -> tuple[ScheduledJob, ...]:
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM scheduled_jobs
                WHERE adapter = ? AND destination_id = ?
                  AND status IN ('active', 'running', 'paused')
                ORDER BY next_run_at, created_at
                """,
                (adapter, destination_id),
            ).fetchall()
        return tuple(_row_to_job(row) for row in rows)

    def cancel(self, job_id: str, *, adapter: str, destination_id: str) -> bool:
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_jobs
                SET status = 'cancelled', lease_until = NULL, claim_token = NULL,
                    updated_at = ?
                WHERE id = ? AND adapter = ? AND destination_id = ?
                  AND status IN ('active', 'running', 'paused')
                """,
                (_encode(_utc_now()), job_id, adapter, destination_id),
            )
        return cursor.rowcount == 1

    def claim_due(
        self,
        *,
        adapter: str,
        now: datetime | None = None,
        limit: int = 10,
        lease_seconds: int = 120,
    ) -> tuple[ScheduledJob, ...]:
        observed = (now or _utc_now()).astimezone(timezone.utc)
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id FROM scheduled_jobs
                WHERE adapter = ?
                  AND next_run_at <= ?
                  AND (
                    status = 'active'
                    OR (status = 'running' AND lease_until < ?)
                  )
                ORDER BY next_run_at
                LIMIT ?
                """,
                (adapter, _encode(observed), _encode(observed), limit),
            ).fetchall()
            claims: list[tuple[str, str]] = []
            for row in rows:
                job_id = str(row["id"])
                claim_token = uuid4().hex
                cursor = conn.execute(
                    """
                    UPDATE scheduled_jobs
                    SET status = 'running', lease_until = ?, claim_token = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND (
                        status = 'active'
                        OR (status = 'running' AND lease_until < ?)
                      )
                    """,
                    (
                        _encode(lease_until),
                        claim_token,
                        _encode(observed),
                        job_id,
                        _encode(observed),
                    ),
                )
                if cursor.rowcount == 1:
                    claims.append((job_id, claim_token))
            claimed = [
                conn.execute(
                    "SELECT * FROM scheduled_jobs WHERE id = ? AND claim_token = ?",
                    (job_id, claim_token),
                ).fetchone()
                for job_id, claim_token in claims
            ]
        return tuple(_row_to_job(row) for row in claimed if row is not None)

    def renew_lease(
        self,
        job_id: str,
        claim_token: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> bool:
        """Extend an unexpired lease only for its current owner."""
        if not claim_token:
            raise ValueError("claim_token is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = (now or _utc_now()).astimezone(timezone.utc)
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_jobs
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND claim_token = ?
                  AND lease_until >= ?
                """,
                (
                    _encode(lease_until),
                    _encode(observed),
                    job_id,
                    claim_token,
                    _encode(observed),
                ),
            )
        return cursor.rowcount == 1

    def complete(
        self,
        job_id: str,
        claim_token: str,
        *,
        finished_at: datetime | None = None,
    ) -> bool:
        """Complete the current unexpired claim and advance its stable cadence."""
        if not claim_token:
            raise ValueError("claim_token is required")
        finished = (finished_at or _utc_now()).astimezone(timezone.utc)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT scheduled_for, interval_seconds
                FROM scheduled_jobs
                WHERE id = ? AND status = 'running' AND claim_token = ?
                  AND lease_until >= ?
                """,
                (job_id, claim_token, _encode(finished)),
            ).fetchone()
            if row is None:
                return False
            interval = row["interval_seconds"]
            if interval is None:
                status = "completed"
                next_run = str(row["scheduled_for"])
                scheduled_for = next_run
            else:
                status = "active"
                anchor = _decode(str(row["scheduled_for"]))
                interval_seconds = int(interval)
                if finished < anchor:
                    intervals = 0
                else:
                    elapsed_seconds = (finished - anchor).total_seconds()
                    intervals = int(elapsed_seconds // interval_seconds) + 1
                next_dt = anchor + timedelta(seconds=interval_seconds * intervals)
                next_run = _encode(next_dt)
                scheduled_for = next_run
            cursor = conn.execute(
                """
                UPDATE scheduled_jobs
                SET status = ?, next_run_at = ?, scheduled_for = ?,
                    lease_until = NULL, claim_token = NULL,
                    last_run_at = ?, last_error = NULL, updated_at = ?
                WHERE id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    status,
                    next_run,
                    scheduled_for,
                    _encode(finished),
                    _encode(finished),
                    job_id,
                    claim_token,
                ),
            )
        return cursor.rowcount == 1

    def fail(
        self,
        job_id: str,
        claim_token: str,
        error: str,
        *,
        retry_seconds: int = 60,
        failed_at: datetime | None = None,
    ) -> bool:
        """Release the current claim for retry without moving its cadence anchor."""
        if not claim_token:
            raise ValueError("claim_token is required")
        if retry_seconds <= 0:
            raise ValueError("retry_seconds must be greater than zero")
        now = (failed_at or _utc_now()).astimezone(timezone.utc)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_jobs
                SET status = 'active', next_run_at = ?, lease_until = NULL,
                    claim_token = NULL,
                    last_error = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND claim_token = ?
                  AND lease_until >= ?
                """,
                (
                    _encode(now + timedelta(seconds=retry_seconds)),
                    error[:500],
                    _encode(now),
                    job_id,
                    claim_token,
                    _encode(now),
                ),
            )
        return cursor.rowcount == 1


def _row_to_job(row: sqlite3.Row) -> ScheduledJob:
    return ScheduledJob(
        id=str(row["id"]),
        adapter=str(row["adapter"]),
        destination_id=str(row["destination_id"]),
        prompt=str(row["prompt"]),
        next_run_at=_decode(str(row["next_run_at"])),
        interval_seconds=int(row["interval_seconds"]) if row["interval_seconds"] is not None else None,
        status=str(row["status"]),
        scheduled_for=_decode(str(row["scheduled_for"])),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        lease_until=_decode(str(row["lease_until"])) if row["lease_until"] else None,
        last_run_at=_decode(str(row["last_run_at"])) if row["last_run_at"] else None,
        last_error=str(row["last_error"]) if row["last_error"] else None,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


__all__ = ["SQLiteScheduleStore"]
