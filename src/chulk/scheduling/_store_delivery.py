"""Delivery-attempt operations for scheduled runs."""

from __future__ import annotations

from datetime import datetime
import sqlite3
from uuid import uuid4

from chulk.scheduling._store_support import (
    AutomationNotFoundError,
    _ScheduleStoreMixin,
    _encode,
    _row_to_delivery_attempt,
    _utc_now,
)
from chulk.scheduling.models import (
    AutomationDeliveryAttempt,
    AutomationDeliveryState,
    AutomationRun,
)


class _ScheduleDeliveryMixin(_ScheduleStoreMixin):
    def delivery_history(
        self,
        run_id: str,
    ) -> tuple[AutomationDeliveryAttempt, ...]:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_delivery_attempts
                WHERE profile_id = ? AND run_id = ?
                ORDER BY created_at, id
                """,
                (self.profile_id, run_id),
            ).fetchall()
        return tuple(_row_to_delivery_attempt(row) for row in rows)

    def mark_delivery(
        self,
        run_id: str,
        state: AutomationDeliveryState,
        *,
        error: str | None = None,
    ) -> AutomationRun:
        selected = AutomationDeliveryState(state)
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_runs SET delivery_state = ?, delivery_error = ?,
                    updated_at = ? WHERE profile_id = ? AND id = ?
                """,
                (
                    selected.value,
                    error[:500] if error else None,
                    _encode(now),
                    self.profile_id,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise AutomationNotFoundError(f"Automation run not found: {run_id}")
            self._delivery_attempt(
                conn,
                run_id=run_id,
                state=selected,
                error=error,
                now=now,
            )
            return self._run_in(conn, run_id)

    def _delivery_attempt(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        state: AutomationDeliveryState,
        now: datetime,
        error: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_delivery_attempts (
                id, run_id, profile_id, state, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                run_id,
                self.profile_id,
                state.value,
                error[:500] if error else None,
                _encode(now),
            ),
        )
