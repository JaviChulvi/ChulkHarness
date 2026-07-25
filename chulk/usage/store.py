"""SQLite-backed immutable usage ledger and transactional reservations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import base64
import binascii
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.usage.models import (
    BudgetExceededError,
    BudgetReservation,
    BudgetScope,
    ExactCost,
    PersistedModelUsage,
    ReservationState,
    ResourceKind,
    RunBudget,
    UnknownCostPolicy,
    UsageDimensions,
    UsageEntry,
    UsageAggregate,
    UsageGroupBy,
    UsagePage,
    UsageQuery,
)


DEFAULT_RESERVATION_TTL = timedelta(minutes=15)


class SQLiteUsageStore:
    """Persist exact usage entries and serialize concurrent budget checks."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl: timedelta = DEFAULT_RESERVATION_TTL,
    ) -> None:
        self.db_path = Path(db_path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if reservation_ttl <= timedelta(0):
            raise ValueError("reservation_ttl must be positive")
        self.reservation_ttl = reservation_ttl
        initialize_sqlite_database(self.db_path)

    def reserve(
        self,
        *,
        idempotency_key: str,
        source_event_id: str,
        resource_kind: ResourceKind,
        dimensions: UsageDimensions,
        budget: RunBudget,
        model_calls: int = 0,
        tool_calls: int = 0,
        tokens: int = 0,
        cost: ExactCost | None = None,
    ) -> BudgetReservation:
        """Atomically check committed plus held usage and reserve allowance."""
        clean_key = idempotency_key.strip()
        clean_source_id = source_event_id.strip()
        if not clean_key or not clean_source_id:
            raise ValueError("reservation identifiers cannot be empty")
        for name, value in (
            ("model_calls", model_calls),
            ("tool_calls", tool_calls),
            ("tokens", tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        reserved_cost = cost or ExactCost(None)
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("usage store clock must return a timezone-aware datetime")
        expires_at = now + self.reservation_ttl
        dimensions.scope_values(budget.scope)

        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM usage_reservations WHERE idempotency_key = ?",
                (clean_key,),
            ).fetchone()
            if existing is not None:
                return _row_to_reservation(existing)
            _release_expired_reservations(conn, now)
            committed = _committed_totals(conn, dimensions, budget.scope)
            held = _reserved_totals(conn, dimensions, budget.scope)
            _enforce_budget(
                budget,
                committed=committed,
                reserved=held,
                requested={
                    "model_calls": Decimal(model_calls),
                    "tool_calls": Decimal(tool_calls),
                    "tokens": Decimal(tokens),
                    "cost": reserved_cost.amount,
                    "unknown_cost": Decimal(int(not reserved_cost.pricing_known)),
                },
                now=now,
            )
            reservation = BudgetReservation(
                id=str(uuid4()),
                idempotency_key=clean_key,
                source_event_id=clean_source_id,
                resource_kind=resource_kind,
                dimensions=dimensions,
                budget=budget,
                state=ReservationState.ACTIVE,
                reserved_model_calls=model_calls,
                reserved_tool_calls=tool_calls,
                reserved_tokens=tokens,
                reserved_cost=reserved_cost,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
            )
            _insert_reservation(conn, reservation)
        return reservation

    def commit(
        self,
        reservation_id: str,
        entries: Iterable[UsageEntry],
    ) -> tuple[UsageEntry, ...]:
        """Insert immutable entries and close a reservation in one transaction."""
        values = tuple(entries)
        if not values:
            raise ValueError("at least one usage entry is required")
        now = self.clock()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM usage_reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"usage reservation {reservation_id!r} does not exist")
            reservation = _row_to_reservation(row)
            if reservation.state is ReservationState.RELEASED:
                raise ValueError("a released usage reservation cannot be committed")
            for entry in values:
                if entry.resource_kind is not reservation.resource_kind:
                    raise ValueError(
                        "usage entry resource kind does not match reservation"
                    )
                _insert_entry(conn, entry)
            stored_entries = tuple(
                _stored_entry(
                    conn,
                    resource_kind=entry.resource_kind,
                    source_event_id=entry.source_event_id,
                )
                for entry in values
            )
            conn.execute(
                """
                UPDATE usage_reservations
                SET state = ?, updated_at = ?, ledger_entry_ids_json = ?
                WHERE id = ?
                """,
                (
                    ReservationState.COMMITTED.value,
                    now.isoformat(),
                    json.dumps(
                        [entry.id for entry in stored_entries],
                        sort_keys=True,
                    ),
                    reservation_id,
                ),
            )
        return stored_entries

    def ingest(self, entry: UsageEntry) -> UsageEntry:
        """Insert one immutable entry without a reservation, idempotently."""
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _insert_entry(conn, entry)
            row = conn.execute(
                """
                SELECT * FROM usage_ledger
                WHERE resource_kind = ? AND source_event_id = ?
                """,
                (entry.resource_kind.value, entry.source_event_id),
            ).fetchone()
        assert row is not None
        return _row_to_entry(row)

    def release(self, reservation_id: str) -> BudgetReservation:
        """Release an active reservation without changing committed usage."""
        now = self.clock()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE usage_reservations
                SET state = ?, updated_at = ?
                WHERE id = ? AND state = ?
                """,
                (
                    ReservationState.RELEASED.value,
                    now.isoformat(),
                    reservation_id,
                    ReservationState.ACTIVE.value,
                ),
            )
            row = conn.execute(
                "SELECT * FROM usage_reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"usage reservation {reservation_id!r} does not exist")
        return _row_to_reservation(row)

    def get_reservation(self, reservation_id: str) -> BudgetReservation:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM usage_reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"usage reservation {reservation_id!r} does not exist")
        return _row_to_reservation(row)

    def active_constraint_reservations(
        self,
        source_event_id: str,
    ) -> tuple[BudgetReservation, ...]:
        """Return shared-scope reservations tied to one primary usage event."""
        clean_source = source_event_id.strip()
        if not clean_source:
            raise ValueError("source_event_id cannot be empty")
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM usage_reservations
                WHERE source_event_id = ? AND state = ?
                  AND idempotency_key LIKE ?
                ORDER BY budget_scope, id
                """,
                (
                    clean_source,
                    ReservationState.ACTIVE.value,
                    f"{clean_source}:constraint:%",
                ),
            ).fetchall()
        return tuple(_row_to_reservation(row) for row in rows)

    def reconcile_committed_constraints(self) -> int:
        """Close shared holds whose primary usage entry is already durable."""
        now = self.clock()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE usage_reservations AS reservations
                SET state = ?, updated_at = ?
                WHERE reservations.state = ?
                  AND reservations.idempotency_key LIKE '%:constraint:%'
                  AND EXISTS (
                      SELECT 1 FROM usage_ledger AS ledger
                      WHERE ledger.resource_kind = reservations.resource_kind
                        AND ledger.source_event_id LIKE
                            reservations.source_event_id || ':%'
                  )
                """,
                (
                    ReservationState.COMMITTED.value,
                    now.isoformat(),
                    ReservationState.ACTIVE.value,
                ),
            )
        return cursor.rowcount

    def list_entries(self, *, limit: int = 100) -> tuple[UsageEntry, ...]:
        clean_limit = max(1, min(limit, 10_000))
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM usage_ledger ORDER BY occurred_at, id LIMIT ?",
                (clean_limit,),
            ).fetchall()
        return tuple(_row_to_entry(row) for row in rows)

    def query(self, query: UsageQuery) -> UsagePage:
        """Return one ascending, cursor-paginated, profile-owned ledger page."""
        clauses = ["profile_id = ?"]
        parameters: list[object] = [query.profile_id]
        if query.start is not None:
            clauses.append("occurred_at >= ?")
            parameters.append(query.start.isoformat())
        if query.end is not None:
            clauses.append("occurred_at < ?")
            parameters.append(query.end.isoformat())
        for column, value in (
            (
                "resource_kind",
                query.resource_kind.value
                if query.resource_kind is not None
                else None,
            ),
            ("channel", query.channel),
            ("conversation_id", query.conversation_id),
            ("goal_id", query.goal_id),
            ("job_id", query.job_id),
            ("child_task_id", query.child_task_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if query.cursor is not None:
            occurred_at, entry_id = _decode_cursor(query.cursor)
            clauses.append("(occurred_at > ? OR (occurred_at = ? AND id > ?))")
            parameters.extend((occurred_at, occurred_at, entry_id))
        parameters.append(query.limit + 1)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM usage_ledger
                WHERE {' AND '.join(clauses)}
                ORDER BY occurred_at, id
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        has_more = len(rows) > query.limit
        page_rows = rows[: query.limit]
        entries = tuple(_row_to_entry(row) for row in page_rows)
        next_cursor = (
            _encode_cursor(
                entries[-1].occurred_at.isoformat(),
                entries[-1].id,
            )
            if has_more and entries
            else None
        )
        return UsagePage(entries, next_cursor)

    def aggregate(
        self,
        query: UsageQuery,
        *,
        group_by: UsageGroupBy,
    ) -> tuple[UsageAggregate, ...]:
        """Aggregate one bounded query without floating-point cost arithmetic."""
        page = self.query(query)
        if page.next_cursor is not None:
            raise ValueError(
                "usage aggregation exceeds the bounded query limit; narrow the range"
            )
        groups: dict[str, list[UsageEntry]] = {}
        for entry in page.entries:
            key = _group_key(entry, group_by)
            groups.setdefault(key, []).append(entry)
        return tuple(
            _aggregate_entries(key, entries)
            for key, entries in sorted(groups.items())
        )

    def list_reservations(
        self,
        *,
        state: ReservationState | None = None,
        limit: int = 100,
    ) -> tuple[BudgetReservation, ...]:
        clean_limit = max(1, min(limit, 10_000))
        where = "WHERE state = ?" if state is not None else ""
        parameters: tuple[object, ...] = (
            (state.value, clean_limit) if state is not None else (clean_limit,)
        )
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM usage_reservations
                {where}
                ORDER BY created_at, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return tuple(_row_to_reservation(row) for row in rows)

    def checkpoint_model_response(
        self,
        reservation_id: str,
        *,
        request_index: int,
        purpose: str,
        usage: dict[str, object] | None,
        cost: dict[str, object] | None,
        fallback_attempts: tuple[dict[str, object], ...],
        provider: str | None,
        model: str | None,
        model_profile_id: str | None,
        credential_ref: str | None,
        trace_path: str | None,
    ) -> datetime | None:
        """Persist the accounting facts before closing their reservation."""
        if request_index < 1:
            raise ValueError("model request index must be positive")
        clean_purpose = purpose.strip()
        if not clean_purpose:
            raise ValueError("model usage purpose cannot be empty")
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("usage store clock must return a timezone-aware datetime")
        checkpoint = {
            "purpose": clean_purpose,
            "usage": usage,
            "cost": cost,
            "fallback_attempts": list(fallback_attempts),
            "provider": provider,
            "model": model,
            "model_profile_id": model_profile_id,
            "credential_ref": credential_ref,
            "trace_path": trace_path,
        }
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            reservation_row = conn.execute(
                "SELECT * FROM usage_reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if reservation_row is None:
                raise KeyError(
                    f"usage reservation {reservation_id!r} does not exist"
                )
            reservation = _row_to_reservation(reservation_row)
            if (
                reservation.dimensions.conversation_id is None
                or reservation.dimensions.turn_id is None
            ):
                return None
            cursor = conn.execute(
                """
                UPDATE conversation_model_requests
                SET usage_json = ?,
                    cost_json = ?,
                    accounting_json = ?,
                    response_created_at = COALESCE(response_created_at, ?)
                WHERE conversation_id = ?
                  AND turn_id = ?
                  AND request_index = ?
                """,
                (
                    json.dumps(usage, sort_keys=True) if usage is not None else None,
                    json.dumps(cost, sort_keys=True) if cost is not None else None,
                    json.dumps(checkpoint, sort_keys=True),
                    now.isoformat(),
                    reservation.dimensions.conversation_id,
                    reservation.dimensions.turn_id,
                    request_index,
                ),
            )
        return now if cursor.rowcount == 1 else None

    def recoverable_model_usage(
        self,
        *,
        profile_id: str,
        limit: int = 10_000,
    ) -> tuple[PersistedModelUsage, ...]:
        """Load active model holds whose provider result is already durable."""
        clean_profile_id = profile_id.strip()
        if not clean_profile_id:
            raise ValueError("usage recovery profile_id cannot be empty")
        clean_limit = max(1, min(limit, 10_000))
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT reservations.*,
                       requests.request_index AS recovery_request_index,
                       requests.request_json AS recovery_request_json,
                       requests.usage_json AS recovery_usage_json,
                       requests.cost_json AS recovery_cost_json,
                       requests.accounting_json AS recovery_accounting_json,
                       requests.response_created_at AS recovery_occurred_at
                FROM usage_reservations AS reservations
                JOIN conversation_model_requests AS requests
                  ON requests.conversation_id = reservations.conversation_id
                 AND requests.turn_id = reservations.turn_id
                 AND reservations.source_event_id = (
                     'model:' || requests.conversation_id || ':' ||
                     requests.turn_id || ':' || requests.request_index
                 )
                WHERE reservations.state = ?
                  AND reservations.resource_kind = ?
                  AND reservations.profile_id = ?
                  AND requests.response_created_at IS NOT NULL
                ORDER BY reservations.created_at, reservations.id
                LIMIT ?
                """,
                (
                    ReservationState.ACTIVE.value,
                    ResourceKind.MODEL.value,
                    clean_profile_id,
                    clean_limit,
                ),
            ).fetchall()
        return tuple(_row_to_persisted_model_usage(row) for row in rows)

    def release_expired(self) -> int:
        """Release expired active reservations and return the affected count."""
        now = self.clock()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return _release_expired_reservations(conn, now)


def _insert_entry(conn: sqlite3.Connection, entry: UsageEntry) -> None:
    units = {key: str(value) for key, value in entry.units.items()}
    conn.execute(
        """
        INSERT INTO usage_ledger (
            id, resource_kind, source_event_id, provider, model,
            tool_or_service, credential_ref, model_profile_id,
            profile_id, channel, conversation_id, turn_id, goal_id, job_id,
            child_task_id, purpose, occurred_at, billing_period, units_json,
            model_calls, tool_calls, total_tokens, cost_amount, currency,
            pricing_known, cost_estimated, cost_reported, usage_estimated,
            trace_path, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resource_kind, source_event_id) DO NOTHING
        """,
        (
            entry.id,
            entry.resource_kind.value,
            entry.source_event_id,
            entry.provider,
            entry.model,
            entry.tool_or_service,
            entry.credential_ref,
            entry.model_profile_id,
            entry.dimensions.profile_id,
            entry.dimensions.channel,
            entry.dimensions.conversation_id,
            entry.dimensions.turn_id,
            entry.dimensions.goal_id,
            entry.dimensions.job_id,
            entry.dimensions.child_task_id,
            entry.purpose,
            entry.occurred_at.isoformat(),
            entry.billing_period,
            json.dumps(units, sort_keys=True),
            int(entry.units.get("model_calls", Decimal(0))),
            int(entry.units.get("tool_calls", Decimal(0))),
            int(entry.units.get("total_tokens", Decimal(0))),
            str(entry.cost.amount) if entry.cost.amount is not None else None,
            entry.cost.currency,
            int(entry.cost.pricing_known),
            int(entry.cost.estimated),
            int(entry.cost.reported),
            int(entry.usage_estimated),
            entry.trace_path,
            json.dumps(dict(entry.metadata), sort_keys=True),
        ),
    )


def _stored_entry(
    conn: sqlite3.Connection,
    *,
    resource_kind: ResourceKind,
    source_event_id: str,
) -> UsageEntry:
    row = conn.execute(
        """
        SELECT * FROM usage_ledger
        WHERE resource_kind = ? AND source_event_id = ?
        """,
        (resource_kind.value, source_event_id),
    ).fetchone()
    if row is None:  # pragma: no cover - guarded by the preceding insert
        raise RuntimeError("usage ledger insert did not produce a durable row")
    return _row_to_entry(row)


def _insert_reservation(
    conn: sqlite3.Connection,
    reservation: BudgetReservation,
) -> None:
    conn.execute(
        """
        INSERT INTO usage_reservations (
            id, idempotency_key, source_event_id, resource_kind,
            profile_id, channel, conversation_id, turn_id, goal_id, job_id,
            child_task_id, budget_scope, budget_json, state,
            reserved_model_calls, reserved_tool_calls, reserved_tokens,
            reserved_cost_amount, currency, pricing_known, unknown_cost,
            created_at, updated_at, expires_at, ledger_entry_ids_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?)
        """,
        (
            reservation.id,
            reservation.idempotency_key,
            reservation.source_event_id,
            reservation.resource_kind.value,
            reservation.dimensions.profile_id,
            reservation.dimensions.channel,
            reservation.dimensions.conversation_id,
            reservation.dimensions.turn_id,
            reservation.dimensions.goal_id,
            reservation.dimensions.job_id,
            reservation.dimensions.child_task_id,
            reservation.budget.scope.value,
            json.dumps(reservation.budget.to_dict(), sort_keys=True),
            reservation.state.value,
            reservation.reserved_model_calls,
            reservation.reserved_tool_calls,
            reservation.reserved_tokens,
            (
                str(reservation.reserved_cost.amount)
                if reservation.reserved_cost.amount is not None
                else None
            ),
            reservation.reserved_cost.currency,
            int(reservation.reserved_cost.pricing_known),
            int(not reservation.reserved_cost.pricing_known),
            _iso(reservation.created_at),
            _iso(reservation.updated_at),
            _iso(reservation.expires_at),
            "[]",
        ),
    )


def _committed_totals(
    conn: sqlite3.Connection,
    dimensions: UsageDimensions,
    scope: BudgetScope,
) -> dict[str, Decimal]:
    where, params = _scope_where(dimensions, scope)
    rows = conn.execute(
        f"""
        SELECT model_calls, tool_calls, total_tokens, cost_amount, pricing_known
        FROM usage_ledger
        WHERE {where}
        """,
        params,
    ).fetchall()
    return _sum_totals(rows)


def _reserved_totals(
    conn: sqlite3.Connection,
    dimensions: UsageDimensions,
    scope: BudgetScope,
) -> dict[str, Decimal]:
    where, params = _scope_where(dimensions, scope)
    rows = conn.execute(
        f"""
        SELECT reserved_model_calls AS model_calls,
               reserved_tool_calls AS tool_calls,
               reserved_tokens AS total_tokens,
               reserved_cost_amount AS cost_amount,
               pricing_known
        FROM usage_reservations
        WHERE state = ? AND budget_scope = ? AND {where}
        """,
        (ReservationState.ACTIVE.value, scope.value, *params),
    ).fetchall()
    return _sum_totals(rows)


def _sum_totals(rows: Iterable[sqlite3.Row]) -> dict[str, Decimal]:
    totals = {
        "model_calls": Decimal(0),
        "tool_calls": Decimal(0),
        "tokens": Decimal(0),
        "cost": Decimal(0),
        "unknown_cost": Decimal(0),
    }
    for row in rows:
        totals["model_calls"] += Decimal(int(row["model_calls"]))
        totals["tool_calls"] += Decimal(int(row["tool_calls"]))
        totals["tokens"] += Decimal(int(row["total_tokens"]))
        if row["cost_amount"] is not None:
            totals["cost"] += Decimal(str(row["cost_amount"]))
        if not bool(row["pricing_known"]):
            totals["unknown_cost"] += Decimal(1)
    return totals


def _scope_where(
    dimensions: UsageDimensions,
    scope: BudgetScope,
) -> tuple[str, tuple[str, ...]]:
    values = dimensions.scope_values(scope)
    clauses = [f"{key} = ?" for key in values]
    return " AND ".join(clauses), tuple(values.values())


def _enforce_budget(
    budget: RunBudget,
    *,
    committed: dict[str, Decimal],
    reserved: dict[str, Decimal],
    requested: dict[str, Decimal | None],
    now: datetime,
) -> None:
    if budget.deadline is not None and now >= budget.deadline:
        raise BudgetExceededError(
            scope=budget.scope,
            dimension="deadline",
            limit=budget.deadline.isoformat(),
            committed="0",
            reserved="0",
            requested=now.isoformat(),
        )
    limits: tuple[tuple[str, int | Decimal | None], ...] = (
        ("model_calls", budget.max_model_calls),
        ("tool_calls", budget.max_tool_calls),
        ("tokens", budget.max_tokens),
        (
            "cost",
            budget.max_cost.amount if budget.max_cost is not None else None,
        ),
    )
    if (
        budget.max_cost is not None
        and budget.unknown_cost_policy is UnknownCostPolicy.FAIL_CLOSED
        and (
            committed["unknown_cost"]
            + reserved["unknown_cost"]
            + (requested["unknown_cost"] or Decimal(0))
        )
        > 0
    ):
        raise BudgetExceededError(
            scope=budget.scope,
            dimension="cost",
            limit=str(budget.max_cost.amount),
            committed="unknown",
            reserved="unknown",
            requested="unknown",
        )
    for dimension, raw_limit in limits:
        if raw_limit is None:
            continue
        limit = Decimal(raw_limit)
        committed_value = committed[dimension]
        reserved_value = reserved[dimension]
        requested_value = requested[dimension] or Decimal(0)
        if committed_value + reserved_value + requested_value > limit:
            raise BudgetExceededError(
                scope=budget.scope,
                dimension=dimension,
                limit=str(limit),
                committed=str(committed_value),
                reserved=str(reserved_value),
                requested=str(requested_value),
            )


def _release_expired_reservations(
    conn: sqlite3.Connection,
    now: datetime,
) -> int:
    cursor = conn.execute(
        """
        UPDATE usage_reservations
        SET state = ?, updated_at = ?
        WHERE state = ? AND expires_at <= ?
        """,
        (
            ReservationState.RELEASED.value,
            now.isoformat(),
            ReservationState.ACTIVE.value,
            now.isoformat(),
        ),
    )
    return cursor.rowcount


def _row_to_entry(row: sqlite3.Row) -> UsageEntry:
    units_payload = json.loads(str(row["units_json"]))
    return UsageEntry(
        id=str(row["id"]),
        resource_kind=ResourceKind(str(row["resource_kind"])),
        source_event_id=str(row["source_event_id"]),
        dimensions=_dimensions_from_row(row),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        billing_period=str(row["billing_period"]),
        purpose=str(row["purpose"]),
        units={
            str(key): Decimal(str(value)) for key, value in units_payload.items()
        },
        cost=ExactCost(
            Decimal(str(row["cost_amount"]))
            if row["cost_amount"] is not None
            else None,
            currency=str(row["currency"]),
            pricing_known=bool(row["pricing_known"]),
            estimated=bool(row["cost_estimated"]),
            reported=bool(row["cost_reported"]),
        ),
        provider=row["provider"],
        model=row["model"],
        tool_or_service=row["tool_or_service"],
        credential_ref=row["credential_ref"],
        model_profile_id=row["model_profile_id"],
        usage_estimated=bool(row["usage_estimated"]),
        trace_path=row["trace_path"],
        metadata=json.loads(str(row["metadata_json"])),
    )


def _row_to_reservation(row: sqlite3.Row) -> BudgetReservation:
    budget_payload = json.loads(str(row["budget_json"]))
    max_cost_payload = budget_payload.get("max_cost")
    budget = RunBudget(
        scope=BudgetScope(str(budget_payload["scope"])),
        max_model_calls=budget_payload.get("max_model_calls"),
        max_tool_calls=budget_payload.get("max_tool_calls"),
        max_tokens=budget_payload.get("max_tokens"),
        max_cost=(
            ExactCost(
                Decimal(str(max_cost_payload["amount"])),
                currency=str(max_cost_payload.get("currency") or "USD"),
                pricing_known=True,
            )
            if isinstance(max_cost_payload, dict)
            and max_cost_payload.get("amount") is not None
            else None
        ),
        deadline=(
            datetime.fromisoformat(str(budget_payload["deadline"]))
            if budget_payload.get("deadline")
            else None
        ),
        unknown_cost_policy=UnknownCostPolicy(
            str(budget_payload["unknown_cost_policy"])
        ),
    )
    return BudgetReservation(
        id=str(row["id"]),
        idempotency_key=str(row["idempotency_key"]),
        source_event_id=str(row["source_event_id"]),
        resource_kind=ResourceKind(str(row["resource_kind"])),
        dimensions=_dimensions_from_row(row),
        budget=budget,
        state=ReservationState(str(row["state"])),
        reserved_model_calls=int(row["reserved_model_calls"]),
        reserved_tool_calls=int(row["reserved_tool_calls"]),
        reserved_tokens=int(row["reserved_tokens"]),
        reserved_cost=ExactCost(
            Decimal(str(row["reserved_cost_amount"]))
            if row["reserved_cost_amount"] is not None
            else None,
            currency=str(row["currency"]),
            pricing_known=bool(row["pricing_known"]),
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        expires_at=datetime.fromisoformat(str(row["expires_at"])),
    )


def _dimensions_from_row(row: sqlite3.Row) -> UsageDimensions:
    return UsageDimensions(
        profile_id=str(row["profile_id"]),
        channel=row["channel"],
        conversation_id=row["conversation_id"],
        turn_id=row["turn_id"],
        goal_id=row["goal_id"],
        job_id=row["job_id"],
        child_task_id=row["child_task_id"],
    )


def _row_to_persisted_model_usage(row: sqlite3.Row) -> PersistedModelUsage:
    accounting = _json_object(row["recovery_accounting_json"])
    request = _json_object(row["recovery_request_json"])
    usage = _mapping_or_none(accounting.get("usage")) or _json_object_or_none(
        row["recovery_usage_json"]
    )
    cost = _mapping_or_none(accounting.get("cost")) or _json_object_or_none(
        row["recovery_cost_json"]
    )
    raw_attempts = accounting.get("fallback_attempts")
    attempts = (
        tuple(value for value in raw_attempts if isinstance(value, dict))
        if isinstance(raw_attempts, list)
        else ()
    )
    occurred_at = datetime.fromisoformat(str(row["recovery_occurred_at"]))
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    raw_purpose = accounting.get("purpose")
    if isinstance(raw_purpose, str) and raw_purpose.strip():
        purpose = raw_purpose
    else:
        request_purpose = request.get("purpose")
        context_report = request.get("context_report")
        context_purpose = (
            context_report.get("purpose")
            if isinstance(context_report, dict)
            else None
        )
        purpose = (
            _clean_string(request_purpose)
            or _clean_string(context_purpose)
            or "agent_action"
        )
    return PersistedModelUsage(
        reservation=_row_to_reservation(row),
        request_index=int(row["recovery_request_index"]),
        purpose=purpose,
        occurred_at=occurred_at,
        usage=usage,
        cost=cost,
        fallback_attempts=attempts,
        provider=_optional_string(accounting.get("provider")),
        model=_optional_string(accounting.get("model")),
        model_profile_id=_optional_string(accounting.get("model_profile_id")),
        credential_ref=_optional_string(accounting.get("credential_ref")),
        trace_path=_optional_string(accounting.get("trace_path")),
    )


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_object_or_none(value: object) -> dict[str, object] | None:
    payload = _json_object(value)
    return payload or None


def _mapping_or_none(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _group_key(entry: UsageEntry, group_by: UsageGroupBy) -> str:
    if group_by is UsageGroupBy.RESOURCE_KIND:
        return entry.resource_kind.value
    if group_by is UsageGroupBy.MODEL:
        provider = entry.provider or "unknown"
        model = entry.model or "unknown"
        return f"{provider}:{model}"
    if group_by is UsageGroupBy.TOOL_SERVICE:
        return entry.tool_or_service or "unknown"
    if group_by is UsageGroupBy.PROFILE:
        return entry.dimensions.profile_id
    if group_by is UsageGroupBy.CHANNEL:
        return entry.dimensions.channel or "unknown"
    if group_by is UsageGroupBy.GOAL:
        return entry.dimensions.goal_id or "none"
    if group_by is UsageGroupBy.JOB:
        return entry.dimensions.job_id or "none"
    return entry.dimensions.child_task_id or "none"


def _aggregate_entries(
    key: str,
    entries: list[UsageEntry],
) -> UsageAggregate:
    currencies = {entry.cost.currency for entry in entries}
    known_amount = sum(
        (
            entry.cost.amount
            for entry in entries
            if entry.cost.amount is not None
        ),
        start=Decimal(0),
    )
    unknown = sum(
        1
        for entry in entries
        if not entry.cost.pricing_known or entry.cost.amount is None
    )
    currency = next(iter(currencies)) if len(currencies) == 1 else "MIXED"
    return UsageAggregate(
        key=key,
        entry_count=len(entries),
        model_calls=sum(
            int(entry.units.get("model_calls", Decimal(0))) for entry in entries
        ),
        tool_calls=sum(
            int(entry.units.get("tool_calls", Decimal(0))) for entry in entries
        ),
        total_tokens=sum(
            int(entry.units.get("total_tokens", Decimal(0))) for entry in entries
        ),
        cost=ExactCost(
            known_amount if len(currencies) == 1 else None,
            currency=currency,
            pricing_known=unknown == 0 and len(currencies) == 1,
            estimated=any(entry.cost.estimated for entry in entries),
            reported=all(entry.cost.reported for entry in entries),
        ),
        unknown_cost_entries=unknown,
    )


def _encode_cursor(occurred_at: str, entry_id: str) -> str:
    payload = json.dumps(
        {"occurred_at": occurred_at, "id": entry_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[str, str]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode((value + padding).encode()).decode()
        )
        occurred_at = str(payload["occurred_at"])
        entry_id = str(payload["id"])
        parsed_at = datetime.fromisoformat(occurred_at)
        if parsed_at.tzinfo is None:
            raise ValueError("usage cursor timestamp must be timezone-aware")
    except (
        binascii.Error,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("invalid usage cursor") from exc
    if not entry_id:
        raise ValueError("invalid usage cursor")
    return occurred_at, entry_id


__all__ = ["DEFAULT_RESERVATION_TTL", "SQLiteUsageStore"]
