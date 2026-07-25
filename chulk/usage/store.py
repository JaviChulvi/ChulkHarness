"""SQLite-backed immutable usage ledger and transactional reservations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
    ReservationState,
    ResourceKind,
    RunBudget,
    UnknownCostPolicy,
    UsageDimensions,
    UsageEntry,
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
            conn.execute(
                """
                UPDATE usage_reservations
                SET state = ?, updated_at = ?, ledger_entry_ids_json = ?
                WHERE id = ?
                """,
                (
                    ReservationState.COMMITTED.value,
                    now.isoformat(),
                    json.dumps([entry.id for entry in values], sort_keys=True),
                    reservation_id,
                ),
            )
        return values

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

    def list_entries(self, *, limit: int = 100) -> tuple[UsageEntry, ...]:
        clean_limit = max(1, min(limit, 10_000))
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM usage_ledger ORDER BY occurred_at, id LIMIT ?",
                (clean_limit,),
            ).fetchall()
        return tuple(_row_to_entry(row) for row in rows)

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
        WHERE state = ? AND {where}
        """,
        (ReservationState.ACTIVE.value, *params),
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


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = ["DEFAULT_RESERVATION_TTL", "SQLiteUsageStore"]
