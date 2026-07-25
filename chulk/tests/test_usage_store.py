from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
from pathlib import Path
from threading import Barrier

import pytest

from chulk.usage import (
    BudgetExceededError,
    BudgetScope,
    ExactCost,
    ReservationState,
    ResourceKind,
    RunBudget,
    SQLiteUsageStore,
    UnknownCostPolicy,
    UsageGroupBy,
    UsageLedger,
    UsageQuery,
    UsageDimensions,
    UsageEntry,
)


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def _dimensions(
    *,
    profile_id: str = "work",
    conversation_id: str = "conversation-1",
    turn_id: str = "turn-1",
) -> UsageDimensions:
    return UsageDimensions(
        profile_id=profile_id,
        channel="cli",
        conversation_id=conversation_id,
        turn_id=turn_id,
    )


def _budget(
    amount: str = "1.00",
    *,
    policy: UnknownCostPolicy = UnknownCostPolicy.FAIL_CLOSED,
) -> RunBudget:
    return RunBudget(
        scope=BudgetScope.TURN,
        max_model_calls=2,
        max_tokens=10_000,
        max_cost=ExactCost(
            Decimal(amount),
            currency="USD",
            pricing_known=True,
        ),
        unknown_cost_policy=policy,
    )


def _entry(
    source_event_id: str,
    *,
    amount: str = "0.125",
    dimensions: UsageDimensions | None = None,
    occurred_at: datetime = NOW,
    credential_ref: str | None = None,
    model: str = "gpt-4.1-mini",
) -> UsageEntry:
    return UsageEntry(
        id=f"entry-{source_event_id}",
        resource_kind=ResourceKind.MODEL,
        source_event_id=source_event_id,
        dimensions=dimensions or _dimensions(),
        occurred_at=occurred_at,
        billing_period="2026-07",
        purpose="agent_action",
        units={
            "model_calls": Decimal(1),
            "input_tokens": Decimal(100),
            "output_tokens": Decimal(20),
            "total_tokens": Decimal(120),
        },
        cost=ExactCost(
            Decimal(amount),
            currency="usd",
            pricing_known=True,
            estimated=True,
        ),
        provider="openai",
        model=model,
        credential_ref=credential_ref,
        model_profile_id="fast",
    )


def test_usage_entry_preserves_exact_decimal_cost_and_typed_units(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)

    stored = store.ingest(_entry("request-1", amount="0.123456789"))

    assert stored.cost.amount == Decimal("0.123456789")
    assert stored.cost.currency == "USD"
    assert stored.units["total_tokens"] == Decimal(120)
    assert stored.dimensions.profile_id == "work"


def test_usage_ingestion_is_idempotent_by_kind_and_source_event(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)

    first = store.ingest(_entry("request-1", amount="0.10"))
    repeated = store.ingest(_entry("request-1", amount="9.99"))

    assert first.id == repeated.id
    assert repeated.cost.amount == Decimal("0.10")
    assert len(store.list_entries()) == 1


def test_reservation_counts_committed_and_active_allowance(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    dimensions = _dimensions()
    first = store.reserve(
        idempotency_key="request-1",
        source_event_id="request-1",
        resource_kind=ResourceKind.MODEL,
        dimensions=dimensions,
        budget=_budget("0.30"),
        model_calls=1,
        tokens=200,
        cost=ExactCost(Decimal("0.20"), pricing_known=True),
    )

    with pytest.raises(BudgetExceededError, match="budget exhausted for cost"):
        store.reserve(
            idempotency_key="request-2",
            source_event_id="request-2",
            resource_kind=ResourceKind.MODEL,
            dimensions=dimensions,
            budget=_budget("0.30"),
            model_calls=1,
            tokens=200,
            cost=ExactCost(Decimal("0.20"), pricing_known=True),
        )

    store.commit(first.id, (_entry("request-1", amount="0.20"),))
    with pytest.raises(BudgetExceededError, match="committed 0.20"):
        store.reserve(
            idempotency_key="request-2",
            source_event_id="request-2",
            resource_kind=ResourceKind.MODEL,
            dimensions=dimensions,
            budget=_budget("0.30"),
            model_calls=1,
            tokens=200,
            cost=ExactCost(Decimal("0.20"), pricing_known=True),
        )


def test_concurrent_reservations_cannot_spend_the_same_remaining_budget(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "store.sqlite"
    SQLiteUsageStore(db_path, clock=lambda: NOW)
    barrier = Barrier(2)

    def reserve(index: int) -> str:
        store = SQLiteUsageStore(db_path, clock=lambda: NOW)
        barrier.wait()
        try:
            reservation = store.reserve(
                idempotency_key=f"request-{index}",
                source_event_id=f"request-{index}",
                resource_kind=ResourceKind.MODEL,
                dimensions=_dimensions(),
                budget=_budget("0.30"),
                model_calls=1,
                tokens=200,
                cost=ExactCost(Decimal("0.20"), pricing_known=True),
            )
        except BudgetExceededError:
            return "blocked"
        return reservation.state.value

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reserve, (1, 2)))

    assert sorted(outcomes) == ["active", "blocked"]


def test_reservation_idempotency_returns_the_original_hold(tmp_path: Path) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    first = store.reserve(
        idempotency_key="same-request",
        source_event_id="request-1",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(),
        budget=_budget(),
        model_calls=1,
        tokens=100,
        cost=ExactCost(Decimal("0.10"), pricing_known=True),
    )

    repeated = store.reserve(
        idempotency_key="same-request",
        source_event_id="different-source",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(),
        budget=_budget(),
        model_calls=2,
        tokens=999,
        cost=ExactCost(Decimal("0.99"), pricing_known=True),
    )

    assert repeated == first


def test_unknown_pricing_follows_the_budget_policy(tmp_path: Path) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)

    with pytest.raises(BudgetExceededError) as exc_info:
        store.reserve(
            idempotency_key="closed",
            source_event_id="closed",
            resource_kind=ResourceKind.MODEL,
            dimensions=_dimensions(),
            budget=_budget(policy=UnknownCostPolicy.FAIL_CLOSED),
            model_calls=1,
            cost=ExactCost(None),
        )
    assert exc_info.value.requested == "unknown"

    reservation = store.reserve(
        idempotency_key="open",
        source_event_id="open",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(turn_id="turn-2"),
        budget=_budget(policy=UnknownCostPolicy.FAIL_OPEN),
        model_calls=1,
        cost=ExactCost(None),
    )
    assert reservation.state is ReservationState.ACTIVE


def test_release_and_expiry_restore_available_allowance(tmp_path: Path) -> None:
    current = [NOW]
    store = SQLiteUsageStore(
        tmp_path / "store.sqlite",
        clock=lambda: current[0],
        reservation_ttl=timedelta(seconds=30),
    )
    reservation = store.reserve(
        idempotency_key="request-1",
        source_event_id="request-1",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(),
        budget=_budget("0.20"),
        model_calls=1,
        cost=ExactCost(Decimal("0.20"), pricing_known=True),
    )
    released = store.release(reservation.id)
    assert released.state is ReservationState.RELEASED

    store.reserve(
        idempotency_key="request-2",
        source_event_id="request-2",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(),
        budget=_budget("0.20"),
        model_calls=1,
        cost=ExactCost(Decimal("0.20"), pricing_known=True),
    )
    current[0] = NOW + timedelta(minutes=1)

    assert store.release_expired() == 1
    replacement = store.reserve(
        idempotency_key="request-3",
        source_event_id="request-3",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(),
        budget=_budget("0.20"),
        model_calls=1,
        cost=ExactCost(Decimal("0.20"), pricing_known=True),
    )
    assert replacement.state is ReservationState.ACTIVE


def test_budget_scope_isolated_by_profile_conversation_and_turn(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    budget = RunBudget(scope=BudgetScope.TURN, max_model_calls=1)
    first = store.reserve(
        idempotency_key="work-1",
        source_event_id="work-1",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(profile_id="work", turn_id="turn-1"),
        budget=budget,
        model_calls=1,
    )
    assert first.state is ReservationState.ACTIVE

    second = store.reserve(
        idempotency_key="personal-1",
        source_event_id="personal-1",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(profile_id="personal", turn_id="turn-1"),
        budget=budget,
        model_calls=1,
    )
    third = store.reserve(
        idempotency_key="work-2",
        source_event_id="work-2",
        resource_kind=ResourceKind.MODEL,
        dimensions=_dimensions(profile_id="work", turn_id="turn-2"),
        budget=budget,
        model_calls=1,
    )
    assert second.state is ReservationState.ACTIVE
    assert third.state is ReservationState.ACTIVE


def test_budget_scope_requires_its_owning_dimension() -> None:
    dimensions = UsageDimensions(profile_id="work")

    with pytest.raises(ValueError, match="turn budget scope requires conversation_id"):
        dimensions.scope_values(BudgetScope.TURN)


def test_usage_query_is_profile_owned_and_cursor_paginated(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    store.ingest(
        _entry(
            "work-1",
            dimensions=_dimensions(profile_id="work", turn_id="turn-1"),
            occurred_at=NOW - timedelta(hours=2),
        )
    )
    store.ingest(
        _entry(
            "work-2",
            dimensions=_dimensions(profile_id="work", turn_id="turn-2"),
            occurred_at=NOW - timedelta(hours=1),
        )
    )
    store.ingest(
        _entry(
            "personal-1",
            dimensions=_dimensions(profile_id="personal"),
        )
    )

    first = store.query(
        UsageQuery(
            profile_id="work",
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            limit=1,
        )
    )
    second = store.query(
        UsageQuery(
            profile_id="work",
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            limit=1,
            cursor=first.next_cursor,
        )
    )

    assert [entry.source_event_id for entry in first.entries] == ["work-1"]
    assert first.next_cursor is not None
    assert [entry.source_event_id for entry in second.entries] == ["work-2"]
    assert second.next_cursor is None
    with pytest.raises(ValueError, match="invalid usage cursor"):
        store.query(UsageQuery(profile_id="work", cursor="not-a-cursor"))
    with pytest.raises(ValueError, match="invalid usage cursor"):
        store.query(UsageQuery(profile_id="work", cursor="_"))


def test_usage_query_normalizes_timezone_boundaries(tmp_path: Path) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    madrid = timezone(timedelta(hours=2))
    store.ingest(
        _entry(
            "boundary",
            occurred_at=datetime(2026, 7, 25, 0, 30, tzinfo=madrid),
        )
    )

    page = store.query(
        UsageQuery(
            profile_id="work",
            start=datetime(2026, 7, 24, 22, 0, tzinfo=timezone.utc),
            end=datetime(2026, 7, 24, 23, 0, tzinfo=timezone.utc),
        )
    )

    assert [entry.source_event_id for entry in page.entries] == ["boundary"]
    assert page.entries[0].occurred_at == datetime(
        2026,
        7,
        24,
        22,
        30,
        tzinfo=timezone.utc,
    )


def test_usage_grouping_preserves_decimal_totals_and_unknown_cost(
    tmp_path: Path,
) -> None:
    store = SQLiteUsageStore(tmp_path / "store.sqlite", clock=lambda: NOW)
    store.ingest(_entry("small-1", amount="0.1"))
    store.ingest(
        _entry(
            "large-1",
            amount="0.2",
            model="gpt-4.1",
            dimensions=_dimensions(turn_id="turn-2"),
        )
    )
    unknown = _entry(
        "small-unknown",
        dimensions=_dimensions(turn_id="turn-3"),
    )
    store.ingest(replace(unknown, cost=ExactCost(None)))

    groups = store.aggregate(
        UsageQuery(profile_id="work", limit=100),
        group_by=UsageGroupBy.MODEL,
    )

    assert [group.key for group in groups] == [
        "openai:gpt-4.1",
        "openai:gpt-4.1-mini",
    ]
    assert groups[0].cost.amount == Decimal("0.2")
    assert groups[1].cost.amount == Decimal("0.1")
    assert groups[1].unknown_cost_entries == 1
    assert not groups[1].cost.pricing_known


def test_bounded_exports_exclude_credential_references(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "store.sqlite"
    store = SQLiteUsageStore(db_path, clock=lambda: NOW)
    store.ingest(_entry("request-1", credential_ref="env:SECRET_NAME"))
    ledger = UsageLedger(db_path, profile_id="work")

    json_path = ledger.export(tmp_path / "usage.json", format="json")
    csv_path = ledger.export(tmp_path / "usage.csv", format="csv")

    json_content = json_path.read_text()
    csv_content = csv_path.read_text()
    assert "credential_ref" not in json_content
    assert "SECRET_NAME" not in json_content
    assert "credential_ref" not in csv_content
    assert "SECRET_NAME" not in csv_content
    if os.name == "posix":
        assert json_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        ledger.export(json_path, format="json")


def test_export_refuses_to_write_a_partial_bounded_result(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    store = SQLiteUsageStore(db_path, clock=lambda: NOW)
    store.ingest(_entry("request-1"))
    store.ingest(_entry("request-2", dimensions=_dimensions(turn_id="turn-2")))
    destination = tmp_path / "too-small.json"

    with pytest.raises(ValueError, match="exceeds max_entries"):
        UsageLedger(db_path, profile_id="work").export(
            destination,
            format="json",
            max_entries=1,
        )

    assert not destination.exists()
