from __future__ import annotations

from datetime import datetime, timedelta, timezone

from chulk.telegram.ledger import (
    SQLiteAdapterUpdateLedger,
    UNCERTAIN_EXECUTION_RESPONSE,
)


def test_update_execution_is_claimed_once_and_response_is_checkpointed(tmp_path) -> None:
    ledger = SQLiteAdapterUpdateLedger(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 24, 8, tzinfo=timezone.utc)

    first = ledger.begin_execution(
        adapter="telegram",
        update_id=7,
        destination_id="9",
        now=now,
    )
    concurrent = ledger.begin_execution(
        adapter="telegram",
        update_id=7,
        destination_id="9",
        now=now,
    )

    assert first.should_execute
    assert not concurrent.should_execute
    assert concurrent.record.status == "processing"
    assert first.execution_token is not None
    assert ledger.record_response(
        adapter="telegram",
        update_id=7,
        execution_token=first.execution_token,
        response_parts=("first", "second"),
    )

    delivery = ledger.claim_delivery(adapter="telegram", update_id=7, now=now)
    assert delivery is not None
    assert delivery.delivery_token is not None
    assert ledger.claim_delivery(adapter="telegram", update_id=7, now=now) is None
    assert ledger.mark_response_part_delivered(
        adapter="telegram",
        update_id=7,
        delivery_token=delivery.delivery_token,
        expected_part=0,
    )
    assert ledger.get(adapter="telegram", update_id=7).next_response_part == 1
    assert ledger.mark_response_part_delivered(
        adapter="telegram",
        update_id=7,
        delivery_token=delivery.delivery_token,
        expected_part=1,
    )
    completed = ledger.get(adapter="telegram", update_id=7)
    assert completed is not None
    assert completed.status == "delivered"
    assert completed.next_response_part == 2


def test_expired_execution_is_quarantined_instead_of_reexecuted(tmp_path) -> None:
    ledger = SQLiteAdapterUpdateLedger(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 24, 8, tzinfo=timezone.utc)
    first = ledger.begin_execution(
        adapter="telegram",
        update_id=8,
        destination_id="9",
        now=now,
        lease_seconds=10,
    )
    assert first.should_execute

    recovered = ledger.begin_execution(
        adapter="telegram",
        update_id=8,
        destination_id="9",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )

    assert not recovered.should_execute
    assert recovered.record.status == "executed"
    assert recovered.record.response_parts == (UNCERTAIN_EXECUTION_RESPONSE,)
    assert recovered.record.last_error == "execution lease expired"


def test_execution_renewal_prevents_premature_quarantine(tmp_path) -> None:
    ledger = SQLiteAdapterUpdateLedger(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 24, 8, tzinfo=timezone.utc)
    claim = ledger.begin_execution(
        adapter="telegram",
        update_id=9,
        destination_id="9",
        now=now,
        lease_seconds=10,
    )
    assert claim.execution_token is not None
    assert ledger.renew_execution(
        adapter="telegram",
        update_id=9,
        execution_token=claim.execution_token,
        now=now + timedelta(seconds=5),
        lease_seconds=10,
    )

    duplicate = ledger.begin_execution(
        adapter="telegram",
        update_id=9,
        destination_id="9",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )

    assert not duplicate.should_execute
    assert duplicate.record.status == "processing"


def test_failed_delivery_retries_from_last_checkpoint(tmp_path) -> None:
    ledger = SQLiteAdapterUpdateLedger(tmp_path / "store.sqlite")
    execution = ledger.begin_execution(
        adapter="telegram",
        update_id=10,
        destination_id="9",
    )
    assert execution.execution_token is not None
    assert ledger.record_response(
        adapter="telegram",
        update_id=10,
        execution_token=execution.execution_token,
        response_parts=("first", "second"),
    )
    delivery = ledger.claim_delivery(adapter="telegram", update_id=10)
    assert delivery is not None
    assert delivery.delivery_token is not None
    assert ledger.mark_response_part_delivered(
        adapter="telegram",
        update_id=10,
        delivery_token=delivery.delivery_token,
        expected_part=0,
    )
    assert ledger.release_delivery(
        adapter="telegram",
        update_id=10,
        delivery_token=delivery.delivery_token,
        error="TelegramError",
    )

    retried = ledger.claim_delivery(adapter="telegram", update_id=10)

    assert retried is not None
    assert retried.next_response_part == 1
    assert retried.response_parts == ("first", "second")


def test_ignored_updates_are_retained_then_purged_with_a_bound(tmp_path) -> None:
    ledger = SQLiteAdapterUpdateLedger(tmp_path / "store.sqlite")
    record = ledger.ignore(adapter="telegram", update_id=11, destination_id="9")

    assert record.status == "ignored"
    assert ledger.purge_terminal(
        before=datetime.now(timezone.utc) + timedelta(seconds=1),
        limit=1,
    ) == 1
    assert ledger.get(adapter="telegram", update_id=11) is None
