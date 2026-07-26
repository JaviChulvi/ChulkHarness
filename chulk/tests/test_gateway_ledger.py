"""Durability and ordering tests for the owner-controlled gateway ledger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    InboundEnvelope,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    TextPart,
    TrustLevel,
)


NOW = datetime(2026, 7, 25, 10, tzinfo=timezone.utc)


def _inbound(
    event_id: str,
    *,
    destination_id: str = "chat-9",
    principal_id: str = "user-7",
) -> InboundEnvelope:
    return InboundEnvelope(
        event_id=event_id,
        idempotency_key=f"telegram:primary:{event_id}",
        identity=ChannelIdentity("telegram", "primary", principal_id),
        destination_id=destination_id,
        parts=(TextPart(f"message {event_id}"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
    )


def _outbound(inbox_id: str, *, profile_id: str = "work") -> OutboundEnvelope:
    return OutboundEnvelope(
        envelope_id=f"reply-{inbox_id}",
        profile_id=profile_id,
        conversation_id=f"conversation-{inbox_id}",
        target=DeliveryTarget("telegram", "primary", "chat-9"),
        text="done",
    )


def test_adapter_lease_has_one_live_owner_and_owned_cursor(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    first = ledger.start_adapter("telegram", "primary", now=NOW, lease_seconds=10)

    with pytest.raises(RuntimeError, match="already running"):
        ledger.start_adapter(
            "telegram",
            "primary",
            now=NOW + timedelta(seconds=5),
            lease_seconds=10,
        )

    assert first.instance_token is not None
    assert not ledger.save_cursor(
        "telegram",
        "primary",
        "20",
        instance_token="not-the-owner",
    )
    assert ledger.save_cursor(
        "telegram",
        "primary",
        "20",
        instance_token=first.instance_token,
    )
    status = ledger.adapter_status("telegram", "primary")
    assert status is not None
    assert status.cursor == "20"
    assert ledger.stop_adapter("telegram", "primary", first.instance_token)


def test_cooperative_stop_prevents_further_lease_renewal(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    status = ledger.start_adapter(
        "telegram",
        "primary",
        now=NOW,
        lease_seconds=10,
    )
    assert status.instance_token is not None
    assert ledger.request_adapter_stop("telegram", "primary")
    assert not ledger.renew_adapter(
        "telegram",
        "primary",
        status.instance_token,
        now=NOW + timedelta(seconds=1),
        lease_seconds=10,
    )


def test_ingest_is_idempotent_and_round_trips_envelope(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    envelope = _inbound("1")

    first = ledger.ingest(envelope, profile_id="work")
    duplicate = ledger.ingest(envelope, profile_id="other")

    assert first.created
    assert not duplicate.created
    assert duplicate.record.id == first.record.id
    assert duplicate.record.profile_id == "work"
    assert duplicate.record.envelope == envelope

    collision = _inbound("1", destination_id="different")
    with pytest.raises(ValueError, match="idempotency key collision"):
        ledger.ingest(collision, profile_id="work")


def test_execution_claims_preserve_fifo_and_enforce_concurrency_limits(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    first = ledger.ingest(_inbound("1"), profile_id="work").record
    second = ledger.ingest(_inbound("2"), profile_id="work").record
    other = ledger.ingest(
        _inbound("3", destination_id="chat-10"),
        profile_id="personal",
    ).record

    claim = ledger.claim_execution(global_limit=2, profile_limit=1, now=NOW)
    assert claim is not None
    assert claim.record.id == first.id

    next_claim = ledger.claim_execution(global_limit=2, profile_limit=1, now=NOW)
    assert next_claim is not None
    assert next_claim.record.id == other.id
    assert ledger.claim_execution(global_limit=2, profile_limit=1, now=NOW) is None

    assert ledger.complete_execution(
        first.id,
        claim.execution_token,
        (_outbound(first.id),),
    )
    resumed = ledger.claim_execution(global_limit=2, profile_limit=1, now=NOW)
    assert resumed is not None
    assert resumed.record.id == second.id


def test_execution_claim_respects_queue_wave_cutoff(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    first = ledger.ingest(_inbound("1"), profile_id="work").record
    second = ledger.ingest(
        _inbound("2", destination_id="chat-10"),
        profile_id="personal",
    ).record

    claim = ledger.claim_execution(
        global_limit=2,
        profile_limit=1,
        queued_before=first.created_at,
    )

    assert claim is not None
    assert claim.record.id == first.id
    assert ledger.complete_execution(
        first.id,
        claim.execution_token,
        (_outbound(first.id),),
    )
    assert (
        ledger.claim_execution(
            global_limit=2,
            profile_limit=1,
            queued_before=first.created_at,
        )
        is None
    )
    next_claim = ledger.claim_execution(global_limit=2, profile_limit=1)
    assert next_claim is not None
    assert next_claim.record.id == second.id


def test_expired_execution_is_quarantined_and_never_reclaimed(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    record = ledger.ingest(_inbound("1"), profile_id="work").record
    claim = ledger.claim_execution(
        global_limit=1,
        profile_limit=1,
        now=NOW,
        lease_seconds=10,
    )
    assert claim is not None

    recovered = ledger.recover_expired_executions(
        now=NOW + timedelta(seconds=11),
    )

    assert [item.id for item in recovered] == [record.id]
    assert recovered[0].state == "uncertain"
    assert recovered[0].last_error == "execution lease expired"
    notice = ledger.claim_delivery(now=NOW + timedelta(seconds=11))
    assert notice is not None
    assert "uncertain execution checkpoint" in (notice.envelope.text or "")
    assert (
        ledger.claim_execution(
            global_limit=1,
            profile_limit=1,
            now=NOW + timedelta(seconds=12),
        )
        is None
    )


def test_outbox_delivery_is_retryable_and_records_attempt_checkpoints(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    inbox = ledger.ingest(_inbound("1"), profile_id="work").record
    execution = ledger.claim_execution(global_limit=1, profile_limit=1, now=NOW)
    assert execution is not None
    response = _outbound(inbox.id)
    assert ledger.complete_execution(
        inbox.id,
        execution.execution_token,
        (response,),
    )

    first = ledger.claim_delivery(now=NOW)
    assert first is not None
    assert first.delivery_token is not None
    assert first.attempt_count == 1
    assert ledger.record_delivery(
        first.id,
        first.delivery_token,
        DeliveryReceipt(
            envelope_id=first.id,
            state=DeliveryState.RETRYABLE,
            attempt=1,
            checkpoint="part-0",
            retry_after_seconds=5,
            error_message="offline",
        ),
        now=NOW,
    )
    assert ledger.claim_delivery(now=NOW + timedelta(seconds=4)) is None

    retried = ledger.claim_delivery(now=NOW + timedelta(seconds=6))
    assert retried is not None
    assert retried.delivery_token is not None
    assert retried.attempt_count == 2
    assert retried.checkpoint == "part-0"
    assert ledger.record_delivery(
        retried.id,
        retried.delivery_token,
        DeliveryReceipt(
            envelope_id=retried.id,
            state=DeliveryState.DELIVERED,
            attempt=2,
            checkpoint="complete",
        ),
        now=NOW + timedelta(seconds=6),
    )
    completed = ledger.get_outbox(response.envelope_id)
    assert completed is not None
    assert completed.state == "delivered"
    assert completed.checkpoint == "complete"


def test_transport_acceptance_is_not_a_final_delivery_checkpoint(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    inbox = ledger.ingest(_inbound("1"), profile_id="work").record
    execution = ledger.claim_execution(global_limit=1, profile_limit=1, now=NOW)
    assert execution is not None
    response = _outbound(inbox.id)
    assert ledger.complete_execution(
        inbox.id,
        execution.execution_token,
        (response,),
    )
    delivery = ledger.claim_delivery(now=NOW)
    assert delivery is not None
    assert delivery.delivery_token is not None

    assert ledger.record_delivery(
        delivery.id,
        delivery.delivery_token,
        DeliveryReceipt(
            envelope_id=delivery.id,
            state=DeliveryState.ACCEPTED,
            attempt=1,
        ),
        now=NOW,
    )
    assert not ledger.inbox_complete(inbox.id)
    assert ledger.claim_delivery(now=NOW + timedelta(milliseconds=500)) is None
    assert ledger.claim_delivery(now=NOW + timedelta(seconds=1)) is not None


def test_backpressure_counts_executed_work_with_pending_delivery(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    inbox = ledger.ingest(
        _inbound("1"),
        profile_id="work",
        max_pending=1,
    ).record
    execution = ledger.claim_execution(global_limit=1, profile_limit=1)
    assert execution is not None
    assert ledger.complete_execution(
        inbox.id,
        execution.execution_token,
        (_outbound(inbox.id),),
    )

    with pytest.raises(RuntimeError, match="pending limit"):
        ledger.ingest(
            _inbound("2", destination_id="chat-10"),
            profile_id="work",
            max_pending=1,
        )


def test_cancellation_is_durable_for_queued_and_active_work(tmp_path) -> None:
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    queued = ledger.ingest(_inbound("1"), profile_id="work").record
    assert ledger.request_cancellation(queued.id)
    queued_record = ledger.get_inbox(queued.id)
    assert queued_record is not None
    assert queued_record.state == "cancelled"

    active = ledger.ingest(
        _inbound("2", destination_id="chat-10"),
        profile_id="work",
    ).record
    claim = ledger.claim_execution(global_limit=1, profile_limit=1)
    assert claim is not None
    assert claim.record.id == active.id
    assert ledger.request_cancellation(active.id)
    active_record = ledger.get_inbox(active.id)
    assert active_record is not None
    assert active_record.cancellation_requested
    assert ledger.mark_execution_cancelled(active.id, claim.execution_token)
    cancelled = ledger.get_inbox(active.id)
    assert cancelled is not None
    assert cancelled.state == "cancelled"
