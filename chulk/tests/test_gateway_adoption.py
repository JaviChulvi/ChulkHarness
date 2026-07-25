"""Crash-safe one-time adoption of the legacy Telegram update ledger."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chulk.gateway import SQLiteGatewayLedger, adopt_legacy_telegram_state
from chulk.sessions import SQLiteSessionStore
from chulk.telegram.ledger import SQLiteAdapterUpdateLedger


NOW = datetime(2026, 7, 25, 12, tzinfo=timezone.utc)


def test_adoption_copies_cursor_delivery_checkpoint_and_terminal_rows_once(tmp_path) -> None:
    legacy_path = tmp_path / "profile.sqlite"
    control_path = tmp_path / "control.sqlite"
    sessions = SQLiteSessionStore(legacy_path)
    sessions.save_adapter_cursor("telegram", 42)
    legacy = SQLiteAdapterUpdateLedger(legacy_path)

    claim = legacy.begin_execution(
        adapter="telegram",
        update_id=1,
        destination_id="9",
        now=NOW,
    )
    assert claim.execution_token is not None
    assert legacy.record_response(
        adapter="telegram",
        update_id=1,
        execution_token=claim.execution_token,
        response_parts=("first", "second"),
    )
    delivery = legacy.claim_delivery(adapter="telegram", update_id=1, now=NOW)
    assert delivery is not None
    assert delivery.delivery_token is not None
    assert legacy.mark_response_part_delivered(
        adapter="telegram",
        update_id=1,
        delivery_token=delivery.delivery_token,
        expected_part=0,
    )
    legacy.ignore(adapter="telegram", update_id=2, destination_id="9")

    result = adopt_legacy_telegram_state(
        control_db_path=control_path,
        profile_db_path=legacy_path,
        profile_id="work",
        now=NOW,
    )
    repeated = adopt_legacy_telegram_state(
        control_db_path=control_path,
        profile_db_path=legacy_path,
        profile_id="work",
        now=NOW,
    )

    ledger = SQLiteGatewayLedger(control_path)
    assert result.adopted
    assert result.cursor == "42"
    assert result.inbox_records == 2
    assert result.outbox_records == 2
    assert not repeated.adopted
    assert ledger.adapter_status("telegram", "primary").cursor == "42"
    pending = ledger.claim_delivery(now=NOW)
    assert pending is not None
    assert pending.envelope.text == "second"
    assert pending.envelope.extensions["legacy_adopted"] is True


def test_adoption_quarantines_in_flight_execution_and_never_queues_it(tmp_path) -> None:
    legacy_path = tmp_path / "profile.sqlite"
    control_path = tmp_path / "control.sqlite"
    legacy = SQLiteAdapterUpdateLedger(legacy_path)
    legacy.begin_execution(
        adapter="telegram",
        update_id=3,
        destination_id="9",
        now=NOW,
    )

    adopt_legacy_telegram_state(
        control_db_path=control_path,
        profile_db_path=legacy_path,
        profile_id="work",
        now=NOW,
    )

    ledger = SQLiteGatewayLedger(control_path)
    assert ledger.pending_count() == 0
    assert ledger.claim_execution(global_limit=1, profile_limit=1, now=NOW) is None
    notice = ledger.claim_delivery(now=NOW)
    assert notice is not None
    assert "uncertain execution checkpoint" in (notice.envelope.text or "")


def test_adoption_refuses_to_copy_live_adapter_state(tmp_path) -> None:
    legacy_path = tmp_path / "profile.sqlite"
    control_path = tmp_path / "control.sqlite"
    SQLiteSessionStore(legacy_path)
    ledger = SQLiteGatewayLedger(control_path)
    ledger.start_adapter("telegram", "primary", now=NOW)

    with pytest.raises(RuntimeError, match="while stopped"):
        adopt_legacy_telegram_state(
            control_db_path=control_path,
            profile_db_path=legacy_path,
            profile_id="work",
            now=NOW,
        )
