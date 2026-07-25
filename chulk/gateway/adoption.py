"""One-time adoption of legacy adapter state into the gateway control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from chulk.gateway.ledger import (
    UNCERTAIN_EXECUTION_MESSAGE,
    _bounded_json,
    _encode,
    _inbound_to_dict,
    _outbound_to_dict,
)
from chulk.gateway.models import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryTarget,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
    TrustLevel,
)
from chulk.profiles.store import CONTROL_MIGRATIONS
from chulk.storage import initialize_sqlite_database, sqlite_connection


@dataclass(frozen=True, slots=True)
class LegacyAdoptionResult:
    adopted: bool
    cursor: str | None
    inbox_records: int
    outbox_records: int


def adopt_legacy_telegram_state(
    *,
    control_db_path: Path | str,
    profile_db_path: Path | str,
    profile_id: str,
    account_id: str = "primary",
    now: datetime | None = None,
) -> LegacyAdoptionResult:
    """Copy legacy Telegram cursor/update rows once while the adapter is stopped."""
    control_path = Path(control_db_path).expanduser().resolve()
    legacy_path = Path(profile_db_path).expanduser().resolve()
    initialize_sqlite_database(control_path, migrations=CONTROL_MIGRATIONS)
    initialize_sqlite_database(legacy_path)
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    with sqlite_connection(legacy_path) as legacy:
        cursor_row = legacy.execute(
            "SELECT cursor FROM adapter_cursors WHERE adapter = 'telegram'"
        ).fetchone()
        cursor = str(cursor_row["cursor"]) if cursor_row is not None else None
        updates = legacy.execute(
            """
            SELECT * FROM adapter_updates
            WHERE adapter = 'telegram'
            ORDER BY update_id
            """
        ).fetchall()

    inbox_count = 0
    outbox_count = 0
    with sqlite_connection(control_path) as control:
        control.execute("BEGIN IMMEDIATE")
        adapter_row = control.execute(
            """
            SELECT * FROM gateway_adapters
            WHERE adapter = 'telegram' AND account_id = ?
            """,
            (account_id,),
        ).fetchone()
        if adapter_row is not None and adapter_row["legacy_adopted_at"] is not None:
            return LegacyAdoptionResult(
                adopted=False,
                cursor=str(adapter_row["cursor"])
                if adapter_row["cursor"] is not None
                else None,
                inbox_records=0,
                outbox_records=0,
            )
        if adapter_row is not None and adapter_row["state"] == "running":
            raise RuntimeError("legacy Telegram state can only be adopted while stopped")

        for row in updates:
            update_id = int(row["update_id"])
            destination_id = str(row["destination_id"])
            status = str(row["status"])
            inbox_id = f"legacy-telegram-{account_id}-{update_id}"
            inbound = InboundEnvelope(
                event_id=str(update_id),
                idempotency_key=f"telegram:{account_id}:update:{update_id}",
                identity=ChannelIdentity("telegram", account_id, "_legacy"),
                destination_id=destination_id,
                parts=(TextPart(f"Legacy Telegram update {update_id}"),),
                scope=ChannelScope.DIRECT,
                authentication=AuthenticationState.UNKNOWN,
                trust=TrustLevel.UNTRUSTED,
                received_at=str(row["created_at"]),
                extensions={"legacy_adopted": True, "update_id": update_id},
            )
            inbox_state, last_error = _legacy_inbox_state(status, row["last_error"])
            inserted = control.execute(
                """
                INSERT OR IGNORE INTO gateway_inbox (
                    id, profile_id, adapter, account_id, event_id,
                    idempotency_key, conversation_key, principal_id,
                    destination_id, thread_id, envelope_json, state,
                    last_error, created_at, updated_at, executed_at
                ) VALUES (?, ?, 'telegram', ?, ?, ?, ?, '_legacy', ?, NULL, ?,
                          ?, ?, ?, ?, ?)
                """,
                (
                    inbox_id,
                    profile_id,
                    account_id,
                    str(update_id),
                    inbound.idempotency_key,
                    f"telegram\x1f{account_id}\x1f{destination_id}\x1f",
                    destination_id,
                    _bounded_json(_inbound_to_dict(inbound)),
                    inbox_state,
                    last_error,
                    str(row["created_at"]),
                    str(row["updated_at"]),
                    row["executed_at"],
                ),
            )
            inbox_count += max(inserted.rowcount, 0)
            if status == "ignored":
                continue
            parts = _legacy_parts(status, row["response_parts"])
            delivered_through = int(row["next_response_part"])
            for sequence, text in enumerate(parts):
                delivered = status == "delivered" or sequence < delivered_through
                outbound = OutboundEnvelope(
                    envelope_id=f"{inbox_id}-part-{sequence}",
                    profile_id=profile_id,
                    conversation_id=f"telegram:{destination_id}",
                    target=DeliveryTarget(
                        "telegram",
                        account_id,
                        destination_id,
                    ),
                    text=text,
                    reply_to_event_id=str(update_id),
                    sequence=sequence,
                    final=sequence == len(parts) - 1,
                    extensions={"legacy_adopted": True},
                )
                inserted_outbox = control.execute(
                    """
                    INSERT OR IGNORE INTO gateway_outbox (
                        id, inbox_id, profile_id, sequence, envelope_json,
                        state, attempt_count, last_error, created_at,
                        updated_at, delivered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        outbound.envelope_id,
                        inbox_id,
                        profile_id,
                        sequence,
                        _bounded_json(_outbound_to_dict(outbound)),
                        "delivered" if delivered else "pending",
                        row["last_error"],
                        str(row["created_at"]),
                        str(row["updated_at"]),
                        row["delivered_at"] if delivered else None,
                    ),
                )
                outbox_count += max(inserted_outbox.rowcount, 0)

        control.execute(
            """
            INSERT INTO gateway_adapters (
                adapter, account_id, state, cursor, legacy_adopted_at,
                stopped_at, updated_at
            ) VALUES ('telegram', ?, 'stopped', ?, ?, ?, ?)
            ON CONFLICT(adapter, account_id) DO UPDATE SET
                cursor = COALESCE(gateway_adapters.cursor, excluded.cursor),
                legacy_adopted_at = excluded.legacy_adopted_at,
                stopped_at = excluded.stopped_at,
                updated_at = excluded.updated_at
            """,
            (
                account_id,
                cursor,
                _encode(observed),
                _encode(observed),
                _encode(observed),
            ),
        )
    return LegacyAdoptionResult(
        adopted=True,
        cursor=cursor,
        inbox_records=inbox_count,
        outbox_records=outbox_count,
    )


def _legacy_inbox_state(status: str, error: object) -> tuple[str, str | None]:
    if status == "ignored":
        return "ignored", str(error) if error is not None else None
    if status == "processing":
        return "uncertain", "legacy execution state was uncertain during adoption"
    return "executed", str(error) if error is not None else None


def _legacy_parts(status: str, raw_parts: object) -> tuple[str, ...]:
    if status == "processing":
        return (UNCERTAIN_EXECUTION_MESSAGE,)
    try:
        decoded = json.loads(str(raw_parts))
    except json.JSONDecodeError:
        decoded = []
    if not isinstance(decoded, list):
        return ()
    return tuple(item for item in decoded if isinstance(item, str) and item)


__all__ = ["LegacyAdoptionResult", "adopt_legacy_telegram_state"]
