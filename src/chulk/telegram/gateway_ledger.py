"""Compatibility view over Telegram records now owned by the gateway ledger."""

from __future__ import annotations

from chulk.gateway import SQLiteGatewayLedger
from chulk.telegram.ledger import AdapterUpdateRecord


class TelegramGatewayLedgerView:
    """Expose the former read API while storage lives in the control database."""

    def __init__(
        self,
        ledger: SQLiteGatewayLedger,
        *,
        account_id: str = "primary",
    ) -> None:
        self.ledger = ledger
        self.account_id = account_id

    def get(self, *, adapter: str, update_id: int) -> AdapterUpdateRecord | None:
        record = self.ledger.find_inbox(
            adapter=adapter,
            account_id=self.account_id,
            idempotency_key=f"{adapter}:{self.account_id}:update:{update_id}",
        )
        if record is None:
            return None
        outbox = self.ledger.list_outbox(record.id)
        status = record.state
        if status == "queued":
            status = "processing"
        elif status == "uncertain":
            status = "executed"
        elif outbox:
            if all(item.state == "delivered" for item in outbox):
                status = "delivered"
            elif any(item.state == "delivering" for item in outbox):
                status = "delivering"
            else:
                status = "executed"
        response_parts = tuple(
            item.envelope.text
            for item in outbox
            if item.envelope.text is not None
        )
        next_part = 0
        for item in outbox:
            if item.state != "delivered":
                break
            next_part += 1
        active_delivery = next(
            (item for item in outbox if item.state == "delivering"),
            None,
        )
        return AdapterUpdateRecord(
            adapter=adapter,
            update_id=update_id,
            destination_id=record.envelope.destination_id,
            status=status,
            response_parts=response_parts,
            next_response_part=next_part,
            execution_token=record.execution_token,
            execution_lease_until=record.execution_lease_until,
            delivery_token=(
                active_delivery.delivery_token
                if active_delivery is not None
                else None
            ),
            delivery_lease_until=(
                active_delivery.delivery_lease_until
                if active_delivery is not None
                else None
            ),
            last_error=record.last_error
            or next(
                (
                    item.last_error
                    for item in outbox
                    if item.last_error is not None
                ),
                None,
            ),
        )


__all__ = ["TelegramGatewayLedgerView"]
