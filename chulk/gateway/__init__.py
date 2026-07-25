"""Public channel-neutral gateway contracts."""

from chulk.gateway.ledger import (
    ExecutionClaim,
    GatewayAdapterStatus,
    InboxRecord,
    IngestResult,
    OutboxRecord,
    SQLiteGatewayLedger,
    UNCERTAIN_EXECUTION_MESSAGE,
    conversation_key_for,
)
from chulk.gateway.models import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    InboundEnvelope,
    InboundPart,
    MediaPart,
    MediaReference,
    OutboundEnvelope,
    ReactionPart,
    ReplyPart,
    TextPart,
    TrustLevel,
)
from chulk.gateway.protocol import ChannelAdapter


__all__ = [
    "AuthenticationState",
    "ChannelAdapter",
    "ChannelIdentity",
    "ChannelScope",
    "DeliveryReceipt",
    "DeliveryState",
    "DeliveryTarget",
    "ExecutionClaim",
    "GatewayAdapterStatus",
    "InboundEnvelope",
    "InboundPart",
    "InboxRecord",
    "IngestResult",
    "MediaPart",
    "MediaReference",
    "OutboundEnvelope",
    "OutboxRecord",
    "ReactionPart",
    "ReplyPart",
    "SQLiteGatewayLedger",
    "TextPart",
    "TrustLevel",
    "UNCERTAIN_EXECUTION_MESSAGE",
    "conversation_key_for",
]
