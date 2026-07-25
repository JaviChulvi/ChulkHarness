"""Public channel-neutral gateway contracts."""

from chulk.gateway.adoption import (
    LegacyAdoptionResult,
    adopt_legacy_telegram_state,
)
from chulk.gateway.commands import (
    ChannelCommand,
    ChannelCommandSpec,
    SHARED_CHANNEL_COMMANDS,
    parse_channel_command,
    shared_command_help,
    shared_command_spec,
)
from chulk.gateway.ledger import (
    ExecutionClaim,
    GatewayAdapterStatus,
    GatewayBackpressureError,
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
from chulk.gateway.routing import (
    GatewayRoute,
    PairingChallenge,
    SQLiteGatewayRouter,
)
from chulk.gateway.runtime import (
    EnvelopeExecutor,
    GatewayLimits,
    GatewayRuntime,
    UNROUTED_PROFILE_ID,
)


__all__ = [
    "AuthenticationState",
    "ChannelAdapter",
    "ChannelCommand",
    "ChannelCommandSpec",
    "ChannelIdentity",
    "ChannelScope",
    "DeliveryReceipt",
    "DeliveryState",
    "DeliveryTarget",
    "EnvelopeExecutor",
    "ExecutionClaim",
    "GatewayAdapterStatus",
    "GatewayBackpressureError",
    "GatewayLimits",
    "GatewayRoute",
    "GatewayRuntime",
    "InboundEnvelope",
    "InboundPart",
    "InboxRecord",
    "IngestResult",
    "LegacyAdoptionResult",
    "MediaPart",
    "MediaReference",
    "OutboundEnvelope",
    "OutboxRecord",
    "PairingChallenge",
    "ReactionPart",
    "ReplyPart",
    "SQLiteGatewayLedger",
    "SQLiteGatewayRouter",
    "SHARED_CHANNEL_COMMANDS",
    "TextPart",
    "TrustLevel",
    "UNROUTED_PROFILE_ID",
    "UNCERTAIN_EXECUTION_MESSAGE",
    "adopt_legacy_telegram_state",
    "conversation_key_for",
    "parse_channel_command",
    "shared_command_help",
    "shared_command_spec",
]
