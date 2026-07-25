"""Optional Discord gateway host."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from chulk.config import Config, ConfigValueError, load_config
from chulk.discord.adapter import (
    DiscordChannelAdapter,
    split_discord_envelope,
)
from chulk.discord.client import (
    DiscordDependencyError,
    DiscordPyTransport,
    DiscordTransportError,
)
from chulk.discord.config import (
    DiscordConfig,
    DiscordConfigError,
    load_discord_config,
)
from chulk.gateway import (
    GatewayLimits,
    GatewayRuntime,
    InboundEnvelope,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    SQLiteGatewayRouter,
)
from chulk.profiles import ProfileRuntimeFactory
from chulk.server.channel_runtime import ChannelConversationExecutor
from chulk.server.dispatcher import ConversationDispatcher


LOGGER = logging.getLogger(__name__)


async def run_discord_runtime(
    config: Config,
    discord_config: DiscordConfig,
    *,
    control_db_path: Path | str | None = None,
    transport: Any = None,
    profile_runtime_factory: ProfileRuntimeFactory | None = None,
) -> None:
    """Run Discord through the shared durable gateway and dispatcher."""
    control_path = Path(
        control_db_path or config.runtime_dir / "control.sqlite"
    ).resolve()
    runtime_factory = profile_runtime_factory or ProfileRuntimeFactory(config)
    dispatcher = ConversationDispatcher(runtime_factory)
    adapter = DiscordChannelAdapter(
        transport or DiscordPyTransport(discord_config.bot_token),
        account_id=discord_config.account_id,
    )
    conversation_executor = ChannelConversationExecutor(dispatcher)

    async def execute(
        profile_id: str,
        envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        responses = await conversation_executor(profile_id, envelope)
        split: list[OutboundEnvelope] = []
        for response in responses:
            split.extend(split_discord_envelope(response))
        return tuple(split)

    runtime = GatewayRuntime(
        ledger=SQLiteGatewayLedger(control_path),
        router=SQLiteGatewayRouter(control_path),
        adapters=(adapter,),
        executor=execute,
        limits=GatewayLimits(
            global_concurrency=4,
            profile_concurrency=1,
            max_pending=discord_config.max_pending,
        ),
    )
    try:
        await runtime.run_forever()
    finally:
        await dispatcher.close()


def run_discord_gateway(
    config: Config,
    *,
    control_db_path: Path | str | None = None,
    profile_runtime_factory: ProfileRuntimeFactory | None = None,
) -> int:
    try:
        discord_config = load_discord_config(env_file=config.project_root / ".env")
        asyncio.run(
            run_discord_runtime(
                config,
                discord_config,
                control_db_path=control_db_path,
                profile_runtime_factory=profile_runtime_factory,
            )
        )
    except (DiscordConfigError, DiscordDependencyError) as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    except DiscordTransportError as exc:
        LOGGER.error("Discord gateway stopped: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Discord gateway stopped")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = load_config()
    except ConfigValueError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    return run_discord_gateway(config)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_discord_gateway", "run_discord_runtime"]
