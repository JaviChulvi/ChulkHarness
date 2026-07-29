"""Programmatic Alembic entry points for the PostgreSQL reference schema."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine


def alembic_config() -> Config:
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).with_name("alembic")),
    )
    return config


def upgrade_postgres(engine: Engine, revision: str = "head") -> None:
    """Upgrade one PostgreSQL database inside a caller-visible transaction."""

    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


__all__ = ["alembic_config", "upgrade_postgres"]
