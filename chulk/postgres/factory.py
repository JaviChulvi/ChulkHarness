"""SQLAlchemy engine factories with production-safe PostgreSQL defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_postgres_engine(
    url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_recycle: int = 1_800,
    connect_args: Mapping[str, Any] | None = None,
) -> Engine:
    """Create a synchronous psycopg 3 engine with liveness checks."""

    _require_postgresql_url(url)
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        connect_args=dict(connect_args or {}),
    )
    return engine


def create_async_postgres_engine(
    url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_recycle: int = 1_800,
    connect_args: Mapping[str, Any] | None = None,
) -> AsyncEngine:
    """Create a native-async psycopg 3 engine with liveness checks."""

    _require_postgresql_url(url)
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        connect_args=dict(connect_args or {}),
    )
    return engine


def _require_postgresql_url(url: str) -> None:
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("the relational reference requires a PostgreSQL URL")
    if parsed.get_driver_name() != "psycopg":
        raise ValueError(
            "the relational reference requires the psycopg 3 driver; "
            "use postgresql+psycopg://"
        )


__all__ = ["create_async_postgres_engine", "create_postgres_engine"]
