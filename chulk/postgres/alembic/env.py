"""Alembic environment for the optional PostgreSQL reference."""

from __future__ import annotations

from alembic import context


def run_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if connection is None:
        raise RuntimeError(
            "Chulk PostgreSQL migrations require upgrade_postgres(engine)"
        )
    context.configure(
        connection=connection,
        target_metadata=None,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations()
