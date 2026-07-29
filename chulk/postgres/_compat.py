"""Small SQLAlchemy connection shim for the established relational stores."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import re
import sqlite3
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError


class PostgreSQLResult:
    """Expose the subset of the sqlite cursor API used by store owners."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.rowcount = result.rowcount

    def fetchone(self) -> Any | None:
        row = self._result.mappings().fetchone()
        return None if row is None else PostgreSQLRow(row)

    def fetchall(self) -> list[Any]:
        return [PostgreSQLRow(row) for row in self._result.mappings().fetchall()]


class PostgreSQLRow(Mapping[str, Any]):
    """Mapping row that also preserves sqlite's positional indexing."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row
        self._values = tuple(row.values())

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)


class PostgreSQLConnection:
    """Translate the stores' deliberately small SQL subset to PostgreSQL."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def execute(
        self,
        statement: str,
        parameters: Sequence[object] = (),
    ) -> PostgreSQLResult:
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            return PostgreSQLResult(self.connection.execute(text("SELECT 1 WHERE false")))
        sql = _postgres_sql(statement)
        sql, bindings = _bind(sql, parameters)
        try:
            return PostgreSQLResult(self.connection.execute(text(sql), bindings))
        except IntegrityError as exc:
            raise sqlite3.IntegrityError(str(exc)) from exc

    def _lock_run(self, run_id: str) -> None:
        self.execute(
            "SELECT id FROM durable_runs WHERE id = ? FOR UPDATE",
            (run_id,),
        )

    def _lock_run_sequence_allocation(self, run_id: str) -> None:
        self._lock_run(run_id)


class PostgreSQLConnectionOwner:
    """Mixin that lets existing stores execute on a SQLAlchemy engine."""

    engine: Engine
    _bound_connection: ContextVar[Connection | None]

    def _initialize_postgres(self, engine: Engine) -> None:
        self.engine = engine
        self._bound_connection = ContextVar(
            f"postgres_connection_{id(self)}",
            default=None,
        )

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        bound = self._bound_connection.get()
        if bound is not None:
            yield PostgreSQLConnection(bound)
            return
        with self.engine.begin() as connection:
            yield PostgreSQLConnection(connection)

    @contextmanager
    def _using_connection(self, connection: Connection) -> Iterator[None]:
        token = self._bound_connection.set(connection)
        try:
            yield
        finally:
            self._bound_connection.reset(token)

    def _connection_is_bound(self) -> bool:
        return self._bound_connection.get() is not None


def _bind(
    statement: str,
    parameters: Sequence[object],
) -> tuple[str, dict[str, object]]:
    bindings: dict[str, object] = {}
    parts = statement.split("?")
    if len(parts) - 1 != len(parameters):
        if parameters:
            raise ValueError("SQL placeholder count does not match parameters")
        return statement, bindings
    rendered = parts[0]
    for index, value in enumerate(parameters):
        name = f"p{index}"
        rendered += f":{name}{parts[index + 1]}"
        bindings[name] = value
    return rendered, bindings


def _postgres_sql(statement: str) -> str:
    sql = re.sub(r"\bdatetime\(\?\)", "?", statement, flags=re.IGNORECASE)
    sql = re.sub(
        r"\bCASE\s+WHEN\s+\?\s+THEN\b",
        "CASE WHEN ? = 1 THEN",
        sql,
        flags=re.IGNORECASE,
    )
    match = re.match(
        r"(\s*)INSERT\s+OR\s+IGNORE\s+INTO\s+",
        sql,
        flags=re.IGNORECASE,
    )
    if match is not None:
        sql = re.sub(
            r"INSERT\s+OR\s+IGNORE\s+INTO\s+",
            "INSERT INTO ",
            sql,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()
        sql += " ON CONFLICT DO NOTHING"
    sql = re.sub(
        r"json_extract\(([\w.]+),\s*'\$\.([A-Za-z0-9_]+)'\)",
        r"(CAST(\1 AS jsonb) ->> '\2')",
        sql,
        flags=re.IGNORECASE,
    )
    normalized = " ".join(sql.split()).lower()
    if (
        "from durable_runs" in normalized
        and "status = 'queued'" in normalized
        and "limit 1" in normalized
    ):
        sql = sql.rstrip() + " FOR UPDATE SKIP LOCKED"
    elif (
        "from gateway_inbox as candidate" in normalized
        and "candidate.state = 'queued'" in normalized
        and "limit 1" in normalized
    ):
        sql = sql.rstrip() + " FOR UPDATE OF candidate SKIP LOCKED"
    elif (
        "from gateway_outbox as candidate" in normalized
        and "limit 1" in normalized
    ):
        sql = sql.rstrip() + " FOR UPDATE OF candidate SKIP LOCKED"
    return sql


__all__ = [
    "PostgreSQLConnection",
    "PostgreSQLConnectionOwner",
    "PostgreSQLResult",
]
