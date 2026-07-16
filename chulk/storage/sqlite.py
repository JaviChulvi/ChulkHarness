"""Shared SQLite connection, migration, and backup policy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from chulk.storage.migrations import SQLITE_MIGRATIONS, SQLiteMigration


SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_JOURNAL_MODE = "wal"
SQLITE_SYNCHRONOUS = "normal"
SQLITE_WAL_AUTOCHECKPOINT_PAGES = 1_000
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


class UnsupportedSQLiteSchemaVersionError(RuntimeError):
    """Raised when a database was created by a newer Chulk version."""

    def __init__(self, path: Path, found_version: int, supported_version: int) -> None:
        self.path = path
        self.found_version = found_version
        self.supported_version = supported_version
        super().__init__(
            f"SQLite schema version {found_version} at {path} is newer than "
            f"the supported version {supported_version}"
        )


class SQLiteMigrationError(RuntimeError):
    """Raised after a schema migration is rolled back."""

    def __init__(
        self,
        path: Path,
        target_version: int,
        *,
        backup_path: Path | None,
    ) -> None:
        self.path = path
        self.target_version = target_version
        self.backup_path = backup_path
        backup_note = f"; pre-migration backup: {backup_path}" if backup_path is not None else ""
        super().__init__(f"SQLite migration to version {target_version} failed and was rolled back{backup_note}")


class SQLiteBackupError(RuntimeError):
    """Raised when an online backup cannot be validated."""


@dataclass(frozen=True, slots=True)
class SQLiteMigrationReport:
    """Result of initializing or upgrading a SQLite database."""

    from_version: int
    to_version: int
    backup_path: Path | None = None


def initialize_sqlite_database(
    db_path: Path | str,
    *,
    migrations: Sequence[SQLiteMigration] | None = None,
) -> SQLiteMigrationReport:
    """Apply pending forward-only migrations under a database writer lock."""
    path = Path(db_path)
    migration_plan = tuple(SQLITE_MIGRATIONS if migrations is None else migrations)
    _validate_migration_plan(migration_plan)
    latest_version = migration_plan[-1].version if migration_plan else 0

    _prepare_private_file(path)
    with sqlite_connection(path) as conn:
        observed_version = _user_version(conn)
        _reject_future_version(path, observed_version, latest_version)
        if observed_version == latest_version:
            return SQLiteMigrationReport(observed_version, observed_version)

        conn.execute("BEGIN IMMEDIATE")
        current_version = _user_version(conn)
        _reject_future_version(path, current_version, latest_version)
        if current_version == latest_version:
            return SQLiteMigrationReport(current_version, current_version)

        backup_path = None
        if _has_application_schema(conn):
            backup_path = _create_locked_backup(path, current_version)

        target_version = current_version
        try:
            for migration in migration_plan:
                if migration.version <= current_version:
                    continue
                target_version = migration.version
                migration.apply(conn)
                conn.execute(f"PRAGMA user_version = {migration.version}")
            _validate_database(conn)
        except BaseException as exc:
            conn.rollback()
            raise SQLiteMigrationError(
                path,
                target_version,
                backup_path=backup_path,
            ) from exc

        return SQLiteMigrationReport(current_version, latest_version, backup_path)


@contextmanager
def sqlite_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a transactional connection with Chulk's explicit SQLite policy."""
    path = Path(db_path)
    _prepare_private_file(path)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        journal_mode = str(conn.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE}").fetchone()[0]).lower()
        if journal_mode != SQLITE_JOURNAL_MODE:
            raise sqlite3.OperationalError(
                f"SQLite journal mode is {journal_mode!r}; expected {SQLITE_JOURNAL_MODE!r}"
            )
        conn.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS}")
        conn.execute(f"PRAGMA wal_autocheckpoint = {SQLITE_WAL_AUTOCHECKPOINT_PAGES}")
        yield conn
        conn.commit()
    except BaseException:
        if conn is not None and conn.in_transaction:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
        _restrict_database_files(path)


def create_sqlite_backup(
    db_path: Path | str,
    destination: Path | str | None = None,
) -> Path:
    """Create and validate a consistent online backup of a live database."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    backup_path = Path(destination) if destination is not None else _new_backup_path(path, _read_user_version(path))
    return _backup_database(path, backup_path)


def _create_locked_backup(path: Path, version: int) -> Path:
    """Back up while the caller holds a writer lock on another connection."""
    return _backup_database(path, _new_backup_path(path, version))


def _backup_database(path: Path, backup_path: Path) -> Path:
    if backup_path.exists():
        raise FileExistsError(backup_path)
    _prepare_private_file(backup_path)
    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    try:
        source = sqlite3.connect(path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000)
        source.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        destination = sqlite3.connect(backup_path)
        source.backup(destination)
        destination.commit()
        _validate_backup(destination)
    except BaseException:
        if destination is not None:
            destination.close()
            destination = None
        if source is not None:
            source.close()
            source = None
        _remove_database_files(backup_path)
        raise
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        _restrict_database_files(path)
        _restrict_database_files(backup_path)
    return backup_path


def _validate_backup(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA integrity_check").fetchall()
    if [str(row[0]).lower() for row in result] != ["ok"]:
        raise SQLiteBackupError("SQLite backup failed integrity validation")


def _validate_database(conn: sqlite3.Connection) -> None:
    integrity = conn.execute("PRAGMA integrity_check").fetchall()
    if [str(row[0]).lower() for row in integrity] != ["ok"]:
        raise sqlite3.IntegrityError("Migrated SQLite database failed integrity_check")
    foreign_key_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_violations:
        raise sqlite3.IntegrityError("Migrated SQLite database contains foreign-key violations")


def _validate_migration_plan(migrations: Sequence[SQLiteMigration]) -> None:
    versions = [migration.version for migration in migrations]
    expected = list(range(1, len(migrations) + 1))
    if versions != expected:
        raise ValueError(f"SQLite migration versions must be contiguous from 1; found {versions}")


def _reject_future_version(path: Path, current_version: int, latest_version: int) -> None:
    if current_version > latest_version:
        raise UnsupportedSQLiteSchemaVersionError(path, current_version, latest_version)


def _has_application_schema(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
          AND type IN ('table', 'index', 'view', 'trigger')
        LIMIT 1
        """
    ).fetchone()
    return row is not None


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _read_user_version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return _user_version(conn)


def _new_backup_path(path: Path, version: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = uuid4().hex[:8]
    return path.with_name(f"{path.name}.backup-v{version}-{timestamp}-{nonce}.sqlite")


def _database_runtime_paths(path: Path) -> tuple[Path, ...]:
    return (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    )


def _restrict_database_files(path: Path) -> None:
    for candidate in _database_runtime_paths(path):
        if candidate.exists():
            _restrict_private_file(candidate)


def _remove_database_files(path: Path) -> None:
    for candidate in _database_runtime_paths(path):
        candidate.unlink(missing_ok=True)


def _prepare_private_file(path: Path) -> None:
    """Create a database file privately and repair its mode if it exists."""
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    if os.name == "posix" and not parent_existed:
        os.chmod(path.parent, _PRIVATE_DIRECTORY_MODE)

    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
    except FileExistsError:
        pass
    else:
        os.close(descriptor)
    _restrict_private_file(path)


def _restrict_private_file(path: Path) -> None:
    if os.name == "posix":
        os.chmod(path, _PRIVATE_FILE_MODE)


__all__ = [
    "SQLITE_BUSY_TIMEOUT_MS",
    "SQLITE_JOURNAL_MODE",
    "SQLITE_SYNCHRONOUS",
    "SQLITE_WAL_AUTOCHECKPOINT_PAGES",
    "SQLiteBackupError",
    "SQLiteMigrationError",
    "SQLiteMigrationReport",
    "UnsupportedSQLiteSchemaVersionError",
    "create_sqlite_backup",
    "initialize_sqlite_database",
    "sqlite_connection",
]
