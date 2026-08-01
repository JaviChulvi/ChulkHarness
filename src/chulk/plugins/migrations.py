"""Forward-only migrations confined to one plugin-owned SQLite database."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from chulk.plugins.manifest import resolve_plugin_resource
from chulk.plugins.models import PluginManifest


_FORBIDDEN_SQL = re.compile(
    r"\b(?:attach|detach|load_extension)\b"
    r"|\bpragma\s+(?:writable_schema|temp_store_directory|data_store_directory)"
    r"|\bvacuum\s+into\b",
    re.IGNORECASE,
)
_MAX_MIGRATION_BYTES = 2_000_000


class PluginMigrationError(RuntimeError):
    """A plugin migration could not be applied or recovered safely."""


class PluginMigrationManager:
    """Own plugin databases, validated backups, and migration state."""

    def __init__(self, runtime_dir: Path | str) -> None:
        root = Path(runtime_dir).expanduser().resolve() / "plugins"
        self.data_dir = root / "data"
        self.backup_dir = root / "backups"

    def database_path(self, plugin_name: str) -> Path:
        return self.data_dir / f"{plugin_name}.sqlite"

    def apply(
        self,
        *,
        manifest: PluginManifest,
        package_root: Path,
        previous_manifest: PluginManifest | None = None,
        previous_root: Path | None = None,
    ) -> Path | None:
        """Apply only new immutable migrations, restoring on any failure."""
        if previous_manifest is not None:
            if previous_root is None:
                raise PluginMigrationError(
                    "previous package root is required for update migrations"
                )
            validate_forward_only_migrations(
                previous_manifest=previous_manifest,
                previous_root=previous_root,
                candidate_manifest=manifest,
                candidate_root=package_root,
            )
        db_path = self.database_path(manifest.name)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _owner_private(self.data_dir, directory=True)
        if db_path.exists() and (
            db_path.is_symlink() or not db_path.is_file()
        ):
            raise PluginMigrationError(
                "plugin database must be a regular file"
            )
        existed = db_path.exists()
        backup = self._backup(db_path, manifest.version) if existed else None
        try:
            with closing(sqlite3.connect(db_path)) as connection:
                with connection:
                    connection.execute("PRAGMA foreign_keys = ON")
                    connection.execute("PRAGMA trusted_schema = OFF")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chulk_plugin_migrations (
                            migration_path TEXT PRIMARY KEY,
                            content_digest TEXT NOT NULL,
                            plugin_version TEXT NOT NULL,
                            applied_at TEXT NOT NULL
                        )
                        """
                    )
                    applied = {
                        str(row[0]): str(row[1])
                        for row in connection.execute(
                            """
                            SELECT migration_path, content_digest
                            FROM chulk_plugin_migrations
                            """
                        )
                    }
                    scripts: list[tuple[str, str, str]] = []
                    for relative in manifest.migrations:
                        path = resolve_plugin_resource(
                            package_root,
                            relative,
                        )
                        content = path.read_bytes()
                        if len(content) > _MAX_MIGRATION_BYTES:
                            raise PluginMigrationError(
                                f"plugin migration exceeds "
                                f"{_MAX_MIGRATION_BYTES} bytes: {relative}"
                            )
                        try:
                            sql = content.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise PluginMigrationError(
                                f"plugin migration must be UTF-8: {relative}"
                            ) from exc
                        if _FORBIDDEN_SQL.search(sql):
                            raise PluginMigrationError(
                                f"plugin migration contains forbidden SQLite "
                                f"authority: {relative}"
                            )
                        digest = f"sha256:{sha256(content).hexdigest()}"
                        prior_digest = applied.get(relative)
                        if prior_digest is not None:
                            if prior_digest != digest:
                                raise PluginMigrationError(
                                    "an applied plugin migration was "
                                    f"modified: {relative}"
                                )
                            continue
                        scripts.append((relative, digest, sql))
                    if scripts:
                        connection.executescript(
                            _migration_script(
                                scripts,
                                plugin_version=manifest.version,
                            )
                        )
                    integrity = connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                    if integrity is None or integrity[0] != "ok":
                        raise PluginMigrationError(
                            "plugin database failed integrity validation"
                        )
            _owner_private(db_path, directory=False)
        except (OSError, sqlite3.Error, PluginMigrationError) as exc:
            self.restore(db_path, backup, remove_without_backup=not existed)
            if isinstance(exc, PluginMigrationError):
                raise
            raise PluginMigrationError(
                f"plugin migration failed: {exc}"
            ) from exc
        return backup

    def backup_current(
        self,
        plugin_name: str,
        version: str,
    ) -> Path | None:
        """Create a validated recovery backup when plugin data exists."""
        database = self.database_path(plugin_name)
        if not database.exists():
            return None
        if database.is_symlink() or not database.is_file():
            raise PluginMigrationError(
                "plugin database must be a regular file"
            )
        return self._backup(database, version)

    def restore(
        self,
        database: Path,
        backup: Path | None,
        *,
        remove_without_backup: bool = False,
    ) -> None:
        """Restore a validated backup after a failed transaction or rollback."""
        if backup is None:
            if remove_without_backup:
                database.unlink(missing_ok=True)
            return
        _validate_sqlite(backup)
        database.parent.mkdir(parents=True, exist_ok=True)
        temporary_handle, temporary_name = tempfile.mkstemp(
            prefix=f".{database.name}.restore-",
            dir=database.parent,
        )
        os.close(temporary_handle)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(backup, temporary)
            _validate_sqlite(temporary)
            os.replace(temporary, database)
            _owner_private(database, directory=False)
        finally:
            temporary.unlink(missing_ok=True)

    def _backup(self, database: Path, version: str) -> Path:
        plugin_dir = self.backup_dir / database.stem
        plugin_dir.mkdir(parents=True, exist_ok=True)
        _owner_private(plugin_dir, directory=True)
        timestamp = (
            datetime.now(timezone.utc)
            .strftime("%Y%m%dT%H%M%S%fZ")
        )
        destination = plugin_dir / f"{timestamp}-{version}.sqlite"
        with closing(sqlite3.connect(database)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
        _validate_sqlite(destination)
        _owner_private(destination, directory=False)
        return destination.resolve(strict=True)


def validate_forward_only_migrations(
    *,
    previous_manifest: PluginManifest,
    previous_root: Path,
    candidate_manifest: PluginManifest,
    candidate_root: Path,
) -> None:
    """Require old migrations to remain an unchanged ordered prefix."""
    previous = previous_manifest.migrations
    candidate = candidate_manifest.migrations
    if candidate[: len(previous)] != previous:
        raise PluginMigrationError(
            "plugin migrations are forward-only; existing paths cannot "
            "be removed, renamed, or reordered"
        )
    for relative in previous:
        old = resolve_plugin_resource(previous_root, relative).read_bytes()
        new = resolve_plugin_resource(candidate_root, relative).read_bytes()
        if sha256(old).digest() != sha256(new).digest():
            raise PluginMigrationError(
                "plugin migrations are forward-only; an existing migration "
                f"changed: {relative}"
            )


def _migration_script(
    scripts: list[tuple[str, str, str]],
    *,
    plugin_version: str,
) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    statements = ["BEGIN IMMEDIATE;"]
    for relative, digest, sql in scripts:
        statements.append(sql)
        statements.append(
            """
            INSERT INTO chulk_plugin_migrations (
                migration_path,
                content_digest,
                plugin_version,
                applied_at
            ) VALUES (
                '{path}',
                '{digest}',
                '{version}',
                '{timestamp}'
            );
            """.format(
                path=_sql_literal(relative),
                digest=_sql_literal(digest),
                version=_sql_literal(plugin_version),
                timestamp=_sql_literal(timestamp),
            )
        )
    statements.append("COMMIT;")
    return "\n".join(statements)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _validate_sqlite(path: Path) -> None:
    try:
        with closing(
            sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        ) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        raise PluginMigrationError(
            f"plugin database backup is invalid: {exc}"
        ) from exc
    if result is None or result[0] != "ok":
        raise PluginMigrationError(
            "plugin database backup failed integrity validation"
        )


def _owner_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600)


__all__ = [
    "PluginMigrationError",
    "PluginMigrationManager",
    "validate_forward_only_migrations",
]
