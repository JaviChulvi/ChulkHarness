"""Owner-only control database for durable agent profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from chulk.config import Config
from chulk.profiles.models import (
    AgentProfile,
    AuxiliaryModelProfiles,
    CredentialRef,
    DEFAULT_PROFILE_ID,
    normalize_profile_id,
)
from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.storage.migrations import SQLiteMigration


class ProfileNotFoundError(LookupError):
    """Raised when a requested profile does not exist."""


class ProfileAlreadyExistsError(ValueError):
    """Raised when attempting to replace an immutable profile id."""


class ProfileOwnershipError(ValueError):
    """Raised when two profiles claim the same persistent resource."""


@dataclass(frozen=True, slots=True)
class StoredAgentProfile:
    """A profile plus owner-database timestamps."""

    profile: AgentProfile
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.profile.to_dict(), "created_at": self.created_at}


def _create_control_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE agent_profiles (
            id TEXT PRIMARY KEY,
            project_root TEXT NOT NULL,
            runtime_dir TEXT NOT NULL UNIQUE,
            store_path TEXT NOT NULL UNIQUE,
            traces_dir TEXT NOT NULL UNIQUE,
            memory_namespace TEXT,
            permission_profile TEXT NOT NULL,
            model_profile_id TEXT NOT NULL,
            execution_backend_id TEXT NOT NULL,
            allowed_skills_json TEXT,
            allowed_mcp_servers_json TEXT,
            credential_refs_json TEXT NOT NULL DEFAULT '[]',
            auxiliary_models_json TEXT NOT NULL DEFAULT '{}',
            system_prompt TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE control_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _add_model_profile_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE model_profiles (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            credential_ref TEXT,
            endpoint_ref TEXT,
            fallback_profile_ids_json TEXT NOT NULL DEFAULT '[]',
            required_capabilities_json TEXT NOT NULL DEFAULT '{}',
            context_window_tokens INTEGER,
            response_reserve_tokens INTEGER,
            max_output_tokens INTEGER,
            max_cost_per_turn TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE agent_model_selections (
            agent_profile_id TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT '',
            model_profile_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (agent_profile_id, channel)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE provider_health (
            health_key TEXT PRIMARY KEY,
            model_profile_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            credential_ref TEXT,
            endpoint_ref TEXT,
            state TEXT NOT NULL,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            cooldown_until TEXT,
            last_error_category TEXT,
            last_error_at TEXT,
            last_success_at TEXT,
            successful_requests INTEGER NOT NULL DEFAULT 0,
            failed_requests INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_provider_health_profile ON provider_health(model_profile_id)"
    )


def _add_gateway_control_schema(conn: sqlite3.Connection) -> None:
    """Create owner-controlled channel routing and delivery state."""
    conn.executescript(
        """
        CREATE TABLE gateway_adapters (
            adapter TEXT NOT NULL,
            account_id TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'stopped',
            instance_token TEXT,
            lease_until TEXT,
            cursor TEXT,
            legacy_adopted_at TEXT,
            started_at TEXT,
            stopped_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (adapter, account_id),
            CHECK (state IN ('stopped', 'running'))
        );

        CREATE TABLE gateway_inbox (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            adapter TEXT NOT NULL,
            account_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            conversation_key TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            destination_id TEXT NOT NULL,
            thread_id TEXT,
            envelope_json TEXT NOT NULL,
            state TEXT NOT NULL,
            execution_token TEXT,
            execution_lease_until TEXT,
            cancellation_requested INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            executed_at TEXT,
            UNIQUE (adapter, account_id, idempotency_key),
            CHECK (
                state IN (
                    'queued', 'processing', 'executed', 'ignored',
                    'cancelled', 'uncertain'
                )
            ),
            CHECK (cancellation_requested IN (0, 1))
        );
        CREATE INDEX idx_gateway_inbox_queue
        ON gateway_inbox(state, created_at, id);
        CREATE INDEX idx_gateway_inbox_profile_queue
        ON gateway_inbox(profile_id, state, created_at, id);
        CREATE INDEX idx_gateway_inbox_conversation
        ON gateway_inbox(conversation_key, state, created_at, id);

        CREATE TABLE gateway_outbox (
            id TEXT PRIMARY KEY,
            inbox_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            envelope_json TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            checkpoint TEXT,
            delivery_token TEXT,
            delivery_lease_until TEXT,
            next_attempt_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            delivered_at TEXT,
            UNIQUE (inbox_id, sequence),
            FOREIGN KEY (inbox_id) REFERENCES gateway_inbox(id) ON DELETE CASCADE,
            CHECK (sequence >= 0),
            CHECK (attempt_count >= 0),
            CHECK (state IN ('pending', 'delivering', 'delivered', 'failed'))
        );
        CREATE INDEX idx_gateway_outbox_delivery
        ON gateway_outbox(state, next_attempt_at, created_at, id);
        CREATE INDEX idx_gateway_outbox_profile
        ON gateway_outbox(profile_id, state, created_at, id);

        CREATE TABLE gateway_delivery_events (
            id TEXT PRIMARY KEY,
            outbox_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            state TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (outbox_id) REFERENCES gateway_outbox(id) ON DELETE CASCADE,
            CHECK (attempt >= 1)
        );
        CREATE INDEX idx_gateway_delivery_events_outbox
        ON gateway_delivery_events(outbox_id, recorded_at, id);

        CREATE TABLE gateway_routes (
            id TEXT PRIMARY KEY,
            adapter TEXT NOT NULL,
            account_id TEXT NOT NULL,
            principal_id TEXT NOT NULL DEFAULT '',
            destination_id TEXT NOT NULL DEFAULT '',
            thread_id TEXT NOT NULL DEFAULT '',
            profile_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (
                adapter, account_id, principal_id, destination_id, thread_id
            ),
            CHECK (enabled IN (0, 1))
        );
        CREATE INDEX idx_gateway_routes_lookup
        ON gateway_routes(adapter, account_id, enabled);

        CREATE TABLE gateway_pairings (
            id TEXT PRIMARY KEY,
            code_digest TEXT NOT NULL UNIQUE,
            adapter TEXT NOT NULL,
            account_id TEXT NOT NULL,
            principal_id TEXT,
            profile_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            consumed_at TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX idx_gateway_pairings_target
        ON gateway_pairings(adapter, account_id, expires_at);
        """
    )


CONTROL_MIGRATIONS = (
    SQLiteMigration(1, "agent profile control database", _create_control_schema),
    SQLiteMigration(2, "model profiles and provider health", _add_model_profile_schema),
    SQLiteMigration(3, "channel gateway control ledger", _add_gateway_control_schema),
)


class SQLiteProfileStore:
    """Persist explicit profiles and the local CLI selection."""

    def __init__(self, db_path: Path | str, *, base_config: Config) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.base_config = base_config
        initialize_sqlite_database(self.db_path, migrations=CONTROL_MIGRATIONS)

    @property
    def default_profile(self) -> AgentProfile:
        """Return the compatibility profile without moving legacy state."""
        return AgentProfile(
            id=DEFAULT_PROFILE_ID,
            project_root=self.base_config.project_root,
            runtime_dir=self.base_config.runtime_dir,
            store_path=self.base_config.store_path,
            traces_dir=self.base_config.traces_dir,
            memory_namespace=None,
            permission_profile=self.base_config.permission_profile,
            model_profile_id=DEFAULT_PROFILE_ID,
            execution_backend_id="host",
            implicit=True,
        )

    def create(self, profile: AgentProfile) -> StoredAgentProfile:
        if profile.id == DEFAULT_PROFILE_ID:
            raise ProfileAlreadyExistsError(
                "the implicit default profile cannot be replaced"
            )
        if profile.implicit:
            raise ValueError("explicit profiles cannot be marked implicit")
        self._validate_owned_paths(profile)
        with sqlite_connection(self.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM agent_profiles WHERE id = ?",
                (profile.id,),
            ).fetchone()
        if existing is not None:
            raise ProfileAlreadyExistsError(f"profile {profile.id!r} already exists")
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_profiles (
                        id, project_root, runtime_dir, store_path, traces_dir,
                        memory_namespace, permission_profile, model_profile_id,
                        execution_backend_id, allowed_skills_json,
                        allowed_mcp_servers_json, credential_refs_json,
                        auxiliary_models_json, system_prompt, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.id,
                        str(profile.project_root),
                        str(profile.runtime_dir),
                        str(profile.store_path),
                        str(profile.traces_dir),
                        profile.memory_namespace,
                        profile.permission_profile,
                        profile.model_profile_id,
                        profile.execution_backend_id,
                        _optional_json(profile.allowed_skills),
                        _optional_json(profile.allowed_mcp_servers),
                        json.dumps(
                            [
                                reference.to_dict()
                                for reference in profile.credential_refs
                            ],
                            sort_keys=True,
                        ),
                        json.dumps(profile.auxiliary_models.to_dict(), sort_keys=True),
                        profile.system_prompt,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc).lower()
            if "agent_profiles.id" in message:
                raise ProfileAlreadyExistsError(
                    f"profile {profile.id!r} already exists"
                ) from exc
            raise ProfileOwnershipError(
                "profile persistent paths must be uniquely owned"
            ) from exc
        return StoredAgentProfile(profile=profile, created_at=created_at)

    def create_profile(
        self,
        profile_id: str,
        *,
        project_root: Path | str,
        permission_profile: str | None = None,
        model_profile_id: str = DEFAULT_PROFILE_ID,
        execution_backend_id: str = "host",
        allowed_skills: tuple[str, ...] | None = None,
        allowed_mcp_servers: tuple[str, ...] | None = None,
        credential_refs: tuple[CredentialRef, ...] = (),
        auxiliary_models: AuxiliaryModelProfiles | None = None,
        system_prompt: str | None = None,
    ) -> StoredAgentProfile:
        """Create a profile with deterministic owner-controlled paths."""
        normalized_id = normalize_profile_id(profile_id)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("profile project_root must be an existing directory")
        runtime_dir = (
            self.base_config.runtime_dir / "profiles" / normalized_id
        ).resolve()
        return self.create(
            AgentProfile(
                id=normalized_id,
                project_root=root,
                runtime_dir=runtime_dir,
                store_path=runtime_dir / "store.sqlite",
                traces_dir=runtime_dir / "traces",
                memory_namespace=f"profile:{normalized_id}",
                permission_profile=permission_profile
                or self.base_config.permission_profile,
                model_profile_id=model_profile_id,
                execution_backend_id=execution_backend_id,
                allowed_skills=allowed_skills,
                allowed_mcp_servers=allowed_mcp_servers,
                credential_refs=credential_refs,
                auxiliary_models=auxiliary_models or AuxiliaryModelProfiles(),
                system_prompt=system_prompt,
            )
        )

    def get(self, profile_id: str) -> StoredAgentProfile:
        normalized_id = normalize_profile_id(profile_id)
        if normalized_id == DEFAULT_PROFILE_ID:
            return StoredAgentProfile(profile=self.default_profile, created_at="")
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM agent_profiles WHERE id = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            raise ProfileNotFoundError(f"profile {normalized_id!r} does not exist")
        profile = _row_to_profile(row)
        self._validate_owned_paths(profile)
        return StoredAgentProfile(profile=profile, created_at=str(row["created_at"]))

    def list(self) -> tuple[StoredAgentProfile, ...]:
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM agent_profiles ORDER BY id").fetchall()
        explicit_profiles: list[StoredAgentProfile] = []
        for row in rows:
            profile = _row_to_profile(row)
            self._validate_owned_paths(profile)
            explicit_profiles.append(
                StoredAgentProfile(profile=profile, created_at=str(row["created_at"]))
            )
        return (
            StoredAgentProfile(profile=self.default_profile, created_at=""),
            *explicit_profiles,
        )

    def selected_cli_profile_id(self) -> str:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM control_settings WHERE key = 'cli.default_profile'"
            ).fetchone()
        if row is None:
            return DEFAULT_PROFILE_ID
        profile_id = normalize_profile_id(str(row["value"]))
        self.get(profile_id)
        return profile_id

    def use(self, profile_id: str) -> StoredAgentProfile:
        selected = self.get(profile_id)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO control_settings (key, value, updated_at)
                VALUES ('cli.default_profile', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (selected.profile.id, now),
            )
        return selected

    def resolve(self, profile_id: str | None = None) -> StoredAgentProfile:
        return self.get(profile_id or self.selected_cli_profile_id())

    def _validate_owned_paths(self, profile: AgentProfile) -> None:
        control_path = self.db_path
        owned = {
            profile.runtime_dir,
            profile.store_path,
            profile.traces_dir,
        }
        if control_path in owned:
            raise ProfileOwnershipError("a profile cannot own the control database")
        default = self.default_profile
        default_paths = {default.runtime_dir, default.store_path, default.traces_dir}
        if profile.id != DEFAULT_PROFILE_ID and owned & default_paths:
            raise ProfileOwnershipError(
                "explicit profiles cannot reuse default-profile paths"
            )
        if profile.id != DEFAULT_PROFILE_ID:
            profiles_root = (self.base_config.runtime_dir / "profiles").resolve()
            expected_runtime_dir = profiles_root / profile.id
            if profile.runtime_dir != expected_runtime_dir:
                raise ProfileOwnershipError(
                    "explicit profile runtime_dir must match its immutable profile id"
                )
            if profile.store_path != profile.runtime_dir / "store.sqlite":
                raise ProfileOwnershipError(
                    "explicit profile store_path must use its owned runtime database"
                )
            if profile.traces_dir != profile.runtime_dir / "traces":
                raise ProfileOwnershipError(
                    "explicit profile traces_dir must use its owned trace directory"
                )


def _row_to_profile(row: sqlite3.Row) -> AgentProfile:
    credential_values = json.loads(str(row["credential_refs_json"]))
    auxiliary_values = json.loads(str(row["auxiliary_models_json"]))
    return AgentProfile(
        id=str(row["id"]),
        project_root=Path(str(row["project_root"])),
        runtime_dir=Path(str(row["runtime_dir"])),
        store_path=Path(str(row["store_path"])),
        traces_dir=Path(str(row["traces_dir"])),
        memory_namespace=row["memory_namespace"],
        permission_profile=str(row["permission_profile"]),
        model_profile_id=str(row["model_profile_id"]),
        execution_backend_id=str(row["execution_backend_id"]),
        allowed_skills=_optional_tuple(row["allowed_skills_json"]),
        allowed_mcp_servers=_optional_tuple(row["allowed_mcp_servers_json"]),
        credential_refs=tuple(
            CredentialRef.from_dict(value) for value in credential_values
        ),
        auxiliary_models=AuxiliaryModelProfiles.from_dict(auxiliary_values),
        system_prompt=row["system_prompt"],
    )


def _optional_json(value: tuple[str, ...] | None) -> str | None:
    return None if value is None else json.dumps(list(value), sort_keys=True)


def _optional_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError("stored profile allowlist is invalid")
    return tuple(parsed)


__all__ = [
    "CONTROL_MIGRATIONS",
    "ProfileAlreadyExistsError",
    "ProfileNotFoundError",
    "ProfileOwnershipError",
    "SQLiteProfileStore",
    "StoredAgentProfile",
]
