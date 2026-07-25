"""Control-database persistence for named model profiles and provider health."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from chulk.config import Config
from chulk.model_profiles.models import (
    DiagnosticCategory,
    EndpointRef,
    ModelCapabilityRequirements,
    ModelProfile,
    ProviderHealth,
    ProviderHealthStatus,
)
from chulk.profiles import CredentialRef, normalize_profile_id
from chulk.profiles.store import CONTROL_MIGRATIONS
from chulk.storage import initialize_sqlite_database, sqlite_connection


class ModelProfileNotFoundError(LookupError):
    pass


class ModelProfileAlreadyExistsError(ValueError):
    pass


class ModelProfileStore:
    """Persist model profiles in the owner-only profile control database."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        base_config: Config,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.base_config = base_config
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        initialize_sqlite_database(self.db_path, migrations=CONTROL_MIGRATIONS)

    @property
    def implicit_profiles(self) -> dict[str, ModelProfile]:
        fallback_ids = tuple(
            f"default-fallback-{index}"
            for index, _item in enumerate(
                self.base_config.llm_fallback_providers,
                start=1,
            )
        )
        profiles = {
            "default": _implicit_profile(
                "default",
                self.base_config.llm_provider,
                self.base_config.model,
                fallback_ids=fallback_ids,
            )
        }
        for index, fallback in enumerate(
            self.base_config.llm_fallback_providers,
            start=1,
        ):
            profile_id = f"default-fallback-{index}"
            profiles[profile_id] = _implicit_profile(
                profile_id,
                fallback.provider,
                fallback.model,
            )
        return profiles

    def create(self, profile: ModelProfile) -> ModelProfile:
        if profile.implicit or profile.id in self.implicit_profiles:
            raise ModelProfileAlreadyExistsError(
                "implicit model profiles cannot be replaced"
            )
        try:
            with sqlite_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO model_profiles (
                        id, provider, model, credential_ref, endpoint_ref,
                        fallback_profile_ids_json, required_capabilities_json,
                        context_window_tokens, response_reserve_tokens,
                        max_output_tokens, max_cost_per_turn, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.id,
                        profile.provider,
                        profile.model,
                        profile.credential_ref.uri if profile.credential_ref else None,
                        profile.endpoint_ref.uri if profile.endpoint_ref else None,
                        json.dumps(list(profile.fallback_profile_ids)),
                        json.dumps(
                            profile.required_capabilities.to_dict(), sort_keys=True
                        ),
                        profile.context_window_tokens,
                        profile.response_reserve_tokens,
                        profile.max_output_tokens,
                        str(profile.max_cost_per_turn)
                        if profile.max_cost_per_turn
                        else None,
                        self.clock().isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ModelProfileAlreadyExistsError(
                f"model profile {profile.id!r} already exists"
            ) from exc
        return profile

    def get(self, profile_id: str) -> ModelProfile:
        profile_id = normalize_profile_id(profile_id)
        implicit = self.implicit_profiles.get(profile_id)
        if implicit is not None:
            return implicit
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
        if row is None:
            raise ModelProfileNotFoundError(
                f"model profile {profile_id!r} does not exist"
            )
        return _row_to_model_profile(row)

    def list(self) -> tuple[ModelProfile, ...]:
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM model_profiles ORDER BY id").fetchall()
        return (
            *self.implicit_profiles.values(),
            *(_row_to_model_profile(row) for row in rows),
        )

    def selected_for_agent(
        self,
        agent_profile_id: str,
        *,
        default: str,
        channel: str | None = None,
    ) -> str:
        channel_key = _normalize_channel(channel)
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT model_profile_id
                FROM agent_model_selections
                WHERE agent_profile_id = ? AND channel = ?
                """,
                (normalize_profile_id(agent_profile_id), channel_key),
            ).fetchone()
            if row is None and channel_key:
                row = conn.execute(
                    """
                    SELECT model_profile_id
                    FROM agent_model_selections
                    WHERE agent_profile_id = ? AND channel = ''
                    """,
                    (normalize_profile_id(agent_profile_id),),
                ).fetchone()
        return str(row["model_profile_id"]) if row is not None else default

    def use_for_agent(
        self,
        agent_profile_id: str,
        model_profile_id: str,
        *,
        channel: str | None = None,
    ) -> None:
        profile = self.get(model_profile_id)
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_model_selections (
                    agent_profile_id, channel, model_profile_id, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_profile_id, channel) DO UPDATE SET
                    model_profile_id = excluded.model_profile_id,
                    updated_at = excluded.updated_at
                """,
                (
                    normalize_profile_id(agent_profile_id),
                    _normalize_channel(channel),
                    profile.id,
                    self.clock().isoformat(),
                ),
            )

    def health(self, profile: ModelProfile) -> ProviderHealth:
        key = health_key(profile)
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM provider_health WHERE health_key = ?",
                (key,),
            ).fetchone()
        health = (
            _row_to_health(row, model_profile_id=profile.id)
            if row is not None
            else _healthy(profile, key)
        )
        if (
            health.status is ProviderHealthStatus.COOLDOWN
            and health.cooldown_until is not None
            and health.cooldown_until <= self.clock()
        ):
            return replace(
                health,
                status=ProviderHealthStatus.DEGRADED,
                cooldown_until=None,
            )
        return health

    def record_success(self, profile: ModelProfile) -> ProviderHealth:
        current = self.health(profile)
        now = self.clock()
        health = ProviderHealth(
            health_key=health_key(profile),
            model_profile_id=profile.id,
            provider=profile.provider,
            credential_ref=profile.credential_ref.uri
            if profile.credential_ref
            else None,
            endpoint_ref=profile.endpoint_ref.uri if profile.endpoint_ref else None,
            status=ProviderHealthStatus.HEALTHY,
            consecutive_failures=0,
            last_error_category=current.last_error_category,
            last_error_at=current.last_error_at,
            last_success_at=now,
            successful_requests=current.successful_requests + 1,
            failed_requests=current.failed_requests,
        )
        self._write_health(health, now)
        return health

    def record_failure(
        self,
        profile: ModelProfile,
        category: DiagnosticCategory,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        max_cooldown_seconds: int = 900,
    ) -> ProviderHealth:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be greater than zero")
        if cooldown_seconds < 1:
            raise ValueError("cooldown_seconds must be greater than zero")
        if max_cooldown_seconds < cooldown_seconds:
            raise ValueError("max_cooldown_seconds must be at least cooldown_seconds")
        current = self.health(profile)
        now = self.clock()
        circuit_categories = {
            DiagnosticCategory.AUTHENTICATION,
            DiagnosticCategory.BILLING,
            DiagnosticCategory.RATE_LIMIT,
            DiagnosticCategory.TIMEOUT,
            DiagnosticCategory.INVALID_MODEL,
            DiagnosticCategory.UNAVAILABLE_ENDPOINT,
            DiagnosticCategory.UNSUPPORTED_CAPABILITY,
            DiagnosticCategory.UNSUPPORTED_SCHEMA,
        }
        affects_circuit = category in circuit_categories
        failures = current.consecutive_failures + 1 if affects_circuit else 0
        immediate = category in {
            DiagnosticCategory.AUTHENTICATION,
            DiagnosticCategory.BILLING,
        }
        enter_cooldown = affects_circuit and (
            immediate or failures >= failure_threshold
        )
        cooldown_until = None
        status = ProviderHealthStatus.DEGRADED
        if enter_cooldown:
            from datetime import timedelta

            multiplier = min(max(1, failures - failure_threshold + 1), 8)
            duration = min(cooldown_seconds * multiplier, max_cooldown_seconds)
            cooldown_until = now + timedelta(seconds=duration)
            status = ProviderHealthStatus.COOLDOWN
        health = ProviderHealth(
            health_key=health_key(profile),
            model_profile_id=profile.id,
            provider=profile.provider,
            credential_ref=profile.credential_ref.uri
            if profile.credential_ref
            else None,
            endpoint_ref=profile.endpoint_ref.uri if profile.endpoint_ref else None,
            status=status,
            consecutive_failures=failures,
            cooldown_until=cooldown_until,
            last_error_category=category,
            last_error_at=now,
            last_success_at=current.last_success_at,
            successful_requests=current.successful_requests,
            failed_requests=current.failed_requests + 1,
        )
        self._write_health(health, now)
        return health

    def reset_health(self, profile: ModelProfile) -> ProviderHealth:
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                "DELETE FROM provider_health WHERE health_key = ?",
                (health_key(profile),),
            )
        return self.health(profile)

    def _write_health(self, health: ProviderHealth, now: datetime) -> None:
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO provider_health (
                    health_key, model_profile_id, provider, credential_ref,
                    endpoint_ref, state, consecutive_failures, cooldown_until,
                    last_error_category, last_error_at, last_success_at,
                    successful_requests, failed_requests, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(health_key) DO UPDATE SET
                    model_profile_id = excluded.model_profile_id,
                    provider = excluded.provider,
                    credential_ref = excluded.credential_ref,
                    endpoint_ref = excluded.endpoint_ref,
                    state = excluded.state,
                    consecutive_failures = excluded.consecutive_failures,
                    cooldown_until = excluded.cooldown_until,
                    last_error_category = excluded.last_error_category,
                    last_error_at = excluded.last_error_at,
                    last_success_at = excluded.last_success_at,
                    successful_requests = excluded.successful_requests,
                    failed_requests = excluded.failed_requests,
                    updated_at = excluded.updated_at
                """,
                (
                    health.health_key,
                    health.model_profile_id,
                    health.provider,
                    health.credential_ref,
                    health.endpoint_ref,
                    health.status.value,
                    health.consecutive_failures,
                    health.cooldown_until.isoformat()
                    if health.cooldown_until
                    else None,
                    (
                        health.last_error_category.value
                        if health.last_error_category is not None
                        else None
                    ),
                    health.last_error_at.isoformat() if health.last_error_at else None,
                    health.last_success_at.isoformat()
                    if health.last_success_at
                    else None,
                    health.successful_requests,
                    health.failed_requests,
                    now.isoformat(),
                ),
            )


def health_key(profile: ModelProfile) -> str:
    value = "|".join(
        (
            profile.provider,
            profile.credential_ref.uri if profile.credential_ref else "",
            profile.endpoint_ref.uri if profile.endpoint_ref else "",
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _implicit_profile(
    profile_id: str,
    provider: str,
    model: str,
    *,
    fallback_ids: tuple[str, ...] = (),
) -> ModelProfile:
    credential_name = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai-compatible": "CHULK_OPENAI_COMPATIBLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
        "gemini": "GEMINI_API_KEY",
    }.get(provider)
    return ModelProfile(
        id=profile_id,
        provider=provider,
        model=model,
        credential_ref=(
            CredentialRef(credential_name) if credential_name is not None else None
        ),
        endpoint_ref=EndpointRef(provider),
        fallback_profile_ids=fallback_ids,
        implicit=True,
    )


def _row_to_model_profile(row: sqlite3.Row) -> ModelProfile:
    fallback_ids = json.loads(str(row["fallback_profile_ids_json"]))
    requirements = json.loads(str(row["required_capabilities_json"]))
    return ModelProfile(
        id=str(row["id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        credential_ref=(
            CredentialRef.parse(str(row["credential_ref"]))
            if row["credential_ref"] is not None
            else None
        ),
        endpoint_ref=(
            EndpointRef.parse(str(row["endpoint_ref"]))
            if row["endpoint_ref"] is not None
            else None
        ),
        fallback_profile_ids=tuple(str(value) for value in fallback_ids),
        required_capabilities=ModelCapabilityRequirements.from_dict(requirements),
        context_window_tokens=row["context_window_tokens"],
        response_reserve_tokens=row["response_reserve_tokens"],
        max_output_tokens=row["max_output_tokens"],
        max_cost_per_turn=(
            Decimal(str(row["max_cost_per_turn"]))
            if row["max_cost_per_turn"] is not None
            else None
        ),
    )


def _healthy(profile: ModelProfile, key: str) -> ProviderHealth:
    return ProviderHealth(
        health_key=key,
        model_profile_id=profile.id,
        provider=profile.provider,
        credential_ref=profile.credential_ref.uri if profile.credential_ref else None,
        endpoint_ref=profile.endpoint_ref.uri if profile.endpoint_ref else None,
        status=ProviderHealthStatus.HEALTHY,
        consecutive_failures=0,
        successful_requests=0,
        failed_requests=0,
    )


def _row_to_health(
    row: sqlite3.Row,
    *,
    model_profile_id: str | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        health_key=str(row["health_key"]),
        model_profile_id=model_profile_id or str(row["model_profile_id"]),
        provider=str(row["provider"]),
        credential_ref=row["credential_ref"],
        endpoint_ref=row["endpoint_ref"],
        status=ProviderHealthStatus(str(row["state"])),
        consecutive_failures=int(row["consecutive_failures"]),
        cooldown_until=_datetime(row["cooldown_until"]),
        last_error_category=(
            DiagnosticCategory(str(row["last_error_category"]))
            if row["last_error_category"] is not None
            else None
        ),
        last_error_at=_datetime(row["last_error_at"]),
        last_success_at=_datetime(row["last_success_at"]),
        successful_requests=int(row["successful_requests"]),
        failed_requests=int(row["failed_requests"]),
    )


def _datetime(value: Any) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value is not None else None


def _normalize_channel(value: str | None) -> str:
    channel = (value or "").strip().lower()
    if len(channel) > 64:
        raise ValueError("channel cannot exceed 64 characters")
    if channel and not all(
        character.isalnum() or character in "._-" for character in channel
    ):
        raise ValueError("channel contains unsupported characters")
    return channel


__all__ = [
    "ModelProfileAlreadyExistsError",
    "ModelProfileNotFoundError",
    "ModelProfileStore",
    "health_key",
]
