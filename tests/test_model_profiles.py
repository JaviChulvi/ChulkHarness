"""Tests for durable model profiles, diagnostics, selection, and health."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from chulk._sdk.results import run_result_from_runtime
from chulk.config import load_config
from chulk.core import TraceEvent
from tests.core_agent import build_core_agent as Agent
from chulk.llm import LLMClient, LLMError, LLMResponse
from chulk.llm.base import classify_provider_exception
from chulk.model_profiles import (
    DiagnosticCategory,
    EndpointRef,
    ModelCapabilityRequirements,
    ModelProfile,
    ModelProfileService,
    ModelProfileStore,
    ModelProfileValidator,
    ProviderHealthStatus,
    discover_endpoint_models,
)
from chulk.profiles import CredentialRef, ProfileRuntimeFactory
from chulk.storage import sqlite_connection


def _runtime(tmp_path: Path, *, environ: dict[str, str] | None = None):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    profiles = ProfileRuntimeFactory(config)
    model_store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    service = ModelProfileService(
        model_store,
        ModelProfileValidator(config, environ=environ or {}),
    )
    return config, profiles.resolve().profile, model_store, service


def _openai_profile(
    profile_id: str,
    credential_name: str,
    *,
    fallbacks: tuple[str, ...] = (),
) -> ModelProfile:
    return ModelProfile(
        id=profile_id,
        provider="openai",
        model="gpt-4.1-mini",
        credential_ref=CredentialRef(credential_name),
        fallback_profile_ids=fallbacks,
    )


def test_typed_credential_and_endpoint_references_reject_arbitrary_values() -> None:
    assert CredentialRef.parse("env:MODEL_KEY").uri == "env:MODEL_KEY"
    assert CredentialRef.parse("keyring:chulk/work").uri == "keyring:chulk/work"
    assert EndpointRef.parse("host:private-gateway").uri == "host:private-gateway"

    with pytest.raises(ValueError):
        CredentialRef.parse("literal-secret")
    with pytest.raises(ValueError):
        CredentialRef.parse("https://secret.example")
    with pytest.raises(ValueError):
        EndpointRef.parse("https://models.example")


def test_model_profiles_persist_references_but_never_resolved_secrets(
    tmp_path: Path,
) -> None:
    secret = "highly-sensitive-value"
    _config, _agent_profile, store, service = _runtime(
        tmp_path,
        environ={"MODEL_KEY": secret},
    )
    profile = service.create(_openai_profile("coding", "MODEL_KEY"))

    diagnostic = service.diagnose(profile.id)
    persisted = store.get(profile.id)

    assert diagnostic.ok
    assert persisted.credential_ref == CredentialRef("MODEL_KEY")
    assert persisted.to_dict()["credential_ref"] == "env:MODEL_KEY"
    assert secret.encode() not in store.db_path.read_bytes()
    assert secret not in json.dumps(diagnostic.to_dict())


def test_implicit_default_maps_legacy_provider_model_and_fallbacks(
    tmp_path: Path,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openai",
            "CHULK_MODEL": "gpt-4.1-mini",
            "CHULK_LLM_FALLBACK_PROVIDERS": "deepseek:deepseek-chat",
        }
    )
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )

    primary = store.get("default")
    fallback = store.get("default-fallback-1")

    assert primary.implicit
    assert (primary.provider, primary.model) == ("openai", "gpt-4.1-mini")
    assert primary.fallback_profile_ids == ("default-fallback-1",)
    assert (fallback.provider, fallback.model) == ("deepseek", "deepseek-chat")


def test_implicit_default_always_uses_the_legacy_compatibility_builder(
    tmp_path: Path,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "OPENAI_API_KEY": "configured",
        }
    )
    agent_profile = ProfileRuntimeFactory(config).resolve().profile
    service = ModelProfileService(
        ModelProfileStore(
            config.runtime_dir / "control.sqlite",
            base_config=config,
        ),
        ModelProfileValidator(
            config,
            environ={"OPENAI_API_KEY": "configured"},
        ),
    )

    runtime = service.resolve_for_agent(agent_profile)

    assert runtime.legacy_compatibility
    assert runtime.candidates == ()
    assert runtime.selection.selected_profile_id == "default"


def test_definition_validation_covers_catalog_capabilities_and_limits(
    tmp_path: Path,
) -> None:
    _config, _agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"MODEL_KEY": "available"},
    )

    with pytest.raises(ValueError, match="token capability metadata"):
        service.create(
            ModelProfile(
                id="unknown",
                provider="openai",
                model="not-a-real-model",
                credential_ref=CredentialRef("MODEL_KEY"),
            )
        )
    with pytest.raises(ValueError, match="hosted_mcp_tools"):
        service.create(
            ModelProfile(
                id="unsupported",
                provider="local",
                model="qwen/qwen3.5-35b-a3b",
                required_capabilities=ModelCapabilityRequirements(
                    hosted_mcp_tools=True
                ),
            )
        )
    with pytest.raises(ValueError, match="context"):
        service.create(
            ModelProfile(
                id="bad-limits",
                provider="openai",
                model="gpt-4.1-mini",
                credential_ref=CredentialRef("MODEL_KEY"),
                context_window_tokens=100,
            )
        )
    with pytest.raises(ValueError, match="positive finite Decimal"):
        ModelProfile(
            id="bad-cost",
            provider="openai",
            model="gpt-4.1-mini",
            credential_ref=CredentialRef("MODEL_KEY"),
            max_cost_per_turn=Decimal("NaN"),
        )


def test_missing_credentials_are_sanitized_and_skip_to_fallback(tmp_path: Path) -> None:
    _config, agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"SECONDARY_KEY": "secondary-secret"},
    )
    service.create(_openai_profile("secondary", "SECONDARY_KEY"))
    service.create(
        _openai_profile(
            "primary",
            "MISSING_PRIMARY_KEY",
            fallbacks=("secondary",),
        )
    )

    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="primary",
    )

    assert runtime.selection.selected_profile_id == "secondary"
    assert (
        runtime.selection.skipped[0].category is DiagnosticCategory.MISSING_CREDENTIAL
    )
    serialized = json.dumps(runtime.to_dict())
    assert "secondary-secret" not in serialized


def test_endpoint_diagnostics_reject_urls_with_embedded_query_credentials(
    tmp_path: Path,
) -> None:
    _config, _agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"LOCAL_ENDPOINT": "http://127.0.0.1:11434/v1?token=secret"},
    )
    profile = service.create(
        ModelProfile(
            id="unsafe-endpoint",
            provider="local",
            model="qwen/qwen3.5-35b-a3b",
            endpoint_ref=EndpointRef("LOCAL_ENDPOINT", source="env"),
        )
    )

    diagnostic = service.diagnose(profile.id)

    assert diagnostic.category is DiagnosticCategory.UNAVAILABLE_ENDPOINT
    assert "token=secret" not in json.dumps(diagnostic.to_dict())


def test_fallback_cycles_are_detected_even_if_storage_is_tampered(
    tmp_path: Path,
) -> None:
    _config, agent_profile, store, service = _runtime(
        tmp_path,
        environ={"MODEL_KEY": "available"},
    )
    service.create(_openai_profile("alpha", "MODEL_KEY"))
    service.create(_openai_profile("beta", "MODEL_KEY", fallbacks=("alpha",)))
    with sqlite_connection(store.db_path) as conn:
        conn.execute(
            "UPDATE model_profiles SET fallback_profile_ids_json = ? WHERE id = ?",
            ('["beta"]', "alpha"),
        )

    with pytest.raises(ValueError, match="alpha -> beta -> alpha"):
        service.resolve_for_agent(
            agent_profile,
            requested_profile_id="alpha",
        )


def test_channel_selection_falls_back_to_agent_selection(tmp_path: Path) -> None:
    _config, agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"MODEL_KEY": "available"},
    )
    service.create(_openai_profile("general", "MODEL_KEY"))
    service.create(_openai_profile("telegram", "MODEL_KEY"))

    service.use_for_agent(agent_profile, "general")
    service.use_for_agent(agent_profile, "telegram", channel="telegram")

    assert service.selected_profile_id(agent_profile, channel="cli") == "general"
    assert service.selected_profile_id(agent_profile, channel="telegram") == "telegram"


def test_explicit_agent_profile_constrains_channel_model_selection(
    tmp_path: Path,
) -> None:
    config, _default_profile, _store, service = _runtime(
        tmp_path,
        environ={"MODEL_KEY": "available"},
    )
    service.create(_openai_profile("allowed", "MODEL_KEY"))
    service.create(_openai_profile("denied", "MODEL_KEY"))
    project = tmp_path / "explicit-project"
    project.mkdir()
    profile_store = ProfileRuntimeFactory(config).profile_store
    explicit = profile_store.create_profile(
        "restricted",
        project_root=project,
        model_profile_id="allowed",
    ).profile

    assert service.use_for_agent(explicit, "allowed", channel="telegram").id == (
        "allowed"
    )
    with pytest.raises(ValueError, match="not allowed"):
        service.use_for_agent(explicit, "denied", channel="telegram")


class _MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 25, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def test_cooldown_is_deterministic_bounded_observable_and_resettable(
    tmp_path: Path,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    clock = _MutableClock()
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
        clock=clock,
    )
    profile = _openai_profile("coding", "MODEL_KEY")
    store.create(profile)

    first = store.record_failure(profile, DiagnosticCategory.RATE_LIMIT)
    second = store.record_failure(profile, DiagnosticCategory.RATE_LIMIT)
    third = store.record_failure(profile, DiagnosticCategory.RATE_LIMIT)

    assert first.status is ProviderHealthStatus.DEGRADED
    assert second.status is ProviderHealthStatus.DEGRADED
    assert third.status is ProviderHealthStatus.COOLDOWN
    assert third.cooldown_until == clock.now + timedelta(seconds=60)
    assert third.failed_requests == 3
    assert third.cooldown_until is not None
    clock.now = third.cooldown_until
    expired = store.health(profile)
    assert expired.status is ProviderHealthStatus.DEGRADED
    assert expired.cooldown_until is None

    reset = store.reset_health(profile)
    assert reset.status is ProviderHealthStatus.HEALTHY
    assert reset.failed_requests == 0


def test_unknown_failures_are_counted_without_opening_the_circuit(
    tmp_path: Path,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    profile = store.create(_openai_profile("coding", "MODEL_KEY"))

    for _index in range(5):
        health = store.record_failure(profile, DiagnosticCategory.UNKNOWN)

    assert health.status is ProviderHealthStatus.DEGRADED
    assert health.consecutive_failures == 0
    assert health.failed_requests == 5


def test_circuit_state_is_shared_by_provider_and_credential_reference(
    tmp_path: Path,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    first = store.create(_openai_profile("first", "SHARED_KEY"))
    second = store.create(_openai_profile("second", "SHARED_KEY"))

    store.record_failure(first, DiagnosticCategory.AUTHENTICATION)

    assert store.health(second).status is ProviderHealthStatus.COOLDOWN
    assert store.health(second).failed_requests == 1


class _FailingProfileClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return self.complete_response(
            messages,
            max_output_tokens=max_output_tokens,
        ).content

    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        raise LLMError(
            "invalid credential",
            code="authentication_error",
            retryable=False,
            fallback_eligible=False,
        )


class _SuccessfulProfileClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return self.complete_response(
            messages,
            max_output_tokens=max_output_tokens,
        ).content

    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="ok", provider=self.provider, model=self.model)


def test_profile_chain_falls_back_on_terminal_health_errors_and_opens_circuit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config, agent_profile, store, service = _runtime(
        tmp_path,
        environ={"PRIMARY_KEY": "bad", "SECONDARY_KEY": "good"},
    )
    service.create(_openai_profile("secondary", "SECONDARY_KEY"))
    primary_profile = service.create(
        _openai_profile("primary", "PRIMARY_KEY", fallbacks=("secondary",))
    )
    primary = _FailingProfileClient()
    secondary = _SuccessfulProfileClient()

    def create_client(**kwargs):
        return primary if kwargs["connection"].api_key == "bad" else secondary

    monkeypatch.setattr(
        "chulk.model_profiles.service.create_llm_client",
        create_client,
    )
    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="primary",
    )
    chain = service.create_chain(service.validator.config, runtime)

    assert chain.complete_response([]).content == "ok"
    assert store.health(primary_profile).status is ProviderHealthStatus.COOLDOWN
    assert chain.complete_response([]).content == "ok"
    assert primary.calls == 1
    assert secondary.calls == 2
    assert chain.last_attempts[0].error_code == "circuit_open"
    assert chain.last_attempts[0].model_profile_id == "primary"


def test_credential_reference_is_refreshed_for_each_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = {"ROTATING_KEY": "first"}
    _config, agent_profile, _store, service = _runtime(
        tmp_path,
        environ=environment,
    )
    service.create(_openai_profile("rotating", "ROTATING_KEY"))
    observed_keys: list[str | None] = []

    def create_client(**kwargs):
        observed_keys.append(kwargs["connection"].api_key)
        return _SuccessfulProfileClient()

    monkeypatch.setattr(
        "chulk.model_profiles.service.create_llm_client",
        create_client,
    )
    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="rotating",
    )
    chain = service.create_chain(service.validator.config, runtime)

    assert chain.complete_response([]).content == "ok"
    environment["ROTATING_KEY"] = "second"
    assert chain.complete_response([]).content == "ok"
    assert observed_keys == ["first", "second"]


def test_local_profile_uses_the_configured_deployment_context_window(
    tmp_path: Path,
) -> None:
    config, agent_profile, _store, service = _runtime(tmp_path)
    service.create(
        ModelProfile(
            id="local-runtime",
            provider="local",
            model="qwen/qwen3.5-35b-a3b",
        )
    )
    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="local-runtime",
    )

    chain = service.create_chain(config, runtime)

    assert chain.model_capabilities is not None
    assert (
        chain.model_capabilities.context_window_tokens
        == config.local_context_window_tokens
    )


class _SecretEchoingClient(LLMClient):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        raise LLMError(
            f"provider rejected key {self.secret}",
            code="authentication_error",
            retryable=False,
            fallback_eligible=False,
        )


def test_resolved_secret_is_redacted_from_provider_errors_and_attempts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "secret-value-that-must-not-leak"
    _config, agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"PRIVATE_KEY": secret},
    )
    service.create(_openai_profile("private", "PRIVATE_KEY"))
    monkeypatch.setattr(
        "chulk.model_profiles.service.create_llm_client",
        lambda **_kwargs: _SecretEchoingClient(secret),
    )
    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="private",
    )
    chain = service.create_chain(service.validator.config, runtime)

    with pytest.raises(LLMError) as raised:
        chain.complete_response([])

    assert secret not in str(raised.value)
    assert secret not in json.dumps(
        [attempt.to_dict() for attempt in chain.last_attempts]
    )
    assert "[REDACTED]" in str(raised.value)


class _FinalClient(LLMClient):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps({"type": "final_answer", "content": "done"})


def test_selection_reason_is_exposed_in_public_result_and_trace() -> None:
    events: list[tuple[str, dict]] = []
    selection = {
        "requested_profile_id": "primary",
        "selected_profile_id": "fallback",
        "fallback_path": ["primary", "fallback"],
        "reason": "primary unavailable",
        "skipped": [],
    }
    agent = Agent(
        _FinalClient(),
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
        runtime_metadata={"model_selection": selection},
    )

    agent.run_turn("hello")
    result = run_result_from_runtime(agent)

    assert result.extension_metadata["model_selection"]["reason"] == (
        "primary unavailable"
    )
    selected_event = next(
        payload
        for event_type, payload in events
        if event_type == TraceEvent.MODEL_PROFILE_SELECTED
    )
    assert selected_event["selected_profile_id"] == "fallback"


class _SuccessfulActionProfileClient(_SuccessfulProfileClient):
    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=json.dumps(
                {"type": "final_answer", "content": "fallback response"}
            ),
            provider=self.provider,
            model=self.model,
        )


def test_runtime_fallback_outcome_updates_public_result_and_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config, agent_profile, _store, service = _runtime(
        tmp_path,
        environ={"PRIMARY_KEY": "bad", "SECONDARY_KEY": "good"},
    )
    service.create(_openai_profile("secondary", "SECONDARY_KEY"))
    service.create(_openai_profile("primary", "PRIMARY_KEY", fallbacks=("secondary",)))
    primary = _FailingProfileClient()
    secondary = _SuccessfulActionProfileClient()
    monkeypatch.setattr(
        "chulk.model_profiles.service.create_llm_client",
        lambda **kwargs: (
            primary if kwargs["connection"].api_key == "bad" else secondary
        ),
    )
    runtime = service.resolve_for_agent(
        agent_profile,
        requested_profile_id="primary",
    )
    chain = service.create_chain(service.validator.config, runtime)
    events: list[tuple[str, dict]] = []
    agent = Agent(
        chain,
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
        runtime_metadata={"model_selection": runtime.selection.to_dict()},
    )

    assert agent.run_turn("hello") == "fallback response"
    result = run_result_from_runtime(agent)

    selection = result.extension_metadata["model_selection"]
    assert selection["requested_profile_id"] == "primary"
    assert selection["selected_profile_id"] == "secondary"
    assert selection["reason"].startswith("runtime fallback selected")
    attempts = result.extension_metadata["model_attempts"]
    assert [attempt["model_profile_id"] for attempt in attempts] == [
        "primary",
        "secondary",
    ]
    runtime_event = next(
        payload
        for event_type, payload in events
        if event_type == TraceEvent.MODEL_PROFILE_SELECTED
        and payload.get("phase") == "runtime_fallback"
    )
    assert runtime_event["selected_profile_id"] == "secondary"


def test_billing_errors_have_a_stable_operational_category() -> None:
    error_type = type("BillingError", (Exception,), {})
    error = error_type("quota exhausted")
    error.status_code = 402  # type: ignore[attr-defined]

    classification = classify_provider_exception(error)

    assert classification.code == "billing_error"
    assert classification.retryable is False
    assert classification.fallback_eligible is False


class _DiscoveryResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit: int) -> bytes:
        return b'{"data":[{"id":"zeta"},{"id":"alpha"},{"id":"alpha"}]}'


def test_endpoint_discovery_is_explicit_bounded_and_provider_limited() -> None:
    requests = []
    profile = ModelProfile(
        id="local-models",
        provider="local",
        model="qwen/qwen3.5-35b-a3b",
        endpoint_ref=EndpointRef("LOCAL_URL", source="env"),
    )

    models = discover_endpoint_models(
        profile,
        connection=type(
            "Connection",
            (),
            {
                "api_key": None,
                "base_url": "http://127.0.0.1:11434/v1",
            },
        )(),
        opener=lambda request, timeout: (
            requests.append((request, timeout)) or _DiscoveryResponse()
        ),
    )

    assert models == ("alpha", "zeta")
    assert requests[0][0].full_url == "http://127.0.0.1:11434/v1/models"
    assert requests[0][1] == 5.0

    with pytest.raises(ValueError, match="supported only"):
        discover_endpoint_models(
            _openai_profile("hosted", "MODEL_KEY"),
            type(
                "Connection",
                (),
                {
                    "api_key": None,
                    "base_url": "https://api.example/v1",
                },
            )(),
        )
