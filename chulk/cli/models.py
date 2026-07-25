"""CLI operations for durable model profiles."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
import json

from chulk.model_profiles import (
    EndpointRef,
    ModelCapabilityRequirements,
    ModelProfile,
    ModelProfileAlreadyExistsError,
    ModelProfileNotFoundError,
    ModelProfileService,
    bounded_provider_probe,
    discover_endpoint_models,
)
from chulk.profiles import AgentProfile, CredentialRef


def run_model_command(
    command: str,
    *,
    service: ModelProfileService,
    agent_profile: AgentProfile,
    profile_id: str | None,
    provider: str | None,
    model: str | None,
    credential_ref: str | None,
    endpoint_ref: str | None,
    fallback_profile_ids: tuple[str, ...],
    required_capabilities: ModelCapabilityRequirements,
    context_window_tokens: int | None,
    response_reserve_tokens: int | None,
    max_output_tokens: int | None,
    max_cost_per_turn: str | None,
    channel: str | None,
    probe: bool,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """Execute one deterministic model-profile control operation."""
    try:
        if command == "create":
            if profile_id is None or provider is None or model is None:
                raise ValueError("model create requires an id, provider, and model")
            created = service.create(
                ModelProfile(
                    id=profile_id,
                    provider=provider,
                    model=model,
                    credential_ref=(
                        CredentialRef.parse(credential_ref)
                        if credential_ref is not None
                        else None
                    ),
                    endpoint_ref=(
                        EndpointRef.parse(endpoint_ref)
                        if endpoint_ref is not None
                        else None
                    ),
                    fallback_profile_ids=fallback_profile_ids,
                    required_capabilities=required_capabilities,
                    context_window_tokens=context_window_tokens,
                    response_reserve_tokens=response_reserve_tokens,
                    max_output_tokens=max_output_tokens,
                    max_cost_per_turn=(
                        Decimal(max_cost_per_turn)
                        if max_cost_per_turn is not None
                        else None
                    ),
                )
            )
            return _emit(
                {"ok": True, "status": "created", "profile": created.to_dict()},
                json_output=json_output,
                text=f"Created model profile {created.id}",
                output_func=output_func,
            )
        if command == "list":
            selected_id = service.selected_profile_id(
                agent_profile,
                channel=channel,
            )
            profiles = [
                {
                    **profile.to_dict(),
                    "selected": profile.id == selected_id,
                    "health": service.store.health(profile).to_dict(),
                }
                for profile in service.store.list()
            ]
            return _emit(
                {"ok": True, "profiles": profiles, "channel": channel},
                json_output=json_output,
                text=_format_list(profiles),
                output_func=output_func,
            )
        if command == "use":
            if profile_id is None:
                raise ValueError("model use requires an id")
            selected = service.use_for_agent(
                agent_profile,
                profile_id,
                channel=channel,
            )
            return _emit(
                {
                    "ok": True,
                    "status": "selected",
                    "profile_id": selected.id,
                    "agent_profile_id": agent_profile.id,
                    "channel": channel,
                },
                json_output=json_output,
                text=f"Selected model profile {selected.id}",
                output_func=output_func,
            )
        if command == "inspect":
            if profile_id is None:
                raise ValueError("model inspect requires an id")
            profile = service.store.get(profile_id)
            diagnostic = service.diagnose(
                profile.id,
                probe=probe,
                probe_callback=(
                    lambda item, connection, timeout: bounded_provider_probe(
                        service.validator.config,
                        item,
                        connection,
                        timeout,
                    )
                )
                if probe
                else None,
            )
            payload = {
                "ok": diagnostic.ok,
                "profile": profile.to_dict(),
                "diagnostic": diagnostic.to_dict(),
                "health": service.store.health(profile).to_dict(),
                "probe_may_consume_quota": probe,
            }
            _emit(
                payload,
                json_output=json_output,
                text=_format_inspect(payload),
                output_func=output_func,
            )
            return 0 if diagnostic.ok else 2
        if command == "health":
            health_profiles = (
                (service.store.get(profile_id),)
                if profile_id is not None
                else service.store.list()
            )
            health_items = [
                service.store.health(profile).to_dict() for profile in health_profiles
            ]
            return _emit(
                {"ok": True, "health": health_items},
                json_output=json_output,
                text=_format_health(health_items),
                output_func=output_func,
            )
        if command == "reset":
            if profile_id is None:
                raise ValueError("model reset requires an id")
            reset_health = service.store.reset_health(service.store.get(profile_id))
            return _emit(
                {
                    "ok": True,
                    "status": "reset",
                    "health": reset_health.to_dict(),
                },
                json_output=json_output,
                text=f"Reset model health for {profile_id}",
                output_func=output_func,
            )
        if command == "discover":
            if profile_id is None:
                raise ValueError("model discover requires an id")
            profile = service.store.get(profile_id)
            diagnostic, connection = service.validator.diagnose(profile)
            if not diagnostic.ok or connection is None:
                raise ValueError(diagnostic.message)
            models = discover_endpoint_models(profile, connection)
            return _emit(
                {
                    "ok": True,
                    "profile_id": profile.id,
                    "models": list(models),
                    "network_used": True,
                },
                json_output=json_output,
                text=_format_discovery(profile.id, models),
                output_func=output_func,
            )
        raise ValueError(f"unknown model command: {command}")
    except (
        ArithmeticError,
        OSError,
        ModelProfileAlreadyExistsError,
        ModelProfileNotFoundError,
        ValueError,
    ) as exc:
        payload = {"ok": False, "status": "model_profile_error", "error": str(exc)}
        if json_output:
            output_func(_json(payload))
        else:
            error_func(f"model profile error: {exc}")
        return 2


def _emit(
    payload: object,
    *,
    json_output: bool,
    text: str,
    output_func: Callable[[str], None],
) -> int:
    output_func(_json(payload) if json_output else text)
    return 0


def _format_list(profiles: list[dict[str, object]]) -> str:
    lines = ["Model profiles:"]
    for profile in profiles:
        marker = "*" if profile["selected"] else " "
        lines.append(
            f"  {marker} {profile['id']}  {profile['provider']}:{profile['model']}"
        )
    return "\n".join(lines)


def _format_inspect(payload: dict[str, object]) -> str:
    profile = payload["profile"]
    diagnostic = payload["diagnostic"]
    health = payload["health"]
    assert isinstance(profile, dict)
    assert isinstance(diagnostic, dict)
    assert isinstance(health, dict)
    lines = [f"Model profile {profile['id']}:"]
    for key, value in profile.items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  diagnostic: {diagnostic['category']} ({diagnostic['message']})")
    lines.append(f"  health: {health['status']}")
    return "\n".join(lines)


def _format_health(health: list[dict[str, object]]) -> str:
    lines = ["Model health:"]
    for item in health:
        lines.append(
            f"  {item['model_profile_id']}: {item['status']} "
            f"({item['consecutive_failures']} consecutive failures)"
        )
    return "\n".join(lines)


def _format_discovery(profile_id: str, models: tuple[str, ...]) -> str:
    lines = [f"Models discovered for {profile_id}:"]
    lines.extend(f"  {model}" for model in models)
    return "\n".join(lines)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


__all__ = ["run_model_command"]
