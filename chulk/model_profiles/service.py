"""Named model selection, fallback resolution, and runtime construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from chulk.config import Config
from chulk.llm import (
    BindableLLM,
    FallbackChain,
    LLMClient,
    LLMError,
    LLMModelCapabilities,
    create_llm_client,
)
from chulk.llm.capabilities import resolve_runtime_model_capabilities
from chulk.llm.public import ProviderAttempt
from chulk.llm.base import LLMErrorCode
from chulk.model_profiles.client import RefreshingLLMClient, RequestClientLease
from chulk.model_profiles.diagnostics import (
    ModelProfileValidator,
    diagnostic_category_from_error_code,
)
from chulk.model_profiles.models import (
    DiagnosticCategory,
    ModelDiagnostic,
    ModelProfile,
    ModelSelectionResult,
    ModelSelectionSkip,
    ProviderHealthStatus,
)
from chulk.model_profiles.store import ModelProfileStore
from chulk.profiles import AgentProfile


@dataclass(frozen=True, slots=True)
class ResolvedModelCandidate:
    profile: ModelProfile


@dataclass(frozen=True, slots=True)
class ResolvedModelRuntime:
    selection: ModelSelectionResult
    candidates: tuple[ResolvedModelCandidate, ...]
    legacy_compatibility: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection.to_dict(),
            "candidates": [
                candidate.profile.to_dict() for candidate in self.candidates
            ],
            "legacy_compatibility": self.legacy_compatibility,
        }


class ModelProfileService:
    """Coordinate model profiles without persisting or exposing secret values."""

    def __init__(
        self,
        store: ModelProfileStore,
        validator: ModelProfileValidator,
    ) -> None:
        self.store = store
        self.validator = validator

    def create(self, profile: ModelProfile) -> ModelProfile:
        self.validator.validate_definition(profile)
        for fallback_id in profile.fallback_profile_ids:
            self.store.get(fallback_id)
        self._flatten(profile, extra={profile.id: profile})
        return self.store.create(profile)

    def allowed_profile_ids(self, agent_profile: AgentProfile) -> set[str] | None:
        if agent_profile.implicit:
            return None
        values = {agent_profile.model_profile_id}
        values.update(
            value
            for value in agent_profile.auxiliary_models.to_dict().values()
            if value is not None
        )
        return values

    def use_for_agent(
        self,
        agent_profile: AgentProfile,
        model_profile_id: str,
        *,
        channel: str | None = None,
    ) -> ModelProfile:
        selected = self.store.get(model_profile_id)
        allowed = self.allowed_profile_ids(agent_profile)
        if allowed is not None and selected.id not in allowed:
            raise ValueError(
                f"model profile {selected.id!r} is not allowed by agent profile "
                f"{agent_profile.id!r}"
            )
        self.store.use_for_agent(agent_profile.id, selected.id, channel=channel)
        return selected

    def selected_profile_id(
        self,
        agent_profile: AgentProfile,
        *,
        channel: str | None = None,
    ) -> str:
        return self.store.selected_for_agent(
            agent_profile.id,
            default=agent_profile.model_profile_id,
            channel=channel,
        )

    def resolve_for_agent(
        self,
        agent_profile: AgentProfile,
        *,
        requested_profile_id: str | None = None,
        channel: str | None = None,
    ) -> ResolvedModelRuntime:
        requested_id = requested_profile_id or self.selected_profile_id(
            agent_profile,
            channel=channel,
        )
        allowed = self.allowed_profile_ids(agent_profile)
        if allowed is not None and requested_id not in allowed:
            raise ValueError(
                f"model profile {requested_id!r} is not allowed by agent profile "
                f"{agent_profile.id!r}"
            )
        requested = self.store.get(requested_id)
        path = self._flatten(requested)
        if requested.implicit and requested.id == "default":
            return self._legacy_runtime(requested, path)
        candidates: list[ResolvedModelCandidate] = []
        skipped: list[ModelSelectionSkip] = []
        now = self.store.clock()
        for profile in path:
            health = self.store.health(profile)
            if (
                health.status is ProviderHealthStatus.COOLDOWN
                and health.cooldown_until is not None
                and health.cooldown_until > now
            ):
                skipped.append(
                    ModelSelectionSkip(
                        profile.id,
                        f"provider is cooling down until {health.cooldown_until.isoformat()}",
                        DiagnosticCategory.COOLDOWN,
                    )
                )
                continue
            diagnostic, connection = self.validator.diagnose(profile)
            if not diagnostic.ok or connection is None:
                skipped.append(
                    ModelSelectionSkip(
                        profile.id,
                        diagnostic.message,
                        diagnostic.category,
                    )
                )
                continue
            candidates.append(ResolvedModelCandidate(profile))
        if not candidates:
            detail = "; ".join(f"{item.profile_id}: {item.reason}" for item in skipped)
            raise ValueError(
                f"no usable model profile remains in the fallback path: {detail}"
            )
        selected = candidates[0].profile
        reason = (
            "requested model profile selected"
            if selected.id == requested.id
            else f"selected fallback after {len(skipped)} unavailable profile(s)"
        )
        return ResolvedModelRuntime(
            selection=ModelSelectionResult(
                requested.id,
                selected.id,
                tuple(profile.id for profile in path),
                reason,
                tuple(skipped),
            ),
            candidates=tuple(candidates),
        )

    def diagnose(
        self,
        profile_id: str,
        *,
        probe: bool = False,
        probe_callback=None,
    ) -> ModelDiagnostic:
        profile = self.store.get(profile_id)
        diagnostic, _connection = self.validator.diagnose(
            profile,
            probe=probe,
            probe_callback=probe_callback,
        )
        return diagnostic

    def create_chain(
        self,
        config: Config,
        runtime: ResolvedModelRuntime,
    ) -> FallbackChain:
        if runtime.legacy_compatibility:
            raise ValueError(
                "legacy runtime must use the existing CLI compatibility builder"
            )
        clients: list[LLMClient | BindableLLM] = []
        profiles_by_id: dict[str, ModelProfile] = {}
        for candidate in runtime.candidates:
            profile = candidate.profile
            base = resolve_runtime_model_capabilities(
                profile.provider,
                profile.model,
                local_context_window_tokens=(
                    profile.context_window_tokens or config.local_context_window_tokens
                ),
            )
            model_capabilities = base
            if (
                profile.context_window_tokens is not None
                or profile.response_reserve_tokens is not None
                or profile.max_output_tokens is not None
            ):
                context = profile.context_window_tokens or base.context_window_tokens
                model_capabilities = LLMModelCapabilities(
                    provider=profile.provider,
                    model=profile.model,
                    context_window_tokens=context,
                    default_response_reserve_tokens=(
                        profile.response_reserve_tokens
                        or base.default_response_reserve_tokens
                    ),
                    max_input_tokens=(
                        min(base.max_input_tokens, context)
                        if base.max_input_tokens is not None
                        else None
                    ),
                    max_output_tokens=(
                        profile.max_output_tokens
                        if profile.max_output_tokens is not None
                        else base.max_output_tokens
                    ),
                )

            def create_request_client(
                profile: ModelProfile = profile,
            ) -> RequestClientLease:
                diagnostic, connection = self.validator.diagnose(profile)
                if not diagnostic.ok or connection is None:
                    raise LLMError(
                        diagnostic.message,
                        provider=profile.provider,
                        model=profile.model,
                        code=_error_code_for_diagnostic(diagnostic.category),
                        retryable=False,
                        fallback_eligible=False,
                    )
                return RequestClientLease(
                    create_llm_client(
                        provider=profile.provider,
                        model=profile.model,
                        connection=connection,
                        local_context_window_tokens=(
                            profile.context_window_tokens
                            or config.local_context_window_tokens
                        ),
                        timeout_seconds=config.llm_timeout_seconds,
                        max_retries=config.llm_max_retries,
                    ),
                    sensitive_values=tuple(
                        value
                        for value in (connection.api_key, connection.base_url)
                        if value
                    ),
                )

            client = RefreshingLLMClient(
                provider=profile.provider,
                model=profile.model,
                model_profile_id=profile.id,
                model_capabilities=model_capabilities,
                client_factory=create_request_client,
            )
            clients.append(client)
            profiles_by_id[profile.id] = profile

        def record_attempt(attempt: ProviderAttempt) -> None:
            profile_id = getattr(attempt, "model_profile_id", None)
            profile = (
                profiles_by_id.get(profile_id) if isinstance(profile_id, str) else None
            )
            if profile is None:
                return
            if attempt.success:
                self.store.record_success(profile)
                return
            self.store.record_failure(
                profile,
                diagnostic_category_from_error_code(attempt.error_code),
            )

        def provider_available(client: LLMClient) -> bool:
            profile_id = getattr(client, "model_profile_id", None)
            profile = (
                profiles_by_id.get(profile_id) if isinstance(profile_id, str) else None
            )
            if profile is None:
                return True
            health = self.store.health(profile)
            return not (
                health.status is ProviderHealthStatus.COOLDOWN
                and health.cooldown_until is not None
                and health.cooldown_until > self.store.clock()
            )

        chain = FallbackChain(
            providers=clients,
            attempt_callback=record_attempt,
            provider_available=provider_available,
            fallback_error_codes=frozenset(
                {"authentication_error", "billing_error", "model_not_found"}
            ),
        )
        chain.selection_result = runtime.selection
        chain.model_profile_id = runtime.selection.selected_profile_id
        return chain

    def _flatten(
        self,
        profile: ModelProfile,
        *,
        extra: dict[str, ModelProfile] | None = None,
    ) -> tuple[ModelProfile, ...]:
        result: list[ModelProfile] = []
        visited: set[str] = set()

        def visit(current: ModelProfile, stack: tuple[str, ...]) -> None:
            if current.id in stack:
                cycle = " -> ".join((*stack, current.id))
                raise ValueError(f"model fallback cycle detected: {cycle}")
            if current.id in visited:
                return
            visited.add(current.id)
            result.append(current)
            for fallback_id in current.fallback_profile_ids:
                fallback = (
                    extra[fallback_id]
                    if extra is not None and fallback_id in extra
                    else self.store.get(fallback_id)
                )
                visit(fallback, (*stack, current.id))

        visit(profile, ())
        return tuple(result)

    def _legacy_runtime(
        self,
        requested: ModelProfile,
        path: tuple[ModelProfile, ...],
    ) -> ResolvedModelRuntime:
        return ResolvedModelRuntime(
            selection=ModelSelectionResult(
                requested.id,
                requested.id,
                tuple(profile.id for profile in path),
                "implicit environment configuration preserved for compatibility",
            ),
            candidates=(),
            legacy_compatibility=True,
        )


def _error_code_for_diagnostic(
    category: DiagnosticCategory,
) -> LLMErrorCode:
    code = {
        DiagnosticCategory.MISSING_CREDENTIAL: "authentication_error",
        DiagnosticCategory.AUTHENTICATION: "authentication_error",
        DiagnosticCategory.BILLING: "billing_error",
        DiagnosticCategory.INVALID_MODEL: "model_not_found",
        DiagnosticCategory.UNAVAILABLE_ENDPOINT: "connection_error",
        DiagnosticCategory.TIMEOUT: "timeout",
        DiagnosticCategory.RATE_LIMIT: "rate_limit",
        DiagnosticCategory.UNSUPPORTED_CAPABILITY: "unsupported_feature",
        DiagnosticCategory.UNSUPPORTED_SCHEMA: "action_shape_error",
        DiagnosticCategory.CONFIGURATION: "configuration_error",
    }.get(category, "unknown")
    return cast(LLMErrorCode, code)


__all__ = [
    "ModelProfileService",
    "ResolvedModelCandidate",
    "ResolvedModelRuntime",
]
