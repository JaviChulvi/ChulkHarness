"""Static and explicitly probed diagnostics for named model profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Protocol
from urllib.parse import urlsplit

from chulk.config import Config
from chulk.llm import (
    LLMError,
    LLMProviderConnection,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
    resolve_model_capabilities,
    supported_llm_providers,
)
from chulk.llm.lifecycle import close_resources
from chulk.model_profiles.models import (
    DiagnosticCategory,
    EndpointRef,
    ModelDiagnostic,
    ModelProfile,
)
from chulk.profiles import CredentialRef


class SecretProvider(Protocol):
    """Host-owned secret lookup for non-environment references."""

    def resolve(self, reference: CredentialRef) -> str | None: ...


class EndpointProvider(Protocol):
    """Host-owned endpoint lookup for non-environment references."""

    def resolve(self, reference: EndpointRef) -> str | None: ...


class ProviderProbe(Protocol):
    """Explicit bounded live diagnostic callback supplied by the host."""

    def __call__(
        self,
        profile: ModelProfile,
        connection: LLMProviderConnection,
        timeout_seconds: float,
    ) -> ModelDiagnostic: ...


@dataclass(frozen=True, slots=True)
class ModelProfileValidator:
    """Resolve refs and validate catalog/capability constraints without network."""

    config: Config
    environ: Mapping[str, str] | None = None
    secret_provider: SecretProvider | None = None
    endpoint_provider: EndpointProvider | None = None

    def validate_definition(self, profile: ModelProfile) -> None:
        if profile.provider not in supported_llm_providers():
            raise ValueError(f"unsupported model provider: {profile.provider}")
        if profile.provider != "local" and profile.credential_ref is None:
            raise ValueError(
                f"model profile {profile.id!r} requires a typed credential reference"
            )
        if (
            profile.endpoint_ref is not None
            and profile.endpoint_ref.source == "config"
            and profile.endpoint_ref.name != profile.provider
        ):
            raise ValueError(
                "config endpoint references must name the model profile provider"
            )
        capabilities = resolve_model_capabilities(profile.provider, profile.model)
        provider_features = provider_capabilities(profile.provider)
        requirements = profile.required_capabilities
        required_features = (
            (
                "structured_output",
                requirements.structured_output,
                provider_features.supports_structured_output,
            ),
            ("json_mode", requirements.json_mode, provider_features.supports_json_mode),
            ("streaming", requirements.streaming, provider_features.supports_streaming),
            (
                "native_tool_calling",
                requirements.native_tool_calling,
                provider_features.supports_native_tool_calling,
            ),
            (
                "hosted_mcp_tools",
                requirements.hosted_mcp_tools,
                provider_features.supports_hosted_mcp_tools,
            ),
        )
        missing = [
            name
            for name, required, available in required_features
            if required and not available
        ]
        if missing:
            raise ValueError(
                f"provider {profile.provider!r} does not support required capability: "
                + ", ".join(missing)
            )
        if (
            profile.context_window_tokens is not None
            and profile.context_window_tokens > capabilities.context_window_tokens
        ):
            raise ValueError("profile context_window_tokens exceeds catalog limits")
        effective_context = (
            profile.context_window_tokens or capabilities.context_window_tokens
        )
        effective_response_reserve = (
            profile.response_reserve_tokens
            or capabilities.default_response_reserve_tokens
        )
        if effective_response_reserve >= effective_context:
            raise ValueError("profile response_reserve_tokens exceeds context limits")
        if (
            profile.max_output_tokens is not None
            and capabilities.max_output_tokens is not None
            and profile.max_output_tokens > capabilities.max_output_tokens
        ):
            raise ValueError("profile max_output_tokens exceeds catalog limits")
        if (
            profile.max_output_tokens is not None
            and profile.max_output_tokens >= effective_context
        ):
            raise ValueError("profile max_output_tokens exceeds context limits")

    def diagnose(
        self,
        profile: ModelProfile,
        *,
        probe: bool = False,
        probe_callback: ProviderProbe | None = None,
        timeout_seconds: float = 5.0,
    ) -> tuple[ModelDiagnostic, LLMProviderConnection | None]:
        try:
            self.validate_definition(profile)
        except ValueError as exc:
            message = str(exc)
            category = (
                DiagnosticCategory.UNSUPPORTED_CAPABILITY
                if "capability" in message
                else DiagnosticCategory.INVALID_MODEL
                if (
                    "token capability metadata" in message
                    or "catalog limits" in message
                )
                else DiagnosticCategory.CONFIGURATION
            )
            return (
                ModelDiagnostic(
                    profile.id,
                    category,
                    False,
                    message,
                    profile.provider,
                    profile.model,
                ),
                None,
            )

        connection = self.resolve_connection(profile)
        if profile.credential_ref is not None and not connection.api_key:
            return (
                ModelDiagnostic(
                    profile.id,
                    DiagnosticCategory.MISSING_CREDENTIAL,
                    False,
                    f"credential reference {profile.credential_ref.uri} is not available",
                    profile.provider,
                    profile.model,
                ),
                None,
            )
        if _requires_endpoint(profile.provider) and not connection.base_url:
            reference = profile.endpoint_ref.uri if profile.endpoint_ref else "none"
            return (
                ModelDiagnostic(
                    profile.id,
                    DiagnosticCategory.UNAVAILABLE_ENDPOINT,
                    False,
                    f"endpoint reference {reference} is not available",
                    profile.provider,
                    profile.model,
                ),
                None,
            )
        if connection.base_url is not None and not _valid_endpoint(connection.base_url):
            return (
                ModelDiagnostic(
                    profile.id,
                    DiagnosticCategory.UNAVAILABLE_ENDPOINT,
                    False,
                    "resolved endpoint must be an HTTP(S) URL without embedded credentials",
                    profile.provider,
                    profile.model,
                ),
                None,
            )
        if not probe:
            return (
                ModelDiagnostic(
                    profile.id,
                    DiagnosticCategory.READY,
                    True,
                    "static model-profile checks passed; no network probe was performed",
                    profile.provider,
                    profile.model,
                    details={"network_used": False},
                ),
                connection,
            )
        if probe_callback is None:
            return (
                ModelDiagnostic(
                    profile.id,
                    DiagnosticCategory.CONFIGURATION,
                    False,
                    "a host probe callback is required for --probe",
                    profile.provider,
                    profile.model,
                    probed=False,
                    details={"network_used": False, "may_consume_quota": True},
                ),
                connection,
            )
        diagnostic = probe_callback(profile, connection, timeout_seconds)
        return diagnostic, connection

    def resolve_connection(self, profile: ModelProfile) -> LLMProviderConnection:
        if profile.implicit:
            return provider_connection_from_config(profile.provider, self.config)
        api_key = self._resolve_credential(profile.credential_ref)
        base_url = self._resolve_endpoint(profile)
        default_connection = provider_connection_from_config(
            profile.provider, self.config
        )
        return LLMProviderConnection(
            api_key=api_key,
            base_url=base_url if base_url is not None else default_connection.base_url,
        )

    def _resolve_credential(self, reference: CredentialRef | None) -> str | None:
        if reference is None:
            return None
        if reference.source == "environment":
            environment = os.environ if self.environ is None else self.environ
            value = environment.get(reference.name)
            return value if value is not None and value.strip() else None
        if self.secret_provider is None:
            return None
        value = self.secret_provider.resolve(reference)
        return value if value is not None and value.strip() else None

    def _resolve_endpoint(self, profile: ModelProfile) -> str | None:
        reference = profile.endpoint_ref
        if reference is None:
            return None
        if reference.source == "config":
            return provider_connection_from_config(
                reference.name,
                self.config,
            ).base_url
        if reference.source == "env":
            environment = os.environ if self.environ is None else self.environ
            value = environment.get(reference.name)
            return value.strip() if value is not None and value.strip() else None
        if self.endpoint_provider is None:
            return None
        value = self.endpoint_provider.resolve(reference)
        return value.strip() if value is not None and value.strip() else None


def diagnostic_category_from_error_code(error_code: str | None) -> DiagnosticCategory:
    return {
        "authentication_error": DiagnosticCategory.AUTHENTICATION,
        "billing_error": DiagnosticCategory.BILLING,
        "rate_limit": DiagnosticCategory.RATE_LIMIT,
        "timeout": DiagnosticCategory.TIMEOUT,
        "connection_error": DiagnosticCategory.UNAVAILABLE_ENDPOINT,
        "circuit_open": DiagnosticCategory.COOLDOWN,
        "model_not_found": DiagnosticCategory.INVALID_MODEL,
        "unsupported_feature": DiagnosticCategory.UNSUPPORTED_CAPABILITY,
        "action_shape_error": DiagnosticCategory.UNSUPPORTED_SCHEMA,
        "configuration_error": DiagnosticCategory.CONFIGURATION,
    }.get(error_code or "", DiagnosticCategory.UNKNOWN)


def bounded_provider_probe(
    config: Config,
    profile: ModelProfile,
    connection: LLMProviderConnection,
    timeout_seconds: float,
) -> ModelDiagnostic:
    """Perform the explicit minimal live request used by CLI diagnostics."""
    client = None
    try:
        client = create_llm_client(
            provider=profile.provider,
            model=profile.model,
            connection=connection,
            local_context_window_tokens=(
                profile.context_window_tokens or config.local_context_window_tokens
            ),
            timeout_seconds=min(timeout_seconds, config.llm_timeout_seconds),
            max_retries=0,
        )
        client.complete_response(
            [{"role": "user", "content": "Reply OK."}],
            max_output_tokens=1,
        )
    except LLMError as exc:
        category = diagnostic_category_from_error_code(exc.code)
        return ModelDiagnostic(
            profile.id,
            category,
            False,
            f"live provider probe failed ({category.value})",
            profile.provider,
            profile.model,
            probed=True,
            details={"network_used": True, "may_consume_quota": True},
        )
    except Exception:
        return ModelDiagnostic(
            profile.id,
            DiagnosticCategory.UNKNOWN,
            False,
            "live provider probe failed (unknown)",
            profile.provider,
            profile.model,
            probed=True,
            details={"network_used": True, "may_consume_quota": True},
        )
    finally:
        if client is not None:
            close_resources((client,))
    return ModelDiagnostic(
        profile.id,
        DiagnosticCategory.READY,
        True,
        "live provider probe succeeded",
        profile.provider,
        profile.model,
        probed=True,
        details={"network_used": True, "may_consume_quota": True},
    )


def _requires_endpoint(provider: str) -> bool:
    return provider in {"openai-compatible", "bedrock"}


def _valid_endpoint(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


__all__ = [
    "EndpointProvider",
    "ModelProfileValidator",
    "ProviderProbe",
    "SecretProvider",
    "bounded_provider_probe",
    "diagnostic_category_from_error_code",
]
