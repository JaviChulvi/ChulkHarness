"""Durable named model profiles, diagnostics, health, and selection."""

from chulk.model_profiles.client import RefreshingLLMClient, RequestClientLease
from chulk.model_profiles.diagnostics import (
    EndpointProvider,
    ModelProfileValidator,
    ProviderProbe,
    SecretProvider,
    bounded_provider_probe,
    diagnostic_category_from_error_code,
)
from chulk.model_profiles.discovery import DiscoveryOpener, discover_endpoint_models
from chulk.model_profiles.models import (
    DEFAULT_MODEL_PROFILE_ID,
    DiagnosticCategory,
    EndpointRef,
    EndpointSource,
    ModelCapabilityRequirements,
    ModelDiagnostic,
    ModelProfile,
    ModelSelectionResult,
    ModelSelectionSkip,
    ProviderHealth,
    ProviderHealthStatus,
)
from chulk.model_profiles.service import (
    ModelProfileService,
    ResolvedModelCandidate,
    ResolvedModelRuntime,
)
from chulk.model_profiles.store import (
    ModelProfileAlreadyExistsError,
    ModelProfileNotFoundError,
    ModelProfileStore,
    health_key,
)


__all__ = [
    "DEFAULT_MODEL_PROFILE_ID",
    "DiagnosticCategory",
    "DiscoveryOpener",
    "EndpointProvider",
    "EndpointRef",
    "EndpointSource",
    "ModelCapabilityRequirements",
    "ModelDiagnostic",
    "ModelProfile",
    "ModelProfileAlreadyExistsError",
    "ModelProfileNotFoundError",
    "ModelProfileService",
    "ModelProfileStore",
    "ModelProfileValidator",
    "ModelSelectionResult",
    "ModelSelectionSkip",
    "ProviderHealth",
    "ProviderHealthStatus",
    "ProviderProbe",
    "RefreshingLLMClient",
    "RequestClientLease",
    "ResolvedModelCandidate",
    "ResolvedModelRuntime",
    "SecretProvider",
    "bounded_provider_probe",
    "diagnostic_category_from_error_code",
    "discover_endpoint_models",
    "health_key",
]
