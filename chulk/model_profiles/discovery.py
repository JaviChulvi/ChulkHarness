"""Explicit model discovery for authoritative OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from chulk.llm import LLMProviderConnection
from chulk.model_profiles.models import ModelProfile


DiscoveryOpener = Callable[..., Any]
_MAX_DISCOVERY_BYTES = 2_000_000


def discover_endpoint_models(
    profile: ModelProfile,
    connection: LLMProviderConnection,
    *,
    timeout_seconds: float = 5.0,
    opener: DiscoveryOpener = urlopen,
) -> tuple[str, ...]:
    """Fetch a bounded `/models` response when that endpoint is authoritative."""
    if profile.provider not in {"local", "openai-compatible"}:
        raise ValueError(
            "model discovery is supported only for local and openai-compatible profiles"
        )
    if connection.base_url is None:
        raise ValueError("model discovery requires an available endpoint reference")
    endpoint = urljoin(connection.base_url.rstrip("/") + "/", "models")
    headers = {"Accept": "application/json"}
    if connection.api_key:
        headers["Authorization"] = f"Bearer {connection.api_key}"
    request = Request(endpoint, headers=headers, method="GET")
    with opener(request, timeout=timeout_seconds) as response:
        payload = response.read(_MAX_DISCOVERY_BYTES + 1)
    if len(payload) > _MAX_DISCOVERY_BYTES:
        raise ValueError("model discovery response exceeds the safe size limit")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model discovery returned invalid JSON") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
        raise ValueError("model discovery returned an unsupported response shape")
    model_ids = {
        item["id"].strip()
        for item in decoded["data"]
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    }
    return tuple(sorted(model_ids))


__all__ = ["DiscoveryOpener", "discover_endpoint_models"]
