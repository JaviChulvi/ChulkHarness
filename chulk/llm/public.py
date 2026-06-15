"""Public LLM provider specs and fallback chains."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Literal, Protocol

from chulk.llm.base import LLMClient, LLMError
from chulk.llm.factory import create_llm_client

if TYPE_CHECKING:
    from chulk.config import Config


FallbackStrategy = Literal["first_success", "round_robin", "lowest_latency"]


class BindableLLM(Protocol):
    """Provider spec that can create an LLM client from runtime config."""

    provider: str
    model: str

    def bind_config(self, config: "Config") -> LLMClient:
        """Return a configured LLM client."""


@dataclass(frozen=True)
class OpenAIProvider:
    """OpenAI provider spec for the public API."""

    model: str
    api_key: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "openai"

    def bind_config(self, config: "Config") -> LLMClient:
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            openai_api_key=self.api_key or config.openai_api_key,
            deepseek_api_key=config.deepseek_api_key,
            deepseek_base_url=config.deepseek_base_url,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class DeepSeekProvider:
    """DeepSeek provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "deepseek"

    def bind_config(self, config: Config) -> LLMClient:
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            openai_api_key=config.openai_api_key,
            deepseek_api_key=self.api_key or config.deepseek_api_key,
            deepseek_base_url=self.base_url or config.deepseek_base_url,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class LocalProvider:
    """Local OpenAI-compatible provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "local"

    def bind_config(self, config: Config) -> LLMClient:
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            openai_api_key=config.openai_api_key,
            deepseek_api_key=config.deepseek_api_key,
            deepseek_base_url=config.deepseek_base_url,
            local_api_key=self.api_key or config.local_api_key,
            local_base_url=self.base_url or config.local_base_url,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class ProviderAttempt:
    """One provider attempt inside a fallback request."""

    provider: str
    model: str | None
    success: bool
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }


@dataclass
class FallbackChain(LLMClient):
    """LLM client that tries providers in order until one succeeds."""

    providers: list[LLMClient | BindableLLM]
    strategy: FallbackStrategy = "first_success"
    attempts: list[ProviderAttempt] = field(default_factory=list)
    last_attempts: list[ProviderAttempt] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FallbackChain requires at least one provider")
        if self.strategy != "first_success":
            raise NotImplementedError(f"FallbackChain strategy is not implemented yet: {self.strategy}")

    def bind_config(self, config: "Config") -> "FallbackChain":
        bound: list[LLMClient] = []
        for provider in self.providers:
            if isinstance(provider, LLMClient):
                bound.append(provider)
            elif hasattr(provider, "bind_config"):
                bound.append(provider.bind_config(config))
            else:
                raise TypeError(f"Unsupported fallback provider: {provider!r}")
        return FallbackChain(bound, strategy=self.strategy)

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        return self._try_providers(lambda provider: _complete(provider, messages, max_output_tokens=max_output_tokens))

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        return self._try_providers(
            lambda provider: _complete_action_once(provider, messages, max_output_tokens=max_output_tokens)
        )

    def _try_providers(self, call) -> str:
        self.last_attempts = []
        errors: list[str] = []
        for provider in self.providers:
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            try:
                response = call(provider)
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                attempt = ProviderAttempt(provider_name, model, False, latency, error=error)
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(provider_name, model, True, latency)
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            return response
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(f"All fallback providers failed: {detail}")


def _complete(provider: LLMClient, messages: list[dict[str, str]], *, max_output_tokens: int | None) -> str:
    try:
        if max_output_tokens is None:
            return provider.complete(messages)
        return provider.complete(messages, max_output_tokens=max_output_tokens)
    except TypeError:
        return provider.complete(messages)


def _complete_action_once(provider: LLMClient, messages: list[dict[str, str]], *, max_output_tokens: int | None) -> str:
    if max_output_tokens is None:
        return provider._complete_action_once(messages)
    return provider._complete_action_once(messages, max_output_tokens=max_output_tokens)


def _provider_identity(provider: object) -> tuple[str, str | None]:
    provider_name = getattr(provider, "provider", None) or getattr(provider, "name", None) or type(provider).__name__
    model = getattr(provider, "model", None)
    return str(provider_name), str(model) if model is not None else None


__all__ = [
    "DeepSeekProvider",
    "FallbackChain",
    "FallbackStrategy",
    "LocalProvider",
    "OpenAIProvider",
    "ProviderAttempt",
]
