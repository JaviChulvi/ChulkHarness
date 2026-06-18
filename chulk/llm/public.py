"""Public LLM provider specs and fallback chains."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Literal, Protocol

from chulk.llm.base import LLMActionResult, LLMClient, LLMError, LLMStreamChunk
from chulk.llm.factory import create_llm_client
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage

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
    usage: LLMUsage | None = None
    cost: LLMCost | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "cost": self.cost.to_dict() if self.cost is not None else None,
        }


@dataclass
class FallbackChain(LLMClient):
    """LLM client that tries providers in order until one succeeds."""

    providers: list[LLMClient | BindableLLM]
    strategy: FallbackStrategy = "first_success"
    attempts: list[ProviderAttempt] = field(default_factory=list)
    last_attempts: list[ProviderAttempt] = field(default_factory=list)
    last_success_provider: LLMClient | None = field(default=None, init=False, repr=False)
    _action_attempts: list[ProviderAttempt] | None = field(default=None, init=False, repr=False)

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
        return self.complete_response(messages, max_output_tokens=max_output_tokens).content

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        return self._try_provider_responses(
            lambda provider: _complete_response(provider, messages, max_output_tokens=max_output_tokens)
        )

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMActionResult:
        self._action_attempts = []
        try:
            return super().complete_action(
                messages,
                max_repair_attempts=max_repair_attempts,
                max_output_tokens=max_output_tokens,
                tools=tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )
        finally:
            if self._action_attempts is not None:
                self.last_attempts = self._action_attempts
                self._action_attempts = None

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        yield from self._stream_providers(messages, max_output_tokens=max_output_tokens)

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        return self._complete_action_response_once(messages, max_output_tokens=max_output_tokens).content

    def _complete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMResponse:
        return self._try_provider_responses(
            lambda provider: _complete_action_response_once(
                provider,
                messages,
                max_output_tokens=max_output_tokens,
                tools=tools,
                hosted_mcp_servers=hosted_mcp_servers if _supports_hosted_mcp(provider) else None,
                mcp_approval_callback=mcp_approval_callback if _supports_hosted_mcp(provider) else None,
            )
        )

    def _try_provider_responses(self, call) -> LLMResponse:
        self.last_attempts = []
        self.last_success_provider = None
        errors: list[str] = []
        for provider in self.providers:
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            try:
                response: LLMResponse = call(provider)
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                attempt = ProviderAttempt(
                    provider_name,
                    model,
                    False,
                    latency,
                    error=error,
                    usage=getattr(exc, "usage", None),
                    cost=getattr(exc, "cost", None),
                )
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                if self._action_attempts is not None:
                    self._action_attempts.append(attempt)
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(provider_name, model, True, latency, usage=response.usage, cost=response.cost)
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            if self._action_attempts is not None:
                self._action_attempts.append(attempt)
            self.last_success_provider = provider
            return response
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(f"All fallback providers failed: {detail}")

    def _stream_providers(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None,
    ) -> Iterator[LLMStreamChunk]:
        self.last_attempts = []
        self.last_success_provider = None
        errors: list[str] = []
        for provider in self.providers:
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            emitted_text = False
            usage: LLMUsage | None = None
            cost: LLMCost | None = None
            try:
                for chunk in _stream_complete(provider, messages, max_output_tokens=max_output_tokens):
                    if chunk.type == "text_delta" and chunk.text:
                        emitted_text = True
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.cost is not None:
                        cost = chunk.cost
                    yield chunk
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                attempt = ProviderAttempt(
                    provider_name,
                    model,
                    False,
                    latency,
                    error=error,
                    usage=getattr(exc, "usage", None),
                    cost=getattr(exc, "cost", None),
                )
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                if emitted_text:
                    raise LLMError(f"{provider_name}/{model or 'unknown'} stream failed after emitting text: {error}") from exc
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(provider_name, model, True, latency, usage=usage, cost=cost)
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            self.last_success_provider = provider
            return
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(f"All fallback providers failed: {detail}")


def _complete_response(provider: LLMClient, messages: list[dict[str, str]], *, max_output_tokens: int | None) -> LLMResponse:
    try:
        if max_output_tokens is None:
            return provider.complete_response(messages)
        return provider.complete_response(messages, max_output_tokens=max_output_tokens)
    except TypeError:
        return provider.complete_response(messages)


def _stream_complete(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
) -> Iterator[LLMStreamChunk]:
    try:
        if max_output_tokens is None:
            yield from provider.stream_complete(messages)
            return
        yield from provider.stream_complete(messages, max_output_tokens=max_output_tokens)
    except TypeError:
        yield from provider.stream_complete(messages)


def _complete_action_response_once(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
    tools: list[object] | None,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
    mcp_approval_callback: Callable[[dict], bool] | None = None,
) -> LLMResponse:
    if max_output_tokens is None:
        return provider._complete_action_response_once(
            messages,
            tools=tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
    return provider._complete_action_response_once(
        messages,
        max_output_tokens=max_output_tokens,
        tools=tools,
        hosted_mcp_servers=hosted_mcp_servers,
        mcp_approval_callback=mcp_approval_callback,
    )


def _supports_hosted_mcp(provider: object) -> bool:
    capabilities = getattr(provider, "capabilities", None)
    return bool(getattr(capabilities, "supports_hosted_mcp_tools", False))


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
