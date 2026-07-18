"""Public LLM provider specs and fallback chains."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
import json
import time
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from chulk.core.actions import (
    FinalAnswerAction,
    PlanAction,
    PlanStepUpdateAction,
    ToolCallAction,
)
from chulk.llm.base import (
    LLMActionResult,
    LLMClient,
    LLMError,
    LLMStreamChunk,
    call_async_with_supported_kwargs,
    call_with_supported_kwargs,
)
from chulk.llm.capabilities import LLMModelCapabilities, conservative_model_capabilities
from chulk.llm.factory import create_llm_client, provider_connection_from_config
from chulk.llm.tools import PlanningToolAvailability
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage

if TYPE_CHECKING:
    from chulk.config import Config


FallbackStrategy = Literal["first_success", "round_robin", "lowest_latency"]
ResultT = TypeVar("ResultT")
_CUSTOM_ACTION_RESULT_METADATA_KEY = "_chulk_custom_action_result"


class BindableLLM(Protocol):
    """Provider spec that can create an LLM client from runtime config."""

    @property
    def provider(self) -> str:
        """Provider registry name."""

    @property
    def model(self) -> str:
        """Configured model identifier."""

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
        connection = provider_connection_from_config(self.provider, config).with_overrides(api_key=self.api_key or None)
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
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
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class LocalProvider:
    """Local OpenAI-compatible provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    context_window_tokens: int | None = field(default=None, kw_only=True)
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "local"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            local_context_window_tokens=(
                self.context_window_tokens
                if self.context_window_tokens is not None
                else config.local_context_window_tokens
            ),
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """User-selected hosted OpenAI-compatible provider spec."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "openai-compatible"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class OpenRouterProvider:
    """OpenRouter provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "openrouter"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class AnthropicProvider:
    """Anthropic provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "anthropic"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class BedrockProvider:
    """AWS Bedrock OpenAI-compatible provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "bedrock"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
            timeout_seconds=self.timeout_seconds or config.llm_timeout_seconds,
            max_retries=self.max_retries if self.max_retries is not None else config.llm_max_retries,
        )


@dataclass(frozen=True)
class GeminiProvider:
    """Google Gemini provider spec for the public API."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    provider: str = "gemini"

    def bind_config(self, config: Config) -> LLMClient:
        connection = provider_connection_from_config(self.provider, config).with_overrides(
            api_key=self.api_key or None,
            base_url=self.base_url or None,
        )
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            connection=connection,
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
    error_code: str | None = None
    retryable: bool | None = None
    fallback_eligible: bool | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "fallback_eligible": self.fallback_eligible,
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
    model_capabilities: LLMModelCapabilities | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FallbackChain requires at least one provider")
        if self.strategy != "first_success":
            raise NotImplementedError(f"FallbackChain strategy is not implemented yet: {self.strategy}")
        records = [getattr(provider, "model_capabilities", None) for provider in self.providers]
        typed_records = [record for record in records if isinstance(record, LLMModelCapabilities)]
        if records and len(typed_records) == len(records):
            self.model_capabilities = conservative_model_capabilities(typed_records)

    def bind_config(self, config: "Config") -> "FallbackChain":
        bound: list[LLMClient] = []
        for provider in self.providers:
            if isinstance(provider, LLMClient):
                bound.append(provider)
            elif hasattr(provider, "bind_config"):
                bound.append(provider.bind_config(config))
            else:
                raise TypeError(f"Unsupported fallback provider: {provider!r}")
        return FallbackChain(providers=list(bound), strategy=self.strategy)

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

    async def acomplete_response(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._atry_provider_responses(
            lambda provider: _acomplete_response(
                provider,
                messages,
                max_output_tokens=max_output_tokens,
            )
        )

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMActionResult:
        self._action_attempts = []
        try:
            result = super().complete_action(
                messages,
                max_repair_attempts=max_repair_attempts,
                max_output_tokens=max_output_tokens,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )
            return _restore_custom_action_result(result)
        finally:
            if self._action_attempts is not None:
                self.last_attempts = self._action_attempts
                self._action_attempts = None

    async def acomplete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMActionResult:
        self._action_attempts = []
        try:
            result = await super().acomplete_action(
                messages,
                max_repair_attempts=max_repair_attempts,
                max_output_tokens=max_output_tokens,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers,
                mcp_approval_callback=mcp_approval_callback,
            )
            return _restore_custom_action_result(result)
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
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMResponse:
        return self._try_provider_responses(
            lambda provider: _complete_action_response_once(
                provider,
                messages,
                max_output_tokens=max_output_tokens,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers if _supports_hosted_mcp(provider) else None,
                mcp_approval_callback=mcp_approval_callback if _supports_hosted_mcp(provider) else None,
            )
        )

    async def _acomplete_action_response_once(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        planning_tools: PlanningToolAvailability | None = None,
        hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
        mcp_approval_callback: Callable[[dict], bool] | None = None,
    ) -> LLMResponse:
        return await self._atry_provider_responses(
            lambda provider: _acomplete_action_response_once(
                provider,
                messages,
                max_output_tokens=max_output_tokens,
                tools=tools,
                planning_tools=planning_tools,
                hosted_mcp_servers=hosted_mcp_servers if _supports_hosted_mcp(provider) else None,
                mcp_approval_callback=mcp_approval_callback if _supports_hosted_mcp(provider) else None,
            )
        )

    def _try_provider_responses(
        self,
        call: Callable[[LLMClient], ResultT],
    ) -> ResultT:
        self.last_attempts = []
        self.last_success_provider = None
        errors: list[str] = []
        for provider in self.providers:
            if not isinstance(provider, LLMClient):
                raise TypeError("FallbackChain must be bound before use")
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            try:
                response = call(provider)
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                error_code, retryable, fallback_eligible = _provider_error_metadata(exc)
                attempt = ProviderAttempt(
                    provider_name,
                    model,
                    False,
                    latency,
                    error=error,
                    error_code=error_code,
                    retryable=retryable,
                    fallback_eligible=fallback_eligible,
                    usage=getattr(exc, "usage", None),
                    cost=getattr(exc, "cost", None),
                )
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                if self._action_attempts is not None:
                    self._action_attempts.append(attempt)
                if not isinstance(exc, LLMError) or not exc.fallback_eligible:
                    raise
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(
                provider_name,
                model,
                True,
                latency,
                usage=getattr(response, "usage", None),
                cost=getattr(response, "cost", None),
            )
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            if self._action_attempts is not None:
                self._action_attempts.append(attempt)
            self.last_success_provider = provider
            return response
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(
            f"All fallback providers failed: {detail}",
            code="fallback_exhausted",
            retryable=any(attempt.retryable is True for attempt in self.last_attempts),
        )

    async def _atry_provider_responses(
        self,
        call: Callable[[LLMClient], Awaitable[ResultT]],
    ) -> ResultT:
        self.last_attempts = []
        self.last_success_provider = None
        errors: list[str] = []
        for provider in self.providers:
            if not isinstance(provider, LLMClient):
                raise TypeError("FallbackChain must be bound before use")
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            try:
                response = await call(provider)
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                error_code, retryable, fallback_eligible = _provider_error_metadata(exc)
                attempt = ProviderAttempt(
                    provider_name,
                    model,
                    False,
                    latency,
                    error=error,
                    error_code=error_code,
                    retryable=retryable,
                    fallback_eligible=fallback_eligible,
                    usage=getattr(exc, "usage", None),
                    cost=getattr(exc, "cost", None),
                )
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                if self._action_attempts is not None:
                    self._action_attempts.append(attempt)
                if not isinstance(exc, LLMError) or not exc.fallback_eligible:
                    raise
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(
                provider_name,
                model,
                True,
                latency,
                usage=getattr(response, "usage", None),
                cost=getattr(response, "cost", None),
            )
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            if self._action_attempts is not None:
                self._action_attempts.append(attempt)
            self.last_success_provider = provider
            return response
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(
            f"All fallback providers failed: {detail}",
            code="fallback_exhausted",
            retryable=any(attempt.retryable is True for attempt in self.last_attempts),
        )

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
            if not isinstance(provider, LLMClient):
                raise TypeError("FallbackChain must be bound before use")
            started_at = time.monotonic()
            provider_name, model = _provider_identity(provider)
            emitted_chunk = False
            usage: LLMUsage | None = None
            cost: LLMCost | None = None
            try:
                for chunk in _stream_complete(provider, messages, max_output_tokens=max_output_tokens):
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.cost is not None:
                        cost = chunk.cost
                    emitted_chunk = True
                    yield chunk
            except Exception as exc:
                latency = time.monotonic() - started_at
                error = str(exc)
                error_code, retryable, fallback_eligible = _provider_error_metadata(exc)
                attempt = ProviderAttempt(
                    provider_name,
                    model,
                    False,
                    latency,
                    error=error,
                    error_code=error_code,
                    retryable=retryable,
                    fallback_eligible=fallback_eligible,
                    usage=getattr(exc, "usage", None),
                    cost=getattr(exc, "cost", None),
                )
                self.last_attempts.append(attempt)
                self.attempts.append(attempt)
                if emitted_chunk:
                    source_error = exc if isinstance(exc, LLMError) else None
                    raise LLMError(
                        f"{provider_name}/{model or 'unknown'} stream failed after yielding a chunk: {error}",
                        provider=provider_name,
                        model=model,
                        code=source_error.code if source_error is not None else "unknown",
                        retryable=source_error.retryable if source_error is not None else False,
                        fallback_eligible=False,
                    ) from exc
                if not isinstance(exc, LLMError) or not exc.fallback_eligible:
                    raise
                errors.append(f"{provider_name}/{model or 'unknown'}: {error}")
                continue
            latency = time.monotonic() - started_at
            attempt = ProviderAttempt(provider_name, model, True, latency, usage=usage, cost=cost)
            self.last_attempts.append(attempt)
            self.attempts.append(attempt)
            self.last_success_provider = provider
            return
        detail = "; ".join(errors) if errors else "no providers were available"
        raise LLMError(
            f"All fallback providers failed: {detail}",
            code="fallback_exhausted",
            retryable=any(attempt.retryable is True for attempt in self.last_attempts),
        )


def _complete_response(provider: LLMClient, messages: list[dict[str, str]], *, max_output_tokens: int | None) -> LLMResponse:
    kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
    return call_with_supported_kwargs(provider.complete_response, messages, **kwargs)


async def _acomplete_response(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
) -> LLMResponse:
    kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
    return await call_async_with_supported_kwargs(provider.acomplete_response, messages, **kwargs)


def _stream_complete(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
) -> Iterator[LLMStreamChunk]:
    kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
    yield from call_with_supported_kwargs(provider.stream_complete, messages, **kwargs)


def _uses_custom_sync_action(provider: LLMClient) -> bool:
    return type(provider).complete_action is not LLMClient.complete_action


def _uses_custom_async_action(provider: LLMClient) -> bool:
    return (
        type(provider).acomplete_action is not LLMClient.acomplete_action
        or _uses_custom_sync_action(provider)
    )


def _complete_action(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_repair_attempts: int,
    max_output_tokens: int | None,
    tools: list[object] | None,
    planning_tools: PlanningToolAvailability | None,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None,
    mcp_approval_callback: Callable[[dict], bool] | None,
) -> LLMActionResult:
    kwargs: dict[str, object] = {
        "max_repair_attempts": max_repair_attempts,
        "tools": tools,
        "planning_tools": planning_tools,
        "hosted_mcp_servers": hosted_mcp_servers,
        "mcp_approval_callback": mcp_approval_callback,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return call_with_supported_kwargs(provider.complete_action, messages, **kwargs)


async def _acomplete_action(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_repair_attempts: int,
    max_output_tokens: int | None,
    tools: list[object] | None,
    planning_tools: PlanningToolAvailability | None,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None,
    mcp_approval_callback: Callable[[dict], bool] | None,
) -> LLMActionResult:
    kwargs: dict[str, object] = {
        "max_repair_attempts": max_repair_attempts,
        "tools": tools,
        "planning_tools": planning_tools,
        "hosted_mcp_servers": hosted_mcp_servers,
        "mcp_approval_callback": mcp_approval_callback,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return await call_async_with_supported_kwargs(
        provider.acomplete_action,
        messages,
        **kwargs,
    )


def _complete_action_response_once(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
    tools: list[object] | None,
    planning_tools: PlanningToolAvailability | None = None,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
    mcp_approval_callback: Callable[[dict], bool] | None = None,
) -> LLMResponse:
    if _uses_custom_sync_action(provider):
        result = _complete_action(
            provider,
            messages,
            max_repair_attempts=0,
            max_output_tokens=max_output_tokens,
            tools=tools,
            planning_tools=planning_tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        return _response_from_custom_action(result, provider=provider)
    kwargs: dict[str, object] = {
        "tools": tools,
        "planning_tools": planning_tools,
        "hosted_mcp_servers": hosted_mcp_servers,
        "mcp_approval_callback": mcp_approval_callback,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return call_with_supported_kwargs(provider._complete_action_response_once, messages, **kwargs)


async def _acomplete_action_response_once(
    provider: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int | None,
    tools: list[object] | None,
    planning_tools: PlanningToolAvailability | None = None,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
    mcp_approval_callback: Callable[[dict], bool] | None = None,
) -> LLMResponse:
    if _uses_custom_async_action(provider):
        result = await _acomplete_action(
            provider,
            messages,
            max_repair_attempts=0,
            max_output_tokens=max_output_tokens,
            tools=tools,
            planning_tools=planning_tools,
            hosted_mcp_servers=hosted_mcp_servers,
            mcp_approval_callback=mcp_approval_callback,
        )
        return _response_from_custom_action(result, provider=provider)
    kwargs: dict[str, object] = {
        "tools": tools,
        "planning_tools": planning_tools,
        "hosted_mcp_servers": hosted_mcp_servers,
        "mcp_approval_callback": mcp_approval_callback,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return await call_async_with_supported_kwargs(
        provider._acomplete_action_response_once,
        messages,
        **kwargs,
    )


def _response_from_custom_action(
    result: LLMActionResult,
    *,
    provider: LLMClient,
) -> LLMResponse:
    metadata = dict(result.metadata)
    metadata[_CUSTOM_ACTION_RESULT_METADATA_KEY] = result
    provider_name, model = _provider_identity(provider)
    return LLMResponse(
        content=json.dumps(_action_payload(result), sort_keys=True),
        usage=result.usage,
        cost=result.cost,
        provider=provider_name,
        model=model,
        metadata=metadata,
    )


def _restore_custom_action_result(result: LLMActionResult) -> LLMActionResult:
    custom_result = result.metadata.get(_CUSTOM_ACTION_RESULT_METADATA_KEY)
    if not isinstance(custom_result, LLMActionResult):
        return result
    metadata = dict(result.metadata)
    metadata.pop(_CUSTOM_ACTION_RESULT_METADATA_KEY, None)
    return LLMActionResult(
        action=custom_result.action,
        raw_response=custom_result.raw_response,
        repair_attempts=result.repair_attempts + custom_result.repair_attempts,
        errors=[*result.errors, *custom_result.errors],
        usage=result.usage,
        cost=result.cost,
        metadata=metadata,
    )


def _action_payload(result: LLMActionResult) -> dict[str, object]:
    action = result.action
    if isinstance(action, FinalAnswerAction):
        return {"type": action.type, "content": action.content}
    if isinstance(action, ToolCallAction):
        return {
            "type": action.type,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }
    if isinstance(action, PlanAction):
        return {"type": action.type, "plan": action.plan.to_dict()}
    if isinstance(action, PlanStepUpdateAction):
        return {
            "type": action.type,
            "step_update": {
                "step_id": action.step_id,
                "status": action.status,
                "evidence": action.evidence,
                "reason": action.reason,
            },
        }
    raise TypeError(f"Unsupported custom action result: {type(action).__name__}")


def _supports_hosted_mcp(provider: object) -> bool:
    capabilities = getattr(provider, "capabilities", None)
    return bool(getattr(capabilities, "supports_hosted_mcp_tools", False))


def _provider_identity(provider: object) -> tuple[str, str | None]:
    provider_name = getattr(provider, "provider", None) or getattr(provider, "name", None) or type(provider).__name__
    model = getattr(provider, "model", None)
    return str(provider_name), str(model) if model is not None else None


def _provider_error_metadata(exc: Exception) -> tuple[str | None, bool | None, bool | None]:
    if not isinstance(exc, LLMError):
        return None, None, None
    return exc.code, exc.retryable, exc.fallback_eligible


__all__ = [
    "AnthropicProvider",
    "BedrockProvider",
    "BindableLLM",
    "DeepSeekProvider",
    "FallbackChain",
    "FallbackStrategy",
    "GeminiProvider",
    "LocalProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderAttempt",
]
