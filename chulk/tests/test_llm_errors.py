"""Focused tests for typed provider failure and fallback semantics."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from chulk.llm import (
    FallbackChain,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMStreamChunk,
    LocalOpenAICompatibleClient,
    OpenAIResponsesClient,
)
from chulk.llm.base import classify_provider_exception, provider_error_from_exception


MESSAGES = [{"role": "user", "content": "hello"}]


def _sdk_error(name: str, message: str, *, status_code: int | None = None, code: str | None = None) -> Exception:
    error_type = type(name, (Exception,), {})
    error = error_type(message)
    if status_code is not None:
        error.status_code = status_code  # type: ignore[attr-defined]
    if code is not None:
        error.code = code  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize(
    ("error", "code", "retryable", "fallback_eligible"),
    [
        (_sdk_error("AuthenticationError", "bad key", status_code=401), "authentication_error", False, False),
        (_sdk_error("PermissionDeniedError", "forbidden", status_code=403), "permission_denied", False, False),
        (_sdk_error("BadRequestError", "invalid input", status_code=400), "invalid_request", False, False),
        (_sdk_error("NotFoundError", "model missing", status_code=404), "model_not_found", False, False),
        (_sdk_error("RateLimitError", "slow down", status_code=429), "rate_limit", True, True),
        (_sdk_error("APITimeoutError", "timed out"), "timeout", True, True),
        (_sdk_error("APIConnectionError", "connection failed"), "connection_error", True, True),
        (_sdk_error("InternalServerError", "unavailable", status_code=503), "server_error", True, True),
    ],
)
def test_openai_style_sdk_errors_have_explicit_typed_semantics(
    error: Exception,
    code: str,
    retryable: bool,
    fallback_eligible: bool,
) -> None:
    classification = classify_provider_exception(error)

    assert classification.code == code
    assert classification.retryable is retryable
    assert classification.fallback_eligible is fallback_eligible


def test_provider_wrapper_preserves_typed_error_metadata_and_adds_identity() -> None:
    original = LLMError(
        "limited",
        code="rate_limit",
        retryable=True,
        fallback_eligible=True,
    )

    wrapped = provider_error_from_exception(
        original,
        message="Local request failed",
        provider="local",
        model="test-model",
    )

    assert wrapped is original
    assert wrapped.provider == "local"
    assert wrapped.model == "test-model"
    assert wrapped.code == "rate_limit"
    assert wrapped.retryable is True
    assert wrapped.fallback_eligible is True


class _FailingClient(LLMClient):
    provider = "primary"
    model = "primary-model"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        raise self.error


class _SuccessfulClient(LLMClient):
    provider = "secondary"
    model = "secondary-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_response(self, messages, *, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="secondary answer", provider=self.provider, model=self.model)


def test_fallback_chain_does_not_hide_authentication_failure() -> None:
    primary_error = LLMError(
        "invalid key",
        code="authentication_error",
        retryable=False,
        fallback_eligible=False,
    )
    primary = _FailingClient(primary_error)
    secondary = _SuccessfulClient()
    chain = FallbackChain([primary, secondary])

    with pytest.raises(LLMError) as raised:
        chain.complete_response(MESSAGES)

    assert raised.value is primary_error
    assert primary.calls == 1
    assert secondary.calls == 0
    assert chain.last_attempts[0].error_code == "authentication_error"
    assert chain.last_attempts[0].fallback_eligible is False


def test_fallback_chain_does_not_advance_on_arbitrary_exception() -> None:
    primary_error = RuntimeError("implementation bug")
    primary = _FailingClient(primary_error)
    secondary = _SuccessfulClient()
    chain = FallbackChain([primary, secondary])

    with pytest.raises(RuntimeError, match="implementation bug") as raised:
        chain.complete_response(MESSAGES)

    assert raised.value is primary_error
    assert secondary.calls == 0
    assert chain.last_attempts[0].fallback_eligible is None


@pytest.mark.parametrize("code", ["rate_limit", "server_error"])
def test_fallback_chain_advances_only_on_explicit_transient_provider_failure(code: str) -> None:
    primary = _FailingClient(
        LLMError(
            "temporarily unavailable",
            code=code,  # type: ignore[arg-type]
            retryable=True,
            fallback_eligible=True,
        )
    )
    secondary = _SuccessfulClient()
    chain = FallbackChain([primary, secondary])

    response = chain.complete_response(MESSAGES)

    assert response.content == "secondary answer"
    assert primary.calls == 1
    assert secondary.calls == 1
    assert [attempt.success for attempt in chain.last_attempts] == [False, True]
    assert chain.last_attempts[0].error_code == code


class _PartialFailingStreamClient(LLMClient):
    provider = "stream-primary"
    model = "stream-model"

    def __init__(self, first_chunk: LLMStreamChunk) -> None:
        self.first_chunk = first_chunk

    def stream_complete(self, messages, *, max_output_tokens=None) -> Iterator[LLMStreamChunk]:
        yield self.first_chunk
        raise LLMError(
            "connection lost",
            code="connection_error",
            retryable=True,
            fallback_eligible=True,
        )


class _RecordingStreamClient(LLMClient):
    provider = "stream-secondary"
    model = "stream-model"

    def __init__(self) -> None:
        self.calls = 0

    def stream_complete(self, messages, *, max_output_tokens=None) -> Iterator[LLMStreamChunk]:
        self.calls += 1
        yield LLMStreamChunk(type="text_delta", text="duplicate")
        yield LLMStreamChunk(type="completed")


@pytest.mark.parametrize(
    "first_chunk",
    [
        LLMStreamChunk(type="text_delta", text="partial"),
        LLMStreamChunk(type="completed"),
    ],
)
def test_fallback_stream_never_switches_provider_after_yielding_any_chunk(first_chunk: LLMStreamChunk) -> None:
    secondary = _RecordingStreamClient()
    chain = FallbackChain([_PartialFailingStreamClient(first_chunk), secondary])
    stream = chain.stream_complete(MESSAGES)

    assert next(stream) is first_chunk
    with pytest.raises(LLMError, match="after yielding a chunk") as raised:
        next(stream)

    assert raised.value.fallback_eligible is False
    assert raised.value.code == "connection_error"
    assert secondary.calls == 0


class _FailingNativeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (_sdk_error("AuthenticationError", "bad key", status_code=401), "authentication_error"),
        (_sdk_error("RateLimitError", "slow down", status_code=429), "rate_limit"),
        (_sdk_error("InternalServerError", "down", status_code=503), "server_error"),
    ],
)
def test_native_tool_failures_do_not_json_retry_auth_rate_or_server_errors(
    error: Exception,
    expected_code: str,
) -> None:
    completions = _FailingNativeCompletions(error)
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = LocalOpenAICompatibleClient(model="local/test-model", client=sdk_client)
    tool = SimpleNamespace(
        name="calculator",
        description="calculate",
        args_schema={"type": "object", "properties": {}},
    )

    with pytest.raises(LLMError) as raised:
        client.complete_action(MESSAGES, tools=[tool])

    assert raised.value.code == expected_code
    assert raised.value.provider == "local"
    assert raised.value.model == "local/test-model"
    assert len(completions.calls) == 1
    assert "tools" in completions.calls[0]


class _FailingNativeResponses:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


def test_openai_native_tool_auth_failure_is_not_retried_as_json() -> None:
    responses = _FailingNativeResponses(
        _sdk_error("AuthenticationError", "bad key", status_code=401),
    )
    client = OpenAIResponsesClient(
        model="gpt-test",
        client=SimpleNamespace(responses=responses),
    )
    tool = SimpleNamespace(
        name="calculator",
        description="calculate",
        args_schema={"type": "object", "properties": {}},
    )

    with pytest.raises(LLMError) as raised:
        client.complete_action(MESSAGES, tools=[tool])

    assert raised.value.code == "authentication_error"
    assert raised.value.provider == "openai"
    assert raised.value.model == "gpt-test"
    assert len(responses.calls) == 1
    assert "tools" in responses.calls[0]
