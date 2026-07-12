"""Async LLM transport, orchestration, and cancellation tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from chulk import AgentConfig, AsyncAgent
from chulk.core.actions import FinalAnswerAction
from chulk.llm.base import LLMActionResult, LLMClient, classify_provider_exception
from chulk.llm.providers.anthropic import AnthropicMessagesClient
from chulk.llm.providers.deepseek import DeepSeekChatCompletionsClient
from chulk.llm.providers.gemini import GeminiGenerateContentClient
from chulk.llm.providers.openai import OpenAIResponsesClient
from chulk.llm.public import FallbackChain
from chulk.llm import public as public_llm
from chulk.llm.usage import LLMResponse, LLMUsage


MESSAGES = [{"role": "user", "content": "hello"}]


def test_public_llm_exports_every_provider_wrapper() -> None:
    assert {
        "OpenAIProvider",
        "DeepSeekProvider",
        "LocalProvider",
        "OpenAICompatibleProvider",
        "OpenRouterProvider",
        "AnthropicProvider",
        "BedrockProvider",
        "GeminiProvider",
    } <= set(public_llm.__all__)


class _FailingSyncEndpoint:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError(f"sync provider method was called: {kwargs}")

    def generate_content(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError(f"sync provider method was called: {kwargs}")


class _AsyncEndpoint:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _AsyncModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.outcomes.pop(0)


class _SyncModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError(f"sync Gemini method was called: {kwargs}")


class _AsyncActionScript(LLMClient):
    provider = "async-script"
    model = "async-script"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.sync_calls = 0
        self.async_messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        self.sync_calls += 1
        raise AssertionError("sync LLM method was called")

    async def _acomplete_action_response_once(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> LLMResponse:
        self.async_messages.append(messages)
        content = self.responses.pop(0)
        usage = LLMUsage(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            estimated=False,
            source="provider",
        )
        return LLMResponse(
            content=content,
            usage=usage,
            provider=self.provider,
            model=self.model,
        )


class _SyncActionOnly(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        raise AssertionError("complete should not replace the custom action hook")

    def complete_action(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> LLMActionResult:
        self.calls += 1
        return LLMActionResult(
            action=FinalAnswerAction(type="final_answer", content="sync compatibility"),
            raw_response=_final_answer("sync compatibility"),
        )


def _final_answer(content: str) -> str:
    return json.dumps({"type": "final_answer", "content": content})


@pytest.mark.parametrize(
    ("exception_name", "expected_code"),
    [
        ("ReadTimeout", "timeout"),
        ("ConnectTimeout", "timeout"),
        ("ConnectError", "connection_error"),
        ("RemoteProtocolError", "connection_error"),
    ],
)
def test_http_client_transient_errors_are_classified_without_sdk_imports(
    exception_name: str,
    expected_code: str,
) -> None:
    exception_type = type(exception_name, (RuntimeError,), {})

    classification = classify_provider_exception(exception_type("transient"))

    assert classification.code == expected_code
    assert classification.retryable is True
    assert classification.fallback_eligible is True


@pytest.mark.asyncio
async def test_async_action_parser_repairs_and_aggregates_usage_without_sync_calls() -> None:
    client = _AsyncActionScript(["not json", _final_answer("repaired")])

    result = await client.acomplete_action(MESSAGES, max_repair_attempts=1)

    assert result.action == FinalAnswerAction(type="final_answer", content="repaired")
    assert result.repair_attempts == 1
    assert result.usage is not None
    assert result.usage.total_tokens == 10
    assert client.sync_calls == 0
    assert "Previous invalid response" in client.async_messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_custom_sync_complete_action_remains_async_compatible() -> None:
    client = _SyncActionOnly()

    result = await client.acomplete_action(MESSAGES, tools=[])

    assert result.action == FinalAnswerAction(type="final_answer", content="sync compatibility")
    assert client.calls == 1


@pytest.mark.asyncio
async def test_openai_uses_injected_native_async_responses_client() -> None:
    sync = _FailingSyncEndpoint()
    async_endpoint = _AsyncEndpoint([SimpleNamespace(output_text="openai async", usage=None)])
    client = OpenAIResponsesClient(
        model="openai-test",
        client=SimpleNamespace(responses=sync),
        async_client=SimpleNamespace(responses=async_endpoint),
    )

    response = await client.acomplete_response(MESSAGES, max_output_tokens=41)

    assert response.content == "openai async"
    assert sync.calls == 0
    assert async_endpoint.calls[0]["max_output_tokens"] == 41


@pytest.mark.asyncio
async def test_shared_chat_transport_uses_injected_native_async_client() -> None:
    sync = _FailingSyncEndpoint()
    async_endpoint = _AsyncEndpoint(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="chat async"))],
                usage=None,
            )
        ]
    )
    client = DeepSeekChatCompletionsClient(
        model="deepseek-test",
        api_key="test",
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        async_client=SimpleNamespace(chat=SimpleNamespace(completions=async_endpoint)),
    )

    response = await client.acomplete_response(MESSAGES, max_output_tokens=42)

    assert response.content == "chat async"
    assert sync.calls == 0
    assert async_endpoint.calls[0]["max_tokens"] == 42


@pytest.mark.asyncio
async def test_anthropic_uses_injected_native_async_messages_client() -> None:
    sync = _FailingSyncEndpoint()
    async_endpoint = _AsyncEndpoint(
        [
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text="anthropic async")],
                usage=None,
            )
        ]
    )
    client = AnthropicMessagesClient(
        model="claude-test",
        client=SimpleNamespace(messages=sync),
        async_client=SimpleNamespace(messages=async_endpoint),
    )

    response = await client.acomplete_response(MESSAGES, max_output_tokens=43)

    assert response.content == "anthropic async"
    assert sync.calls == 0
    assert async_endpoint.calls[0]["max_tokens"] == 43


@pytest.mark.asyncio
async def test_gemini_uses_injected_native_async_generate_content_client() -> None:
    sync = _SyncModels()
    async_models = _AsyncModels([SimpleNamespace(text="gemini async", usage_metadata=None)])
    client = GeminiGenerateContentClient(
        model="gemini-test",
        client=SimpleNamespace(models=sync),
        async_client=SimpleNamespace(models=async_models),
    )

    response = await client.acomplete_response(MESSAGES, max_output_tokens=44)

    assert response.content == "gemini async"
    assert sync.calls == 0
    assert async_models.calls[0]["config"] == {"max_output_tokens": 44}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "chat", "anthropic", "gemini"])
async def test_provider_native_async_action_path_never_calls_sync_transport(provider: str) -> None:
    sync = _FailingSyncEndpoint()
    if provider == "openai":
        async_endpoint = _AsyncEndpoint(
            [SimpleNamespace(id="resp_1", output_text="openai action", output=[], usage=None)]
        )
        client: LLMClient = OpenAIResponsesClient(
            model="openai-test",
            client=SimpleNamespace(responses=sync),
            async_client=SimpleNamespace(responses=async_endpoint),
        )
    elif provider == "chat":
        async_endpoint = _AsyncEndpoint(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="chat action", tool_calls=[]),
                        )
                    ],
                    usage=None,
                )
            ]
        )
        client = DeepSeekChatCompletionsClient(
            model="deepseek-test",
            client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
            async_client=SimpleNamespace(chat=SimpleNamespace(completions=async_endpoint)),
        )
    elif provider == "anthropic":
        async_endpoint = _AsyncEndpoint(
            [
                SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="anthropic action")],
                    usage=None,
                )
            ]
        )
        client = AnthropicMessagesClient(
            model="claude-test",
            client=SimpleNamespace(messages=sync),
            async_client=SimpleNamespace(messages=async_endpoint),
        )
    else:
        async_models = _AsyncModels([SimpleNamespace(text="gemini action", usage_metadata=None)])
        client = GeminiGenerateContentClient(
            model="gemini-test",
            client=SimpleNamespace(models=sync),
            async_client=SimpleNamespace(models=async_models),
        )

    result = await client.acomplete_action(MESSAGES, tools=[])

    assert isinstance(result.action, FinalAnswerAction)
    assert result.action.content == f"{provider} action"
    assert result.metadata["action_transport"] == "provider_native"
    assert sync.calls == 0


@pytest.mark.asyncio
async def test_async_agent_and_planned_turn_never_call_sync_llm(tmp_path) -> None:
    config = AgentConfig(project_root=tmp_path)
    run_client = _AsyncActionScript([_final_answer("async run")])
    run_agent = AsyncAgent(config=config, llm=run_client, tools=[], skills=[])

    run_result = await run_agent.run_result("run async")

    plan_payload = {
        "type": "plan",
        "content": None,
        "tool_name": None,
        "arguments_json": "{}",
        "plan_json": json.dumps({
            "summary": "Implement native async planning",
            "steps": [
                {
                    "id": "1",
                    "title": "Implement",
                    "description": "Update the agent action loop to await the native async provider path.",
                    "status": "pending",
                    "acceptance_criteria": ["Async planned turns do not call sync LLM methods."],
                    "retry_limit": 0,
                }
            ],
        }),
        "step_update_json": "{}",
    }
    plan_client = _AsyncActionScript([json.dumps(plan_payload)])
    plan_agent = AsyncAgent(config=config, llm=plan_client, tools=[], skills=[])

    plan_result = await plan_agent.plan_result("plan async")

    assert run_result.content == "async run"
    assert run_client.sync_calls == 0
    assert plan_result.status == "waiting_for_approval"
    assert plan_client.sync_calls == 0


@pytest.mark.asyncio
async def test_cancelling_native_async_provider_does_not_fallback_or_call_sync() -> None:
    started = asyncio.Event()
    sync = _FailingSyncEndpoint()

    class HangingResponses:
        async def create(self, **kwargs: object) -> object:
            started.set()
            await asyncio.Future()
            raise AssertionError(f"unreachable: {kwargs}")

    primary = OpenAIResponsesClient(
        model="primary",
        client=SimpleNamespace(responses=sync),
        async_client=SimpleNamespace(responses=HangingResponses()),
    )
    secondary = _AsyncActionScript([_final_answer("secondary")])
    chain = FallbackChain([primary, secondary])

    task = asyncio.create_task(chain.acomplete_action(MESSAGES, tools=[]))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sync.calls == 0
    assert secondary.sync_calls == 0
    assert secondary.async_messages == []
    assert chain.last_success_provider is None
    assert chain.last_attempts == []
