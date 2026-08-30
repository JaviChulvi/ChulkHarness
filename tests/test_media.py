from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

import pytest

from chulk import Agent as SDKAgent
from chulk import UserInput as PublicUserInput
from chulk.config import load_config
from tests.core_agent import build_core_agent as Agent
from chulk.core.actions import FinalAnswerAction
from chulk.llm import LLMActionResult, LLMClient
from chulk.media import (
    ContentIntegrityError,
    ContentLimitError,
    ContentOwnershipError,
    ContentRef,
    ContentStore,
    GeneratedMedia,
    HostedTranscriptionProcessor,
    LocalTextExtractor,
    LocalTranscriptionProcessor,
    MediaLimits,
    MediaInputPart,
    MediaKind,
    MediaProcessorRegistry,
    RetentionPolicy,
    TextInputPart,
    TransformKind,
    UnsupportedMediaError,
    UserInput,
    media_delivery_tool,
    media_generation_tool,
    media_transform_tool,
)
from chulk.usage import (
    BudgetExceededError,
    ExactCost,
    ModelUsageAccounting,
    RunBudget,
    SQLiteUsageStore,
    UsageDimensions,
)


class ActionClient(LLMClient):
    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def complete_action(self, messages, **kwargs):
        self.messages.append(messages)
        return LLMActionResult(
            FinalAnswerAction(type="final_answer", content="media ok"),
            '{"type":"final_answer","content":"media ok"}',
        )


def test_user_input_projection_contains_only_safe_media_metadata(tmp_path) -> None:
    assert PublicUserInput is UserInput
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )
    item = store.put(
        b"private document bytes",
        kind=MediaKind.DOCUMENT,
        mime_type="text/plain",
        provenance="upload:test",
        file_name="../../notes.txt",
    )
    value = UserInput(
        (
            TextInputPart("Summarize this."),
            MediaInputPart(item, caption="Quarterly notes"),
        )
    )

    projection = value.textual_projection()

    assert "private document bytes" not in projection
    assert item.content_ref.id in projection
    assert item.file_name == "notes.txt"
    assert value.safe_metadata()[1]["media"]["sha256"] == item.sha256


def test_public_sdk_runs_typed_input_with_injected_processors(tmp_path) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openai",
            "CHULK_MODEL": "gpt-4.1-mini",
            "OPENAI_API_KEY": "test",
        }
    )
    store = ContentStore(
        config.store_path,
        config.runtime_dir / "content",
        profile_id="default",
    )
    item = store.put(
        b"sdk document",
        kind="document",
        mime_type="text/plain",
        provenance="test",
    )
    client = ActionClient()
    sdk = SDKAgent(
        config=config,
        llm=client,
        content_store=store,
        media_processors=MediaProcessorRegistry((LocalTextExtractor(),)),
    )

    result = sdk.run_input(
        UserInput((TextInputPart("Read"), MediaInputPart(item)))
    )
    sdk.close()

    assert result == "media ok"
    assert "sdk document" in json.dumps(client.messages)


def test_content_store_enforces_limits_owner_integrity_and_retention(tmp_path) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock_value = [now]
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
        max_content_bytes=16,
        clock=lambda: clock_value[0],
    )
    with pytest.raises(ContentLimitError):
        store.put(
            b"x" * 17,
            kind="document",
            mime_type="text/plain",
            provenance="test",
        )
    with pytest.raises(ValueError, match="signature"):
        store.put(
            b"not-png",
            kind="image",
            mime_type="image/png",
            provenance="test",
        )
    limited = ContentStore(
        tmp_path / "limited.sqlite",
        tmp_path / "limited-content",
        profile_id="alpha",
        limits=MediaLimits(
            max_decompressed_bytes=20,
            max_duration_seconds=5,
            max_pages=2,
        ),
    )
    with pytest.raises(ContentLimitError, match="duration"):
        limited.put(
            b"audio",
            kind="audio",
            mime_type="audio/ogg",
            provenance="test",
            metadata={"duration_seconds": 6},
        )
    with pytest.raises(ContentLimitError, match="page"):
        limited.put(
            b"doc",
            kind="document",
            mime_type="text/plain",
            provenance="test",
            metadata={"page_count": 3},
        )
    item = store.put(
        b"hello",
        kind="document",
        mime_type="text/plain",
        provenance="test",
        retention=RetentionPolicy.TURN,
        retention_seconds=10,
    )
    assert store.read(item.content_ref) == b"hello"
    with pytest.raises(ContentOwnershipError):
        store.read(item.content_ref, profile_id="beta")

    path = tmp_path / "content" / item.content_ref.id.removeprefix("content:")
    path.write_bytes(b"HELLO")
    with pytest.raises(ContentIntegrityError):
        store.read(item.content_ref)
    path.write_bytes(b"hello")
    clock_value[0] = now + timedelta(seconds=11)
    assert store.sweep_expired() == (item.content_ref.id,)
    with pytest.raises(KeyError):
        store.get(item.content_ref)


def test_processor_registry_routes_local_and_hosted_transcription(tmp_path) -> None:
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )
    audio = store.put(
        b"audio",
        kind="audio",
        mime_type="audio/ogg",
        provenance="test",
    )
    local = LocalTranscriptionProcessor(lambda data, mime: f"{mime}:{data.decode()}")
    hosted = HostedTranscriptionProcessor(
        lambda data, mime: "hosted",
        provider="speech-cloud",
        pricing_per_minute=Decimal("0.01"),
    )
    registry = MediaProcessorRegistry((local, hosted))

    result = registry.transform(store, audio, instruction="transcribe")

    assert result.text == "audio/ogg:audio"
    assert result.transform is TransformKind.TRANSCRIPTION
    assert result.metadata["network_access"] is False


def test_text_only_client_rejects_untransformed_media() -> None:
    client = ActionClient()
    item = _media_item_for_contract()
    from chulk.media import ModelRequest

    with pytest.raises(UnsupportedMediaError):
        client.complete_action_request(
            ModelRequest(
                messages=({"role": "user", "content": "inspect"},),
                user_input=UserInput((MediaInputPart(item),)),
            )
        )


def test_optional_media_tools_separate_generation_transform_and_delivery(
    tmp_path,
) -> None:
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )

    class Generator:
        name = "fake_image"
        provider = "fixture"
        network_access = False
        retains_data = False

        def generate(self, prompt: str) -> GeneratedMedia:
            assert prompt == "draw"
            return GeneratedMedia(
                b"\x89PNG\r\n\x1a\nfixture",
                MediaKind.IMAGE,
                "image/png",
                "image.png",
            )

    generation = media_generation_tool(
        store,
        Generator(),
        tool_name="image_generation",
        transform=TransformKind.GENERATION,
    )
    generated = generation.callable({"prompt": "draw"})
    assert generated.success is True
    content_ref = generated.value["content_ref"]

    deliveries: list[tuple[str, str]] = []
    delivery = media_delivery_tool(
        store,
        lambda item, target: deliveries.append((item.content_ref.id, target))
        or {"delivered": True},
    )
    assert delivery.requires_confirmation is True
    receipt = delivery.callable(
        {"content_ref": content_ref, "target": "telegram:42"}
    )
    assert receipt.success is True
    assert deliveries == [(content_ref, "telegram:42")]

    document = store.put(
        b"extract me",
        kind="document",
        mime_type="text/plain",
        provenance="test",
    )
    transform = media_transform_tool(
        store,
        MediaProcessorRegistry((LocalTextExtractor(),)),
    )
    result = transform.callable(
        {
            "content_ref": document.content_ref.id,
            "instruction": "read",
            "transform": "extraction",
        }
    )
    assert result.observation == "extract me"


def test_agent_run_input_extracts_media_without_persisting_bytes(tmp_path) -> None:
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )
    item = store.put(
        b"confidential payload",
        kind="document",
        mime_type="text/plain",
        provenance="test",
        file_name="note.txt",
    )
    client = ActionClient()
    accounting = ModelUsageAccounting(
        SQLiteUsageStore(tmp_path / "store.sqlite"),
        client=client,
        dimensions=UsageDimensions(
            profile_id="alpha",
            conversation_id="conversation",
        ),
        budget=RunBudget(),
        max_output_tokens=100,
    )
    agent = Agent(
        client,
        profile_id="alpha",
        content_store=store,
        media_processors=MediaProcessorRegistry((LocalTextExtractor(),)),
        usage_accounting=accounting,
    )
    agent.state.conversation_id = "conversation"

    answer = agent.run_input(
        UserInput(
            (
                TextInputPart("Summarize"),
                MediaInputPart(item),
            )
        )
    )

    assert answer == "media ok"
    assert "confidential payload" in json.dumps(client.messages)
    persisted = json.dumps(agent.state.to_dict(), sort_keys=True)
    assert "confidential payload" not in persisted
    assert item.content_ref.id in persisted
    entries = SQLiteUsageStore(tmp_path / "store.sqlite").list_entries()
    assert any(entry.resource_kind.value == "media" for entry in entries)


def test_hosted_media_reserves_budget_before_network_processing(tmp_path) -> None:
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )
    item = store.put(
        b"audio",
        kind="audio",
        mime_type="audio/ogg",
        provenance="test",
    )
    calls: list[bytes] = []
    processor = HostedTranscriptionProcessor(
        lambda data, mime: calls.append(data) or "transcript",
        provider="hosted",
    )
    client = ActionClient()
    accounting = ModelUsageAccounting(
        SQLiteUsageStore(tmp_path / "store.sqlite"),
        client=client,
        dimensions=UsageDimensions(
            profile_id="alpha",
            conversation_id="conversation",
        ),
        budget=RunBudget(
            max_cost=ExactCost(Decimal("1"), pricing_known=True),
        ),
        max_output_tokens=100,
    )
    agent = Agent(
        client,
        profile_id="alpha",
        content_store=store,
        media_processors=MediaProcessorRegistry((processor,)),
        usage_accounting=accounting,
    )
    agent.state.conversation_id = "conversation"

    with pytest.raises(BudgetExceededError):
        agent.run_input(UserInput((MediaInputPart(item),)))

    assert calls == []


def test_repeated_media_parts_have_distinct_usage_events(tmp_path) -> None:
    store = ContentStore(
        tmp_path / "store.sqlite",
        tmp_path / "content",
        profile_id="alpha",
    )
    item = store.put(
        b"repeat",
        kind="document",
        mime_type="text/plain",
        provenance="test",
    )
    client = ActionClient()
    accounting = ModelUsageAccounting(
        SQLiteUsageStore(tmp_path / "store.sqlite"),
        client=client,
        dimensions=UsageDimensions(
            profile_id="alpha",
            conversation_id="conversation",
        ),
        budget=RunBudget(),
        max_output_tokens=100,
    )
    agent = Agent(
        client,
        profile_id="alpha",
        content_store=store,
        media_processors=MediaProcessorRegistry((LocalTextExtractor(),)),
        usage_accounting=accounting,
    )
    agent.state.conversation_id = "conversation"

    agent.run_input(
        UserInput((MediaInputPart(item), MediaInputPart(item)))
    )

    media_entries = [
        entry
        for entry in SQLiteUsageStore(tmp_path / "store.sqlite").list_entries()
        if entry.resource_kind.value == "media"
    ]
    assert len(media_entries) == 2
    assert media_entries[0].source_event_id != media_entries[1].source_event_id


def _media_item_for_contract():
    from chulk.media import ContentTrust, MediaItem

    return MediaItem(
        kind=MediaKind.IMAGE,
        mime_type="image/png",
        byte_length=8,
        content_ref=ContentRef("content:" + "a" * 32),
        sha256="0" * 64,
        provenance="test",
        trust=ContentTrust.UNTRUSTED,
    )
