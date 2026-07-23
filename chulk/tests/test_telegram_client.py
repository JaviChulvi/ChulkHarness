from __future__ import annotations

import pytest

from chulk.telegram.client import TELEGRAM_COMMANDS, TelegramClient, TelegramError, split_message


def test_get_updates_extracts_only_text_messages() -> None:
    calls: list[tuple[str, dict[str, object], float]] = []

    def request(url: str, payload: dict[str, object], timeout: float) -> object:
        calls.append((url, payload, timeout))
        return {
            "ok": True,
            "result": [
                {
                    "update_id": 7,
                    "message": {
                        "chat": {"id": 11, "type": "private"},
                        "from": {"id": 13},
                        "text": "hello",
                    },
                },
                {"update_id": 8, "message": {"chat": {"id": 11}, "photo": []}},
            ],
        }

    client = TelegramClient("top-secret", request_json=request)
    updates = client.get_updates(offset=7, timeout_seconds=30)

    assert [
        (item.update_id, item.chat_id, item.user_id, item.text, item.chat_type)
        for item in updates
    ] == [
        (7, 11, 13, "hello", "private")
    ]
    assert calls[0][1] == {
        "offset": 7,
        "timeout": 30,
        "allowed_updates": ["message"],
    }
    assert calls[0][2] >= 35
    assert client.next_offset == 9


def test_send_message_splits_long_responses() -> None:
    payloads: list[dict[str, object]] = []

    def request(_url: str, payload: dict[str, object], _timeout: float) -> object:
        payloads.append(payload)
        return {"ok": True, "result": {}}

    client = TelegramClient("secret", request_json=request)
    client.send_message(42, "abcdef")
    assert payloads == [{"chat_id": 42, "text": "abcdef"}]
    assert split_message("abcd\nefgh", limit=5) == ("abcd", "efgh")


def test_get_updates_normalizes_voice_and_photo_attachments() -> None:
    def request(_url: str, _payload: dict[str, object], _timeout: float) -> object:
        return {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {"id": 11, "type": "private"},
                        "from": {"id": 13},
                        "voice": {"file_id": "voice-1", "mime_type": "audio/ogg"},
                    },
                },
                {
                    "update_id": 2,
                    "message": {
                        "chat": {"id": 11, "type": "private"},
                        "from": {"id": 13},
                        "caption": "What is shown?",
                        "photo": [{"file_id": "small"}, {"file_id": "large"}],
                    },
                },
            ],
        }

    updates = TelegramClient("secret", request_json=request).get_updates(
        offset=None,
        timeout_seconds=1,
    )

    assert updates[0].attachment is not None
    assert (updates[0].attachment.kind, updates[0].attachment.file_id) == ("voice", "voice-1")
    assert updates[1].text == "What is shown?"
    assert updates[1].attachment is not None
    assert updates[1].attachment.file_id == "large"


def test_download_file_resolves_path_and_enforces_bound() -> None:
    calls: list[tuple[str, int]] = []

    def request(_url: str, payload: dict[str, object], _timeout: float) -> object:
        assert payload == {"file_id": "file-1"}
        return {"ok": True, "result": {"file_path": "voice/file.ogg"}}

    def binary(url: str, _timeout: float, max_bytes: int) -> bytes:
        calls.append((url, max_bytes))
        return b"audio"

    client = TelegramClient("secret", request_json=request, request_binary=binary)
    assert client.download_file("file-1", max_bytes=20) == b"audio"
    assert calls[0][0].endswith("/file/botsecret/voice/file.ogg")
    assert calls[0][1] == 20

    oversized = TelegramClient(
        "secret",
        request_json=request,
        request_binary=lambda *_args: b"x" * 21,
    )
    with pytest.raises(TelegramError, match="size limit"):
        oversized.download_file("file-1", max_bytes=20)


def test_chat_action_and_command_registration_use_bot_api_payloads() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def request(url: str, payload: dict[str, object], _timeout: float) -> object:
        calls.append((url.rsplit("/", 1)[-1], payload))
        return {"ok": True, "result": True}

    client = TelegramClient("secret", request_json=request)
    client.send_chat_action(42)
    client.set_commands()

    assert calls[0] == ("sendChatAction", {"chat_id": 42, "action": "typing"})
    assert calls[1] == (
        "setMyCommands",
        {
            "commands": [
                {"command": command, "description": description}
                for command, description in TELEGRAM_COMMANDS
            ]
        },
    )


def test_transport_failure_does_not_expose_token() -> None:
    def request(_url: str, _payload: dict[str, object], _timeout: float) -> object:
        raise RuntimeError("failure mentioning internal URL")

    client = TelegramClient("top-secret", request_json=request)
    with pytest.raises(TelegramError) as caught:
        client.get_updates(offset=None, timeout_seconds=1)

    assert "top-secret" not in str(caught.value)
